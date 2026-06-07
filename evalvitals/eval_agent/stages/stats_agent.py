"""M2 — StatsAnalysisAgent: statistically analyze probe results.

Extends :class:`~evalvitals.eval_agent.analysis.AnalysisModule` with:

- **Protocol-aware interpretation**: focuses the analysis narrative on what
  the user's experiment was actually testing.
- **Richer output** (:class:`StatsAnalysisReport`): adds ``conclusion``,
  ``evidence_chain``, and ``qualitative_findings`` on top of the base
  :class:`~evalvitals.eval_agent.analysis.AnalysisReport` fields so M3
  (DiagnosisAgent) receives richer context.
- **LLM-guided path** (``judge=`` set): the LLM synthesizes a protocol-specific
  narrative from the raw findings.  The threshold-rules pass always runs first;
  the LLM *enriches*, not replaces, its output.

Two analysis paths:

a. **Basic** (default, no judge): apply built-in threshold rules + build
   conclusion/evidence_chain from the flagged anomalies.  Fast, deterministic,
   no API key needed.

b. **LLM-guided** (``judge=`` + ``protocol=`` at call time): judge reads the
   protocol + raw findings and writes a protocol-specific conclusion.
   Falls back to the basic path if the LLM call fails.

:class:`StatsAnalysisReport` inherits from :class:`AnalysisReport` so it is
accepted everywhere an ``AnalysisReport`` is expected (e.g. ``DiagnosisAgent``).

Usage::

    # Basic path
    agent  = StatsAnalysisAgent()
    report = agent.analyze(probe_results, model_name="qwen3-vl-8b")

    # Protocol-guided (richer conclusion)
    agent  = StatsAnalysisAgent(judge=gemini)
    report = agent.analyze(probe_results, protocol=protocol)
    print(report.conclusion)
    print(report.evidence_chain)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalvitals.eval_agent.stages.analysis import AnalysisModule, AnalysisReport
from evalvitals.eval_agent.stages.stats_tool_agent import StatsToolAgent as LegacyStatsToolAgent
from evalvitals.eval_agent.stages.stats_tools import (
    STATS_TOOL_CATALOG,
    StatsInput,
    StatsToolResult,
    build_stats_input,
    default_plan,
    describe_data,
    fdr_correct,
    has_testable_data,
    plot_effects,
    run_stats_tool,
)

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template (LLM-guided path)
# ---------------------------------------------------------------------------

_ANALYSIS_PROMPT = """\
You are an expert in ML failure analysis for vision-language models and agentic systems.

Experiment protocol:
{protocol_text}

Task domain: {task_domain}

Analyzer summary:
{narrative}

Raw per-analyzer findings (JSON):
{findings_json}

Based on the protocol and the findings above, write:

CONCLUSION: <one paragraph — what is the root cause of failures, given what was tested>
EVIDENCE_CHAIN:
- <step 1: which analyzer and metric first caught your attention, and why>
- <step 2: how it connects to the protocol's stated failure patterns>
- <step 3: any corroborating or contradicting signals from other analyzers>
QUALITATIVE:
- <observation 1: a pattern not captured by numbers alone>
- <observation 2: anything unexpected or surprising>

Keep each bullet to one sentence. If the model looks healthy given the protocol, say so clearly."""


# ---------------------------------------------------------------------------
# Prompt template (stats-tool selection)
# ---------------------------------------------------------------------------

_TOOL_SELECT_PROMPT = """\
You are selecting statistical tools to test why a model fails, given the data on hand.

Experiment protocol:
{protocol_text}

Available statistical tools:
{tool_catalog}

Data shape available for testing:
{data_shape}

Pick the tools whose data requirements are satisfied and that best test the \
protocol's question. Return ONLY a JSON object, no other text:
{{"tools": ["name1", "name2", ...], "rationale": "one sentence"}}"""


def _select_tools_via_llm(
    judge: "Model",
    protocol: "ExperimentProtocol",
    inp: StatsInput,
    valid_names: set[str],
) -> set[str]:
    """Ask the judge to choose tool names from the catalog. Returns a name set.

    Raises on any failure so the caller can fall back to the deterministic plan.
    """
    import inspect

    catalog = "\n".join(f"  - {name}: {desc}" for name, desc in STATS_TOOL_CATALOG.items())
    prompt = _TOOL_SELECT_PROMPT.format(
        protocol_text=protocol.description,
        tool_catalog=catalog,
        data_shape=json.dumps(describe_data(inp), indent=2),
    )
    sig = inspect.signature(judge.generate)
    if "temperature" in sig.parameters:
        raw = judge.generate(prompt, temperature=0)
    else:
        raw = judge.generate(prompt)

    cleaned = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.DOTALL).strip()
    match = re.search(r"\{[^{}]*\}", cleaned)
    if match:
        data = json.loads(match.group())
        names = {n.lower() for n in data.get("tools", []) if n.lower() in valid_names}
        if names:
            return names
    # soft fallback: scan for any known tool name
    low = cleaned.lower()
    return {n for n in valid_names if n in low}


def _parse_llm_analysis(
    raw: str,
    base: AnalysisReport,
) -> tuple[str, list[str], list[str]]:
    """Parse LLM output into (conclusion, evidence_chain, qualitative_findings).

    Falls back to the base narrative/findings if a section is missing.
    """
    conclusion = ""
    evidence: list[str] = []
    qualitative: list[str] = []
    section: str | None = None

    for line in raw.splitlines():
        s = line.strip()
        upper = s.upper()
        if upper.startswith("CONCLUSION:"):
            conclusion = s[len("CONCLUSION:"):].strip()
            section = "conclusion"
        elif upper.startswith("EVIDENCE_CHAIN:"):
            section = "evidence"
        elif upper.startswith("QUALITATIVE:"):
            section = "qualitative"
        elif s.startswith("- "):
            content = s[2:].strip()
            if section == "evidence":
                evidence.append(content)
            elif section == "qualitative":
                qualitative.append(content)
        elif section == "conclusion" and s and not upper.startswith(("EVIDENCE", "QUALITATIVE")):
            conclusion = (conclusion + " " + s).strip()

    if not conclusion:
        first_line = base.narrative.split("\n")[0] if base.narrative else ""
        conclusion = first_line or "No conclusion produced."

    return conclusion, evidence, qualitative


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class StatsAnalysisReport(AnalysisReport):
    """M2 output: a richer, protocol-aware analysis report.

    Inherits all :class:`~evalvitals.eval_agent.analysis.AnalysisReport`
    fields so it is accepted everywhere that type is expected.

    Additional attributes:
        conclusion:             NL paragraph summarising what was found.
        evidence_chain:         Step-by-step derivation of the conclusion.
        qualitative_findings:   Patterns or anomalies noted in free text.
        stats_tool:             ``"threshold_rules"`` (basic), ``"llm_guided"``
                                (basic + LLM narrative), or ``"selected_tools"``
                                (statistical tools from the catalog were run).
        stats_results:          Per-tool :class:`StatsToolResult` verdicts.
        stats_plan:             ``[{tool, config, rationale}]`` — which tools
                                were selected/generated and why (M2's analog of
                                M1's ProbingSchema).
        corrected_rejections:   e-BH FDR correction across all tool e-values.
        figures:                Paths to generated figures (forest plot, …).
        stats_tool_results:     Backward-compatible JSON-safe stats summaries.
        visualizations:         Backward-compatible figure/spec list.
        protocol:               The protocol that guided this analysis, if any.
    """

    conclusion: str = ""
    evidence_chain: list[str] = field(default_factory=list)
    qualitative_findings: list[str] = field(default_factory=list)
    stats_tool: str = "threshold_rules"
    stats_results: list[StatsToolResult] = field(default_factory=list)
    stats_plan: list[dict[str, Any]] = field(default_factory=list)
    corrected_rejections: dict[str, Any] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    stats_tool_results: list[dict[str, Any]] = field(default_factory=list)
    visualizations: list[dict[str, Any]] = field(default_factory=list)
    protocol: "ExperimentProtocol | None" = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "conclusion": self.conclusion,
            "evidence_chain": self.evidence_chain,
            "qualitative_findings": self.qualitative_findings,
            "stats_tool": self.stats_tool,
            "stats_results": [r.to_dict() for r in self.stats_results],
            "stats_plan": self.stats_plan,
            "corrected_rejections": self.corrected_rejections,
            "figures": self.figures,
            "stats_tool_results": self.stats_tool_results,
            "visualizations": self.visualizations,
        })
        return d


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class StatsAnalysisAgent:
    """M2: statistically analyze probe results guided by the experiment protocol.

    Args:
        judge:       Optional LLM for the LLM-guided analysis path.
                     Any :class:`~evalvitals.core.model.Model` with
                     ``Capability.GENERATE``.  Pass ``None`` to use
                     threshold rules only.
        extra_rules: Additional ``{analyzer_name: [_Rule, …]}`` entries
                     merged with the built-in rules.
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        extra_rules: dict | None = None,
        stats_tool_agent: Any | None = None,
        enable_stats_tools: bool = True,
        figure_dir: "str | None" = None,
        max_signal_tools: int = 4,
        allow_codegen: bool = False,
    ) -> None:
        self._base = AnalysisModule(extra_rules)
        self._judge = judge
        self._stats_tool_agent = stats_tool_agent or LegacyStatsToolAgent()
        self._enable_stats_tools = enable_stats_tools
        self._figure_dir = figure_dir
        self._max_signal_tools = max_signal_tools
        # Placeholder for the deferred tier-(b) code-generation path.
        self._allow_codegen = allow_codegen

    def analyze(
        self,
        results: "dict[str, Result]",
        model_name: str = "",
        protocol: "ExperimentProtocol | None" = None,
        data: "CaseBatch | None" = None,
    ) -> StatsAnalysisReport:
        """Analyze *results* into a :class:`StatsAnalysisReport`.

        Args:
            results:    ``{analyzer_name: Result}`` dict from M1.
            model_name: Human-readable model identifier.
            protocol:   Experiment protocol for context-aware interpretation.
                        When ``None``, the analysis is protocol-agnostic.
            data:       Labeled cases from M1.  Required to run the statistical
                        tool layer (per-case fail/pass labels).  When ``None``
                        or unlabeled, M2 falls back to threshold rules only.

        Returns:
            :class:`StatsAnalysisReport` — backward-compatible with
            :class:`~evalvitals.eval_agent.analysis.AnalysisReport`.
        """
        base = self._base.analyze(results, model_name)
        legacy_tool_results = self._run_legacy_stats_tools(results, protocol)

        # ── Statistical tool layer (select → run → FDR-correct → plot) ──
        stats_results, stats_plan, corrected, figures = self._run_stats_tools(
            results, data, protocol
        )

        if self._judge is not None and protocol is not None:
            try:
                return self._analyze_llm_guided(
                    base, results, protocol,
                    stats_results, stats_plan, corrected, figures,
                    legacy_tool_results,
                )
            except Exception as exc:
                logger.warning("LLM-guided M2 analysis failed, falling back: %s", exc)

        return self._to_stats_report(
            base, protocol, stats_results, stats_plan, corrected, figures,
            legacy_tool_results,
        )

    # ------------------------------------------------------------------
    # Statistical tool layer
    # ------------------------------------------------------------------

    def _run_stats_tools(
        self,
        results: "dict[str, Result]",
        data: "CaseBatch | None",
        protocol: "ExperimentProtocol | None",
    ) -> tuple[list[StatsToolResult], list[dict[str, Any]], dict[str, Any], list[str]]:
        """Build input, select and run tools, FDR-correct, and plot.

        Returns ``(stats_results, stats_plan, corrected, figures)``.  All empty
        when there is no labeled/grouped data to test (pure backward compat).
        """
        inp = build_stats_input(results, data)
        if not self._enable_stats_tools or not has_testable_data(inp):
            return [], [], {}, []

        plan = self._select_plan(inp, protocol)

        stats_results: list[StatsToolResult] = []
        for tool, cfg, rationale in plan:
            try:
                r = run_stats_tool(tool, inp, cfg)
            except Exception as exc:  # never let one tool sink the analysis
                logger.warning("stats tool %r failed: %s", tool, exc)
                r = StatsToolResult(
                    tool=tool, config=cfg, ok=False,
                    error=str(exc), summary=f"{tool}: {exc}",
                )
            r.details.setdefault("rationale", rationale)
            stats_results.append(r)

        corrected = fdr_correct(stats_results)

        figures: list[str] = []
        if self._figure_dir:
            fig = plot_effects(
                stats_results, os.path.join(self._figure_dir, "m2_effects.png")
            )
            if fig:
                figures.append(fig)

        stats_plan = [
            {"tool": t, "config": c, "rationale": rat} for t, c, rat in plan
        ]
        return stats_results, stats_plan, corrected, figures

    def _run_legacy_stats_tools(
        self,
        results: "dict[str, Result]",
        protocol: "ExperimentProtocol | None",
    ) -> list[Any]:
        """Run the older lightweight M2 tools for backward-compatible reports."""
        if not self._enable_stats_tools or self._stats_tool_agent is None:
            return []
        try:
            return list(self._stats_tool_agent.analyze(results, protocol=protocol))
        except Exception as exc:
            logger.warning("legacy M2 stats-tool analysis failed: %s", exc)
            return []

    def _select_plan(
        self,
        inp: StatsInput,
        protocol: "ExperimentProtocol | None",
    ) -> list[tuple[str, dict, str]]:
        """Select tools: LLM picks from the catalog, deterministic plan otherwise.

        The deterministic plan is always the backbone (it guarantees coverage);
        when a judge + protocol are present, the LLM *narrows* it to the chosen
        tools.  Any failure or empty selection falls back to the full plan.
        """
        plan = default_plan(inp, max_signals=self._max_signal_tools)
        if self._judge is None or protocol is None or not plan:
            return plan
        try:
            chosen = _select_tools_via_llm(
                self._judge, protocol, inp, set(STATS_TOOL_CATALOG)
            )
        except Exception as exc:
            logger.debug("LLM stats-tool selection failed, using default plan: %s", exc)
            return plan
        if not chosen:
            return plan
        filtered = [p for p in plan if p[0] in chosen]
        return filtered or plan

    # ------------------------------------------------------------------
    # Report assembly
    # ------------------------------------------------------------------

    def _to_stats_report(
        self,
        base: AnalysisReport,
        protocol: "ExperimentProtocol | None",
        stats_results: list[StatsToolResult] | None = None,
        stats_plan: list[dict[str, Any]] | None = None,
        corrected: dict[str, Any] | None = None,
        figures: list[str] | None = None,
        legacy_tool_results: list[Any] | None = None,
    ) -> StatsAnalysisReport:
        """Basic path: wrap AnalysisReport + stats-tool results into a report."""
        stats_results = stats_results or []
        stats_plan = stats_plan or []
        corrected = corrected or {}
        figures = figures or []
        legacy_tool_results = legacy_tool_results or []

        chain: list[str] = []
        if base.findings:
            chain.append(
                f"Applied threshold rules across {len(base.raw_results)} "
                f"analyzer(s): {', '.join(sorted(base.raw_results))}"
            )
            for f in base.findings:
                cmp = ">" if f.direction == "above" else "<"
                chain.append(
                    f"  [{f.severity.upper()}] {f.analyzer}.{f.metric}="
                    f"{f.value:.3g} {cmp} {f.threshold}: {f.message}"
                )
            chain.append(f"Overall severity: {base.severity}")
        elif base.raw_results:
            chain.append(
                f"All metrics within normal ranges across "
                f"{len(base.raw_results)} analyzer(s)."
            )

        # Append statistical verdicts to the evidence chain.
        for r in stats_results:
            if r.ok:
                chain.append(f"[stats:{r.tool}] {r.summary}")
            else:
                chain.append(f"[stats:{r.tool}] N/A ({r.error})")
        if corrected.get("rejected_tools"):
            chain.append(
                f"FDR (e-BH) survivors: {', '.join(corrected['rejected_tools'])}"
            )

        qualitative = [str(f) for f in base.findings]
        conclusion = _build_conclusion(base, protocol, stats_results)
        stats_tool = "selected_tools" if stats_results else "threshold_rules"

        return StatsAnalysisReport(
            # AnalysisReport fields
            model_name=base.model_name,
            findings=base.findings,
            severity=base.severity,
            narrative=base.narrative,
            raw_results=base.raw_results,
            # StatsAnalysisReport extras
            conclusion=conclusion,
            evidence_chain=chain,
            qualitative_findings=qualitative,
            stats_tool=stats_tool,
            stats_results=stats_results,
            stats_plan=stats_plan,
            corrected_rejections=corrected,
            figures=figures,
            stats_tool_results=(
                _serialize_legacy_tool_results(legacy_tool_results)
                or _compat_stats_tool_results(stats_results)
            ),
            visualizations=(
                _legacy_visualizations(legacy_tool_results)
                or [{"type": "figure", "path": p} for p in figures]
            ),
            protocol=protocol,
        )

    def _analyze_llm_guided(
        self,
        base: AnalysisReport,
        results: "dict[str, Result]",
        protocol: "ExperimentProtocol",
        stats_results: list[StatsToolResult],
        stats_plan: list[dict[str, Any]],
        corrected: dict[str, Any],
        figures: list[str],
        legacy_tool_results: list[Any],
    ) -> StatsAnalysisReport:
        """LLM-guided path: judge synthesises protocol-specific insights.

        The judge now also sees the statistical verdicts so its conclusion is
        grounded in effect sizes + e-values, not just raw findings.
        """
        findings_json = json.dumps(
            {r: res.findings for r, res in results.items()},
            indent=2,
            default=str,
        )
        stats_block = _format_stats_for_prompt(stats_results, corrected)
        prompt = _ANALYSIS_PROMPT.format(
            protocol_text=protocol.description,
            task_domain=protocol.task_domain or "general",
            narrative=base.narrative + stats_block,
            findings_json=findings_json,
        )
        raw = self._judge.generate(prompt)  # type: ignore[union-attr]
        conclusion, evidence, qualitative = _parse_llm_analysis(str(raw), base)

        report = self._to_stats_report(
            base, protocol, stats_results, stats_plan, corrected, figures,
            legacy_tool_results,
        )
        if conclusion:
            report.conclusion = conclusion
        if evidence:
            report.evidence_chain = evidence
        if qualitative:
            report.qualitative_findings = qualitative
        report.stats_tool = "llm_guided"
        return report


def _build_conclusion(
    base: AnalysisReport,
    protocol: "ExperimentProtocol | None",
    stats_results: list[StatsToolResult] | None = None,
) -> str:
    """Build a short NL conclusion from the base report + stats + optional protocol."""
    domain = f"For {protocol.task_domain} tasks: " if (protocol and protocol.task_domain) else ""

    # Prefer a statistically supported verdict when one exists.
    supported = [
        r for r in (stats_results or [])
        if r.ok and r.reject and r.effect is not None
    ]
    stat_msg = ""
    if supported:
        top = max(supported, key=lambda r: abs(r.effect or 0.0))
        stat_msg = f" Statistically supported: {top.summary}"

    if not base.findings:
        if stat_msg:
            return f"{domain}No threshold violations, but a statistical test fired.{stat_msg}"
        msg = "No threshold violations detected; all metrics within normal ranges."
        return f"{domain}{msg}"

    top_f = base.findings[0]
    msg = (
        f"Primary anomaly: {top_f.message} "
        f"({top_f.analyzer}.{top_f.metric}={top_f.value:.3g}, severity={top_f.severity})."
    )
    if len(base.findings) > 1:
        msg += f" {len(base.findings) - 1} additional finding(s) also flagged."
    return f"{domain}{msg}{stat_msg}"


def _compat_stats_tool_results(stats_results: list[StatsToolResult]) -> list[dict[str, Any]]:
    """Return the legacy JSON shape used by older logs/examples."""
    out: list[dict[str, Any]] = []
    for result in stats_results:
        payload = result.to_dict()
        payload["name"] = result.tool
        payload["conclusion"] = result.summary or result.error or ""
        payload["visualizations"] = (
            [{"type": "figure", "path": result.figure_path}]
            if result.figure_path
            else []
        )
        out.append(payload)
    return out


def _serialize_legacy_tool_results(tool_results: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for result in tool_results:
        if hasattr(result, "to_dict"):
            payloads.append(result.to_dict())
        elif isinstance(result, dict):
            payloads.append(dict(result))
    return payloads


def _legacy_visualizations(tool_results: list[Any]) -> list[dict[str, Any]]:
    visualizations: list[dict[str, Any]] = []
    for result in tool_results:
        visualizations.extend(getattr(result, "visualizations", []) or [])
    return visualizations


def _format_stats_for_prompt(
    stats_results: list[StatsToolResult],
    corrected: dict[str, Any],
) -> str:
    """Render statistical verdicts as a block appended to the LLM narrative."""
    if not stats_results:
        return ""
    lines = ["", "Statistical test results (effect-sized, FDR-aware):"]
    for r in stats_results:
        if r.ok:
            lines.append(f"  - {r.summary}")
        else:
            lines.append(f"  - {r.tool}: not run ({r.error})")
    if corrected.get("rejected_tools"):
        lines.append(
            f"  After e-BH FDR correction, surviving tools: "
            f"{', '.join(corrected['rejected_tools'])}"
        )
    return "\n".join(lines)
