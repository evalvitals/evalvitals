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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from evalvitals.eval_agent.stages.analysis import AnalysisModule, AnalysisReport

if TYPE_CHECKING:
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
        stats_tool:             ``"threshold_rules"`` or ``"llm_guided"``.
        protocol:               The protocol that guided this analysis, if any.
    """

    conclusion: str = ""
    evidence_chain: list[str] = field(default_factory=list)
    qualitative_findings: list[str] = field(default_factory=list)
    stats_tool: Literal["threshold_rules", "llm_guided"] = "threshold_rules"
    protocol: "ExperimentProtocol | None" = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "conclusion": self.conclusion,
            "evidence_chain": self.evidence_chain,
            "qualitative_findings": self.qualitative_findings,
            "stats_tool": self.stats_tool,
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
    ) -> None:
        self._base = AnalysisModule(extra_rules)
        self._judge = judge

    def analyze(
        self,
        results: "dict[str, Result]",
        model_name: str = "",
        protocol: "ExperimentProtocol | None" = None,
    ) -> StatsAnalysisReport:
        """Analyze *results* into a :class:`StatsAnalysisReport`.

        Args:
            results:    ``{analyzer_name: Result}`` dict from M1.
            model_name: Human-readable model identifier.
            protocol:   Experiment protocol for context-aware interpretation.
                        When ``None``, the analysis is protocol-agnostic.

        Returns:
            :class:`StatsAnalysisReport` — backward-compatible with
            :class:`~evalvitals.eval_agent.analysis.AnalysisReport`.
        """
        base = self._base.analyze(results, model_name)

        if self._judge is not None and protocol is not None:
            try:
                return self._analyze_llm_guided(base, results, protocol)
            except Exception as exc:
                logger.warning("LLM-guided M2 analysis failed, falling back: %s", exc)

        return self._to_stats_report(base, protocol)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_stats_report(
        self,
        base: AnalysisReport,
        protocol: "ExperimentProtocol | None",
    ) -> StatsAnalysisReport:
        """Basic path: wrap AnalysisReport into StatsAnalysisReport."""
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

        qualitative = [str(f) for f in base.findings]
        conclusion = _build_conclusion(base, protocol)

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
            stats_tool="threshold_rules",
            protocol=protocol,
        )

    def _analyze_llm_guided(
        self,
        base: AnalysisReport,
        results: "dict[str, Result]",
        protocol: "ExperimentProtocol",
    ) -> StatsAnalysisReport:
        """LLM-guided path: judge synthesises protocol-specific insights."""
        findings_json = json.dumps(
            {r: res.findings for r, res in results.items()},
            indent=2,
            default=str,
        )
        prompt = _ANALYSIS_PROMPT.format(
            protocol_text=protocol.description,
            task_domain=protocol.task_domain or "general",
            narrative=base.narrative,
            findings_json=findings_json,
        )
        raw = self._judge.generate(prompt)  # type: ignore[union-attr]
        conclusion, evidence, qualitative = _parse_llm_analysis(str(raw), base)

        report = self._to_stats_report(base, protocol)
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
) -> str:
    """Build a short NL conclusion from the base report + optional protocol."""
    domain = f"For {protocol.task_domain} tasks: " if (protocol and protocol.task_domain) else ""

    if not base.findings:
        msg = "No threshold violations detected; all metrics within normal ranges."
        return f"{domain}{msg}"

    top = base.findings[0]
    msg = (
        f"Primary anomaly: {top.message} "
        f"({top.analyzer}.{top.metric}={top.value:.3g}, severity={top.severity})."
    )
    if len(base.findings) > 1:
        msg += f" {len(base.findings) - 1} additional finding(s) also flagged."
    return f"{domain}{msg}"
