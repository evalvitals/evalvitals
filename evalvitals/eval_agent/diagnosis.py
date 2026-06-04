"""M3 — DiagnosisAgent: synthesize analysis findings into falsifiable hypotheses.

The agent reads the structured :class:`~evalvitals.eval_agent.analysis.AnalysisReport`
produced by M2, formats it as a prompt, sends it to an LLM judge (Gemini by
default), and parses the response into :class:`~evalvitals.eval_agent.hypothesis.Hypothesis`
objects.

The judge model defaults to **Gemini** when ``GEMINI_API_KEY`` is set in the
environment and no explicit judge is passed.

Usage::

    # Gemini default (set GEMINI_API_KEY in env)
    agent = DiagnosisAgent()
    diag  = agent.diagnose(analysis_report)

    # Explicit judge (any Model with Capability.GENERATE)
    agent = DiagnosisAgent(judge=my_model)
    diag  = agent.diagnose(analysis_report)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalvitals.eval_agent.hypothesis import Hypothesis

if TYPE_CHECKING:
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.analysis import AnalysisReport


_DIAGNOSE_PROMPT = """\
You are an expert ML diagnostician. Based on the analysis report below, propose
specific, falsifiable hypotheses about the root cause of the model's failures.
{prior_section}
Model: {model_name}
Overall severity: {severity}

Analysis narrative:
{narrative}

Raw findings (JSON):
{findings_json}

Propose 1-3 hypotheses. For each write exactly two lines:
HYPOTHESIS: <one-sentence falsifiable claim about the failure mode>
FAILURE_MODE: <short tag, e.g. attention_sink / hallucination / loop / low_consistency>

Only include hypotheses clearly supported by the analysis.
Do NOT repeat hypotheses already listed in the prior cycles above.
If the model looks healthy, respond with: NO_ISSUE"""


def _format_prior_section(prior_cycles: list[dict]) -> str:
    """Render prior cycle history as a context block for the diagnosis prompt."""
    if not prior_cycles:
        return ""
    lines = [
        "\nPrior investigation cycles — do NOT repeat these hypotheses; propose "
        "distinct or refined ones that address what remains unexplained:",
    ]
    for c in prior_cycles:
        lines.append(f"\nCycle {c['cycle']} (severity={c['severity']}):")
        for h in c.get("hypotheses", []):
            lines.append(
                f"  [{h['status'].upper()}] {h['statement']} "
                f"(failure_mode: {h['failure_mode']})"
            )
    return "\n".join(lines) + "\n"


@dataclass
class DiagnosisResult:
    """Output of :class:`DiagnosisAgent`.

    Attributes:
        model_name:       ``repr()`` of the analysed model.
        hypotheses:       Proposed :class:`~evalvitals.eval_agent.hypothesis.Hypothesis`
                          objects for M4.
        findings_summary: The findings dict forwarded to the judge.
        raw_judge_output: Verbatim LLM response (useful for debugging).
    """

    model_name: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    findings_summary: dict[str, Any] = field(default_factory=dict)
    raw_judge_output: str = ""


def _parse_hypotheses(raw: str, model_name: str) -> list[Hypothesis]:
    """Extract ``HYPOTHESIS:`` / ``FAILURE_MODE:`` pairs from LLM output."""
    hypotheses: list[Hypothesis] = []
    statement: str | None = None
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("HYPOTHESIS:"):
            statement = line[len("HYPOTHESIS:"):].strip()
        elif line.upper().startswith("FAILURE_MODE:") and statement:
            mode = line[len("FAILURE_MODE:"):].strip()
            hypotheses.append(
                Hypothesis(
                    statement=statement,
                    target_model=model_name,
                    predicted_failure_mode=mode,
                )
            )
            statement = None
    return hypotheses


def _default_judge() -> "Model":
    """Return a GeminiModel when GEMINI_API_KEY is available, else raise."""
    import os

    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError(
            "DiagnosisAgent requires a judge model. "
            "Either pass judge= explicitly, or set GEMINI_API_KEY to use the "
            "Gemini default (install with: pip install 'evalvitals[gemini]')."
        )
    from evalvitals.models.blackbox.gemini import GeminiModel

    return GeminiModel()


class DiagnosisAgent:
    """Proposes hypotheses from an :class:`~evalvitals.eval_agent.analysis.AnalysisReport`.

    Args:
        judge:    Any :class:`~evalvitals.core.model.Model` with
                  ``Capability.GENERATE``.  Defaults to a ``GeminiModel``
                  when ``GEMINI_API_KEY`` is in the environment.
        api_key:  Gemini API key — used only when *judge* is ``None`` and no
                  ``GEMINI_API_KEY`` env var is set.
        model_id: Gemini model identifier (default: ``"gemini-2.0-flash"``).
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        api_key: str | None = None,
        model_id: str = "gemini-2.0-flash",
    ) -> None:
        if judge is None and api_key is not None:
            from evalvitals.models.blackbox.gemini import GeminiModel

            judge = GeminiModel(model_id=model_id, api_key=api_key)
        self._judge = judge  # None → resolved lazily on first call

    @property
    def judge(self) -> "Model":
        if self._judge is None:
            self._judge = _default_judge()
        return self._judge

    @judge.setter
    def judge(self, value: "Model") -> None:
        self._judge = value

    def diagnose(
        self,
        analysis: "AnalysisReport | dict[str, Result]",
        model_name: str = "",
        prior_cycles: list[dict] | None = None,
    ) -> DiagnosisResult:
        """Synthesize *analysis* into a set of falsifiable hypotheses.

        Args:
            analysis:     An :class:`~evalvitals.eval_agent.analysis.AnalysisReport`
                          from M2, or (for backward compatibility) a plain
                          ``{analyzer_name: Result}`` dict.
            model_name:   Ignored when *analysis* is an ``AnalysisReport``
                          (the name is taken from the report).
            prior_cycles: Summary of previous M1→M4 cycles produced by
                          :class:`~evalvitals.eval_agent.loop.AutoDiagnoseLoop`.
                          Each entry is ``{"cycle": int, "severity": str,
                          "hypotheses": [{"statement", "failure_mode", "status"}]}``.
                          Injected into the prompt so the judge avoids re-proposing
                          already-tested hypotheses.

        Returns:
            :class:`DiagnosisResult` with zero or more hypotheses.
        """
        from evalvitals.eval_agent.analysis import AnalysisModule, AnalysisReport

        if not isinstance(analysis, AnalysisReport):
            # Backward compat: wrap raw results in a minimal AnalysisReport.
            analysis = AnalysisModule().analyze(analysis, model_name)

        summary = {name: r.findings for name, r in analysis.raw_results.items()}
        prompt = _DIAGNOSE_PROMPT.format(
            prior_section=_format_prior_section(prior_cycles or []),
            model_name=analysis.model_name or model_name,
            severity=analysis.severity,
            narrative=analysis.narrative,
            findings_json=json.dumps(summary, indent=2, default=str),
        )
        raw = self.judge.generate(prompt)
        hypotheses = _parse_hypotheses(str(raw), analysis.model_name or model_name)
        return DiagnosisResult(
            model_name=analysis.model_name or model_name,
            hypotheses=hypotheses,
            findings_summary=summary,
            raw_judge_output=str(raw),
        )
