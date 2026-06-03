"""M3 — DiagnosisAgent: synthesize analysis findings into falsifiable hypotheses.

The agent forwards structured findings to an LLM judge, which proposes
hypotheses in a structured format the parser can extract.  No rule-based
fallback — the judge model is required.

Usage::

    agent = DiagnosisAgent(judge=my_model)
    diag  = agent.diagnose(results, model_name="qwen3-vl-8b")
    for h in diag.hypotheses:
        print(h.statement, "→", h.predicted_failure_mode)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalvitals.eval_agent.hypothesis import Hypothesis

if TYPE_CHECKING:
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result


_DIAGNOSE_PROMPT = """\
You are diagnosing why a model is failing. Analyze the evaluation findings \
below and propose specific, falsifiable hypotheses about the root cause.

Model: {model_name}
Findings (JSON):
{findings_json}

Propose 1-3 hypotheses. For each write exactly two lines:
HYPOTHESIS: <one-sentence falsifiable claim about the failure mode>
FAILURE_MODE: <short tag, e.g. attention_sink / hallucination / loop / low_consistency>

Only include hypotheses clearly supported by the findings.
If the model looks healthy, respond with: NO_ISSUE"""


@dataclass
class DiagnosisResult:
    """Output of :class:`DiagnosisAgent`.

    Attributes:
        model_name:       ``repr()`` of the analysed model.
        hypotheses:       Proposed :class:`~evalvitals.eval_agent.hypothesis.Hypothesis`
                          objects, ready to be handed to the survey agent.
        findings_summary: The ``{analyzer: findings}`` dict forwarded to the judge.
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


class DiagnosisAgent:
    """Proposes hypotheses by asking an LLM judge to read the analysis findings.

    Args:
        judge: Any :class:`~evalvitals.core.model.Model` with
               ``Capability.GENERATE``.  This is typically a capable
               instruction-following model (e.g. a Claude API model or a local
               chat model) rather than the model under evaluation.
    """

    def __init__(self, judge: "Model") -> None:
        self.judge = judge

    def diagnose(
        self,
        results: dict[str, "Result"],
        model_name: str,
    ) -> DiagnosisResult:
        """Synthesize *results* into a set of falsifiable hypotheses.

        Args:
            results:    ``{analyzer_name: Result}`` from the executor (M2).
            model_name: Human-readable name or ``repr()`` of the analysed model.

        Returns:
            :class:`DiagnosisResult` with zero or more hypotheses.
        """
        summary = {name: r.findings for name, r in results.items()}
        prompt = _DIAGNOSE_PROMPT.format(
            model_name=model_name,
            findings_json=json.dumps(summary, indent=2, default=str),
        )
        raw = self.judge.generate(prompt)
        hypotheses = _parse_hypotheses(str(raw), model_name)
        return DiagnosisResult(
            model_name=model_name,
            hypotheses=hypotheses,
            findings_summary=summary,
            raw_judge_output=str(raw),
        )
