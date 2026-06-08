"""M2→M3 handoff: DiagnosisAgent must consume M2's LLM conclusion + evidence
chain + statistical verdicts, not just the threshold narrative.

Regression for the gap where an analyzer surfaced a real failure mode (rich M2
conclusion) but M3 saw only "no anomalies" (severity=none) and returned 0
hypotheses.
"""

from __future__ import annotations

from evalvitals.core.capability import Capability
from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
from evalvitals.eval_agent.stages.stats_agent import StatsAnalysisReport
from evalvitals.eval_agent.stages.stats_tools import StatsToolResult
from tests.conftest import FakeModel


def _stats_report_with_conclusion() -> StatsAnalysisReport:
    # severity=none and no threshold findings, but a rich LLM conclusion +
    # evidence + a statistical verdict — exactly the relative_attention case.
    return StatsAnalysisReport(
        model_name="vlm",
        findings=[],
        severity="none",
        narrative="No anomalies detected — all metrics within normal ranges.",
        raw_results={},
        conclusion="The model ignores the image and answers from language priors.",
        evidence_chain=[
            "relative_attention max weight only 1.69x (near-uniform)",
            "diffuse attention connects to the counting/colour errors",
        ],
        stats_results=[
            StatsToolResult(tool="signal_label_assoc", ok=True,
                            summary="signal vs FAIL: effect=+0.80 -> REJECT H0",
                            effect=0.8, reject=True),
        ],
    )


class CapturingJudge(FakeModel):
    def __init__(self) -> None:
        super().__init__(capabilities={Capability.GENERATE})
        self.prompts: list[str] = []

    def generate(self, inputs, **kw) -> str:
        # diagnose() makes a second adversarial-validation call; record all and
        # let tests inspect the first (the diagnosis prompt).
        self.prompts.append(str(inputs))
        return "HYPOTHESIS: model fails to ground answers in the image\nFAILURE_MODE: weak_visual_grounding"


def test_prompt_includes_conclusion_evidence_and_stats():
    judge = CapturingJudge()
    DiagnosisAgent(judge=judge).diagnose(_stats_report_with_conclusion())
    p = judge.prompts[0]
    assert "ignores the image and answers from language priors" in p   # conclusion
    assert "near-uniform" in p                                          # evidence chain
    assert "REJECT H0" in p                                             # stats verdict
    # The prompt must steer M3 away from trusting threshold severity alone.
    assert "threshold severity" in p.lower()


def test_hypothesis_generated_despite_severity_none():
    # The whole point: a real hypothesis comes out even though severity=none.
    judge = CapturingJudge()
    diag = DiagnosisAgent(judge=judge).diagnose(_stats_report_with_conclusion())
    assert len(diag.hypotheses) == 1
    assert diag.hypotheses[0].predicted_failure_mode == "weak_visual_grounding"
