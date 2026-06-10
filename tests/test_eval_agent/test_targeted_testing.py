"""P3+P4: hypothesis test designs, evidence routing, and depth-tiered stopping.

P3 — M3 attaches a ``test_design`` to each hypothesis; M5 routes evidence by it
(deterministic) before falling back to keywords; M1 folds the designs into
cycle-2 analyzer selection.
P4 — verdicts carry an ``evidence_grade`` (intervention > observational) and
``stopping_criteria_met`` can require intervention-grade evidence.
"""

from __future__ import annotations

import pytest

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.eval_agent import HypothesisTester, ProbeAgent
from evalvitals.eval_agent.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    hypothesis_from_dict,
    hypothesis_to_dict,
)
from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
from evalvitals.eval_agent.stages.hypothesis_tester import (
    HypothesisTestResult,
    _evidence_grade,
)
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol
from evalvitals.eval_agent.stages.stats_agent import StatsAnalysisReport
from evalvitals.eval_agent.stages.stats_tools import StatsToolResult
from tests.conftest import FakeModel

# ── fixtures ────────────────────────────────────────────────────────────────


def _labeled_batch() -> CaseBatch:
    cases = [FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"q{i}"),
                         label=Label.FAIL if i < 2 else Label.PASS) for i in range(4)]
    return CaseBatch(cases)


def _report() -> StatsAnalysisReport:
    """Three signal tools: decisive pope FN, non-decisive attention, decisive
    intervention-derived prompt_contrast repair flag."""
    return StatsAnalysisReport(
        model_name="m",
        findings=[],
        severity="none",
        narrative="",
        raw_results={},
        conclusion="c",
        stats_results=[
            StatsToolResult(tool="signal_label_assoc", ok=True, effect=0.8,
                            ci=(0.5, 1.0), reject=True,
                            config={"signal": "pope.false_negative"},
                            summary="pope.false_negative vs FAIL"),
            StatsToolResult(tool="signal_label_assoc", ok=True, effect=-0.3,
                            ci=(-0.8, 0.2), reject=False,
                            config={"signal": "relative_attention.max_relative_weight"},
                            summary="relative_attention.max_relative_weight vs FAIL"),
            StatsToolResult(tool="signal_label_assoc", ok=True, effect=0.6,
                            ci=(0.4, 0.9), reject=True,
                            config={"signal": "prompt_contrast.fixed_by_describe_first"},
                            summary="prompt_contrast.fixed_by_describe_first vs FAIL"),
        ],
    )


def _hyp(statement: str, mode: str = "mode", design: str = "") -> Hypothesis:
    return Hypothesis(statement=statement, target_model="m",
                      predicted_failure_mode=mode, test_design=design)


# ── evidence grading (P4) ───────────────────────────────────────────────────


def test_evidence_grade_unit():
    assert _evidence_grade("mcnemar_evalue", "") == "intervention"
    assert _evidence_grade("friedman_nemenyi", "") == "intervention"
    assert _evidence_grade("signal_label_assoc", "prompt_contrast.fixed_by_x") == "intervention"
    assert _evidence_grade("signal_label_assoc", "pope.false_negative") == "observational"


def test_invalid_min_grade_raises():
    with pytest.raises(ValueError):
        HypothesisTester(min_evidence_grade="causal")


# ── P3 routing ──────────────────────────────────────────────────────────────


def test_test_design_routes_to_designated_signal():
    # Statement keywords would match pope ("negative"), but the explicit test
    # design names the attention signal — design must win.
    h = _hyp("Failures stem from negative answer bias.",
             design="relative_attention.max_relative_weight")
    tr = HypothesisTester().test([h], _report(), _labeled_batch())[0]
    assert tr.evidence["routed_by"] == "test_design"
    assert tr.evidence["chosen_tool"] == "signal_label_assoc"
    assert tr.status == HypothesisStatus.INCONCLUSIVE  # attention CI crosses 0
    assert tr.effect_size == -0.3


def test_design_routing_to_intervention_signal_grades_intervention():
    h = _hyp("Describing first interferes with perception.",
             design="prompt_contrast describe_first repair")
    tr = HypothesisTester().test([h], _report(), _labeled_batch())[0]
    assert tr.evidence["routed_by"] == "test_design"
    assert tr.status == HypothesisStatus.SUPPORTED
    assert tr.evidence_grade == "intervention"


def test_keyword_fallback_when_no_design():
    h = _hyp("The model produces false negative answers on presence questions.")
    tr = HypothesisTester().test([h], _report(), _labeled_batch())[0]
    assert tr.evidence["routed_by"] == "keywords"
    assert tr.status == HypothesisStatus.SUPPORTED
    assert tr.evidence_grade == "observational"


# ── P4 stopping tiers ───────────────────────────────────────────────────────


def _result(grade: str, confidence: float = 0.5) -> HypothesisTestResult:
    return HypothesisTestResult(
        hypothesis=_hyp("h"), status=HypothesisStatus.SUPPORTED, test_name="t",
        effect_size=0.5, is_consistent_with_protocol=True,
        confidence=confidence, verdict="v", evidence_grade=grade,
    )


def test_observational_min_grade_keeps_plan_a_behavior():
    tester = HypothesisTester()  # default observational
    assert tester.stopping_criteria_met([_result("observational")]) is True


def test_intervention_min_grade_rejects_observational_support():
    tester = HypothesisTester(min_evidence_grade="intervention")
    assert tester.stopping_criteria_met([_result("observational")]) is False
    assert tester.stopping_criteria_met([_result("intervention")]) is True


def test_best_hypotheses_prefers_intervention_grade():
    tester = HypothesisTester()
    obs = _result("observational", confidence=0.9)
    interv = _result("intervention", confidence=0.6)
    best = tester.best_hypotheses([obs, interv])
    assert best[0].evidence_grade == "intervention"


# ── P3 diagnosis: TEST line + available evidence ────────────────────────────


class ScriptedJudge(FakeModel):
    def __init__(self, answer: str) -> None:
        super().__init__(capabilities={Capability.GENERATE})
        self.prompts: list[str] = []
        self._answer = answer

    def generate(self, inputs, **kw) -> str:
        self.prompts.append(str(inputs))
        return self._answer


def test_diagnosis_parses_test_line_into_design():
    judge = ScriptedJudge(
        "HYPOTHESIS: the model ignores the image\n"
        "FAILURE_MODE: visual_blindness\n"
        "TEST: relative_attention.max_relative_weight association\n"
    )
    diag = DiagnosisAgent(judge=judge).diagnose(_report())
    assert len(diag.hypotheses) == 1
    assert diag.hypotheses[0].test_design == "relative_attention.max_relative_weight association"


def test_diagnosis_prompt_lists_available_evidence():
    judge = ScriptedJudge("NO_ISSUE")
    DiagnosisAgent(judge=judge).diagnose(_report())
    p = judge.prompts[0]
    assert "AVAILABLE EVIDENCE" in p
    assert "pope.false_negative" in p
    assert "relative_attention.max_relative_weight" in p


def test_hypothesis_test_design_round_trips():
    h = _hyp("s", design="pope.false_negative")
    assert hypothesis_from_dict(hypothesis_to_dict(h)).test_design == "pope.false_negative"


# ── P3 M1: designs reach cycle-2 selection prompt ───────────────────────────


def test_m1_selection_prompt_includes_test_designs():
    class CapturingJudge(FakeModel):
        def __init__(self) -> None:
            super().__init__(capabilities={Capability.GENERATE})
            self.prompt = ""

        def generate(self, inputs, **kw) -> str:
            self.prompt = str(inputs)
            return '{"analyzers": ["self_consistency"], "rationale": "r"}'

    judge = CapturingJudge()
    model = FakeModel(capabilities={Capability.GENERATE})
    agent = ProbeAgent(judge=judge, max_analyzers=1)
    prior = [_hyp("attention is diffuse", design="run prompt_contrast describe_first")]
    agent.probe(model, _labeled_batch(),
                protocol=ExperimentProtocol(description="d"), prior_hypotheses=prior)
    assert "proposed test: run prompt_contrast describe_first" in judge.prompt