"""Tests for the AutoDiagnose pipeline: M1 (probe), M3 (diagnosis), M4 (survey),
and the full AutoDiagnoseLoop that ties them together.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label, Step, StepRole, Trajectory
from evalvitals.core.registry import registry
from evalvitals.eval_agent import (
    AutoDiagnoseLoop,
    AutoDiagnoseReport,
    DiagnosisAgent,
    DiagnosisResult,
    Hypothesis,
    HypothesisStatus,
    InterventionResult,
    ModelKind,
    StrategyProbe,
    SurveyAgent,
)
from tests.conftest import FakeModel

# ── helpers ────────────────────────────────────────────────────────────────────


class ScriptedModel(FakeModel):
    """FakeModel with deterministic generate() responses."""

    def __init__(self, answers: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._answers = answers
        self._i = 0

    def generate(self, inputs, **kwargs) -> str:
        answer = self._answers[self._i % len(self._answers)]
        self._i += 1
        return answer


def _vlm_model() -> FakeModel:
    return FakeModel(
        capabilities={Capability.GENERATE, Capability.ATTENTION},
        modalities={"text", "image"},
    )


def _agent_model() -> FakeModel:
    return FakeModel(capabilities={Capability.GENERATE, Capability.TOOL_CALLS})


def _llm_model() -> FakeModel:
    return FakeModel(capabilities={Capability.GENERATE, Capability.ATTENTION, Capability.HIDDEN_STATES})


def _traj_batch(n_fail: int = 1, n_pass: int = 1) -> CaseBatch:
    cases = []
    for i in range(n_fail):
        traj = Trajectory(
            sample_id=f"fail_{i}",
            goal="open file",
            outcome=Label.FAIL,
            steps=[
                Step(idx=0, role=StepRole.USER, content="open file"),
                Step(idx=1, role=StepRole.ACTOR, tool_call={"name": "open", "args": {}}),
                Step(idx=2, role=StepRole.TOOL, observation="Error: denied"),
                Step(idx=3, role=StepRole.ACTOR, tool_call={"name": "open", "args": {}}),
            ],
        )
        cases.append(FailureCase(inputs=Inputs(prompt="open"), trajectory=traj, label=Label.FAIL))
    for i in range(n_pass):
        traj = Trajectory(
            sample_id=f"pass_{i}",
            goal="open file",
            outcome=Label.PASS,
            steps=[
                Step(idx=0, role=StepRole.USER, content="open file"),
                Step(idx=1, role=StepRole.ACTOR, tool_call={"name": "open", "args": {}}),
                Step(idx=2, role=StepRole.TOOL, observation="OK"),
            ],
        )
        cases.append(FailureCase(inputs=Inputs(prompt="open"), trajectory=traj, label=Label.PASS))
    return CaseBatch(cases)


# ══════════════════════════════════════════════════════════════════════════════
# M1 — StrategyProbe
# ══════════════════════════════════════════════════════════════════════════════

def test_probe_detects_vlm():
    assert StrategyProbe().detect_kind(_vlm_model()) == ModelKind.VLM


def test_probe_detects_agent():
    assert StrategyProbe().detect_kind(_agent_model()) == ModelKind.AGENT


def test_probe_detects_llm():
    assert StrategyProbe().detect_kind(_llm_model()) == ModelKind.LLM


def test_probe_select_returns_only_compatible_analyzers():
    model = _llm_model()
    names = StrategyProbe().select(model)
    compatible = set(registry.analyzers.names_compatible_with(model))
    assert set(names) <= compatible


def test_probe_select_priority_ordering_for_llm():
    model = FakeModel(
        capabilities={
            Capability.GENERATE,
            Capability.ATTENTION,
            Capability.HIDDEN_STATES,
            Capability.LOGITS,
        }
    )
    names = StrategyProbe().select(model)
    assert names.index("attention") < names.index("cka")


def test_probe_select_priority_ordering_for_agent():
    model = _agent_model()
    names = StrategyProbe().select(model)
    assert names.index("loop_detect") < names.index("ignored_obs")


def test_probe_select_max_analyzers_caps_list():
    model = _llm_model()
    assert len(StrategyProbe().select(model, max_analyzers=2)) == 2


def test_probe_select_only_zero_requires_for_no_capability_model():
    # Analyzers with empty requires (loop_detect, ignored_obs, first_error_judge)
    # are compatible with any model, including one with no capabilities.
    model = FakeModel(capabilities=frozenset())
    names = StrategyProbe().select(model)
    zero_req = {n for n, cls in registry.analyzers.all().items() if not cls.requires}
    assert set(names) == zero_req & set(registry.analyzers.names_compatible_with(model))


def test_probe_custom_priority_override():
    model = FakeModel(capabilities={Capability.GENERATE, Capability.ATTENTION})
    # Both attention_sink and attention are compatible; priority list puts attention_sink first.
    probe = StrategyProbe(priority_override={ModelKind.LLM: ["attention_sink", "attention"]})
    names = probe.select(model)
    assert names[0] == "attention_sink"
    assert names[1] == "attention"


# ══════════════════════════════════════════════════════════════════════════════
# M3 — DiagnosisAgent
# ══════════════════════════════════════════════════════════════════════════════

def _fake_results() -> dict:
    from evalvitals.analyzers.attention.summary import AttentionAnalyzer
    model = FakeModel()
    return {"attention": AttentionAnalyzer().run(model, "probe")}


def test_diagnosis_parses_hypothesis_and_failure_mode():
    judge = ScriptedModel(
        answers=[
            "HYPOTHESIS: model attends too strongly to the BOS token\n"
            "FAILURE_MODE: attention_sink\n"
        ],
        capabilities={Capability.GENERATE},
    )
    diag = DiagnosisAgent(judge=judge).diagnose(_fake_results(), "test-model")
    assert isinstance(diag, DiagnosisResult)
    assert len(diag.hypotheses) == 1
    h = diag.hypotheses[0]
    assert "BOS" in h.statement
    assert h.predicted_failure_mode == "attention_sink"
    assert h.target_model == "test-model"
    assert h.status == HypothesisStatus.PROPOSED


def test_diagnosis_parses_multiple_hypotheses():
    raw = (
        "HYPOTHESIS: high attention entropy\nFAILURE_MODE: diffuse_attention\n"
        "HYPOTHESIS: low self-consistency\nFAILURE_MODE: instability\n"
    )
    judge = ScriptedModel(answers=[raw], capabilities={Capability.GENERATE})
    diag = DiagnosisAgent(judge=judge).diagnose(_fake_results(), "m")
    assert len(diag.hypotheses) == 2


def test_diagnosis_no_issue_returns_empty_hypotheses():
    judge = ScriptedModel(answers=["NO_ISSUE"], capabilities={Capability.GENERATE})
    diag = DiagnosisAgent(judge=judge).diagnose(_fake_results(), "m")
    assert diag.hypotheses == []
    assert "NO_ISSUE" in diag.raw_judge_output


def test_diagnosis_result_carries_findings_summary():
    judge = ScriptedModel(answers=["NO_ISSUE"], capabilities={Capability.GENERATE})
    results = _fake_results()
    diag = DiagnosisAgent(judge=judge).diagnose(results, "m")
    assert "attention" in diag.findings_summary
    assert isinstance(diag.findings_summary["attention"], dict)


def test_diagnosis_partial_output_missing_failure_mode_skipped():
    raw = "HYPOTHESIS: something odd\n(no FAILURE_MODE line)"
    judge = ScriptedModel(answers=[raw], capabilities={Capability.GENERATE})
    diag = DiagnosisAgent(judge=judge).diagnose(_fake_results(), "m")
    assert diag.hypotheses == []


# ══════════════════════════════════════════════════════════════════════════════
# M4 — SurveyAgent
# ══════════════════════════════════════════════════════════════════════════════

def _hypothesis(mode: str = "loop") -> Hypothesis:
    return Hypothesis(statement="test", target_model="m", predicted_failure_mode=mode)


def test_survey_verify_fn_override():
    expected = InterventionResult(
        hypothesis=_hypothesis(),
        status=HypothesisStatus.SUPPORTED,
        fixed=True,
        evidence={"custom": True},
    )
    agent = SurveyAgent(verify_fn=lambda h, m, r, d: expected)
    result = agent.survey(_hypothesis(), None, {}, CaseBatch([]))
    assert result is expected


def test_survey_correlate_supported_when_signal_predicts_failure():
    from evalvitals.analyzers.agent.loop_detect import LoopDetector

    model = _agent_model()
    data = _traj_batch(n_fail=2, n_pass=2)
    results = {"loop_detect": LoopDetector().run(model, data)}
    h = _hypothesis("loop")
    iv = SurveyAgent().survey(h, model, results, data)
    # fail cases have the loop signal; pass cases do not → SUPPORTED
    assert iv.status == HypothesisStatus.SUPPORTED
    assert iv.evidence["fail_rate_signal"] > iv.evidence["fail_rate_control"]
    assert isinstance(iv.new_data, CaseBatch)


def test_survey_correlate_inconclusive_when_no_labels():
    from evalvitals.analyzers.agent.loop_detect import LoopDetector

    model = _agent_model()
    unlabeled = _traj_batch(n_fail=1, n_pass=1)
    for c in unlabeled:
        c.label = None
    results = {"loop_detect": LoopDetector().run(model, unlabeled)}
    iv = SurveyAgent().survey(_hypothesis(), model, results, unlabeled)
    assert iv.status == HypothesisStatus.INCONCLUSIVE
    assert iv.fixed is False


def test_survey_correlate_inconclusive_when_no_per_case_findings():
    from evalvitals.analyzers.attention.summary import AttentionAnalyzer

    model = _llm_model()
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"), label=Label.FAIL)])
    results = {"attention": AttentionAnalyzer().run(model, data)}
    iv = SurveyAgent().survey(_hypothesis("attention_sink"), model, results, data)
    # AttentionAnalyzer emits no per_case → INCONCLUSIVE
    assert iv.status == HypothesisStatus.INCONCLUSIVE


def test_survey_param_sweep_returns_evidence_dict():
    model = FakeModel(capabilities={Capability.GENERATE, Capability.ATTENTION})
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])
    agent = SurveyAgent(analyzer_params={"attention": {"top_k": 2}})
    iv = agent.survey(_hypothesis(), model, {}, data)
    assert iv.status == HypothesisStatus.INCONCLUSIVE
    assert "param_sweep" in iv.evidence
    assert "attention" in iv.evidence["param_sweep"]


def test_survey_fixed_true_when_perfect_separation():
    from evalvitals.analyzers.agent.loop_detect import LoopDetector

    model = _agent_model()
    data = _traj_batch(n_fail=3, n_pass=3)
    results = {"loop_detect": LoopDetector().run(model, data)}
    h = _hypothesis("loop")
    iv = SurveyAgent().survey(h, model, results, data)
    if iv.status == HypothesisStatus.SUPPORTED:
        # fixed = True only when perfect separation
        expected_fixed = (
            iv.evidence["fail_rate_signal"] >= 1.0
            and iv.evidence["fail_rate_control"] == 0.0
        )
        assert iv.fixed == expected_fixed


# ══════════════════════════════════════════════════════════════════════════════
# AutoDiagnoseLoop
# ══════════════════════════════════════════════════════════════════════════════

def test_loop_analysis_only_mode_returns_results():
    model = _llm_model()
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])
    loop = AutoDiagnoseLoop(model=model, max_analyzers=2)  # no diagnosis_agent
    report = loop.run(data)
    assert isinstance(report, AutoDiagnoseReport)
    assert report.resolved is False
    assert report.final_hypotheses == []
    assert len(report.final_results) <= 2


def test_loop_resolves_when_survey_returns_fixed():
    model = _llm_model()
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])

    judge = ScriptedModel(
        answers=["HYPOTHESIS: sink\nFAILURE_MODE: attention_sink\n"],
        capabilities={Capability.GENERATE},
    )

    def always_fixed(h, m, r, d):
        return InterventionResult(h, HypothesisStatus.SUPPORTED, fixed=True, evidence={})

    loop = AutoDiagnoseLoop(
        model=model,
        diagnosis_agent=DiagnosisAgent(judge=judge),
        survey_agent=SurveyAgent(verify_fn=always_fixed),
        max_analyzers=1,
        max_cycles=5,
    )
    report = loop.run(data)
    assert report.resolved is True
    assert report.cycles == 1


def test_loop_stops_when_no_hypotheses():
    model = _llm_model()
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])
    judge = ScriptedModel(answers=["NO_ISSUE"], capabilities={Capability.GENERATE})
    loop = AutoDiagnoseLoop(
        model=model,
        diagnosis_agent=DiagnosisAgent(judge=judge),
        max_analyzers=1,
        max_cycles=5,
    )
    report = loop.run(data)
    assert report.resolved is False
    assert report.final_hypotheses == []


def test_loop_max_cycles_respected():
    model = _llm_model()
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])
    judge = ScriptedModel(
        answers=["HYPOTHESIS: h\nFAILURE_MODE: f\n"],
        capabilities={Capability.GENERATE},
    )
    loop = AutoDiagnoseLoop(
        model=model,
        diagnosis_agent=DiagnosisAgent(judge=judge),
        survey_agent=SurveyAgent(verify_fn=lambda h, m, r, d: InterventionResult(
            h, HypothesisStatus.INCONCLUSIVE, fixed=False, evidence={}
        )),
        max_analyzers=1,
        max_cycles=3,
    )
    report = loop.run(data)
    assert report.resolved is False
    assert report.cycles <= 3


def test_loop_skips_analyzer_needing_mandatory_arg(recwarn):
    """CounterfactualReplay needs rerun_fn; without an override it should be skipped."""
    model = FakeModel(capabilities={Capability.GENERATE, Capability.TOOL_CALLS})
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])
    loop = AutoDiagnoseLoop(model=model, max_cycles=1)
    report = loop.run(data)
    # Should not raise; counterfactual is skipped with a warning
    assert isinstance(report, AutoDiagnoseReport)
    skipped = any("counterfactual" in str(w.message) for w in recwarn.list)
    assert skipped


def test_loop_uses_analyzer_override_for_counterfactual():
    from evalvitals.analyzers.agent.counterfactual import CounterfactualReplay

    model = FakeModel(capabilities={Capability.GENERATE, Capability.TOOL_CALLS})
    data = _traj_batch(n_fail=1, n_pass=0)
    rerun = CounterfactualReplay(rerun_fn=lambda t, i, s: True, n_replays=1)
    loop = AutoDiagnoseLoop(
        model=model,
        max_cycles=1,
        analyzer_overrides={"counterfactual": rerun},
    )
    report = loop.run(data)
    assert "counterfactual" in report.final_results


def test_loop_store_accumulates_results_and_hypotheses():
    model = _llm_model()
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])
    judge = ScriptedModel(
        answers=["HYPOTHESIS: h\nFAILURE_MODE: f\n"],
        capabilities={Capability.GENERATE},
    )
    loop = AutoDiagnoseLoop(
        model=model,
        diagnosis_agent=DiagnosisAgent(judge=judge),
        survey_agent=SurveyAgent(verify_fn=lambda h, m, r, d: InterventionResult(
            h, HypothesisStatus.INCONCLUSIVE, fixed=False, evidence={}
        )),
        max_analyzers=1,
        max_cycles=1,
    )
    report = loop.run(data)
    assert len(report.store.results) > 0
    assert len(report.store.hypotheses) > 0
