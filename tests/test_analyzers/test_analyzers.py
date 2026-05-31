"""Implemented analyzers across the functional taxonomy + modality matching."""

from __future__ import annotations

import evalvitals.analyzers  # noqa: F401  (populate registry)
from evalvitals.analyzers.agent.first_error_judge import FirstErrorJudge
from evalvitals.analyzers.agent.ignored_obs import IgnoredObservationDetector
from evalvitals.analyzers.agent.loop_detect import LoopDetector
from evalvitals.analyzers.attention.rollout import AttentionRolloutAnalyzer
from evalvitals.analyzers.attention.sink import AttentionSinkAnalyzer
from evalvitals.analyzers.geometry.cka import CKAAnalyzer, linear_cka
from evalvitals.analyzers.lens.logit_lens import LogitLensAnalyzer
from evalvitals.analyzers.perturbation.rise import RISEAnalyzer
from evalvitals.analyzers.uncertainty.self_consistency import SelfConsistencyAnalyzer
from evalvitals.analyzers.uncertainty.verbalized_conf import VerbalizedConfidenceAnalyzer
from evalvitals.core import registry
from evalvitals.core.capability import Capability
from evalvitals.core.case import Step, StepRole, Trajectory
from tests.conftest import FakeModel


def _traj_with(actor_obs_pairs) -> Trajectory:
    steps = [Step(idx=0, role=StepRole.USER, content="goal")]
    for name, args, obs in actor_obs_pairs:
        steps.append(Step(idx=len(steps), role=StepRole.ACTOR, tool_call={"name": name, "args": args}))
        steps.append(Step(idx=len(steps), role=StepRole.TOOL, content=name, observation=obs))
    return Trajectory(sample_id="s", goal="goal", steps=steps)


# ---------------- agent heuristics (no model) ----------------
def test_loop_detector_flags_repeated_action():
    traj = _traj_with([("search", {"q": "x"}, "none"), ("search", {"q": "x"}, "none")])
    res = LoopDetector().run(None, traj)
    assert res.findings["n_with_loops"] == 1
    assert res.findings["per_case"][0]["consecutive_repeat"] is True


def test_loop_detector_no_loop_on_distinct_actions():
    traj = _traj_with([("search", {"q": "x"}, "ok"), ("search", {"q": "y"}, "ok")])
    assert LoopDetector().run(None, traj).findings["n_with_loops"] == 0


def test_ignored_obs_flags_repeat_after_error():
    traj = _traj_with([("search", {"q": "x"}, "error: not found"), ("search", {"q": "x"}, "ok")])
    res = IgnoredObservationDetector().run(None, traj)
    assert res.findings["n_with_ignored_obs"] == 1


def test_first_error_judge_marks_step():
    class FakeJudge:
        capabilities = frozenset({Capability.GENERATE})
        def generate(self, prompt, **k):
            return "After review, STEP: 2"

    traj = _traj_with([("search", {"q": "x"}, "none")])  # steps: 0 user,1 actor,2 tool
    res = FirstErrorJudge(judge=FakeJudge()).run(None, traj)
    assert res.findings["per_case"][0]["first_error_step"] == 2
    assert traj.steps[2].is_first_error is True


# ---------------- uncertainty (GENERATE) ----------------
def test_self_consistency_constant_model_is_fully_consistent():
    model = FakeModel(capabilities={Capability.GENERATE})
    res = SelfConsistencyAnalyzer(n=4).run(model, "q")
    assert res.findings["consistency"] == 1.0 and res.findings["n_unique"] == 1


def test_verbalized_confidence_parses_number():
    class ConfModel:
        capabilities = frozenset({Capability.GENERATE})
        def generate(self, inputs, **k):
            return "The answer is Paris.\nConfidence: 80"

    res = VerbalizedConfidenceAnalyzer().run(ConfModel(), "q")
    assert res.findings["verbalized_confidence"] == 0.8


# ---------------- attention (ATTENTION) ----------------
def test_attention_rollout_returns_top_tokens():
    model = FakeModel(capabilities={Capability.ATTENTION})
    res = AttentionRolloutAnalyzer(top_k=3).run(model, "x")
    assert res.findings["seq_len"] == 5 and len(res.findings["top_rollout_tokens"]) == 3


def test_attention_sink_per_layer():
    model = FakeModel(capabilities={Capability.ATTENTION})
    res = AttentionSinkAnalyzer().run(model, "x")
    assert len(res.findings["per_layer_sink"]) == 3
    assert 0.0 <= res.findings["mean_sink_mass"] <= 1.0


# ---------------- lens (HIDDEN_STATES + unembed) ----------------
def test_logit_lens_reads_every_layer():
    model = FakeModel(capabilities={Capability.HIDDEN_STATES})
    res = LogitLensAnalyzer(top_k=2).run(model, "x")
    assert res.findings["n_layers"] == 4  # n_layers + 1 hidden states
    assert all(len(layer["top"]) == 2 for layer in res.findings["per_layer_top"])


# ---------------- geometry (HIDDEN_STATES) ----------------
def test_cka_self_similarity_is_one():
    import torch
    X = torch.rand(10, 6)
    assert abs(linear_cka(X, X) - 1.0) < 1e-5


def test_cka_analyzer_layer_matrix():
    model = FakeModel(capabilities={Capability.HIDDEN_STATES})
    res = CKAAnalyzer().run(model, "x")
    assert len(res.findings["adjacent_layer_cka"]) == 3
    assert 0.0 <= res.findings["mean_offdiagonal_cka"] <= 1.0


# ---------------- perturbation (GENERATE + scorer) ----------------
def test_rise_attributes_importance_to_keyword():
    class EchoModel:
        capabilities = frozenset({Capability.GENERATE})
        modalities = frozenset({"text"})
        def generate(self, inputs, **k):
            return str(inputs)

    score = lambda out: 1.0 if "secret" in out else 0.0  # noqa: E731
    res = RISEAnalyzer(score_fn=score, n_masks=200, seed=0).run(EchoModel(), "the secret word is here")
    assert res.findings["top_tokens"][0]["token"] == "secret"


# ---------------- modality matching ----------------
def test_modality_matching_filters_vision_only_analyzers():
    text_model = FakeModel(capabilities={Capability.GENERATE}, modalities={"text"})
    vlm = FakeModel(capabilities={Capability.GENERATE}, modalities={"text", "image"})
    text_names = registry.analyzers.names_compatible_with(text_model)
    vlm_names = registry.analyzers.names_compatible_with(vlm)
    assert "pope" not in text_names         # applies_to {image}
    assert "pope" in vlm_names
    assert "self_consistency" in text_names  # applies_to {text, image}
