"""Parametrized contract tests for every Analyzer — pyod-style baseline suite.

Every registered analyzer must pass suites 1 and 3 automatically.
Suites 2 and 4 require a manual entry when adding a new analyzer:
add it to _RUNNABLE (if it can run in CI) or _STUBS (if _run raises
NotImplementedError), and document any exclusion in _SKIPPED_REASON.
The coverage test (test_contract_coverage_is_complete) enforces this.

    Suite 1 · CLASS METADATA     all registered analyzers, no model needed.
    Suite 2 · RESULT CONTRACT    runnable analyzers only (see _RUNNABLE).
    Suite 3 · CAPABILITY GATING  every analyzer with non-empty requires.
    Suite 4 · STUB TRACKING      Stage-2 stubs that must raise NotImplementedError.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

import evalvitals  # noqa: F401 — side-effect: registers all analyzers
from evalvitals.analyzers.agent.counterfactual import CounterfactualReplay
from evalvitals.analyzers.agent.first_error_judge import FirstErrorJudge
from evalvitals.analyzers.agent.ignored_obs import IgnoredObservationDetector
from evalvitals.analyzers.agent.loop_detect import LoopDetector
from evalvitals.analyzers.attention.rollout import AttentionRolloutAnalyzer
from evalvitals.analyzers.attention.sink import AttentionSinkAnalyzer
from evalvitals.analyzers.attention.summary import AttentionAnalyzer
from evalvitals.analyzers.geometry.cka import CKAAnalyzer
from evalvitals.analyzers.geometry.linear_probe import LinearProbeAnalyzer
from evalvitals.analyzers.hallucination.chair import CHAIRAnalyzer
from evalvitals.analyzers.hallucination.opera import OPERAAnalyzer
from evalvitals.analyzers.hallucination.pope import POPEAnalyzer
from evalvitals.analyzers.hallucination.vcd import VCDAnalyzer
from evalvitals.analyzers.lens.logit_lens import LogitLensAnalyzer
from evalvitals.analyzers.lens.tuned_lens import TunedLensAnalyzer
from evalvitals.analyzers.patching.causal_trace import CausalTraceAnalyzer
from evalvitals.analyzers.perturbation.mm_shap import MMShapAnalyzer
from evalvitals.analyzers.perturbation.prompt_contrast import PromptContrastAnalyzer
from evalvitals.analyzers.uncertainty.entropy import TokenEntropyAnalyzer
from evalvitals.analyzers.uncertainty.logprob_entropy import LogprobEntropyAnalyzer
from evalvitals.analyzers.uncertainty.self_consistency import SelfConsistencyAnalyzer
from evalvitals.analyzers.uncertainty.verbalized_conf import VerbalizedConfidenceAnalyzer
from evalvitals.core.capability import Capability, CapabilityError
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label, Step, StepRole, Trajectory
from evalvitals.core.registry import registry
from evalvitals.core.result import Result
from evalvitals.datasets import cases_from_records
from tests.conftest import FakeModel

# ── shared fixtures ────────────────────────────────────────────────────────────

# One model with every text-domain capability so any text-path analyzer can run.
_FULL = FakeModel(
    capabilities={
        Capability.GENERATE,
        Capability.ATTENTION,
        Capability.HIDDEN_STATES,
        Capability.LOGITS,
        Capability.LOGPROBS,
        Capability.TOOL_CALLS,
    }
)

_STANDARD = "the quick brown fox"

# Labelled batch for analyzers that need PASS/FAIL supervision (linear_probe).
_LABELLED = CaseBatch([
    FailureCase(inputs=Inputs(prompt=f"probe case {i}"),
                label=Label.FAIL if i % 2 else Label.PASS)
    for i in range(16)
])


class ScriptedFakeModel(FakeModel):
    """FakeModel variant with deterministic generate() outputs for behavior checks."""

    def __init__(self, answers: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._answers = answers
        self._i = 0

    def generate(self, inputs, **kwargs) -> str:
        answer = self._answers[self._i % len(self._answers)]
        self._i += 1
        return answer


_JUDGE_MODEL = ScriptedFakeModel(
    answers=["STEP: 2"],
    capabilities={
        Capability.GENERATE,
        Capability.ATTENTION,
        Capability.HIDDEN_STATES,
        Capability.LOGITS,
        Capability.LOGPROBS,
        Capability.TOOL_CALLS,
    },
)


def _traj_batch() -> CaseBatch:
    """Actor → tool error → actor repeats: exercises loop-detect and ignored-obs."""
    traj = Trajectory(
        sample_id="s0",
        goal="open file",
        outcome=Label.FAIL,
        steps=[
            Step(idx=0, role=StepRole.USER,  content="open file"),
            Step(idx=1, role=StepRole.ACTOR, tool_call={"name": "open", "args": {"path": "/tmp"}}),
            Step(idx=2, role=StepRole.TOOL,  observation="Error: permission denied"),
            Step(idx=3, role=StepRole.ACTOR, tool_call={"name": "open", "args": {"path": "/tmp"}}),
        ],
    )
    case = FailureCase(inputs=Inputs(prompt="open file"), trajectory=traj, label=Label.FAIL)
    return CaseBatch([case])


def _pope_batch() -> CaseBatch:
    return cases_from_records([
        {"question": "Is there a dog?", "pope_label": "yes"},
        {"question": "Is there a cat?", "pope_label": "no"},
    ])


def _chair_batch() -> CaseBatch:
    return cases_from_records(
        [{"caption_goal": "describe the image", "gt_objects": ["dog"]}],
        prompt_key="caption_goal",
    )


def _contrast_batch() -> CaseBatch:
    case = FailureCase(
        inputs=Inputs(prompt="Is there a dog?"),
        expected={"all_of": ["yes"], "none_of": ["no"]},
        label=Label.FAIL,
    )
    return CaseBatch([case])


# ── analyzers excluded from suite 2, with documented reasons ──────────────────
# When adding a new analyzer: either add it to _RUNNABLE/_STUBS below, or add an
# entry here explaining why it is excluded. The coverage test will catch omissions.
_SKIPPED_REASON: dict[str, str] = {
    # Requires a live PIL image; no FakeModel path.
    "relative_attention": "VLM-only: needs case.inputs.image (PIL image)",
    "vl_shap":            "VLM-only: needs case.inputs.image (PIL image)",
    # Requires a backward pass; Capability.GRADIENTS is not available in FakeModel.
    "gradcam":           "needs Capability.GRADIENTS (backward pass)",
    "generic_attention": "needs Capability.GRADIENTS (Chefer relevance propagation)",
    # score_fn is mandatory at run time with no useful default.
    "rise": "requires a user-supplied score_fn(output)->float; no sensible default",
}


# ── suite 2: (analyzer, model, data) triples for runnable analyzers ────────────
# Each call to _traj_batch() / _pope_batch() / _chair_batch() creates a fresh
# CaseBatch so tests are isolated even if an analyzer mutates its input cases.
_RUNNABLE: list[tuple[Any, Any, Any]] = [
    # attention
    (AttentionAnalyzer(),            _FULL, _STANDARD),
    (AttentionRolloutAnalyzer(),     _FULL, _STANDARD),
    (AttentionSinkAnalyzer(),        _FULL, _STANDARD),
    # lens
    (LogitLensAnalyzer(),            _FULL, _STANDARD),
    # uncertainty
    (TokenEntropyAnalyzer(),         _FULL, _STANDARD),
    (LogprobEntropyAnalyzer(),       _FULL, _STANDARD),
    (SelfConsistencyAnalyzer(n=2),   _FULL, _STANDARD),
    (VerbalizedConfidenceAnalyzer(), ScriptedFakeModel(
        answers=["final answer\nConfidence: 80"],
        capabilities={Capability.GENERATE},
    ), _STANDARD),
    # geometry
    (CKAAnalyzer(),                  _FULL, _STANDARD),
    (LinearProbeAnalyzer(epochs=50), _FULL, _LABELLED),
    # perturbation — text path; default scorer falls back to model.logprobs
    (MMShapAnalyzer(n_samples=4),    _FULL, _STANDARD),
    # agent analyzers — all consume a Trajectory-bearing CaseBatch
    (LoopDetector(),                 None,  _traj_batch()),
    (IgnoredObservationDetector(),   None,  _traj_batch()),
    (FirstErrorJudge(),              _JUDGE_MODEL, _traj_batch()),
    (CounterfactualReplay(rerun_fn=lambda traj, idx, seed: True, n_replays=2),
                                     _FULL, _traj_batch()),
    # hallucination — GENERATE-based; modality filtering is in registry discovery,
    # not in _check_capabilities, so a text FakeModel reaches _run correctly.
    (POPEAnalyzer(), ScriptedFakeModel(
        answers=["yes", "no"],
        capabilities={Capability.GENERATE},
        modalities={"image"},
    ), _pope_batch()),
    (CHAIRAnalyzer(object_vocab=["dog", "cat"]), ScriptedFakeModel(
        answers=["a dog"],
        capabilities={Capability.GENERATE},
        modalities={"image"},
    ), _chair_batch()),
    (PromptContrastAnalyzer(), ScriptedFakeModel(
        answers=["yes"],  # repeated for every strategy call
        capabilities={Capability.GENERATE},
    ), _contrast_batch()),
]
_RUNNABLE_IDS = [a.name for a, _, _ in _RUNNABLE]


# pyod-style behavioral contracts: each runnable analyzer declares the stable
# user-facing findings it must produce, plus lightweight invariants on values.
_EXPECTED_FINDING_KEYS: dict[str, set[str]] = {
    "attention": {
        "num_layers", "num_heads", "seq_len", "summary_layer", "summary_head",
        "top_attended_tokens", "mean_attention_entropy",
    },
    "attention_rollout": {"seq_len", "n_layers", "top_rollout_tokens"},
    "attention_sink": {"n_layers", "sink_token", "mean_sink_mass", "per_layer_sink"},
    "logit_lens": {"n_layers", "pos", "per_layer_top", "per_case", "final_norm_applied"},
    "linear_probe": {
        "n_layers", "n_fail", "n_pass", "per_layer_accuracy",
        "best_layer", "best_accuracy", "per_case",
    },
    "token_entropy": {
        "seq_len", "mean_entropy", "max_entropy", "final_token_entropy", "top_next_tokens",
    },
    "logprob_entropy": {
        "n_tokens", "mean_logprob", "perplexity", "min_token_logprob", "mean_top_entropy",
    },
    "self_consistency": {"n_samples", "consistency", "n_unique", "modal_answer", "gen_kwargs"},
    "verbalized_confidence": {"verbalized_confidence", "parsed", "raw_tail"},
    "cka": {"n_layers", "mean_offdiagonal_cka", "adjacent_layer_cka", "_caveat"},
    "mm_shap": {
        "mm_score", "text_contribution", "image_contribution", "has_image",
        "top_text_tokens", "_note",
    },
    "loop_detect": {"n_trajectories", "n_with_loops", "per_case"},
    "ignored_obs": {"n_trajectories", "n_with_ignored_obs", "per_case"},
    "first_error_judge": {"n_trajectories", "judge", "per_case", "_caveat"},
    "counterfactual": {"n_trajectories", "per_case", "_caveat"},
    "pope": {"n", "unparsed", "accuracy", "precision", "recall", "f1", "yes_rate"},
    "chair": {"n", "chair_i", "chair_s"},
    "prompt_contrast": {"n_cases", "n_strategies", "n_unscored", "by_strategy"},
}


def _between(value: float | int | None, lo: float, hi: float) -> bool:
    return value is not None and lo <= float(value) <= hi


def _check_attention(f: dict[str, Any]) -> None:
    assert f["num_layers"] == 3
    assert f["num_heads"] == 4
    assert f["seq_len"] == 5
    assert len(f["top_attended_tokens"]) == 5
    assert f["mean_attention_entropy"] >= 0


def _check_attention_rollout(f: dict[str, Any]) -> None:
    assert f["n_layers"] == 3
    assert f["seq_len"] == 5
    assert len(f["top_rollout_tokens"]) == 5


def _check_attention_sink(f: dict[str, Any]) -> None:
    assert f["n_layers"] == 3
    assert len(f["per_layer_sink"]) == 3
    assert _between(f["mean_sink_mass"], 0, 1)


def _check_logit_lens(f: dict[str, Any]) -> None:
    assert f["n_layers"] == 4
    assert all(len(layer["top"]) == 3 for layer in f["per_layer_top"])
    assert len(f["per_case"]) == 1  # _STANDARD is a single case
    pc = f["per_case"][0]
    assert 0 <= pc["decision_frac"] <= 1
    assert _between(pc["final_top1_prob"], 0, 1)
    assert pc["late_drop"] >= 0


def _check_linear_probe(f: dict[str, Any]) -> None:
    assert f["n_layers"] == 4
    assert len(f["per_layer_accuracy"]) == 4
    assert f["n_fail"] == 8 and f["n_pass"] == 8
    assert _between(f["best_accuracy"], 0, 1)
    assert len(f["per_case"]) == 16
    assert all(_between(p["fail_prob_best_layer"], 0, 1) for p in f["per_case"])


def _check_token_entropy(f: dict[str, Any]) -> None:
    assert f["seq_len"] == 5
    assert f["max_entropy"] >= f["mean_entropy"] >= 0
    assert len(f["top_next_tokens"]) == 5


def _check_logprob_entropy(f: dict[str, Any]) -> None:
    assert f["n_tokens"] == 4
    assert f["perplexity"] > 1
    assert f["mean_top_entropy"] >= 0


def _check_self_consistency(f: dict[str, Any]) -> None:
    assert f["n_samples"] == 2
    assert _between(f["consistency"], 0, 1)
    assert f["n_unique"] >= 1
    # Sampling provenance must travel with the score that depends on it.
    assert isinstance(f["gen_kwargs"], dict)


def _check_verbalized_confidence(f: dict[str, Any]) -> None:
    assert f["parsed"] is True
    assert f["verbalized_confidence"] == 0.8


def _check_cka(f: dict[str, Any]) -> None:
    assert f["n_layers"] == 4
    assert len(f["adjacent_layer_cka"]) == 3
    assert _between(f["mean_offdiagonal_cka"], 0, 1)


def _check_mm_shap(f: dict[str, Any]) -> None:
    assert _between(f["mm_score"], 0, 1)
    assert f["has_image"] is False
    assert len(f["top_text_tokens"]) <= 4


def _check_loop_detect(f: dict[str, Any]) -> None:
    assert f["n_trajectories"] == 1
    assert f["n_with_loops"] == 1
    assert f["per_case"][0]["has_loop"] is True


def _check_ignored_obs(f: dict[str, Any]) -> None:
    assert f["n_trajectories"] == 1
    assert f["n_with_ignored_obs"] == 1
    assert f["per_case"][0]["n_ignored"] == 1


def _check_first_error_judge(f: dict[str, Any]) -> None:
    assert f["n_trajectories"] == 1
    assert f["per_case"][0]["first_error_step"] == 2


def _check_counterfactual(f: dict[str, Any]) -> None:
    assert f["n_trajectories"] == 1
    assert f["per_case"][0]["most_influential_step"] is not None
    assert _between(f["per_case"][0]["most_influential_step"]["flip_rate"], 0, 1)


def _check_pope(f: dict[str, Any]) -> None:
    assert f["n"] == 2
    assert f["unparsed"] == 0
    assert f["accuracy"] == 1.0
    assert f["f1"] == 1.0


def _check_chair(f: dict[str, Any]) -> None:
    assert f["n"] == 1
    assert f["chair_i"] == 0.0
    assert f["chair_s"] == 0.0


def _check_prompt_contrast(f: dict[str, Any]) -> None:
    assert f["n_cases"] == 1
    assert f["n_strategies"] == 3
    assert f["n_unscored"] == 0
    # "yes" under every strategy satisfies the gold-yes rubric.
    assert f["success_rate_baseline"] == 1.0
    assert all(v == {} or set(v.values()) == {1.0} for v in f["by_strategy"].values())
    assert f["n_fixed_by_sensitive"] == 0 and f["n_broken_by_sensitive"] == 0


_FINDING_INVARIANTS: dict[str, Callable[[dict[str, Any]], None]] = {
    "attention": _check_attention,
    "attention_rollout": _check_attention_rollout,
    "attention_sink": _check_attention_sink,
    "logit_lens": _check_logit_lens,
    "linear_probe": _check_linear_probe,
    "token_entropy": _check_token_entropy,
    "logprob_entropy": _check_logprob_entropy,
    "self_consistency": _check_self_consistency,
    "verbalized_confidence": _check_verbalized_confidence,
    "cka": _check_cka,
    "mm_shap": _check_mm_shap,
    "loop_detect": _check_loop_detect,
    "ignored_obs": _check_ignored_obs,
    "first_error_judge": _check_first_error_judge,
    "counterfactual": _check_counterfactual,
    "pope": _check_pope,
    "chair": _check_chair,
    "prompt_contrast": _check_prompt_contrast,
}


# ── suite 4: Stage-2 stubs ─────────────────────────────────────────────────────
# These analyzers are fully registered and their class contracts are sound, but
# _run raises NotImplementedError. When you implement one, move it to _RUNNABLE.
_STUBS: list[tuple[Any, Any, Any]] = [
    (TunedLensAnalyzer(),   _FULL, _STANDARD),
    (CausalTraceAnalyzer(), _FULL, _STANDARD),
    # OPERA and VCD are image-only stubs; _FULL satisfies their capability check
    # (ATTENTION and LOGITS respectively), so _run is reached and raises.
    (OPERAAnalyzer(), _FULL, _STANDARD),
    (VCDAnalyzer(),   _FULL, _STANDARD),
]
_STUBS_IDS = [a.name for a, _, _ in _STUBS]


_CLASS_FACTORIES: dict[str, Callable[[], Any]] = {
    "counterfactual": lambda: CounterfactualReplay(
        rerun_fn=lambda traj, idx, seed: True,
        n_replays=2,
    ),
    "chair": lambda: CHAIRAnalyzer(object_vocab=["dog", "cat"]),
}


def _make_analyzer(cls):
    return _CLASS_FACTORIES.get(cls.name, cls)()


# ══════════════════════════════════════════════════════════════════════════════
# Coverage sentinel — fails when a newly registered analyzer is not yet placed
# ══════════════════════════════════════════════════════════════════════════════

def test_contract_coverage_is_complete():
    """Every registered analyzer must appear in _RUNNABLE, _STUBS, or _SKIPPED_REASON."""
    covered = (
        {a.name for a, _, _ in _RUNNABLE}
        | {a.name for a, _, _ in _STUBS}
        | set(_SKIPPED_REASON)
    )
    uncovered = set(registry.analyzers.list()) - covered
    assert not uncovered, (
        f"Add these to _RUNNABLE, _STUBS, or _SKIPPED_REASON: {sorted(uncovered)}"
    )


def test_runnable_analyzers_have_behavior_contracts():
    """Every runnable analyzer must declare stable findings keys and invariants."""
    runnable = {a.name for a, _, _ in _RUNNABLE}
    missing_key_contract = runnable - set(_EXPECTED_FINDING_KEYS)
    missing_invariant = runnable - set(_FINDING_INVARIANTS)
    assert not missing_key_contract, (
        f"Add expected findings keys for: {sorted(missing_key_contract)}"
    )
    assert not missing_invariant, (
        f"Add findings invariants for: {sorted(missing_invariant)}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Suite 1 · CLASS METADATA  (all 26 registered analyzers)
# ══════════════════════════════════════════════════════════════════════════════

_ALL_CLASSES = list(registry.analyzers.all().values())
_ALL_NAMES   = list(registry.analyzers.all().keys())


@pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_NAMES)
def test_name_is_nonempty_string(cls):
    assert isinstance(cls.name, str) and cls.name


@pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_NAMES)
def test_requires_is_frozenset_of_capabilities(cls):
    assert isinstance(cls.requires, frozenset)
    assert all(isinstance(c, Capability) for c in cls.requires)


@pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_NAMES)
def test_applies_to_modalities_is_nonempty_frozenset(cls):
    assert isinstance(cls.applies_to_modalities, frozenset)
    assert cls.applies_to_modalities


@pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_NAMES)
def test_registered_under_own_name(cls):
    assert registry.analyzers.get(cls.name) is cls


@pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_NAMES)
def test_sklearn_params_roundtrip(cls):
    inst = _make_analyzer(cls)
    params = inst.get_params()
    assert isinstance(params, dict)
    returned = inst.set_params(**params)
    assert returned is inst
    assert inst.get_params() == params


# ══════════════════════════════════════════════════════════════════════════════
# Suite 2 · RESULT CONTRACT  (runnable analyzers)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("analyzer,model,data", _RUNNABLE, ids=_RUNNABLE_IDS)
def test_run_returns_result_instance(analyzer, model, data):
    assert isinstance(analyzer.run(model, data), Result)


@pytest.mark.parametrize("analyzer,model,data", _RUNNABLE, ids=_RUNNABLE_IDS)
def test_result_analyzer_field_matches_name(analyzer, model, data):
    assert analyzer.run(model, data).analyzer == analyzer.name


@pytest.mark.parametrize("analyzer,model,data", _RUNNABLE, ids=_RUNNABLE_IDS)
def test_result_model_field_is_string(analyzer, model, data):
    assert isinstance(analyzer.run(model, data).model, str)


@pytest.mark.parametrize("analyzer,model,data", _RUNNABLE, ids=_RUNNABLE_IDS)
def test_findings_is_dict(analyzer, model, data):
    assert isinstance(analyzer.run(model, data).findings, dict)


@pytest.mark.parametrize("analyzer,model,data", _RUNNABLE, ids=_RUNNABLE_IDS)
def test_findings_is_json_serialisable(analyzer, model, data):
    json.dumps(analyzer.run(model, data).findings)  # must not raise


@pytest.mark.parametrize("analyzer,model,data", _RUNNABLE, ids=_RUNNABLE_IDS)
def test_findings_match_behavior_contract(analyzer, model, data):
    result = analyzer.run(model, data)
    expected = _EXPECTED_FINDING_KEYS[analyzer.name]
    missing = expected - set(result.findings)
    assert not missing, f"{analyzer.name} missing findings keys: {sorted(missing)}"
    _FINDING_INVARIANTS[analyzer.name](result.findings)


@pytest.mark.parametrize("analyzer,model,data", _RUNNABLE, ids=_RUNNABLE_IDS)
def test_to_dict_has_expected_keys(analyzer, model, data):
    required = {"analyzer", "model", "findings", "metadata", "n_cases"}
    assert required <= analyzer.run(model, data).to_dict().keys()


@pytest.mark.parametrize("analyzer,model,data", _RUNNABLE, ids=_RUNNABLE_IDS)
def test_to_json_round_trips(analyzer, model, data):
    result = analyzer.run(model, data)
    parsed = json.loads(result.to_json())
    assert parsed["analyzer"] == analyzer.name


@pytest.mark.parametrize("analyzer,model,data", _RUNNABLE, ids=_RUNNABLE_IDS)
def test_summary_returns_string(analyzer, model, data):
    assert isinstance(analyzer.run(model, data).summary(), str)


# ══════════════════════════════════════════════════════════════════════════════
# Suite 3 · CAPABILITY GATING  (all analyzers with non-empty requires)
# ══════════════════════════════════════════════════════════════════════════════

_CAP_GATED   = [cls for cls in _ALL_CLASSES if cls.requires]
_CAP_GATED_IDS = [cls.name for cls in _CAP_GATED]
_NO_CAP = FakeModel(capabilities=frozenset())


@pytest.mark.parametrize("cls", _CAP_GATED, ids=_CAP_GATED_IDS)
def test_capability_error_when_model_has_no_caps(cls):
    with pytest.raises(CapabilityError):
        _make_analyzer(cls).run(_NO_CAP, _STANDARD)


# ══════════════════════════════════════════════════════════════════════════════
# Suite 4 · STUB TRACKING  (Stage-2 NotImplementedError analyzers)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("analyzer,model,data", _STUBS, ids=_STUBS_IDS)
def test_stub_raises_not_implemented(analyzer, model, data):
    """Stubs raise NotImplementedError until Stage-2 work is done.

    If this fails because you implemented an analyzer, move it from _STUBS to
    _RUNNABLE and add it to the result-contract suite above.
    """
    with pytest.raises(NotImplementedError):
        analyzer.run(model, data)
