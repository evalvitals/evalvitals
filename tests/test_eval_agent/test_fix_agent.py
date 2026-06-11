"""Fix module: intervention-space tiers, L2 tool pipelines, validated repair.

The allowed tier is an input (default L2); no automatic escalation — when no
candidate validates, the outcome recommends raising the tier, routed from the
verified hypotheses' mechanisms.
"""

from __future__ import annotations

import json

import pytest

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.model import Model
from evalvitals.eval_agent import (
    FixAgent,
    FixTier,
    parse_tier,
    route_min_tier,
)
from evalvitals.eval_agent.hypothesis import Hypothesis

# ── tiers ─────────────────────────────────────────────────────────────────────


def test_tier_parse_and_order():
    assert parse_tier("L1") is FixTier.L1_PROMPT
    assert parse_tier("l3a") is FixTier.L3A_INTERNALS_READ
    assert parse_tier("L3") is FixTier.L3A_INTERNALS_READ  # bare L3 = read side
    assert parse_tier(FixTier.L4_PARAMETERS) is FixTier.L4_PARAMETERS
    assert FixTier.L1_PROMPT < FixTier.L2_SCAFFOLD < FixTier.L3A_INTERNALS_READ
    assert FixTier.L3B_INTERNALS_WRITE < FixTier.L4_PARAMETERS
    assert FixTier.L3B_INTERNALS_WRITE.label == "L3b"
    with pytest.raises(ValueError, match="unknown fix tier"):
        parse_tier("L9")


def _hyp(statement: str, mode: str = "", design: str = "") -> Hypothesis:
    return Hypothesis(statement=statement, target_model="m",
                      predicted_failure_mode=mode, test_design=design)


def test_routing_by_mechanism_keywords():
    tier, why = route_min_tier(_hyp(
        "pathologies smaller than one patch are destroyed by downsampling",
        mode="resolution_limit"))
    assert tier is FixTier.L2_SCAFFOLD and "resolution" in why

    tier, _ = route_min_tier(_hyp(
        "suppress the attention sink on structural tokens", mode="attention_sink"))
    assert tier is FixTier.L3B_INTERNALS_WRITE  # write verbs beat bare "attention"

    tier, _ = route_min_tier(_hyp(
        "attention mass never reaches the image region", mode="attention_dispersion"))
    assert tier is FixTier.L3A_INTERNALS_READ

    tier, _ = route_min_tier(_hyp(
        "the model relies on a finding frequency prior from training"))
    assert tier is FixTier.L4_PARAMETERS

    tier, why = route_min_tier(_hyp("the model answers too tersely"))
    assert tier is FixTier.L1_PROMPT and "cheapest" in why


# ── L2 image tools + pipeline executor ───────────────────────────────────────


def _img(size=(64, 48)):
    PIL = pytest.importorskip("PIL")
    from PIL import Image  # noqa: F401

    return PIL.Image.new("L", size, color=100)


def test_image_tools_preserve_or_scale_size():
    from evalvitals.eval_agent.stages.fix_tools import upscale, zoom_center

    img = _img()
    assert zoom_center(img, factor=2.0).size == img.size
    assert upscale(img, factor=2.0).size == (128, 96)


def test_apply_image_ops_skips_unknown_and_loads_paths(tmp_path):
    from evalvitals.eval_agent.stages.fix_tools import apply_image_ops

    path = tmp_path / "x.png"
    _img().save(path)
    out = apply_image_ops(str(path), [
        {"tool": "no_such_tool", "params": {}},
        {"tool": "upscale", "params": {"factor": 2.0}},
    ])
    assert out.size == (128, 96)


def test_pipeline_spec_validation():
    from evalvitals.eval_agent.stages.fix_tools import PipelineSpec

    assert PipelineSpec.from_dict({"name": "x", "prompt_template": "no placeholder"}) is None
    spec = PipelineSpec.from_dict({
        "name": "zoom", "image_ops": [{"tool": "zoom_center", "params": {"factor": 2}},
                                      {"tool": "bogus"}],
        "n_samples": 99,
    })
    assert spec is not None
    assert [op["tool"] for op in spec.image_ops] == ["zoom_center"]  # bogus dropped
    assert spec.n_samples == 5  # capped


# ── fake models ───────────────────────────────────────────────────────────────


class BaselineFailsModel(Model):
    """Answers "no" unless the prompt asks to examine carefully -> then "yes"."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        p = str(getattr(inputs, "prompt", inputs)).lower()
        return "Yes." if "carefully" in p else "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class ZoomSensitiveModel(Model):
    """Answers "yes" only when the image was upscaled (width >= 100)."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        img = getattr(inputs, "image", None)
        w = img.size[0] if img is not None and hasattr(img, "size") else 0
        return "Yes." if w >= 100 else "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class HopelessModel(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class ScriptedJudge(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.prompts: list[str] = []

    def generate(self, inputs, **kwargs) -> str:
        self.prompts.append(str(inputs))
        return self._reply

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def _gold_yes_batch(n: int = 8, image=None) -> CaseBatch:
    yes = {"all_of": ["yes"], "none_of": ["no"]}
    return CaseBatch([
        FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"Is there a lesion {i}?", image=image),
                    expected=yes, label=Label.FAIL)
        for i in range(n)
    ])


# ── FixAgent: L1 repair via judge proposal ───────────────────────────────────


def test_l1_judge_candidate_validates_and_fixes():
    judge = ScriptedJudge(json.dumps([
        {"name": "careful", "prompt_template": "Look very carefully. {prompt}"},
    ]))
    agent = FixAgent(judge=judge, max_tier="L1")
    out = agent.propose_and_validate(BaselineFailsModel(), _gold_yes_batch(),
                                     [_hyp("the prompt phrasing underspecifies the task")])
    assert out.fixed is True
    assert out.best is not None and out.best.candidate.tier is FixTier.L1_PROMPT
    assert out.best.n_fixed == 8 and out.best.n_broken == 0
    assert out.best.effect and out.best.effect > 0
    assert out.recommendation is None
    # max_tier=L1 -> no L2 candidates were attempted
    assert all(v.candidate.tier is FixTier.L1_PROMPT for v in out.attempted)


def test_l2_pipeline_candidate_fixes_zoom_sensitive_model():
    pytest.importorskip("PIL")
    judge = ScriptedJudge(json.dumps([
        {"name": "upscale2x",
         "image_ops": [{"tool": "upscale", "params": {"factor": 2.0}}],
         "prompt_template": "{prompt}", "n_samples": 1},
    ]))
    agent = FixAgent(judge=judge, max_tier="L2")
    out = agent.propose_and_validate(
        ZoomSensitiveModel(), _gold_yes_batch(image=_img()),
        [_hyp("small findings are destroyed by downsampling", mode="resolution_limit")])
    assert out.fixed is True
    assert out.best.candidate.name == "upscale2x"
    assert out.best.candidate.tier is FixTier.L2_SCAFFOLD


# ── FixAgent: unfixable -> recommendation, no auto-escalation ────────────────


def test_unfixable_recommends_routed_tier_above_max():
    pytest.importorskip("PIL")
    agent = FixAgent(judge=None, max_tier="L2")  # defaults only
    out = agent.propose_and_validate(
        HopelessModel(), _gold_yes_batch(image=_img()),
        [_hyp("suppress the attention sink on structural tokens")])
    assert out.fixed is False and out.best is None
    assert out.recommendation is not None
    assert out.recommendation["recommend_tier"] == "L3b"
    assert "beyond the allowed L2" in out.recommendation["reason"]
    # routing recorded per hypothesis
    assert out.routed[0]["min_tier"] == "L3b"


def test_unfixable_with_low_routes_recommends_next_tier():
    pytest.importorskip("PIL")
    agent = FixAgent(judge=None, max_tier="L2")
    out = agent.propose_and_validate(
        HopelessModel(), _gold_yes_batch(image=_img()),
        [_hyp("the prompt phrasing is fine but answers are wrong")])
    assert out.fixed is False
    assert out.recommendation["recommend_tier"] == "L3a"  # next above L2
    assert "no candidate within L2" in out.recommendation["reason"]


def test_at_l4_no_higher_recommendation():
    agent = FixAgent(judge=None, max_tier="L4")
    out = agent.propose_and_validate(HopelessModel(), _gold_yes_batch(),
                                     [_hyp("requires retraining on new data")])
    assert out.fixed is False and out.recommendation is None


# ── FixAgent: robustness ─────────────────────────────────────────────────────


def test_garbage_judge_falls_back_to_defaults():
    pytest.importorskip("PIL")
    agent = FixAgent(judge=ScriptedJudge("I refuse to answer in JSON."), max_tier="L2")
    out = agent.propose_and_validate(HopelessModel(), _gold_yes_batch(image=_img()),
                                     [_hyp("x")])
    sources = {v.candidate.source for v in out.attempted}
    assert sources == {"default"}
    tiers = {v.candidate.tier for v in out.attempted}
    assert tiers == {FixTier.L1_PROMPT, FixTier.L2_SCAFFOLD}


def test_no_rubric_cases_yield_recommendation_not_crash():
    cases = CaseBatch([FailureCase(id="u", inputs=Inputs(prompt="q"), label=Label.FAIL)])
    agent = FixAgent(judge=None, max_tier="L1")
    out = agent.propose_and_validate(HopelessModel(), cases, [_hyp("x")])
    assert out.fixed is False
    assert out.attempted == []
    assert "no case carries a scoring rubric" in out.recommendation["reason"]


def test_broken_cases_counted_and_net_negative_not_fixed():
    """A candidate that repairs nothing and breaks passing cases must not pass."""

    class InvertModel(Model):
        capabilities = frozenset({Capability.GENERATE})
        modalities = frozenset({"text", "image"})

        def generate(self, inputs, **kwargs):
            p = str(getattr(inputs, "prompt", inputs)).lower()
            return "No." if "carefully" in p else "Yes."

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError

    judge = ScriptedJudge(json.dumps([
        {"name": "careful", "prompt_template": "Look very carefully. {prompt}"}]))
    agent = FixAgent(judge=judge, max_tier="L1")
    out = agent.propose_and_validate(InvertModel(), _gold_yes_batch(), [_hyp("x")])
    v = out.attempted[0]
    assert v.n_broken == 8 and v.n_fixed == 0
    assert v.fixed is False and out.fixed is False


def test_outcome_serializes_and_logs(tmp_path):
    from evalvitals.eval_agent import RunLogger

    judge = ScriptedJudge(json.dumps([
        {"name": "careful", "prompt_template": "Look carefully. {prompt}"}]))
    logger = RunLogger(tmp_path / "logs")
    agent = FixAgent(judge=judge, max_tier="L1", run_logger=logger)
    out = agent.propose_and_validate(BaselineFailsModel(), _gold_yes_batch(), [_hyp("x")])
    d = out.to_dict()
    json.dumps(d)  # fully serializable
    assert d["max_tier"] == "L1" and d["fixed"] is True
    log_text = (tmp_path / "logs" / "run_log.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in log_text.splitlines()]
    fix_events = [e for e in events if e.get("event") == "fix"]
    assert len(fix_events) == 1
    assert fix_events[0]["attempted"][0]["name"] == "careful"


# ── loop integration ─────────────────────────────────────────────────────────


def test_run_fix_on_loop_report():
    from evalvitals.eval_agent import VLDiagnoseLoop, VLDiagnoseReport
    from evalvitals.eval_agent.hypothesis import HypothesisStatus
    from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTestResult
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    h = _hyp("prompt phrasing underspecifies the task")
    h.status = HypothesisStatus.SUPPORTED
    report = VLDiagnoseReport(
        cycles=1, stopped_by="max_cycles",
        verified_hypotheses=[HypothesisTestResult(
            hypothesis=h, status=HypothesisStatus.SUPPORTED,
            test_name="fail_rate_comparison", effect_size=0.5,
            is_consistent_with_protocol=True, confidence=0.8, verdict="ok")],
    )
    judge = ScriptedJudge(json.dumps([
        {"name": "careful", "prompt_template": "Look very carefully. {prompt}"}]))
    loop = VLDiagnoseLoop(model=BaselineFailsModel(),
                          protocol=ExperimentProtocol(description="d"))
    out = loop.run_fix(report, _gold_yes_batch(), max_tier="L1",
                       fix_agent=FixAgent(judge=judge, max_tier="L1"))
    assert out.fixed is True
    assert report.fix_outcome is out
