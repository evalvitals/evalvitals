"""Fix module — tiered, validated repair attempts after the diagnosis loop.

Design (intervention-space tiers, see :mod:`fix_tiers`): the allowed tier is
an **input** (default L2).  The agent proposes candidate fixes inside the
allowed tiers, compiles every candidate to the same shape — a per-case success
function, exactly :mod:`~evalvitals.eval_agent.ab_runner`'s *strategy*
contract — and validates each against the unmodified baseline with the paired
machinery from :mod:`evalvitals.stats` (McNemar + e-value, never a bare p).

There is **no automatic escalation**: when no candidate within the allowed
tier validates, the outcome carries a *recommendation* to raise the tier,
routed from the verified hypotheses' mechanisms (:func:`route_min_tier`).

v1 executors cover L1 (prompt transforms) and L2 (tool-catalog pipelines,
:mod:`fix_tools`).  L3a/L3b/L4 exist as routing/recommendation targets; their
executors need host-side internals hooks / training infrastructure and land
separately.

A *fixed* verdict means: paired McNemar rejects with positive net effect —
the candidate repairs significantly more cases than it breaks.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.analyzers.perturbation.prompt_contrast import _default_score
from evalvitals.eval_agent.stages.fix_tiers import FixTier, parse_tier, route_min_tier
from evalvitals.eval_agent.stages.fix_tools import (
    PipelineSpec,
    catalog_text,
    run_pipeline,
)
from evalvitals.stats import compare

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model
    from evalvitals.eval_agent.hypothesis import Hypothesis

logger = logging.getLogger(__name__)

_MAX_JUDGE_CANDIDATES = 3
_EXAMPLE_PROMPTS = 3

_L1_PROMPT = """\
You are designing PROMPT-LEVEL fixes (tier L1: the input space only) for a \
vision-language model failure.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}

EXAMPLE FAILING PROMPTS:
{examples}

Propose up to {k} prompt rewrite strategies that could repair these failures
WITHOUT changing the model or adding pipeline steps.  Each strategy is a
template applied to every case prompt; it MUST contain the literal placeholder
{{prompt}}.

Reply with ONLY a JSON array:
[{{"name": "<short_snake_case>", "prompt_template": "<template with {{prompt}}>"}}]"""

_L2_PROMPT = """\
You are designing SCAFFOLD-LEVEL fixes (tier L2: a pipeline around the \
unchanged model) for a vision-language model failure.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}

EXAMPLE FAILING PROMPTS:
{examples}

AVAILABLE IMAGE TOOLS (applied to the case image before the model sees it):
{catalog}

Propose up to {k} pipelines.  Each may chain image tools, rewrite the prompt
(template MUST contain {{prompt}}), and sample the model n_samples times
(majority vote).  Reply with ONLY a JSON array:
[{{"name": "<short_snake_case>",
   "image_ops": [{{"tool": "<catalog name>", "params": {{}}}}],
   "prompt_template": "{{prompt}}", "n_samples": 1}}]"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FixCandidate:
    """One proposed repair, compiled later to an ab_runner-style strategy.

    Attributes:
        tier:        Intervention space the candidate lives in.
        name:        Short identifier.
        payload:     Tier-specific spec — L1: ``{"prompt_template": ...}``;
                     L2: a :class:`~.fix_tools.PipelineSpec` dict.
        source:      ``"judge"`` or ``"default"`` (deterministic fallback).
    """

    tier: FixTier
    name: str
    payload: "dict[str, Any]"
    source: str = "judge"


@dataclass
class FixValidation:
    """Paired validation of one candidate against the unmodified baseline."""

    candidate: FixCandidate
    n_pairs: int = 0
    n_fixed: int = 0
    n_broken: int = 0
    fixed_cases: "list[str]" = field(default_factory=list)
    broken_cases: "list[str]" = field(default_factory=list)
    effect: "float | None" = None
    reject: bool = False
    fixed: bool = False
    summary: str = ""


@dataclass
class FixOutcome:
    """Everything the fix module did, plus the escalation recommendation.

    ``recommendation`` is ``None`` when a candidate validated; otherwise
    ``{"recommend_tier": "L3a", "reason": ...}`` — the caller decides whether
    to re-run with a higher ``max_tier`` (never automatic).
    """

    max_tier: FixTier
    routed: "list[dict[str, str]]" = field(default_factory=list)
    attempted: "list[FixValidation]" = field(default_factory=list)
    best: "FixValidation | None" = None
    fixed: bool = False
    recommendation: "dict[str, Any] | None" = None

    def to_dict(self) -> "dict[str, Any]":
        return {
            "max_tier": self.max_tier.label,
            "routed": self.routed,
            "attempted": [
                {
                    "tier": v.candidate.tier.label,
                    "name": v.candidate.name,
                    "source": v.candidate.source,
                    "payload": v.candidate.payload,
                    "n_pairs": v.n_pairs,
                    "n_fixed": v.n_fixed,
                    "n_broken": v.n_broken,
                    "fixed_cases": v.fixed_cases,
                    "broken_cases": v.broken_cases,
                    "effect": v.effect,
                    "reject": v.reject,
                    "fixed": v.fixed,
                    "summary": v.summary,
                }
                for v in self.attempted
            ],
            "best": self.best.candidate.name if self.best else None,
            "fixed": self.fixed,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FixAgent:
    """Propose and validate tiered fixes for the loop's verified hypotheses.

    Args:
        judge:      LLM proposing candidates (deterministic defaults when
                    ``None`` or unparseable).
        max_tier:   Highest allowed intervention tier (input, default L2).
        score_fn:   ``(case, output) -> bool | None``; defaults to the
                    rubric scorer shared with prompt_contrast.
        run_logger: Optional RunLogger — records the outcome as a ``fix`` event.
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        max_tier: "str | FixTier" = FixTier.L2_SCAFFOLD,
        score_fn: "Callable[[FailureCase, str], Optional[bool]] | None" = None,
        run_logger: "Any | None" = None,
    ) -> None:
        self._judge = judge
        self.max_tier = parse_tier(max_tier)
        self._score = score_fn or _default_score
        self.run_logger = run_logger

    # -- public ---------------------------------------------------------

    def propose_and_validate(
        self,
        model: "Model",
        data: "CaseBatch",
        hypotheses: "list[Hypothesis]",
    ) -> FixOutcome:
        """Generate candidates within the allowed tiers, validate, recommend."""
        outcome = FixOutcome(max_tier=self.max_tier)
        routed_tiers: "list[FixTier]" = []
        for h in hypotheses:
            tier, why = route_min_tier(h)
            routed_tiers.append(tier)
            outcome.routed.append({
                "hypothesis": getattr(h, "statement", str(h))[:160],
                "min_tier": tier.label,
                "rationale": why,
            })

        baseline = self._baseline(model, data)
        if not any(v is not None for v in baseline.values()):
            logger.warning("FixAgent: no scorable case (no rubrics); nothing to validate")
            outcome.recommendation = self._recommend(routed_tiers, reason_prefix=(
                "no case carries a scoring rubric, so no fix can be validated"
            ))
            self._emit(outcome)
            return outcome

        for candidate in self._propose(hypotheses, data):
            validation = self._validate(candidate, model, data, baseline)
            outcome.attempted.append(validation)

        winners = [v for v in outcome.attempted if v.fixed]
        if winners:
            outcome.best = max(winners, key=lambda v: (v.effect or 0.0, -v.n_broken))
            outcome.fixed = True
        else:
            outcome.recommendation = self._recommend(routed_tiers)
        self._emit(outcome)
        return outcome

    # -- candidate generation --------------------------------------------

    def _propose(
        self, hypotheses: "list[Hypothesis]", data: "CaseBatch"
    ) -> "list[FixCandidate]":
        hyp_lines = "\n".join(
            f"- [{getattr(h, 'predicted_failure_mode', '')}] {getattr(h, 'statement', h)}"
            for h in hypotheses
        ) or "- (no verified hypotheses; failures are unexplained)"
        examples = "\n".join(
            f"- {str(getattr(c.inputs, 'prompt', ''))[:160]}"
            for c in list(data)
            if getattr(getattr(c, "label", None), "value", None) == "fail"
        )[: 1000] or "- (none)"

        candidates = self._l1_candidates(hyp_lines, examples)
        if self.max_tier >= FixTier.L2_SCAFFOLD:
            candidates += self._l2_candidates(hyp_lines, examples)
        if self.max_tier >= FixTier.L3A_INTERNALS_READ:
            logger.info(
                "FixAgent: tiers above L2 are allowed (max=%s) but their "
                "executors are not implemented yet; attempting L1/L2 only",
                self.max_tier.label,
            )
        return candidates

    def _l1_candidates(self, hyp_lines: str, examples: str) -> "list[FixCandidate]":
        proposals = self._ask_judge(_L1_PROMPT.format(
            hypotheses=hyp_lines, examples=examples, k=_MAX_JUDGE_CANDIDATES))
        out: "list[FixCandidate]" = []
        for p in proposals:
            template = str(p.get("prompt_template", ""))
            name = str(p.get("name", "")).strip()
            if name and "{prompt}" in template:
                out.append(FixCandidate(
                    tier=FixTier.L1_PROMPT, name=name,
                    payload={"prompt_template": template}))
        if not out:
            out = [FixCandidate(
                tier=FixTier.L1_PROMPT, name="attend_carefully", source="default",
                payload={"prompt_template": (
                    "Examine the image carefully, including small, subtle and "
                    "low-contrast regions, before answering. {prompt}")})]
        return out[:_MAX_JUDGE_CANDIDATES]

    def _l2_candidates(self, hyp_lines: str, examples: str) -> "list[FixCandidate]":
        proposals = self._ask_judge(_L2_PROMPT.format(
            hypotheses=hyp_lines, examples=examples, k=_MAX_JUDGE_CANDIDATES,
            catalog=catalog_text()))
        out: "list[FixCandidate]" = []
        for p in proposals:
            spec = PipelineSpec.from_dict(p) if isinstance(p, dict) else None
            if spec is not None:
                out.append(FixCandidate(
                    tier=FixTier.L2_SCAFFOLD, name=spec.name, payload=spec.to_dict()))
        if not out:
            out = [
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD, name="zoom_equalize", source="default",
                    payload=PipelineSpec(
                        name="zoom_equalize",
                        image_ops=[{"tool": "zoom_center", "params": {"factor": 1.6}},
                                   {"tool": "equalize", "params": {}}]).to_dict()),
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD, name="upscale_sharpen", source="default",
                    payload=PipelineSpec(
                        name="upscale_sharpen",
                        image_ops=[{"tool": "upscale", "params": {"factor": 2.0}},
                                   {"tool": "sharpen", "params": {"factor": 2.0}}]).to_dict()),
            ]
        return out[:_MAX_JUDGE_CANDIDATES]

    def _ask_judge(self, prompt: str) -> "list[dict[str, Any]]":
        if self._judge is None:
            return []
        try:
            raw = str(self._judge.generate(prompt))
        except Exception as exc:
            logger.warning("FixAgent: judge call failed: %s", exc)
            return []
        match = re.search(r"\[.*\]", re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL),
                          flags=re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("FixAgent: unparseable judge proposal; using defaults")
            return []
        return [p for p in parsed if isinstance(p, dict)] if isinstance(parsed, list) else []

    # -- strategy compilation + validation --------------------------------

    def _baseline(self, model: "Model", data: "CaseBatch") -> "dict[str, Optional[bool]]":
        scores: "dict[str, Optional[bool]]" = {}
        for case in data:
            try:
                output = str(model.generate(case.inputs))
            except Exception as exc:
                logger.debug("FixAgent: baseline generate failed on %s: %s", case.id, exc)
                scores[case.id] = None
                continue
            scores[case.id] = self._score(case, output)
        return scores

    def _strategy(self, candidate: FixCandidate) -> "Callable[[Model, FailureCase], Optional[bool]]":
        """Compile a candidate to a per-case success function (ab_runner shape)."""
        if candidate.tier == FixTier.L1_PROMPT:
            template = candidate.payload["prompt_template"]

            def l1(model: "Model", case: "FailureCase") -> "Optional[bool]":
                from evalvitals.core.case import Inputs

                inp = case.inputs
                new_inputs = Inputs(
                    prompt=template.format(prompt=str(getattr(inp, "prompt", ""))),
                    image=getattr(inp, "image", None))
                try:
                    return self._score(case, str(model.generate(new_inputs)))
                except Exception:
                    return None
            return l1

        spec = PipelineSpec.from_dict(candidate.payload)
        if spec is None:  # already validated at proposal time; belt and braces
            return lambda model, case: None
        return lambda model, case: run_pipeline(model, case, spec, self._score)

    def _validate(
        self,
        candidate: FixCandidate,
        model: "Model",
        data: "CaseBatch",
        baseline: "dict[str, Optional[bool]]",
    ) -> FixValidation:
        strategy = self._strategy(candidate)
        v = FixValidation(candidate=candidate)
        base_vec: "list[bool]" = []
        cand_vec: "list[bool]" = []
        for case in data:
            b = baseline.get(case.id)
            c = strategy(model, case)
            if b is None or c is None:
                continue
            base_vec.append(b)
            cand_vec.append(c)
            if not b and c:
                v.n_fixed += 1
                v.fixed_cases.append(case.id)
            elif b and not c:
                v.n_broken += 1
                v.broken_cases.append(case.id)
        v.n_pairs = len(base_vec)
        if v.n_pairs == 0:
            v.summary = "no scorable pair — candidate unvalidatable"
            return v
        try:
            stat = compare(base_vec, cand_vec, paired=True)
        except Exception as exc:
            v.summary = f"stats failed: {exc}"
            return v
        v.effect = stat.effect
        v.reject = bool(stat.reject)
        # Fixed = the paired test rejects with a net-positive effect: the
        # candidate repairs significantly more cases than it breaks.
        v.fixed = v.reject and (v.effect or 0.0) > 0
        v.summary = stat.summary()
        return v

    # -- recommendation + logging ------------------------------------------

    def _recommend(
        self, routed: "list[FixTier]", reason_prefix: str = ""
    ) -> "dict[str, Any] | None":
        above = sorted(t for t in routed if t > self.max_tier)
        if above:
            target = above[0]
            reason = (
                f"verified hypotheses route to {target.label} "
                f"({target.describe()}), beyond the allowed {self.max_tier.label}"
            )
        elif self.max_tier < FixTier.L4_PARAMETERS:
            target = FixTier(self.max_tier + 1)
            reason = (
                f"no candidate within {self.max_tier.label} validated; the next "
                f"intervention space is {target.describe()}"
            )
        else:
            return None  # already at L4 — nothing above to recommend
        if reason_prefix:
            reason = f"{reason_prefix}; {reason}"
        return {"recommend_tier": target.label, "reason": reason}

    def _emit(self, outcome: FixOutcome) -> None:
        if self.run_logger is None:
            return
        try:
            self.run_logger.log_fix(outcome)
        except Exception as exc:  # logging must never break the fix step
            logger.debug("FixAgent: log_fix failed: %s", exc)
