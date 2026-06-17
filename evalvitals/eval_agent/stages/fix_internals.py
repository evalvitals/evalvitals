"""L3b fix executors — internals-**modifying** repairs (the sandbox boundary).

This module holds only what genuinely cannot be handed to a sandboxed coding
agent: pre-audited, parameterised primitives that **write** to the forward pass.

* **L3a (internals, read)** is intentionally NOT here at all.  Reading attention
  needs no privileged model handle, so the capability is exposed to sandboxed
  coded pipelines via ``model_attend()`` (see :mod:`fix_pipeline`), and the
  agent writes its own peak-find → crop → re-ask scaffold.  The host-side
  capture that backs ``model_attend()`` is
  :func:`~evalvitals.analyzers.attention.relative_attn.attention_heatmap` — a
  generic attention reducer that lives with the analyzers (one reduction shared
  with the white-box probe path), not in this fix module.  (An earlier
  ``attention_guided_crop`` primitive was removed for the same reason — it
  duplicated what the coded path already writes.)
* **L3b (internals, write)**: pre-audited intervention primitives that modify the
  forward pass — **never** free codegen against the model handle, because
  arbitrary hook code with the raw model object cannot be sandboxed.  The judge
  selects and parameterises; it never authors these.  v1 ships *visual embedding
  boost*: a forward hook on the input-embedding layer scaling image-token
  embeddings by ``gamma`` (architecture-agnostic for HF VLMs whose image tokens
  are placeholder ids in ``input_ids``).  Attention-map editing needs
  per-architecture hooks and joins this registry later.

L4 (parameter space) is **defined but TODO**: :class:`FinetuneSpec` captures a
complete fine-tune recipe (dataset construction generalising the verified
hypothesis, method, target, evaluation protocol incl. a regression battery);
no executor exists yet — the fix module records the recipe without running it.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.eval_agent.stages.fix_tiers import FixTier
from evalvitals.eval_agent.stages.fix_tools import score_to_bool

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model

logger = logging.getLogger(__name__)

ScoreFn = Callable[["FailureCase", str], Optional[bool]]


# ---------------------------------------------------------------------------
# L3b — visual embedding boost (forward-hook intervention)
# ---------------------------------------------------------------------------

def _resolve_hf(model: "Model"):
    """Underlying (HF model, image_token_id) or (None, None) when unavailable."""
    hf = getattr(model, "_hf", None)
    hf_model = hf[0] if isinstance(hf, tuple) and hf else None
    if hf_model is None:
        return None, None
    cfg = getattr(hf_model, "config", None)
    for attr in ("image_token_id", "image_token_index"):
        tid = getattr(cfg, attr, None)
        if tid is not None:
            return hf_model, int(tid)
    return None, None


def boost_available(model: "Model") -> bool:
    return _resolve_hf(model)[0] is not None


@contextmanager
def visual_embedding_boost(model: "Model", gamma: float = 1.5):
    """Scale image-token embeddings by *gamma* for every forward inside the block."""
    hf_model, image_token_id = _resolve_hf(model)
    if hf_model is None:
        raise RuntimeError(
            "visual_embedding_boost: backend internals unavailable "
            "(needs a loaded hf_local model exposing image_token_id)"
        )
    embedding = hf_model.get_input_embeddings()
    gamma = float(gamma)

    def _hook(module, args, output):
        input_ids = args[0]
        mask = input_ids == image_token_id
        if mask.any():
            output = output.clone()
            output[mask] = output[mask] * gamma
        return output

    handle = embedding.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def run_visual_embedding_boost(
    model: "Model",
    cases: "CaseBatch",
    score_fn: ScoreFn,
    params: "dict[str, Any] | None" = None,
) -> "dict[str, Optional[bool]]":
    """Generate every case under the boost hook; score against the rubric."""
    gamma = float((params or {}).get("gamma", 1.5))
    scores: "dict[str, Optional[bool]]" = {}
    try:
        with visual_embedding_boost(model, gamma=gamma):
            for case in cases:
                try:
                    out = str(model.generate(case.inputs))
                except Exception as exc:
                    logger.debug("boosted generate failed on %s: %s", case.id, exc)
                    scores[case.id] = None
                    continue
                scores[case.id] = score_to_bool(score_fn(case, out))
    except RuntimeError as exc:
        logger.warning("visual_embedding_boost unavailable: %s", exc)
        return {c.id: None for c in cases}
    return scores


# ---------------------------------------------------------------------------
# Primitive registry (judge selects + parameterises; code is pre-audited)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InternalsPrimitive:
    """A host-side, pre-audited L3 intervention primitive."""

    name: str
    tier: FixTier
    description: str
    params_hint: str
    available: "Callable[[Model], bool]"
    run: "Callable[[Model, CaseBatch, ScoreFn, dict | None], dict[str, Optional[bool]]]"


#: Pre-audited internals-WRITE primitives only.  Reads (L3a) are not here — the
#: agent authors them against the ``model_attend()`` bridge (see module docstring).
INTERNALS_PRIMITIVES: "dict[str, InternalsPrimitive]" = {
    "visual_embedding_boost": InternalsPrimitive(
        name="visual_embedding_boost",
        tier=FixTier.L3B_INTERNALS_WRITE,
        description="scale image-token embeddings by gamma via a forward hook "
                    "(amplifies visual evidence against language priors)",
        params_hint='{"gamma": float > 1 (default 1.5)}',
        available=boost_available,
        run=run_visual_embedding_boost,
    ),
}


def primitives_catalog_text(model: "Model", max_tier: FixTier) -> str:
    """Render available primitives (≤ *max_tier*, supported by *model*)."""
    lines = []
    for prim in INTERNALS_PRIMITIVES.values():
        if prim.tier <= max_tier and prim.available(model):
            lines.append(f"- {prim.name} [{prim.tier.label}]: {prim.description}  "
                         f"[params: {prim.params_hint}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# L4 — parameter space: DEFINED, executor TODO
# ---------------------------------------------------------------------------

@dataclass
class FinetuneSpec:
    """A complete L4 fine-tune recipe — recorded by the fix module, NOT executed.

    TODO(L4 executor): dataset synthesis from ``dataset_recipe``, LoRA/SFT
    training of ``target``, then validation per ``eval_protocol`` (held-out
    split + regression battery against catastrophic forgetting).  Until that
    lands, :class:`~.fix_agent.FixAgent` records the recipe as a non-validated
    candidate so the escalation decision has something concrete to act on.

    Attributes:
        dataset_recipe: How to build training data that *generalises* the
                        verified mechanism (never just the failing cases).
        method:         Training method, e.g. ``"lora"`` / ``"sft"``.
        target:         Component to tune, e.g. ``"vision_encoder"`` /
                        ``"llm"`` / ``"projector"`` / ``"full"``.
        eval_protocol:  How the tuned model must be validated — held-out
                        repair effect AND no-regression on passing cases.
        rationale:      Why parameter-space change is the minimum effective
                        intervention for the hypothesis.
    """

    dataset_recipe: str
    method: str = "lora"
    target: str = "llm"
    eval_protocol: str = (
        "paired McNemar on held-out failures + regression battery on all "
        "baseline-passing cases"
    )
    rationale: str = ""
    metadata: "dict[str, Any]" = field(default_factory=dict)

    def to_dict(self) -> "dict[str, Any]":
        return {
            "dataset_recipe": self.dataset_recipe,
            "method": self.method,
            "target": self.target,
            "eval_protocol": self.eval_protocol,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }
