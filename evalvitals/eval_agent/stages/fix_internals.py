"""L3b fix executors — internals-**modifying** repairs (the sandbox boundary).

This module holds only what genuinely cannot be handed to a sandboxed coding
agent: pre-audited, parameterised primitives that **write** to the forward pass,
plus the host-side :func:`attention_heatmap` helper that backs the **read**-only
``model_attend()`` bridge.

* **L3a (internals, read)** is intentionally NOT a primitive.  Reading attention
  needs no privileged model handle, so the capability is exposed to sandboxed
  coded pipelines via ``model_attend()`` (see :mod:`fix_pipeline`, built on
  :func:`attention_heatmap`) and the agent writes its own peak-find → crop →
  re-ask scaffold.  Anything the agent can author against the bridge does not
  belong in this registry.  (An earlier ``attention_guided_crop`` primitive was
  removed for exactly this reason — it duplicated what the coded path already
  writes.)
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

import numpy as np

from evalvitals.eval_agent.stages.fix_tiers import FixTier
from evalvitals.eval_agent.stages.fix_tools import score_to_bool

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model

logger = logging.getLogger(__name__)

ScoreFn = Callable[["FailureCase", str], Optional[bool]]


# ---------------------------------------------------------------------------
# L3a — attention heatmap (host-side helper that backs the bridged model_attend)
#
# There is deliberately NO canned attention-guided-crop primitive here: the read
# capability is exposed to sandboxed coded pipelines through model_attend()
# (implemented on top of this helper in fix_pipeline.py), and the agent writes
# its own peak-find + crop_region + re-ask scaffold.  Reads need no privileged
# model handle, so they belong on the agent side, not in this pre-audited
# registry — which is reserved for interventions that cannot be sandboxed (L3b).
# ---------------------------------------------------------------------------

def attention_heatmap(
    model: "Model", case: "FailureCase", layer: "int | float" = 0.75
) -> "np.ndarray | None":
    """One ATTENTION forward → (H, W) image-patch heatmap, or ``None``.

    Reduction mirrors the white-box probe capture: head-averaged attention from
    the last query position, restricted to image-token positions, reshaped via
    the backend's ``image_spatial_shape`` (near-square fallback).  ``layer`` is
    resolved by :func:`~evalvitals.analyzers.attention.relative_attn.resolve_attention_layer`
    — a float is a fractional depth (default 0.75, a spatially-grounded
    late-middle layer); the last layer is sink-dominated and localizes poorly.
    """
    from evalvitals.analyzers.attention.relative_attn import resolve_attention_layer
    from evalvitals.core.capability import Capability

    if Capability.ATTENTION not in getattr(model, "capabilities", frozenset()):
        return None
    try:
        trace = model.forward(case.inputs, capture={Capability.ATTENTION})
        attns = trace.require(Capability.ATTENTION)
        layer_idx = resolve_attention_layer(layer, len(attns))
        row = attns[layer_idx].float().mean(dim=0)[-1].cpu().numpy()  # (seq,)
        mask = trace.extras.get("image_token_mask")
        if mask is None:
            return None
        mask = np.asarray(mask.cpu().numpy() if hasattr(mask, "cpu") else mask, dtype=bool)
        if not mask.any() or mask.size != row.size:
            return None
        heat = row[mask].astype(np.float64)
        shape = trace.extras.get("image_spatial_shape")
        if shape is not None and int(shape[0]) * int(shape[1]) == heat.size:
            h, w = int(shape[0]), int(shape[1])
        else:  # near-square fallback
            h = max(1, int(np.sqrt(heat.size)))
            while heat.size % h:
                h -= 1
            w = heat.size // h
        return heat.reshape(h, w)
    except Exception as exc:
        logger.debug("attention_heatmap failed for %s: %s", getattr(case, "id", "?"), exc)
        return None


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
