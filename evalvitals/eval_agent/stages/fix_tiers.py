"""Intervention-space tiers for the fix module (post-loop repair).

A *fix tier* names the space a candidate repair intervenes in.  The ladder is
ordered by invasiveness into the model — each step gives up some deployability
for more causal reach:

    L1   input space        prompt rewrites, instruction strategies
    L2   scaffold space     agent-designed pipelines around the unchanged
                            model: multi-call, external tools (zoom, contrast
                            enhancement, …), aggregation
    L3a  internals (read)   read attention/logits to guide scaffold actions
                            (attention-guided crop, contrastive decoding,
                            confidence routing)
    L3b  internals (write)  modify the forward pass: attention reweighting,
                            sink suppression, activation steering
    L4   parameter space    build a dataset, fine-tune, re-test

The allowed tier is an **input** to the fix module (default L2) — there is no
automatic escalation.  When every candidate within the allowed tier fails
validation, the fix module *recommends* raising the tier instead, routed from
the verified hypotheses via :func:`route_min_tier`.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalvitals.eval_agent.hypothesis import Hypothesis


class FixTier(IntEnum):
    """Where a candidate fix intervenes, ordered by invasiveness."""

    L1_PROMPT = 1
    L2_SCAFFOLD = 2
    L3A_INTERNALS_READ = 3
    L3B_INTERNALS_WRITE = 4
    L4_PARAMETERS = 5

    @property
    def label(self) -> str:
        return _LABELS[self]

    def describe(self) -> str:
        return f"{self.label} ({_DESCRIPTIONS[self]})"


_LABELS = {
    FixTier.L1_PROMPT: "L1",
    FixTier.L2_SCAFFOLD: "L2",
    FixTier.L3A_INTERNALS_READ: "L3a",
    FixTier.L3B_INTERNALS_WRITE: "L3b",
    FixTier.L4_PARAMETERS: "L4",
}

_DESCRIPTIONS = {
    FixTier.L1_PROMPT: "input space: prompt/instruction changes",
    FixTier.L2_SCAFFOLD: "scaffold space: pipelines + external tools around the model",
    FixTier.L3A_INTERNALS_READ: "internals, read-only: attention/logits guide the scaffold",
    FixTier.L3B_INTERNALS_WRITE: "internals, write: attention/activation modification",
    FixTier.L4_PARAMETERS: "parameter space: dataset construction + fine-tuning",
}

_PARSE = {
    "l1": FixTier.L1_PROMPT,
    "l2": FixTier.L2_SCAFFOLD,
    "l3a": FixTier.L3A_INTERNALS_READ,
    "l3b": FixTier.L3B_INTERNALS_WRITE,
    "l3": FixTier.L3A_INTERNALS_READ,   # bare "L3" means the read side
    "l4": FixTier.L4_PARAMETERS,
}


def parse_tier(value: "str | FixTier") -> FixTier:
    """Parse ``"L1"``/``"l3a"``/… (or pass through a :class:`FixTier`)."""
    if isinstance(value, FixTier):
        return value
    key = str(value).strip().lower()
    if key not in _PARSE:
        raise ValueError(
            f"unknown fix tier {value!r}; expected one of "
            f"{sorted({t.label for t in FixTier})}"
        )
    return _PARSE[key]


# ---------------------------------------------------------------------------
# Hypothesis → minimum effective tier routing
# ---------------------------------------------------------------------------
# Keyword tables are checked from the most invasive tier downwards so that
# specific mechanism vocabulary ("suppress attention sink") wins over generic
# vocabulary ("attention") that also appears in lower tiers.  A hypothesis's
# verified mechanism tells you the *space* the failure lives in, hence the
# minimum tier whose interventions can causally address it.

_TIER_KEYWORDS: "list[tuple[FixTier, tuple[str, ...]]]" = [
    (FixTier.L4_PARAMETERS, (
        "train", "finetun", "fine-tun", "retrain", "pretrain", "lora",
        "knowledge gap", "frequency prior", "language prior", "memoriz",
        "dataset bias",
    )),
    (FixTier.L3B_INTERNALS_WRITE, (
        "suppress", "steer", "reweight", "re-weight", "knockout",
        "activation patch", "modify attention", "edit attention",
        "attention interven",
    )),
    (FixTier.L3A_INTERNALS_READ, (
        "attention", "logit", "logprob", "contrastive decoding", "vcd",
        "hidden state", "internal", "entropy", "sink",
    )),
    (FixTier.L2_SCAFFOLD, (
        "resolution", "downsampl", "image token", "patch grid", "zoom",
        "crop", "contrast", "conspicuity", "small", "subtle", "low-contrast",
        "preprocess", "multi-call", "pipeline", "tool", "enhanc",
    )),
]


def route_min_tier(hypothesis: "Hypothesis") -> "tuple[FixTier, str]":
    """Return the minimum tier that can causally address *hypothesis*.

    Matches mechanism keywords in ``predicted_failure_mode`` + ``statement`` +
    ``test_design``; defaults to L1 (cheapest first) when nothing matches.
    """
    text = " ".join(
        str(getattr(hypothesis, attr, "") or "")
        for attr in ("predicted_failure_mode", "statement", "test_design")
    ).lower()
    text = text.replace("training-free", "").replace("train-free", "")
    for tier, keywords in _TIER_KEYWORDS:
        hits = [k for k in keywords if k in text]
        if hits:
            return tier, f"matched {hits[:3]} -> {tier.label}"
    return FixTier.L1_PROMPT, "no mechanism keywords matched -> L1 (cheapest first)"
