"""M1 — StrategyProbe: select which analyzers to run given a model.

The probe inspects the model's capabilities and modalities to determine what
kind of model it is (VLM, agent, or plain LLM), then returns a ranked list of
analyzer names from the registry that are both compatible with the model and
ordered by their diagnostic value for that model kind.

Usage::

    probe = StrategyProbe()
    kind  = probe.detect_kind(model)          # ModelKind.VLM / AGENT / LLM
    names = probe.select(model, max_analyzers=4)  # e.g. ["attention", "logit_lens", ...]
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from evalvitals.core.capability import Capability
from evalvitals.core.registry import registry

if TYPE_CHECKING:
    from evalvitals.core.model import Model


class ModelKind(str, Enum):
    VLM   = "vlm"    # image modality — hallucination + attention analyzers first
    AGENT = "agent"  # TOOL_CALLS — trajectory / behavioral analyzers first
    LLM   = "llm"    # text-only — interpretability analyzers first


# Maps failure-mode tags (from M3 hypotheses) to the analyzers most likely to
# surface evidence for that mode.  Used in cycle 2+ to focus the probe on what
# the diagnosis agent flagged rather than running the same generic priority list.
_FAILURE_MODE_TO_ANALYZERS: dict[str, list[str]] = {
    "attention_sink":            ["attention_sink"],
    "attention":                 ["attention", "attention_rollout"],
    "hallucination":             ["pope", "chair"],
    "low_consistency":           ["self_consistency"],
    "unstable_generation":       ["self_consistency"],
    "overconfidence":            ["verbalized_confidence"],
    "miscalibrated_confidence":  ["verbalized_confidence"],
    "confident_inconsistency":   ["self_consistency", "verbalized_confidence"],
    "loop":                      ["loop_detect"],
    "ignored_obs":               ["ignored_obs"],
    "entropy":                   ["token_entropy", "logprob_entropy"],
    "perplexity":                ["logprob_entropy"],
    "logit_lens":                ["logit_lens"],
    "representational_collapse": ["cka"],
    "numerical_hallucination":   ["self_consistency", "verbalized_confidence"],
}

# Per-kind ordered priority: high → low diagnostic value for false attribution.
# Analyzers not in the list are appended alphabetically after the ranked ones.
_PRIORITY: dict[str, list[str]] = {
    ModelKind.VLM: [
        "pope", "chair",                          # hallucination metrics
        "attention", "attention_rollout",          # where is the model looking?
        "attention_sink",                          # sink collapse?
        "mm_shap",                                # text vs image reliance
        "logprob_entropy", "self_consistency",
    ],
    ModelKind.AGENT: [
        "loop_detect", "ignored_obs",             # behavioral heuristics
        "first_error_judge", "counterfactual",
    ],
    ModelKind.LLM: [
        "attention", "logit_lens",                # interpretability
        "token_entropy", "logprob_entropy",
        "attention_sink", "attention_rollout",
        "cka", "self_consistency", "verbalized_confidence",
    ],
}


class StrategyProbe:
    """Selects analyzers appropriate for a given model.

    Args:
        priority_override: Replaces the built-in per-kind priority tables
            (keyed by ``ModelKind``).  Useful for domain-specific orderings.
    """

    def __init__(self, priority_override: dict[str, list[str]] | None = None) -> None:
        self._priority = priority_override or _PRIORITY

    def detect_kind(self, model: "Model") -> ModelKind:
        """Infer VLM / AGENT / LLM from the model's capabilities and modalities."""
        if Capability.TOOL_CALLS in getattr(model, "capabilities", frozenset()):
            return ModelKind.AGENT
        if "image" in getattr(model, "modalities", frozenset({"text"})):
            return ModelKind.VLM
        return ModelKind.LLM

    def select(
        self,
        model: "Model",
        max_analyzers: int | None = None,
        hint_failure_modes: list[str] | None = None,
    ) -> list[str]:
        """Return compatible analyzer names ranked by diagnostic priority.

        Args:
            model:              The model to analyse.
            max_analyzers:      If given, cap the returned list at this length.
            hint_failure_modes: Failure-mode tags from outstanding M3 hypotheses.
                                Analyzers that match a hint are promoted to the
                                front of the ranked list for focused follow-up.

        Returns:
            Ordered list of registered analyzer names.  Hint-matched items
            come first, then the standard priority-list items, then remaining
            compatible analyzers sorted alphabetically.
        """
        kind = self.detect_kind(model)
        compatible = set(registry.analyzers.names_compatible_with(model))
        priority = self._priority.get(kind, [])

        ranked = [name for name in priority if name in compatible]
        ranked += sorted(compatible - set(ranked))

        if hint_failure_modes:
            # Promote analyzers that map to outstanding failure modes, preserving
            # their relative order and avoiding duplicates.
            boosted = dict.fromkeys(
                a
                for mode in hint_failure_modes
                for a in _FAILURE_MODE_TO_ANALYZERS.get(mode.lower().replace(" ", "_"), [])
                if a in compatible
            )
            ranked = list(boosted) + [a for a in ranked if a not in boosted]

        if max_analyzers is not None:
            ranked = ranked[:max_analyzers]
        return ranked
