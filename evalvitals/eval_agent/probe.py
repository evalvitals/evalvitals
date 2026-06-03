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
    ) -> list[str]:
        """Return compatible analyzer names ranked by diagnostic priority.

        Args:
            model:         The model to analyse.
            max_analyzers: If given, cap the returned list at this length.

        Returns:
            Ordered list of registered analyzer names.  Priority-list items
            come first (in priority order), then any remaining compatible
            analyzers sorted alphabetically.
        """
        kind = self.detect_kind(model)
        compatible = set(registry.analyzers.names_compatible_with(model))
        priority = self._priority.get(kind, [])

        ranked = [name for name in priority if name in compatible]
        ranked += sorted(compatible - set(ranked))

        if max_analyzers is not None:
            ranked = ranked[:max_analyzers]
        return ranked
