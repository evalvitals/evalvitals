"""VL-SHAP — Shapley attribution over masked image regions (Stage 2).

Interpret VLM outputs by scoring masked semantic visual priors with output
probability. ``requires=LOGPROBS`` (coalition scoring); VLM-only; expensive.
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("vl_shap")
class VLShapAnalyzer(Analyzer):
    name = "vl_shap"
    requires = frozenset({Capability.LOGPROBS})
    applies_to_modalities = frozenset({"image"})

    def _run(self, model, cases):
        raise NotImplementedError(
            "Stage 2: Shapley over masked image regions scored by output logprob "
            "(needs LOGPROBS + a segmentation/superpixel prior)."
        )
