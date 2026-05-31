"""Gradient-based saliency analysis (Grad-CAM, VL-SHAP).

Planned for Stage 2. References:
- Grad-CAM: https://github.com/jacobgil/pytorch-grad-cam
- VL-SHAP:  https://github.com/explainability-vl/vl-shap
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("saliency")
class SaliencyAnalyzer(Analyzer):
    """Gradient-based saliency maps (Grad-CAM, guided backprop, VL-SHAP)."""

    name = "saliency"
    requires = frozenset({Capability.GRADIENTS})

    def _run(self, model, cases):
        raise NotImplementedError("SaliencyAnalyzer is planned for Stage 2.")
