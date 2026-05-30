"""VL-SHAP: Interpreting Vision and Language Generative Models.

Planned for Stage 2. References:
- Code: https://github.com/explainability-vl/vl-shap
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("vl_shap")
class VLShapAnalyzer(Analyzer):
    """SHAP-based attribution for black-box VL models."""

    name = "vl_shap"
    requires = frozenset({Capability.GENERATE})

    def _run(self, model, cases):
        raise NotImplementedError("VLShapAnalyzer is planned for Stage 2.")
