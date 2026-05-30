"""Shapley-value modality contribution (MM-SHAP).

Planned for Stage 2. References:
- MM-SHAP: https://github.com/Heidelberg-NLP/MM-SHAP
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("shapley")
class ShapleyAnalyzer(Analyzer):
    """Estimate per-modality Shapley values via input perturbation."""

    name = "shapley"
    requires = frozenset({Capability.GENERATE})

    def _run(self, model, cases):
        raise NotImplementedError("ShapleyAnalyzer is planned for Stage 2.")
