"""Model-level activation pattern analysis.

Planned for Stage 2. References:
- Unveiling Multimodal Processing (activation patterns in MLLMs)
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("activation")
class ActivationAnalyzer(Analyzer):
    """Capture and analyse activation patterns in hidden layers."""

    name = "activation"
    requires = frozenset({Capability.ACTIVATIONS})

    def _run(self, model, cases):
        raise NotImplementedError("ActivationAnalyzer is planned for Stage 2.")
