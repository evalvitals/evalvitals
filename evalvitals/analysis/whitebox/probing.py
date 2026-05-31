"""Linear probing — representation richness per layer.

Planned for Stage 2. References:
- Probing Multimodal LLMs: https://arxiv.org/abs/2402.11574
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("probing")
class ProbingAnalyzer(Analyzer):
    """Train lightweight linear classifiers on hidden states per layer."""

    name = "probing"
    requires = frozenset({Capability.HIDDEN_STATES})

    def _run(self, model, cases):
        raise NotImplementedError("ProbingAnalyzer is planned for Stage 2.")
