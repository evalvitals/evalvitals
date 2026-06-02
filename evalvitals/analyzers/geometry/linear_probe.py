"""Linear probing — per-layer representation richness for a target property (Stage 2).

Train a linear classifier on each layer's hidden states (samples + labels) and
report per-layer accuracy — where in the stack is the property linearly decodable.
Needs a labelled probe set; correlational, not causal.

References:
- Understanding intermediate layers using linear classifier probes
  Alain & Bengio, 2016 — arXiv:1610.01644
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("linear_probe")
class LinearProbeAnalyzer(Analyzer):
    """Per-layer linear-probe accuracy for a labelled property."""

    name = "linear_probe"
    requires = frozenset({Capability.HIDDEN_STATES})
    applies_to_modalities = frozenset({"text", "image"})

    def _run(self, model, cases):
        raise NotImplementedError(
            "Stage 2: collect (hidden_state, label) pairs across cases, fit a per-layer "
            "logistic probe (with a held-out split), report accuracy curve over depth."
        )
