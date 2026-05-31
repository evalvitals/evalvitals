"""Generic Attention-model Explainability for Bi-Modal Transformers.

Planned for Stage 2. References:
- Code: https://github.com/hila-chefer/Transformer-MM-Explainability

Also covers: exCLIP second-order caption-image attribution.
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("transformer_mm")
class TransformerMMAnalyzer(Analyzer):
    """Attention-based explainability for bi-modal encoder-decoder transformers."""

    name = "transformer_mm"
    requires = frozenset({Capability.ATTENTION})

    def _run(self, model, cases):
        raise NotImplementedError("TransformerMMAnalyzer is planned for Stage 2.")
