"""Embedding geometry analysis — CKA, cosine similarity, PCA (DINO).

Planned for Stage 2. References:
- Do VL Encoders Represent the World Similarly?
- Interpreting the Linear Structure of VL Embedding Spaces
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("embedding_geometry")
class EmbeddingGeometryAnalyzer(Analyzer):
    """CKA, cosine similarity, and PCA over model embedding spaces."""

    name = "embedding_geometry"
    requires = frozenset({Capability.HIDDEN_STATES})

    def _run(self, model, cases):
        raise NotImplementedError("EmbeddingGeometryAnalyzer is planned for Stage 2.")
