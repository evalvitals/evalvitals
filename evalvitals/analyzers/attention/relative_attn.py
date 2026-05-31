"""Relative attention to image tokens — "MLLMs Know Where to Look" (Stage 2).

Needs a TokenTypeMap to know which sequence positions are image-patch tokens, then
measures attention to the relevant image region vs the rest.  VLM-only.
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("relative_attention")
class RelativeAttentionAnalyzer(Analyzer):
    """Attention concentration on image-patch tokens (VLM)."""

    name = "relative_attention"
    requires = frozenset({Capability.ATTENTION})
    applies_to_modalities = frozenset({"image"})

    def _run(self, model, cases):
        raise NotImplementedError(
            "Stage 2: requires core.tokentype.TokenTypeMap to locate image-patch positions "
            "(blocked on VLM forward-capture in hf_local)."
        )
