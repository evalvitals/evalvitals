"""POPE — Polling-based Object Probing Evaluation (Li et al.) (Stage 2).

Ask the VLM yes/no questions about object presence and score accuracy/F1 (random /
popular / adversarial splits). Black-box (``GENERATE``); needs a POPE-format probe set.
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("pope")
class POPEAnalyzer(Analyzer):
    name = "pope"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"image"})

    def _run(self, model, cases):
        raise NotImplementedError(
            "Stage 2: pose yes/no object-presence probes, parse the answer, score "
            "accuracy/precision/recall/F1 over the POPE splits."
        )
