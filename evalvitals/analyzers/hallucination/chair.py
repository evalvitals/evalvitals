"""CHAIR — Caption Hallucination Assessment (Rohrbach et al.).

Ships the pure per-instance metric (``chair_score``); the analyzer that extracts
objects from generated captions + ground-truth object lists is Stage 2 (needs an
object vocabulary / extractor and a dataset).
"""

from __future__ import annotations

from typing import Iterable

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


def chair_score(generated_objects: Iterable[str], gt_objects: Iterable[str]) -> dict:
    """Per-instance CHAIR_i: fraction of mentioned objects that are hallucinated."""
    gen = {o.lower() for o in generated_objects}
    gt = {o.lower() for o in gt_objects}
    hallucinated = sorted(gen - gt)
    chair_i = len(hallucinated) / len(gen) if gen else 0.0
    return {"chair_i": round(chair_i, 4), "hallucinated": hallucinated, "n_mentioned": len(gen)}


@register_analyzer("chair")
class CHAIRAnalyzer(Analyzer):
    name = "chair"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"image"})

    def _run(self, model, cases):
        raise NotImplementedError(
            "Stage 2: generate a caption, extract objects (object vocabulary), then "
            "aggregate chair_score() across the dataset (CHAIR_i / CHAIR_s)."
        )
