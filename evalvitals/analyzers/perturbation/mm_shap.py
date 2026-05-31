"""MM-SHAP — modality-contribution metric (Parcalabescu & Frank) (Stage 2).

Measures how much each MODALITY (text vs image, generalises to N) contributes to
the prediction via Shapley over masked tokens. ``requires=LOGPROBS``. Measures
reliance, NOT correctness — report it as such.
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("mm_shap")
class MMShapAnalyzer(Analyzer):
    name = "mm_shap"
    requires = frozenset({Capability.LOGPROBS})
    applies_to_modalities = frozenset({"text", "image"})

    def _run(self, model, cases):
        raise NotImplementedError(
            "Stage 2: Shapley over masked text+image tokens scored by output logprob; "
            "report per-modality contribution share."
        )
