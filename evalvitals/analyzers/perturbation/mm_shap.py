"""MM-SHAP — modality-contribution metric.

Black-box (``LOGPROBS``): Shapley over the input's text tokens + the image (as a
player), scored by the model's output logprob, then aggregated by modality:

    mm_score = |image contribution| / (|text contribution| + |image contribution|)

0 ⇒ the prediction relies entirely on text; 1 ⇒ entirely on the image.  Measures
*reliance*, not correctness — report it as such.

The score is injectable (``score_fn(Inputs) -> float``); the default reduces
``model.logprobs(masked_input)`` to a mean logprob (a generation-confidence proxy
— pass a target-likelihood ``score_fn`` for a more faithful score).

Paper: "MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal
       Contributions in Vision and Language Models"
       Parcalabescu & Frank, ACL 2022 — https://arxiv.org/abs/2212.08158
Code:  https://github.com/coastalcph/mm-shap
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from evalvitals.analyzers.perturbation._shapley import shapley_values
from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.case import Inputs
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model

_IMAGE = "__image__"


@register_analyzer("mm_shap")
class MMShapAnalyzer(Analyzer):
    """Per-modality Shapley contribution + the MM-SHAP image-reliance score."""

    name = "mm_shap"
    requires = frozenset({Capability.LOGPROBS})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        score_fn: Optional[Callable[[Inputs], float]] = None,
        n_samples: int = 64,
        mask_token: str = "___",
        top_k: int = 5,
        seed: int = 0,
    ) -> None:
        super().__init__(score_fn=score_fn, n_samples=n_samples, mask_token=mask_token, top_k=top_k, seed=seed)

    def _default_scorer(self, model: "Model") -> Callable[[Inputs], float]:
        def score(inputs: Inputs) -> float:
            lps = model.logprobs(inputs)
            return sum(t.logprob for t in lps) / len(lps) if lps else 0.0
        return score

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        case = cases[0]
        words = str(case.inputs.prompt).split()
        has_image = case.inputs.image is not None
        score = self.score_fn or self._default_scorer(model)

        players: list = list(range(len(words)))
        if has_image:
            players.append(_IMAGE)

        def value(kept: set) -> float:
            masked = " ".join(w if i in kept else self.mask_token for i, w in enumerate(words))
            img = case.inputs.image if (has_image and _IMAGE in kept) else None
            return score(Inputs(prompt=masked, image=img))

        shap = shapley_values(players, value, n_samples=self.n_samples, seed=self.seed)
        text_contrib = sum(abs(shap[i]) for i in range(len(words)))
        image_contrib = abs(shap.get(_IMAGE, 0.0))
        total = text_contrib + image_contrib or 1.0
        top_text = sorted(
            ({"token": words[i], "shapley": round(shap[i], 4)} for i in range(len(words))),
            key=lambda d: -abs(d["shapley"]),
        )[: self.top_k]
        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"shapley": shap},
            findings={
                "mm_score": round(image_contrib / total, 4),   # 0=text-reliant, 1=image-reliant
                "text_contribution": round(text_contrib, 4),
                "image_contribution": round(image_contrib, 4),
                "has_image": has_image,
                "top_text_tokens": top_text,
                "_note": "measures modality reliance, not correctness",
            },
        )
