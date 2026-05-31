"""Self-consistency — sample N generations and measure agreement.

A near-free black-box uncertainty signal (needs only ``GENERATE``): low agreement
across samples flags brittle/uncertain answers.  Runs on API models too.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model


@register_analyzer("self_consistency")
class SelfConsistencyAnalyzer(Analyzer):
    """Sample ``n`` generations and report the modal-answer fraction (consistency).

    Hyper-parameters:
        n:          number of samples.
        gen_kwargs: passed to ``model.generate`` (e.g. ``{"temperature": 0.7}``).
    """

    name = "self_consistency"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(self, n: int = 5, gen_kwargs: dict | None = None) -> None:
        super().__init__(n=n, gen_kwargs=gen_kwargs or {})

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        case = cases[0]
        samples = [model.generate(case.inputs, **self.gen_kwargs) for _ in range(self.n)]
        norm = [s.strip().lower() for s in samples]
        counts = Counter(norm)
        modal, modal_n = counts.most_common(1)[0]
        return Result(
            analyzer=self.name,
            model=repr(model),
            cases=cases,
            artifacts={"samples": samples},
            findings={
                "n_samples": self.n,
                "consistency": round(modal_n / max(len(samples), 1), 4),
                "n_unique": len(counts),
                "modal_answer": samples[norm.index(modal)],
            },
        )
