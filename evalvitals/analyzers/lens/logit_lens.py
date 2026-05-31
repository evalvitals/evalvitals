"""Logit lens (nostalgebraist) — project intermediate hidden states through the unembed.

Reveals *when* (which layer) the model's prediction forms.  Needs ``HIDDEN_STATES``
plus the model's unembedding matrix (``model.unembed_weight()``).  The cheapest,
highest-value white-box signal — build it first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model


@register_analyzer("logit_lens")
class LogitLensAnalyzer(Analyzer):
    """Top predicted token at the final position, read off each layer's hidden state.

    Hyper-parameters:
        pos:   query position to read (default ``-1``, the last token).
        top_k: number of candidates to report per layer.
    """

    name = "logit_lens"
    requires = frozenset({Capability.HIDDEN_STATES})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(self, pos: int = -1, top_k: int = 3) -> None:
        super().__init__(pos=pos, top_k=top_k)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import torch

        W = model.unembed_weight()
        if W is None:
            raise ValueError(
                f"{type(model).__name__} exposes no unembed_weight(); logit-lens needs it "
                "(white-box local backend)."
            )
        case = cases[0]
        trace = model.forward(case.inputs, capture={Capability.HIDDEN_STATES})
        hidden = trace.require(Capability.HIDDEN_STATES)  # list per layer: (seq, dim)
        W = W.float()  # (vocab, dim)

        per_layer = []
        for layer_idx, h in enumerate(hidden):
            vec = h[self.pos].float()              # (dim,)
            logits = vec @ W.T                     # (vocab,)
            k = min(self.top_k, logits.shape[-1])
            topk = torch.topk(logits, k)
            per_layer.append({
                "layer": layer_idx,
                "top": [{"token_id": int(i), "logit": round(float(v), 3)} for v, i in zip(topk.values, topk.indices)],
            })
        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"hidden_states": hidden},
            findings={"n_layers": len(hidden), "pos": self.pos, "per_layer_top": per_layer},
        )
