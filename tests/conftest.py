"""Shared test fixtures: a fully-mocked Qwen that needs no weights or GPU."""

from __future__ import annotations

import torch

from evalvitals.core.capability import Capability
from evalvitals.core.model import Model, Trace


class FakeModel(Model):
    """A minimal in-memory Model for tests.

    Declares a configurable capability set and returns a deterministic Trace
    from ``forward`` — no HuggingFace, no GPU.
    """

    def __init__(
        self,
        capabilities: set[Capability] | None = None,
        n_layers: int = 3,
        n_heads: int = 4,
        seq_len: int = 5,
    ) -> None:
        self.capabilities = frozenset(
            capabilities
            if capabilities is not None
            else {Capability.GENERATE, Capability.ATTENTION, Capability.HIDDEN_STATES}
        )
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._seq_len = seq_len

    def generate(self, inputs, **kwargs) -> str:
        return "fake-output"

    def forward(self, inputs, capture: set[Capability], spec=None) -> Trace:
        torch.manual_seed(0)
        provided: set[Capability] = set()
        attentions = hidden_states = logits = None
        if Capability.ATTENTION in capture and Capability.ATTENTION in self.capabilities:
            attentions = [
                torch.rand(self._n_heads, self._seq_len, self._seq_len)
                for _ in range(self._n_layers)
            ]
            provided.add(Capability.ATTENTION)
        if Capability.HIDDEN_STATES in capture and Capability.HIDDEN_STATES in self.capabilities:
            hidden_states = [torch.rand(self._seq_len, 8) for _ in range(self._n_layers + 1)]
            provided.add(Capability.HIDDEN_STATES)
        if Capability.LOGITS in capture and Capability.LOGITS in self.capabilities:
            logits = torch.rand(self._seq_len, 32)
            provided.add(Capability.LOGITS)
        return Trace(
            tokens=[f"t{i}" for i in range(self._seq_len)],
            token_ids=list(range(self._seq_len)),
            provided=provided,
            attentions=attentions,
            hidden_states=hidden_states,
            logits=logits,
        )

    def __repr__(self) -> str:
        return f"FakeModel(caps={sorted(c.value for c in self.capabilities)})"
