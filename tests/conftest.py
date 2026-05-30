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

    def forward(self, inputs, capture: set[Capability]) -> Trace:
        torch.manual_seed(0)
        attentions = None
        if Capability.ATTENTION in capture:
            attentions = [
                torch.rand(self._n_heads, self._seq_len, self._seq_len)
                for _ in range(self._n_layers)
            ]
        return Trace(
            tokens=[f"t{i}" for i in range(self._seq_len)],
            token_ids=list(range(self._seq_len)),
            provided={Capability.ATTENTION} if attentions is not None else set(),
            attentions=attentions,
        )

    def __repr__(self) -> str:
        return f"FakeModel(caps={sorted(c.value for c in self.capabilities)})"
