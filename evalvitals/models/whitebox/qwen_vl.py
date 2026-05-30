"""QwenVL vision-language model (white-box, local deployment).

Planned for Stage 2.  When implemented it will declare the same white-box
capabilities as :class:`~evalvitals.models.whitebox.qwen.QwenLLM` plus
multimodal ones, and accept ``Inputs`` with an ``image`` field.
"""

from __future__ import annotations

from typing import Any

from evalvitals.core.capability import Capability
from evalvitals.core.model import Trace
from evalvitals.models.whitebox.base import WhiteboxModel


class QwenVL(WhiteboxModel):
    """QwenVL multimodal model — text + image inputs (Stage 2)."""

    capabilities = frozenset(
        {
            Capability.GENERATE,
            Capability.LOGITS,
            Capability.ATTENTION,
            Capability.HIDDEN_STATES,
        }
    )

    def __init__(
        self,
        checkpoint: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.dtype = dtype

    def load(self) -> None:
        raise NotImplementedError("QwenVL is planned for Stage 2.")

    def generate(self, inputs: Any, **kwargs) -> str:
        raise NotImplementedError("QwenVL is planned for Stage 2.")

    def forward(self, inputs: Any, capture: set[Capability]) -> Trace:
        raise NotImplementedError("QwenVL is planned for Stage 2.")

    def __repr__(self) -> str:
        return f"QwenVL(checkpoint={self.checkpoint!r})"
