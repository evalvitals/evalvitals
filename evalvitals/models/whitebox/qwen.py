"""Qwen language model (white-box, local deployment).

Capabilities provided: GENERATE, LOGITS, ATTENTION, HIDDEN_STATES.
Any analyzer whose ``requires`` is a subset of these runs on Qwen with no
per-analysis wiring — the registry matches them automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evalvitals.core.capability import Capability
from evalvitals.core.case import Inputs
from evalvitals.core.model import Trace
from evalvitals.core.registry import register_model
from evalvitals.models.whitebox.base import WhiteboxModel


@dataclass
class _HFInternals:
    model: Any
    tokenizer: Any


# Maps a requested Capability → the HF forward() flag that produces it.
_CAPTURE_FLAGS: dict[Capability, str] = {
    Capability.ATTENTION: "output_attentions",
    Capability.HIDDEN_STATES: "output_hidden_states",
}


@register_model("qwen")
class QwenLLM(WhiteboxModel):
    """Qwen language model loaded locally via HuggingFace Transformers.

    Args:
        checkpoint: HuggingFace model ID or local path.
        device:     ``"cuda"``, ``"cpu"``, ``"auto"``, or ``"cuda:N"``.
        dtype:      ``"float16"``, ``"bfloat16"``, or ``"float32"``.

    Examples::

        model = QwenLLM()
        print(model.generate("What is the capital of France?"))

        # canonical, analyzer-centric:
        from evalvitals.analysis.whitebox.attention import AttentionAnalyzer
        result = AttentionAnalyzer().run(model, "The Eiffel Tower is in")

        # hybrid convenience shim (same thing, derived from capabilities):
        result = model.call_attention("The Eiffel Tower is in")
    """

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
        checkpoint: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self._hf: _HFInternals | None = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load tokenizer and model weights into memory."""
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        model.eval()
        self._hf = _HFInternals(model=model, tokenizer=tokenizer)

    @property
    def _loaded(self) -> _HFInternals:
        if self._hf is None:
            self.load()
        return self._hf

    def _input_device(self) -> "torch.device":
        return next(self._loaded.model.parameters()).device

    @staticmethod
    def _as_prompt(inputs: Any) -> str:
        """Accept a raw string or an :class:`Inputs` object."""
        return inputs.prompt if isinstance(inputs, Inputs) else str(inputs)

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def generate(self, inputs: Any, max_new_tokens: int = 512, **kwargs) -> str:
        hf = self._loaded
        prompt = self._as_prompt(inputs)
        enc = hf.tokenizer(prompt, return_tensors="pt").to(self._input_device())
        with torch.no_grad():
            out = hf.model.generate(**enc, max_new_tokens=max_new_tokens, **kwargs)
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        return hf.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def forward(self, inputs: Any, capture: set[Capability], spec=None) -> Trace:
        """Run one forward pass, capturing requested internals into a :class:`Trace`.

        ``spec`` (:class:`~evalvitals.core.model.CaptureSpec`) is accepted for
        interface compatibility; this legacy adapter captures all layers.
        """
        hf = self._loaded
        prompt = self._as_prompt(inputs)
        enc = hf.tokenizer(prompt, return_tensors="pt").to(self._input_device())
        token_ids: list[int] = enc["input_ids"][0].tolist()
        tokens: list[str] = [hf.tokenizer.decode([tid]) for tid in token_ids]

        flags = {flag: True for cap, flag in _CAPTURE_FLAGS.items() if cap in capture}
        with torch.no_grad():
            outputs = hf.model(**enc, **flags)

        provided: set[Capability] = set()
        attentions = hidden_states = logits = None

        if Capability.ATTENTION in capture and outputs.attentions is not None:
            attentions = [a.squeeze(0).cpu() for a in outputs.attentions]
            provided.add(Capability.ATTENTION)
        if Capability.HIDDEN_STATES in capture and outputs.hidden_states is not None:
            hidden_states = [h.squeeze(0).cpu() for h in outputs.hidden_states]
            provided.add(Capability.HIDDEN_STATES)
        if Capability.LOGITS in capture:
            logits = outputs.logits.squeeze(0).cpu()
            provided.add(Capability.LOGITS)

        return Trace(
            tokens=tokens,
            token_ids=token_ids,
            provided=provided,
            attentions=attentions,
            hidden_states=hidden_states,
            logits=logits,
        )

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "loaded" if self._hf else "not loaded"
        return f"QwenLLM(checkpoint={self.checkpoint!r}, device={self.device!r}, {status})"
