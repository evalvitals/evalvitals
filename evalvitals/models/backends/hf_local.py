"""HF-local backend — the only InternalsHandle path.

Loads any spec via ``transformers`` and captures internals.  Two-tier capture:

  * **HF flags** (``output_attentions`` / ``output_hidden_states`` / logits) cover
    attention, residual-stream hidden states and logits with NO module-path
    surgery — this is the high-value common path.
  * **hooks + runtime path discovery** (:mod:`evalvitals.models._discover`) are
    only needed beyond what flags give (activation patching, MoE routing,
    gradients) — reserved for Stage 2.

Attention capture REQUIRES eager attention (sdpa/flash silently return ``None``),
so when the model declares ``eager_required_for_attn`` we force
``attn_implementation="eager"``.  ``torch`` / ``transformers`` are imported
lazily so this module imports on a torch-free install.
"""

from __future__ import annotations

from typing import Any

from evalvitals.core.capability import Capability, CapabilityError
from evalvitals.core.case import Inputs
from evalvitals.core.model import Model, Trace
from evalvitals.core.spec import AttnSemantics
from evalvitals.core.tool import ChatTurn
from evalvitals.models.backends.base import Backend, RuntimeConfig

# capability -> HF forward flag
_CAPTURE_FLAGS = {
    Capability.ATTENTION: "output_attentions",
    Capability.HIDDEN_STATES: "output_hidden_states",
}


class HFLocalModel(Model):
    """A locally-loaded HF model, constructed from a :class:`ModelSpec`."""

    def __init__(self, spec, runtime: RuntimeConfig) -> None:
        self.spec = spec
        self.runtime = runtime
        self._hf = None  # (model, processor) — lazy
        caps = {
            Capability.GENERATE,
            Capability.LOGITS,
            Capability.LOGPROBS,
            Capability.HIDDEN_STATES,
        }
        if spec.attn_semantics is not AttnSemantics.NONE:
            caps.add(Capability.ATTENTION)
        # TOOL_CALLS is a CONDITIONAL capability for local models: the backend
        # provides the channel, but tool-calling only works if the model's chat
        # template renders tools (declared per-model via spec.tool_calling).
        if spec.tool_calling:
            caps.add(Capability.TOOL_CALLS)
        self.capabilities = frozenset(caps)
        self.modalities = frozenset({"text", "image"}) if spec.is_vlm else frozenset({"text"})

    @classmethod
    def from_loaded(cls, model, tokenizer, spec=None, runtime: "RuntimeConfig | None" = None) -> "HFLocalModel":
        """Wrap an ALREADY-LOADED HF model + tokenizer (the ``wrap()`` on-ramp).

        Unlike the spec-driven path, no weights are fetched: we inject the live
        ``(model, processor)`` directly and skip :meth:`load`.  The spec is inferred
        from the model when not given, and the attention capability is verified
        against the live ``attn_implementation`` (eager is required, else the model
        returns ``None`` attentions) — flipping it to eager when possible.
        """
        from evalvitals.models.inference import infer_spec

        spec = spec or infer_spec(model, tokenizer)
        self = cls(spec, runtime or RuntimeConfig())
        self._hf = (model, tokenizer)  # bypass lazy load(); a bare tokenizer acts as processor
        if Capability.ATTENTION in self.capabilities:
            self._ensure_eager_attention(model)
        return self

    @staticmethod
    def _ensure_eager_attention(model) -> None:
        """Best-effort switch to eager attention so ``output_attentions`` is populated.

        sdpa / flash_attention_2 silently return ``None`` attentions.  Newer
        transformers expose ``set_attn_implementation``; otherwise we set the config
        flag and warn that a reload may be required for it to take effect.
        """
        config = getattr(model, "config", None)
        current = getattr(config, "_attn_implementation", None) if config is not None else None
        if current == "eager":
            return
        if hasattr(model, "set_attn_implementation"):
            try:
                model.set_attn_implementation("eager")
                return
            except Exception:  # pragma: no cover - falls through to the config flag
                pass
        if config is not None:
            config._attn_implementation = "eager"
            import warnings

            warnings.warn(
                f"wrapped model used attn_implementation={current!r}; set it to 'eager' for "
                "attention capture. If attentions come back empty, reload the model with "
                "from_pretrained(..., attn_implementation='eager').",
                stacklevel=2,
            )

    def unembed_weight(self):
        """The lm_head / unembedding weight ``(vocab, dim)`` for logit-lens."""
        from evalvitals.models._discover import get_unembed

        model, _ = self._loaded
        head = get_unembed(model)
        return getattr(head, "weight", None)

    # -- lazy load -----------------------------------------------------
    def load(self) -> None:
        import torch
        import transformers

        auto_cls = getattr(transformers, self.spec.auto_class)
        proc_cls = getattr(transformers, self.spec.processor_class, transformers.AutoProcessor)

        attn_impl = self.runtime.attn_impl
        if attn_impl is None and self.spec.eager_required_for_attn and Capability.ATTENTION in self.capabilities:
            attn_impl = "eager"  # sdpa/flash return None attentions

        # Use `dtype` (the current transformers param; `torch_dtype` is deprecated).
        kwargs: dict[str, Any] = dict(
            dtype=getattr(torch, self.runtime.dtype),
            trust_remote_code=self.spec.trust_remote_code,
        )
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        device = self.runtime.device
        if device in (None, "auto") or isinstance(device, dict):
            # device_map path (multi-GPU / sharded) — needs accelerate
            model = auto_cls.from_pretrained(self.spec.hf_repo, device_map=device or "auto", **kwargs)
        else:
            # explicit single device ("cuda" / "cuda:0" / "cpu") — no accelerate dependency
            model = auto_cls.from_pretrained(self.spec.hf_repo, **kwargs).to(device)
        model.eval()
        processor = proc_cls.from_pretrained(self.spec.hf_repo, trust_remote_code=self.spec.trust_remote_code)

        # Verify the declared TOOL_CALLS capability against the actual template.
        if self.spec.tool_calling:
            tok = getattr(processor, "tokenizer", processor)
            template = getattr(tok, "chat_template", None) or ""
            if "tools" not in template:
                import warnings

                warnings.warn(
                    f"{self.spec.key!r}: spec.tool_calling=True but the chat template has no "
                    "'tools' handling — tool-calling may not render. Verify the checkpoint."
                )
        self._hf = (model, processor)

    @property
    def _loaded(self):
        if self._hf is None:
            self.load()
        return self._hf

    @staticmethod
    def _as_prompt(inputs: Any) -> str:
        return inputs.prompt if isinstance(inputs, Inputs) else str(inputs)

    def _encode(self, prompt: str):
        import torch  # noqa: F401

        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        enc = tok(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        return {k: v.to(device) for k, v in enc.items()}

    # -- interface -----------------------------------------------------
    def generate(self, inputs: Any, **kwargs) -> str:
        import torch

        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        prompt = self._as_prompt(inputs)
        enc = self._encode(prompt)
        max_new = kwargs.pop("max_new_tokens", self.runtime.max_new_tokens)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, **kwargs)
        new = out[0][enc["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True)

    def chat(self, messages: list, tools=None) -> ChatTurn:
        """Tool-aware turn via the model's chat template.

        transformers' ``apply_chat_template(tools=...)`` accepts OpenAI-format tool
        schemas and renders them into the prompt; the model emits the call as text
        which the (Qwen/Hermes) codec parses out — so ``raw_tool_calls`` is None.
        """
        import torch

        if Capability.TOOL_CALLS not in self.capabilities:
            raise CapabilityError(analyzer="chat", model=repr(self), missing={Capability.TOOL_CALLS})
        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        text = tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False,
            **self.spec.chat_template_kwargs,  # e.g. {"enable_thinking": False} for Qwen3
        )
        enc = tok(text, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=self.runtime.max_new_tokens)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return ChatTurn(text=gen, raw_tool_calls=None)

    def forward(self, inputs: Any, capture: set[Capability], spec=None) -> Trace:
        import torch

        if self.spec.is_vlm:
            raise NotImplementedError(
                f"{self.spec.key}: VLM forward-capture (image tokens + TokenTypeMap) is Stage 2. "
                "Text generation works; white-box image-token capture is not wired yet."
            )
        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        prompt = self._as_prompt(inputs)
        enc = self._encode(prompt)
        token_ids = enc["input_ids"][0].tolist()
        tokens = [tok.decode([t]) for t in token_ids]

        flags = {flag: True for cap, flag in _CAPTURE_FLAGS.items() if cap in capture}
        with torch.no_grad():
            outputs = model(**enc, **flags)

        layers = spec.layers if spec is not None else None
        to_cpu = spec.to_cpu if spec is not None else True

        def _maybe_subset(seq):
            return [seq[i] for i in layers] if layers is not None else list(seq)

        def _move(t):
            return t.cpu() if to_cpu else t

        provided: set[Capability] = set()
        attentions = hidden_states = logits = None
        if Capability.ATTENTION in capture:
            if getattr(outputs, "attentions", None) is None:
                raise RuntimeError(
                    f"{self!r}: ATTENTION was requested but the model returned no attentions. "
                    "Load it with attn_implementation='eager' (sdpa/flash silently return None)."
                )
            attentions = [_move(a.squeeze(0)) for a in _maybe_subset(outputs.attentions)]
            provided.add(Capability.ATTENTION)
        if Capability.HIDDEN_STATES in capture and getattr(outputs, "hidden_states", None) is not None:
            hidden_states = [_move(h.squeeze(0)) for h in _maybe_subset(outputs.hidden_states)]
            provided.add(Capability.HIDDEN_STATES)
        if Capability.LOGITS in capture:
            logits = _move(outputs.logits.squeeze(0))
            provided.add(Capability.LOGITS)

        return Trace(
            tokens=tokens,
            token_ids=token_ids,
            provided=provided,
            attentions=attentions,
            hidden_states=hidden_states,
            logits=logits,
            extras={"attn_semantics": self.spec.attn_semantics.value},
        )

    def __repr__(self) -> str:
        status = "loaded" if self._hf else "lazy"
        return f"HFLocalModel(key={self.spec.key!r}, {status})"


class HFLocalBackend(Backend):
    kind = "hf_local"
    # Superset the backend CAN provide; the actual per-model set is computed in
    # HFLocalModel.__init__ (e.g. TOOL_CALLS only when the spec's template supports it).
    capabilities = frozenset(
        {
            Capability.GENERATE,
            Capability.TOOL_CALLS,
            Capability.LOGITS,
            Capability.LOGPROBS,
            Capability.HIDDEN_STATES,
            Capability.ATTENTION,
        }
    )

    def build(self, spec, runtime: RuntimeConfig) -> HFLocalModel:
        return HFLocalModel(spec, runtime)
