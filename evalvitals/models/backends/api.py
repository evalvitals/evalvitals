"""API backend — remote / OpenAI-compatible models (incl. ``vllm serve``).

Capabilities: ``GENERATE`` + ``TOOL_CALLS`` (+ ``LOGPROBS`` when the endpoint
returns top-logprobs, e.g. OpenAI but not Gemini).  No internals: ``forward``
raises, and capability matching means white-box analyzers are simply never
offered for an API model.

Reuse of your existing engine is by **dependency injection**: pass your
``call_vision_api`` adapted through :func:`call_vision_api_generate_fn` as
``RuntimeConfig.generate_fn`` — no cross-repo import, no rewrite.  A ``vllm serve``
endpoint is just this backend pointed at ``localhost`` (the answer to "is
vllm_local the same as api?": *for serving, yes* — see ``vllm_offline`` for the
distinct in-process batch path).
"""

from __future__ import annotations

from typing import Any, Callable

from evalvitals.core.capability import Capability, CapabilityError
from evalvitals.core.case import Inputs
from evalvitals.core.model import Model, Trace
from evalvitals.models.backends.base import Backend, RuntimeConfig

_WHITEBOX = {
    Capability.ATTENTION,
    Capability.HIDDEN_STATES,
    Capability.LOGITS,
    Capability.GRADIENTS,
    Capability.ACTIVATIONS,
    Capability.EMBEDDINGS,
}


def call_vision_api_generate_fn(
    call_vision_api: Callable,
    *,
    default_sampling: dict | None = None,
) -> Callable[..., str]:
    """Adapt an existing ``call_vision_api(model, messages, sampling_params, tools)``
    into the simple ``generate_fn(prompt, model=..., **kw) -> str`` an APIModel needs.

    This is the one-liner that reuses your XSkill engine verbatim::

        from engine.api_caller import call_vision_api
        rt = RuntimeConfig(generate_fn=call_vision_api_generate_fn(call_vision_api))
    """

    def _fn(prompt: str, model: str, **kw) -> str:
        messages = [{"role": "user", "content": prompt}]
        sampling = {**(default_sampling or {}), **kw}
        resp = call_vision_api(model, messages, sampling)
        if isinstance(resp, dict):  # tool/reasoning shape -> pull text
            return resp.get("content") or str(resp)
        return resp

    return _fn


class APIModel(Model):
    """A model reachable only through a text-in / text-out API."""

    def __init__(self, spec, runtime: RuntimeConfig) -> None:
        self.spec = spec
        self.runtime = runtime
        self._generate_fn = runtime.generate_fn
        caps = {Capability.GENERATE, Capability.TOOL_CALLS}
        if runtime.client_kwargs.get("logprobs"):
            caps.add(Capability.LOGPROBS)
        self.capabilities = frozenset(caps)

    def generate(self, inputs: Any, **kwargs) -> str:
        if self._generate_fn is None:
            raise RuntimeError(
                f"APIModel({self.spec.key!r}) has no generate_fn. Inject one via "
                "RuntimeConfig(generate_fn=...), e.g. call_vision_api_generate_fn(call_vision_api)."
            )
        prompt = inputs.prompt if isinstance(inputs, Inputs) else str(inputs)
        model_name = self.spec.hf_repo or self.spec.key
        return self._generate_fn(prompt, model=model_name, **kwargs)

    def forward(self, inputs: Any, capture: set[Capability], spec=None) -> Trace:
        missing = set(capture) & _WHITEBOX
        raise CapabilityError(analyzer="forward", model=repr(self), missing=missing or set(capture))

    def __repr__(self) -> str:
        return f"APIModel(key={self.spec.key!r}, caps={sorted(c.value for c in self.capabilities)})"


class APIBackend(Backend):
    kind = "api"
    capabilities = frozenset({Capability.GENERATE, Capability.TOOL_CALLS, Capability.LOGPROBS})

    def build(self, spec, runtime: RuntimeConfig) -> APIModel:
        return APIModel(spec, runtime)
