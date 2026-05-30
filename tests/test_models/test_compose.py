"""ModelSpec x Backend composition + capability negotiation."""

from __future__ import annotations

import pytest

from evalvitals.core.capability import Capability, CapabilityError
from evalvitals.models import RuntimeConfig, compose
from evalvitals.models.backends import BACKENDS
from evalvitals.models.backends.api import APIModel, call_vision_api_generate_fn


def test_backends_registered():
    assert set(BACKENDS) == {"api", "hf_local", "vllm_offline"}


def test_capability_from_backend_not_spec():
    # hf_local can provide attention; api cannot — same spec, different caps.
    assert Capability.ATTENTION in BACKENDS["hf_local"].capabilities
    assert Capability.ATTENTION not in BACKENDS["api"].capabilities


def test_negotiation_raises_before_load():
    # Requesting ATTENTION against the api backend fails up front (no weights load).
    with pytest.raises(CapabilityError) as ei:
        compose("qwen3-vl-8b-instruct", "api", want={Capability.ATTENTION})
    assert Capability.ATTENTION in ei.value.missing


def test_api_only_spec_refuses_local_backend():
    with pytest.raises(ValueError):
        compose("step-1o-vision", "hf_local")


def test_api_model_generate_via_injected_fn():
    rt = RuntimeConfig(generate_fn=lambda prompt, model, **k: f"[{model}] {prompt}")
    m = compose("qwen3-8b", "api", rt)
    assert isinstance(m, APIModel)
    assert m.generate("hi") == "[Qwen/Qwen3-8B] hi"
    assert m.capabilities == frozenset({Capability.GENERATE, Capability.TOOL_CALLS})


def test_logprobs_capability_toggled_by_client_kwargs():
    rt = RuntimeConfig(generate_fn=lambda *a, **k: "x", client_kwargs={"logprobs": True})
    m = compose("qwen3-8b", "api", rt)
    assert Capability.LOGPROBS in m.capabilities


def test_api_model_forward_refuses_internals():
    m = compose("qwen3-8b", "api", RuntimeConfig(generate_fn=lambda *a, **k: "x"))
    with pytest.raises(CapabilityError):
        m.forward("hi", capture={Capability.ATTENTION})


def test_call_vision_api_adapter_builds_messages():
    seen = {}

    def fake_engine(model, messages, sampling):
        seen["model"], seen["messages"] = model, messages
        return "engine-reply"

    fn = call_vision_api_generate_fn(fake_engine)
    out = fn("hello", model="gpt-x")
    assert out == "engine-reply"
    assert seen["messages"] == [{"role": "user", "content": "hello"}]


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        compose("qwen3-8b", "nope")
