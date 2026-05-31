"""evalvitals.wrap(model, tokenizer) — the public on-ramp for user-supplied models.

These run on CPU with NO downloads: a tiny ``nn.Module`` + fake tokenizer stand in
for a real HF causal LM, exercising the same ``HFLocalModel`` capture path that the
curated ``load("qwen...")`` route uses.  (Real-weights parity lives behind ``--run-gpu``.)
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import evalvitals
from evalvitals.analyzers.lens.logit_lens import LogitLensAnalyzer
from evalvitals.core.capability import Capability
from evalvitals.models.backends.hf_local import HFLocalModel
from evalvitals.models.inference import infer_spec


# ----------------------------------------------------------------------
# Minimal fakes mimicking the transformers surface wrap() touches
# ----------------------------------------------------------------------
class _Cfg:
    def __init__(self, model_type="fakellm", n_layers=3, attn_impl="eager", name="fake/model"):
        self.model_type = model_type
        self.num_hidden_layers = n_layers
        self._name_or_path = name
        self._attn_implementation = attn_impl
        # NOTE: deliberately no `vision_config` -> looks like a text decoder-only model.


class _Outputs:
    def __init__(self, hidden_states=None, attentions=None, logits=None):
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.logits = logits


class _Layer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)  # makes _discover see a decoder layer


class FakeTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template

    def __call__(self, text, return_tensors="pt"):
        ids = list(range(1, len(text.split()) + 1)) or [1]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, skip_special_tokens=False):
        return f"t{ids[0]}"


class FakeCausalLM(nn.Module):
    """A tiny decoder-only stand-in: real params, HF-shaped forward outputs."""

    def __init__(self, n_layers=3, dim=8, vocab=16, n_heads=4, attn_impl="eager", emit_attn=True):
        super().__init__()
        self.config = _Cfg(n_layers=n_layers, attn_impl=attn_impl)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(dim) for _ in range(n_layers)])
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self._dim, self._vocab, self._n_layers, self._n_heads = dim, vocab, n_layers, n_heads
        self._emit_attn = emit_attn

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, output_hidden_states=False, output_attentions=False, **kw):
        seq = input_ids.shape[1]
        hs = tuple(torch.rand(1, seq, self._dim) for _ in range(self._n_layers + 1)) if output_hidden_states else None
        attns = (
            tuple(torch.rand(1, self._n_heads, seq, seq) for _ in range(self._n_layers))
            if output_attentions and self._emit_attn
            else None
        )
        return _Outputs(hidden_states=hs, attentions=attns, logits=torch.rand(1, seq, self._vocab))


# ----------------------------------------------------------------------
# infer_spec
# ----------------------------------------------------------------------
def test_infer_spec_reads_model_type_and_no_tools():
    spec = infer_spec(FakeCausalLM(), FakeTokenizer())
    assert spec.model_type == "fakellm"
    assert spec.hf_repo == ""          # already in memory -> never reloaded
    assert spec.tool_calling is False
    assert spec.is_vlm is False


def test_infer_spec_detects_tool_calling_from_template():
    spec = infer_spec(FakeCausalLM(), FakeTokenizer(chat_template="...{{ tools }}..."))
    assert spec.tool_calling is True


def test_infer_spec_rejects_vlm():
    model = FakeCausalLM()
    model.config.vision_config = object()  # now looks like a VLM
    with pytest.raises(NotImplementedError, match="VLM"):
        infer_spec(model, FakeTokenizer())


# ----------------------------------------------------------------------
# wrap() construction + capabilities
# ----------------------------------------------------------------------
def test_wrap_returns_hflocal_model():
    m = evalvitals.wrap(FakeCausalLM(), FakeTokenizer())
    assert isinstance(m, HFLocalModel)


def test_wrap_infers_internals_capabilities():
    m = evalvitals.wrap(FakeCausalLM(), FakeTokenizer())
    for cap in (Capability.GENERATE, Capability.LOGITS, Capability.HIDDEN_STATES, Capability.ATTENTION):
        assert cap in m.capabilities
    assert Capability.TOOL_CALLS not in m.capabilities  # no tools in template


def test_wrap_grants_tool_calls_when_template_supports_it():
    m = evalvitals.wrap(FakeCausalLM(), FakeTokenizer(chat_template="{{ tools }}"))
    assert Capability.TOOL_CALLS in m.capabilities


def test_wrap_want_negotiation_passes_for_available_cap():
    m = evalvitals.wrap(FakeCausalLM(), FakeTokenizer(), want={Capability.ATTENTION})
    assert Capability.ATTENTION in m.capabilities


def test_wrap_want_negotiation_rejects_missing_cap():
    from evalvitals.core.capability import CapabilityError

    with pytest.raises(CapabilityError):
        evalvitals.wrap(FakeCausalLM(), FakeTokenizer(), want={Capability.GRADIENTS})


# ----------------------------------------------------------------------
# attention fix-up
# ----------------------------------------------------------------------
def test_wrap_flips_non_eager_attention_and_warns():
    model = FakeCausalLM(attn_impl="sdpa")
    with pytest.warns(UserWarning, match="eager"):
        evalvitals.wrap(model, FakeTokenizer())
    assert model.config._attn_implementation == "eager"


def test_wrap_leaves_eager_attention_untouched():
    model = FakeCausalLM(attn_impl="eager")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning expected
        evalvitals.wrap(model, FakeTokenizer())


def test_forward_raises_actionable_error_when_attentions_missing():
    # Model never emits attentions (e.g. sdpa that didn't actually switch) -> clear error.
    m = evalvitals.wrap(FakeCausalLM(emit_attn=False), FakeTokenizer())
    with pytest.raises(RuntimeError, match="eager"):
        m.forward("hello world", capture={Capability.ATTENTION})


# ----------------------------------------------------------------------
# end-to-end: wrap -> analyze, no weights
# ----------------------------------------------------------------------
def test_wrap_then_logit_lens_runs():
    m = evalvitals.wrap(FakeCausalLM(n_layers=3, dim=8, vocab=16), FakeTokenizer())
    result = LogitLensAnalyzer(top_k=3).run(m, "the capital of france is")
    assert result.findings["n_layers"] == 4  # n_layers + 1 hidden states
    assert all(len(layer["top"]) == 3 for layer in result.findings["per_layer_top"])


def test_wrap_then_forward_captures_attention():
    m = evalvitals.wrap(FakeCausalLM(n_layers=2, n_heads=4), FakeTokenizer())
    trace = m.forward("hello world", capture={Capability.ATTENTION})
    assert Capability.ATTENTION in trace.provided
    assert len(trace.attentions) == 2                 # one per layer
    assert trace.attentions[0].shape[0] == 4          # heads, after squeeze(0)


def test_wrap_unembed_weight_exposed():
    m = evalvitals.wrap(FakeCausalLM(dim=8, vocab=16), FakeTokenizer())
    W = m.unembed_weight()
    assert tuple(W.shape) == (16, 8)  # (vocab, dim)
