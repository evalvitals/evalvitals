"""Tests for QwenLLM (capabilities + forward->Trace) and the load_model factory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from evalvitals.config import ModelConfig, load_config
from evalvitals.core import Capability, Trace
from evalvitals.models import load_model
from evalvitals.models.whitebox.qwen import QwenLLM, _HFInternals


# ------------------------------------------------------------------
# Helpers — mock HuggingFace internals so no weights/GPU are needed.
# ------------------------------------------------------------------

class _MockEncoding(dict):
    def to(self, device):
        return self


def _make_qwen(n_layers=2, n_heads=4, seq_len=5) -> QwenLLM:
    model = QwenLLM(checkpoint="mock/qwen", device="cpu")

    input_ids = torch.tensor([[10, 20, 30, 40, 50]])
    enc = _MockEncoding({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})

    fake_attn = torch.rand(1, n_heads, seq_len, seq_len)
    fake_hidden = torch.rand(1, seq_len, 8)
    mock_outputs = MagicMock()
    mock_outputs.attentions = [fake_attn] * n_layers
    mock_outputs.hidden_states = [fake_hidden] * (n_layers + 1)
    mock_outputs.logits = torch.rand(1, seq_len, 100)

    generated_ids = torch.cat([input_ids, torch.tensor([[999]])], dim=1)

    mock_hf_model = MagicMock()
    mock_hf_model.return_value = mock_outputs
    mock_hf_model.generate.return_value = generated_ids
    mock_hf_model.parameters.return_value = iter([torch.zeros(1)])

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = enc
    mock_tokenizer.decode.side_effect = lambda ids, **kw: f"<tok_{ids[0]}>" if ids else ""

    model._hf = _HFInternals(model=mock_hf_model, tokenizer=mock_tokenizer)
    return model


# ------------------------------------------------------------------
# Capabilities
# ------------------------------------------------------------------

def test_declares_capabilities():
    caps = QwenLLM.capabilities
    assert Capability.ATTENTION in caps
    assert Capability.HIDDEN_STATES in caps
    assert Capability.GENERATE in caps
    assert Capability.GRADIENTS not in caps


def test_supports():
    model = QwenLLM(device="cpu")
    assert model.supports({Capability.ATTENTION})
    assert not model.supports({Capability.GRADIENTS})


# ------------------------------------------------------------------
# forward() -> Trace
# ------------------------------------------------------------------

def test_forward_captures_attention():
    model = _make_qwen(n_layers=3, n_heads=8, seq_len=5)
    trace = model.forward("hello", capture={Capability.ATTENTION})
    assert isinstance(trace, Trace)
    assert Capability.ATTENTION in trace.provided
    assert len(trace.attentions) == 3
    assert trace.attentions[0].shape == torch.Size([8, 5, 5])
    assert trace.attentions[0].device.type == "cpu"


def test_forward_only_captures_requested():
    model = _make_qwen()
    trace = model.forward("hi", capture={Capability.ATTENTION})
    assert trace.attentions is not None
    assert trace.hidden_states is None  # not requested


def test_trace_require_missing_raises():
    model = _make_qwen()
    trace = model.forward("hi", capture={Capability.ATTENTION})
    with pytest.raises(ValueError, match="hidden_states"):
        trace.require(Capability.HIDDEN_STATES)


# ------------------------------------------------------------------
# generate()
# ------------------------------------------------------------------

def test_generate_returns_string():
    model = _make_qwen()
    assert isinstance(model.generate("Hello"), str)


# ------------------------------------------------------------------
# call_attention shim (via registry)
# ------------------------------------------------------------------

def test_call_attention_shim():
    from evalvitals.analysis.whitebox.attention import AttentionResult

    model = _make_qwen(n_layers=2, n_heads=4, seq_len=5)
    result = model.call_attention("The capital of France is")
    assert isinstance(result, AttentionResult)
    assert result.num_layers == 2


# ------------------------------------------------------------------
# load_model factory + repr
# ------------------------------------------------------------------

def test_repr_not_loaded():
    model = QwenLLM(checkpoint="Qwen/Qwen2.5-7B-Instruct")
    assert "not loaded" in repr(model)


def test_load_model_qwen():
    model = load_model(ModelConfig(name="qwen"))
    assert isinstance(model, QwenLLM)
    assert model.checkpoint == "Qwen/Qwen2.5-7B-Instruct"


def test_load_model_alias():
    model = load_model(ModelConfig(name="qwen2.5-14b"))
    assert isinstance(model, QwenLLM)


def test_load_model_unknown_raises():
    cfg = ModelConfig.__new__(ModelConfig)
    cfg.name = "unknown_model_xyz"
    cfg.checkpoint = "some/checkpoint"
    cfg.device = "cpu"
    cfg.dtype = "float32"
    with pytest.raises(ValueError, match="Unknown model"):
        load_model(cfg)


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------

def test_load_config_short_form(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("model: qwen\nanalysis: attention\n")
    cfg = load_config(cfg_file)
    assert cfg.model.name == "qwen"
    assert cfg.model.checkpoint == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.analysis == "attention"


def test_load_config_full_form(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(
        "model:\n"
        "  name: qwen\n"
        "  checkpoint: Qwen/Qwen2.5-72B-Instruct\n"
        "  device: cpu\n"
        "  dtype: float32\n"
        "analysis: attention\n"
        "analysis_kwargs:\n"
        "  layer: 0\n"
        "  top_k: 5\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.model.checkpoint == "Qwen/Qwen2.5-72B-Instruct"
    assert cfg.analysis_kwargs["top_k"] == 5


def test_model_config_unknown_name_raises():
    with pytest.raises(ValueError, match="No default checkpoint"):
        ModelConfig(name="unknown_xyz")
