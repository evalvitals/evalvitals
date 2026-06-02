"""Per-version Qwen factories (thin wrappers over compose(spec, 'hf_local'))."""

from __future__ import annotations

import pytest

from evalvitals.core.capability import Capability
from evalvitals.models.whitebox import qwen as qmod
from evalvitals.models.whitebox import qwen_vl as qvlmod
from evalvitals.specs import list_specs


def test_new_qwen_specs_registered():
    keys = set(list_specs())
    for k in ["qwen2.5-14b-instruct", "qwen2.5-72b-instruct", "qwen3-14b", "qwen3-32b",
              "qwen3-235b-a22b", "qwen2.5-vl-3b-instruct", "qwen2.5-vl-72b-instruct",
              "qwen3-vl-2b-instruct", "qwen3-vl-30b-a3b-instruct", "qwen3-vl-235b-a22b-instruct"]:
        assert k in keys, f"missing spec {k}"


def test_text_factory_builds_lazy_hf_local_model():
    m = qmod.qwen3_8b()  # lazy — no weights loaded
    assert m.spec.key == "qwen3-8b" and not m.spec.is_vlm
    assert Capability.ATTENTION in m.capabilities and Capability.LOGPROBS in m.capabilities
    assert repr(m).endswith("lazy)")


def test_vl_factory_builds_vlm():
    vm = qvlmod.qwen3_vl_8b_instruct()
    assert vm.spec.key == "qwen3-vl-8b-instruct" and vm.spec.is_vlm
    assert "image" in vm.modalities
    moe = qvlmod.qwen3_vl_30b_a3b_instruct()
    assert moe.spec.is_moe and moe.spec.is_vlm


def test_factory_names_exposed():
    for name in ["qwen2_5_7b_instruct", "qwen2_5_72b_instruct", "qwen3_235b_a22b"]:
        assert name in qmod.__all__ and callable(getattr(qmod, name))
    for name in ["qwen3_vl_8b_instruct", "qwen3_vl_30b_a3b_instruct", "qwen2_5_vl_3b_instruct"]:
        assert name in qvlmod.__all__ and callable(getattr(qvlmod, name))


def test_legacy_shims_still_work_with_deprecation():
    with pytest.warns(DeprecationWarning):
        m = qmod.QwenLLM()
    assert m.spec.key == "qwen2.5-7b-instruct"
    with pytest.warns(DeprecationWarning):
        vm = qvlmod.QwenVL()
    assert vm.spec.key == "qwen2.5-vl-7b-instruct" and vm.spec.is_vlm
