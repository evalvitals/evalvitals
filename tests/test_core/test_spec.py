"""ModelSpec layer: backend-orthogonal specs + registry."""

from __future__ import annotations

import pytest

from evalvitals.core.spec import AttnSemantics, ModelSpec, VisionSpec
from evalvitals.specs import REGISTRY, get_spec, list_specs


def test_llm_spec_has_no_vision():
    spec = get_spec("qwen3-8b")
    assert spec.vision is None
    assert spec.is_vlm is False
    assert spec.api_only is False


def test_vlm_spec_has_vision():
    spec = get_spec("qwen3-vl-8b-instruct")
    assert isinstance(spec.vision, VisionSpec)
    assert spec.is_vlm is True
    # token id is read from config by attribute NAME, never baked as a value
    assert spec.vision.image_token_id_attr == "image_token_id"


def test_api_only_spec():
    spec = get_spec("step-1o-vision")
    assert spec.api_only is True
    assert spec.module_paths is None
    assert spec.attn_semantics is AttnSemantics.NONE


def test_mla_models_flagged():
    for key in ("deepseek-v3", "kimi-vl-a3b-thinking"):
        assert get_spec(key).attn_semantics is AttnSemantics.MLA_LATENT


def test_registry_listing_and_unknown():
    assert "qwen3-vl-8b-instruct" in list_specs()
    assert set(list_specs()) == set(REGISTRY)
    with pytest.raises(KeyError):
        get_spec("does-not-exist")


def test_spec_is_frozen_dataclass():
    spec = ModelSpec(key="t", family="t", model_type="t")
    with pytest.raises(Exception):
        spec.key = "other"  # frozen
