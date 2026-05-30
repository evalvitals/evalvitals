"""Tests for capability declaration, matching, and enforcement."""

from __future__ import annotations

import pytest

import evalvitals.analysis  # noqa: F401  (populate analyzer registry)
from evalvitals.core import Capability, CapabilityError, registry
from tests.conftest import FakeModel


def test_model_supports_subset():
    model = FakeModel(capabilities={Capability.GENERATE, Capability.ATTENTION})
    assert model.supports({Capability.ATTENTION})
    assert model.supports({Capability.GENERATE, Capability.ATTENTION})
    assert not model.supports({Capability.GRADIENTS})


def test_registry_compatible_with_includes_attention():
    model = FakeModel(capabilities={Capability.ATTENTION})
    names = registry.analyzers.names_compatible_with(model)
    assert "attention" in names


def test_registry_compatible_with_excludes_gradient_analyzer():
    # Qwen-like model: no GRADIENTS → saliency must be excluded.
    model = FakeModel(
        capabilities={Capability.GENERATE, Capability.ATTENTION, Capability.HIDDEN_STATES}
    )
    names = registry.analyzers.names_compatible_with(model)
    assert "attention" in names
    assert "saliency" not in names      # requires GRADIENTS
    assert "activation" not in names    # requires ACTIVATIONS


def test_capability_error_on_missing():
    from evalvitals.analysis.whitebox.saliency import SaliencyAnalyzer

    model = FakeModel(capabilities={Capability.GENERATE})
    with pytest.raises(CapabilityError) as exc:
        SaliencyAnalyzer().run(model, "x")
    assert Capability.GRADIENTS in exc.value.missing


def test_capability_error_message_is_actionable():
    from evalvitals.analysis.whitebox.probing import ProbingAnalyzer

    model = FakeModel(capabilities={Capability.GENERATE})
    with pytest.raises(CapabilityError, match="hidden_states"):
        ProbingAnalyzer().run(model, "x")
