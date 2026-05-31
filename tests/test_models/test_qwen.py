"""Back-compat tests for the deprecated ``QwenLLM`` shim.

The concrete Qwen class is gone — identity lives in ``evalvitals.specs`` and
construction goes through ``compose``.  ``QwenLLM(...)`` is kept only as a
deprecated alias that builds an ``hf_local`` model.  (HF-local forward/capture
mechanics are covered by ``test_models/test_discover.py`` and the analyzer tests;
spec×backend composition by ``test_models/test_compose.py``.)
"""

from __future__ import annotations

import warnings

import pytest

from evalvitals.core import Capability
from evalvitals.models.backends.hf_local import HFLocalModel
from evalvitals.models.whitebox.qwen import QwenLLM


def test_qwenllm_warns_deprecation():
    with pytest.warns(DeprecationWarning, match="evalvitals.load"):
        QwenLLM()


def test_qwenllm_returns_hf_local_model_with_caps():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = QwenLLM()
    assert isinstance(model, HFLocalModel)
    assert model.spec.key == "qwen2.5-7b-instruct"
    # capabilities come from the hf_local backend + spec (no weights loaded)
    assert Capability.ATTENTION in model.capabilities
    assert Capability.HIDDEN_STATES in model.capabilities
    assert Capability.GENERATE in model.capabilities
    assert Capability.GRADIENTS not in model.capabilities


def test_qwenllm_checkpoint_override():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = QwenLLM(checkpoint="some/other-qwen")
    assert model.spec.hf_repo == "some/other-qwen"
