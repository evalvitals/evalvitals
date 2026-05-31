"""Session-scoped fixtures for GPU integration tests.

Load each model once per test session and share the handle across all
integration test files to avoid redundant weight loads (~2s from cache each).
"""

from __future__ import annotations

import pytest
import torch

from evalvitals.core.capability import Capability
from evalvitals.models import load


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA GPU available")
    for item in items:
        if item.get_closest_marker("gpu"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def qwen_model():
    return load("qwen2.5-7b-instruct", want={Capability.ATTENTION, Capability.HIDDEN_STATES})
