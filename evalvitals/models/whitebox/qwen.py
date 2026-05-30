"""Deprecated ``QwenLLM`` shim.

The concrete white-box Qwen class is gone — model identity now lives in
:mod:`evalvitals.specs` and construction goes through ``compose(spec, backend)``.
``QwenLLM(...)`` is kept only for backward compatibility: it warns and returns a
``compose("qwen2.5-7b-instruct", "hf_local")`` model.

Prefer::

    import evalvitals
    model = evalvitals.load("qwen2.5-7b-instruct")        # friendly
    # or
    from evalvitals.models import compose
    model = compose("qwen2.5-7b-instruct", "hf_local")    # explicit
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = ["QwenLLM"]


def QwenLLM(
    checkpoint: str | None = None,
    device: str = "auto",
    dtype: str = "bfloat16",
    **runtime: Any,
):
    """Deprecated. Build Qwen2.5-7B-Instruct via the unified engine.

    Returns an ``hf_local`` model (capabilities: GENERATE, LOGITS, LOGPROBS,
    HIDDEN_STATES, ATTENTION).  Use :func:`evalvitals.load` instead.
    """
    warnings.warn(
        "QwenLLM is deprecated; use evalvitals.load('qwen2.5-7b-instruct') or "
        "compose('qwen2.5-7b-instruct', 'hf_local').",
        DeprecationWarning,
        stacklevel=2,
    )
    from evalvitals.models import load

    return load(
        "qwen2.5-7b-instruct",
        backend="hf_local",
        checkpoint=checkpoint,
        device=device,
        dtype=dtype,
        **runtime,
    )
