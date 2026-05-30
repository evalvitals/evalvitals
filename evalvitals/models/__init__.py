"""Model construction — one engine: ``compose(spec, backend)``.

``evalvitals.specs`` is the single source of truth for *which* models exist;
``models/backends`` provides the runtimes.  Everything funnels through
:func:`load` (friendly, key-based) and :func:`load_model` (``ModelConfig``-based),
both of which build via ``compose`` — so there is exactly one way a model is
constructed, and the capability set always reflects the chosen backend.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from evalvitals.config import ModelConfig
from evalvitals.core.capability import Capability
from evalvitals.core.model import Model

# Backend layer (torch-free at import; heavy deps are lazy inside build/load).
from evalvitals.models.backends import BACKENDS, RuntimeConfig
from evalvitals.models.base import BaseAgent
from evalvitals.models.compose import compose

# Legacy concrete white-box model (now a thin deprecated alias) — optional on the
# light, pure-API install (its module imports nothing heavy at top level).
try:
    from evalvitals.models import whitebox as _whitebox  # noqa: F401
except ImportError:  # pragma: no cover - light install
    _whitebox = None

__all__ = [
    "Model",
    "BaseAgent",
    "load",
    "load_model",
    "compose",
    "RuntimeConfig",
    "BACKENDS",
    "resolve_spec_key",
]

# Legacy name → canonical spec key.  Keeps old configs/code working.
_SPEC_ALIASES: dict[str, str] = {
    "qwen": "qwen2.5-7b-instruct",
    "qwen2.5-7b": "qwen2.5-7b-instruct",
}


def resolve_spec_key(name: str) -> str:
    """Resolve a spec key or legacy alias to a canonical :mod:`evalvitals.specs` key."""
    from evalvitals.specs import REGISTRY

    key = name.lower()
    if key in REGISTRY:
        return key
    if key in _SPEC_ALIASES:
        return _SPEC_ALIASES[key]
    raise ValueError(
        f"Unknown model {name!r}. Known specs: {sorted(REGISTRY)}; "
        f"aliases: {sorted(_SPEC_ALIASES)}."
    )


def load(
    key: str,
    *,
    backend: str = "hf_local",
    want: "list[str] | set[Capability] | tuple" = (),
    checkpoint: str | None = None,
    **runtime: Any,
) -> Model:
    """Build a model by spec key — the friendly one-liner.

    ``evalvitals.load("qwen2.5-7b-instruct")`` is the modern replacement for the
    old ``QwenLLM()``.  It wraps ``compose(get_spec(key), backend, RuntimeConfig(**runtime))``
    and negotiates capabilities up front (``want``).

    Args:
        key:        Spec key or legacy alias (see :func:`resolve_spec_key`).
        backend:    ``"hf_local"`` (internals) / ``"api"`` / ``"vllm_offline"``.
                    Forced to ``"api"`` for ``api_only`` (closed-weight) specs.
        want:       Capabilities the backend must provide (names or ``Capability``).
        checkpoint: Optional override of the spec's ``hf_repo``.
        **runtime:  Forwarded to :class:`RuntimeConfig` (``device``, ``dtype``, …).
    """
    from evalvitals.specs import get_spec

    spec = get_spec(resolve_spec_key(key))
    if checkpoint:
        spec = replace(spec, hf_repo=checkpoint)
    chosen = "api" if spec.api_only else backend
    caps = {w if isinstance(w, Capability) else Capability(w) for w in want}
    return compose(spec, chosen, RuntimeConfig(**runtime), caps)


def load_model(cfg: ModelConfig) -> Model:
    """Instantiate a model from a :class:`~evalvitals.config.ModelConfig` via :func:`load`."""
    return load(
        cfg.name,
        backend=cfg.backend,
        want=cfg.want,
        checkpoint=cfg.checkpoint,
        device=cfg.device,
        dtype=cfg.dtype,
    )
