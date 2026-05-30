"""Model factory and public re-exports.

Models register themselves with the core registry via ``@register_model``.
``load_model`` looks them up by name, so adding a model never touches this file.
"""

from __future__ import annotations

from evalvitals.config import ModelConfig
from evalvitals.core.model import Model
from evalvitals.core.registry import registry

# Backend layer (torch-free at import; heavy deps are lazy inside build/load).
from evalvitals.models.backends import BACKENDS, RuntimeConfig
from evalvitals.models.base import BaseAgent
from evalvitals.models.compose import compose

# Legacy concrete white-box model (imports torch at module load) — optional on
# the light, pure-API install.  Its @register_model runs only when torch is present.
try:
    from evalvitals.models import whitebox as _whitebox  # noqa: F401
except ImportError:  # pragma: no cover - light install
    _whitebox = None

__all__ = ["Model", "BaseAgent", "load_model", "compose", "RuntimeConfig", "BACKENDS"]


def load_model(cfg: ModelConfig) -> Model:
    """Instantiate a registered model from a :class:`ModelConfig`.

    The model name is resolved against the core model registry (populated by
    ``@register_model``).  Stage-1 registered models: ``qwen``.
    """
    name = cfg.name.lower()

    # Resolve aliases (e.g. "qwen2.5-7b" -> "qwen") to a registered key.
    key = name if registry.models.has(name) else _alias(name)
    if key is None:
        raise ValueError(
            f"Unknown model '{cfg.name}'. Registered: {registry.models.list()}"
        )

    model_cls = registry.models.get(key)
    return model_cls(checkpoint=cfg.checkpoint, device=cfg.device, dtype=cfg.dtype)


def _alias(name: str) -> str | None:
    """Map family prefixes to a registered model key."""
    if name.startswith("qwen") and "vl" not in name:
        return "qwen"
    return None
