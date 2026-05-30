"""Model factory and public re-exports.

Models register themselves with the core registry via ``@register_model``.
``load_model`` looks them up by name, so adding a model never touches this file.
"""

from __future__ import annotations

from evalvitals.config import ModelConfig
from evalvitals.core.model import Model
from evalvitals.core.registry import registry

# Import model modules so their @register_model decorators run.
from evalvitals.models import whitebox as _whitebox  # noqa: F401
from evalvitals.models.base import BaseAgent

__all__ = ["Model", "BaseAgent", "load_model"]


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
