"""Config loading and dataclasses for evalvitals."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Maps short model names → default HuggingFace checkpoints.
_DEFAULT_CHECKPOINTS: dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
}


@dataclass
class ModelConfig:
    name: str
    checkpoint: str | None = None
    device: str = "cuda"
    dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.checkpoint is None:
            resolved = _DEFAULT_CHECKPOINTS.get(self.name.lower())
            if resolved is None:
                known = list(_DEFAULT_CHECKPOINTS)
                raise ValueError(
                    f"No default checkpoint for model '{self.name}'. "
                    f"Set 'checkpoint' explicitly or use a known name: {known}"
                )
            self.checkpoint = resolved


@dataclass
class AnalysisConfig:
    model: ModelConfig
    analysis: str
    analysis_kwargs: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> AnalysisConfig:
    """Load an AnalysisConfig from a YAML file.

    Supports both short form::

        model: qwen
        analysis: attention

    and full form::

        model:
          name: qwen
          checkpoint: Qwen/Qwen2.5-7B-Instruct
          device: cuda
          dtype: float16
        analysis: attention      # registered analyzer name
        analysis_kwargs:
          layer: -1
          head: mean
          top_k: 10
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    model_raw = raw["model"]
    model_cfg = ModelConfig(name=model_raw) if isinstance(model_raw, str) else ModelConfig(**model_raw)

    return AnalysisConfig(
        model=model_cfg,
        analysis=raw["analysis"],
        analysis_kwargs=raw.get("analysis_kwargs", {}),
    )
