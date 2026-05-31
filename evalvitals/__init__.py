"""EvalVitals — failure case analysis for LLMs and VLMs.

Three equivalent ways to run an analysis:

1. Canonical (sklearn-style) — configure an analyzer, run it on a model::

    from evalvitals.models.whitebox.qwen import QwenLLM
    from evalvitals.analysis.whitebox.attention import AttentionAnalyzer

    model = QwenLLM()
    result = AttentionAnalyzer(top_k=5).run(model, "The capital of France is")
    print(result.summary())

2. Config-driven — declare model + analysis in YAML::

    from evalvitals import load_config, run

    config = load_config("configs/qwen_attention.yaml")
    result = run(config, "The capital of France is")

3. Hybrid convenience shim (auto-derived from capabilities)::

    result = model.call_attention("The capital of France is")
"""

# Importing these populates the registry (models + analyzers self-register).
import evalvitals.analysis as _analysis  # noqa: E402,F401
from evalvitals.config import AnalysisConfig, load_config
from evalvitals.core import (
    Analyzer,
    Capability,
    CaseBatch,
    FailureCase,
    Model,
    Result,
    registry,
)
from evalvitals.core.tool import Tool, ToolCall
from evalvitals.models import Agent, RuntimeConfig, compose, load_model
from evalvitals.specs import get_spec, list_specs

__version__ = "0.1.0"
__all__ = [
    "load_config",
    "load_model",
    "compose",
    "RuntimeConfig",
    "Agent",
    "Tool",
    "ToolCall",
    "get_spec",
    "list_specs",
    "run",
    "AnalysisConfig",
    "Model",
    "Analyzer",
    "Result",
    "FailureCase",
    "CaseBatch",
    "Capability",
    "registry",
]


def run(config: AnalysisConfig, data, **kwargs):
    """Run the analysis declared by *config* on *data*.

    Equivalent to::

        analyzer = registry.analyzers.get(config.analysis)(**config.analysis_kwargs)
        analyzer.run(load_model(config.model), data)

    Args:
        config: Loaded :class:`AnalysisConfig` (see :func:`load_config`).
        data:   ``str | FailureCase | CaseBatch | iterable`` to analyse.
        **kwargs: Override or extend ``config.analysis_kwargs`` at call time.

    Returns:
        A :class:`~evalvitals.core.result.Result` subclass.
    """
    # Back-compat: tolerate a leading "call_" in the config (old style).
    name = config.analysis
    if name.startswith("call_"):
        name = name[len("call_"):]

    analyzer_cls = registry.analyzers.get(name)
    analyzer = analyzer_cls(**{**config.analysis_kwargs, **kwargs})
    model = load_model(config.model)
    return analyzer.run(model, data)
