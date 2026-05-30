"""Experiment — a declarative spec that bridges the agent and the core.

The self-evolving agent does not call analyzers directly; it emits *experiments*
("run analyzer X on model Y over cases Z"), which an :class:`ExperimentRunner`
executes, caches, and turns into results.  Making the experiment a first-class
object lets the agent queue, log, replay, and reason about its own actions.

This is intentionally minimal in Stage 1: enough structure for the agent layer
to target, with caching as the one piece of real behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalvitals.core.analyzer import Analyzer
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result


@dataclass
class Experiment:
    """A declarative analysis request.

    Attributes:
        model:    The model to analyse.
        analyzer: The configured analyzer to run.
        data:     Cases (or anything :func:`as_casebatch` accepts).
        metadata: Free-form context (e.g. the hypothesis this tests).
    """

    model: "Model"
    analyzer: "Analyzer"
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """Execute :class:`Experiment` specs, caching results by experiment identity."""

    def __init__(self) -> None:
        self._cache: dict[int, "Result"] = {}

    def run(self, experiment: Experiment, use_cache: bool = True) -> "Result":
        """Run an experiment, returning (and caching) its :class:`Result`."""
        key = id(experiment)
        if use_cache and key in self._cache:
            return self._cache[key]
        result = experiment.analyzer.run(experiment.model, experiment.data)
        result.metadata.setdefault("experiment", experiment.metadata)
        self._cache[key] = result
        return result

    def run_many(self, experiments: list[Experiment]) -> list["Result"]:
        return [self.run(exp) for exp in experiments]
