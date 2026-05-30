"""Closed-loop evaluation orchestrator.

Thin facade over the self-evolving loop, kept for the directory layout in the
design doc. It composes the :class:`~evalvitals.eval_agent.loop.SelfEvolveLoop`
with an :class:`~evalvitals.core.experiment.ExperimentRunner`.

Pipeline: define → A/B test → hypothesize → test → report (Stage 2).
"""

from __future__ import annotations

from evalvitals.core.experiment import ExperimentRunner
from evalvitals.eval_agent.loop import SelfEvolveLoop


class EvalOrchestrator:
    """Drive the full eval pipeline end-to-end (Stage 2)."""

    def __init__(self, loop: SelfEvolveLoop | None = None) -> None:
        self.loop = loop or SelfEvolveLoop()
        self.runner: ExperimentRunner = self.loop.runner

    def run(self, *args, **kwargs) -> None:
        raise NotImplementedError("EvalOrchestrator is planned for Stage 2.")
