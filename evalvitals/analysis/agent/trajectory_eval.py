"""Trajectory-level evaluation of multi-step agent runs.

Planned for Stage 2.

Trajectory components tracked:
  user goal → plan → thoughts → tool calls → tool outputs →
  observations → memory updates → agent messages → final answer
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("trajectory_eval")
class TrajectoryEvaluator(Analyzer):
    """Identify where in a multi-step trajectory a failure first occurred."""

    name = "trajectory_eval"
    requires = frozenset({Capability.GENERATE})

    def _run(self, model, cases):
        raise NotImplementedError("TrajectoryEvaluator is planned for Stage 2.")
