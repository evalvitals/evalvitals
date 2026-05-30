"""Unit-level failure attribution for agent systems.

Planned for Stage 2. Reference:
- https://github.com/ag2ai/Agents_Failure_Attribution

Example questions answered by this module:
  - Did the planner choose the right next action?
  - Did the tool call use the correct arguments?
  - Did the memory module retrieve relevant context?
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("failure_attribution")
class FailureAttributionAnalyzer(Analyzer):
    """Identify which atomic agent component caused a task failure."""

    name = "failure_attribution"
    requires = frozenset({Capability.GENERATE})

    def _run(self, model, cases):
        raise NotImplementedError("FailureAttributionAnalyzer is planned for Stage 2.")
