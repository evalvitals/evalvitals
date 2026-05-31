"""Counterfactual replay — causal step attribution (Stage 2).

Fork a trajectory at a suspect step, correct the args/observation, re-run the
Agent loop, and check whether the outcome flips — turning correlational judging
into intervention.  Reuses ``Agent`` + the executor.

Caveat (build in before claiming causality): with temperature > 0 use n-replay +
a paired test, else "this step was causal" conflates with model noise; prefer
temperature=0 replay.
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("counterfactual")
class CounterfactualReplay(Analyzer):
    """Causal step attribution by forking + replaying a trajectory."""

    name = "counterfactual"
    requires = frozenset({Capability.TOOL_CALLS})
    applies_to_modalities = frozenset({"text"})

    def _run(self, model, cases):
        raise NotImplementedError(
            "Stage 2: fork the trajectory at a suspect step and re-run the Agent loop; "
            "use temperature=0 (or n-replay + paired test) to separate causality from noise."
        )
