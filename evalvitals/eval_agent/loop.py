"""SelfEvolveLoop — the closed-loop controller for automated failure discovery.

This is the top-level orchestrator the whole package is designed to serve. It
ties the pieces together into one cycle:

    ┌─────────────────────────────────────────────────────────────┐
    │  1. HYPOTHESIZE   generator.propose(store.summarize())        │
    │  2. CONSTRUCT     tools.make_cases(hypothesis) -> CaseBatch    │
    │  3. EXPERIMENT    Experiment(model, analyzer, cases)           │
    │  4. RUN           ExperimentRunner.run(exp) -> Result          │
    │  5. ATTRIBUTE     read result.findings; localise the failure   │
    │  6. EVALUATE      stats over results (significance)            │
    │  7. RECORD        store.add_result / add_case / add_hypothesis │
    │  8. MUTATE        generator.mutate(hypothesis, feedback)       │
    └─────────────────────────────────────────────────────────────┘
                         ↑___________________________│  (repeat)

"Self-evolving" = step 8 feeding step 1: each cycle's findings reshape the next
cycle's hypotheses, and the store accumulates so the agent builds on itself.

Stage 1 ships this as a documented skeleton: the cycle and wiring points are
fixed, the intelligence (hypothesis generation, attribution, case synthesis)
arrives in Stage 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evalvitals.core.experiment import ExperimentRunner
from evalvitals.eval_agent.store import InMemoryStore, Store

if TYPE_CHECKING:
    from evalvitals.eval_agent.hypothesis import HypothesisGenerator


class SelfEvolveLoop:
    """Controller that drives the hypothesize→run→attribute→record→mutate cycle.

    Args:
        generator: Proposes and mutates hypotheses (Stage-2 LLM-backed).
        store:     Persistent memory; defaults to an in-memory store.
        runner:    Executes experiments; defaults to a fresh runner.
    """

    def __init__(
        self,
        generator: "HypothesisGenerator | None" = None,
        store: Store | None = None,
        runner: ExperimentRunner | None = None,
    ) -> None:
        self.generator = generator
        self.store = store or InMemoryStore()
        self.runner = runner or ExperimentRunner()

    def step(self) -> list:
        """One cycle: propose hypotheses (from the store's summary) and record them.

        Returns the hypotheses proposed this cycle (empty => nothing new to try).
        The test/attribute/mutate intelligence (case synthesis, LLM judging) plugs
        in here in Stage 2; the propose→record→repeat skeleton is functional now.
        """
        if self.generator is None:
            raise ValueError("SelfEvolveLoop needs a generator (e.g. ManualHypothesisGenerator).")
        proposed = self.generator.propose(self.store.summarize())
        for h in proposed:
            self.store.add_hypothesis(h)
        return list(proposed)

    def run(self, max_cycles: int = 10) -> list:
        """Run until the generator stops proposing (converged) or *max_cycles*."""
        history: list = []
        for _ in range(max_cycles):
            proposed = self.step()
            history.append(proposed)
            if not proposed:  # dry — nothing new
                break
        return history
