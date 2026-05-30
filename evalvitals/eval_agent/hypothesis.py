"""Hypotheses — the unit the self-evolving agent proposes, tests, and mutates.

A hypothesis is a falsifiable claim about *when/why* a model fails (e.g. "Qwen
mis-binds entities when two names share a surname"). The loop turns it into
cases + an experiment, runs it, and updates its status from the findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"        # generated, not yet tested
    TESTING = "testing"          # experiment in flight
    SUPPORTED = "supported"      # evidence backs it
    REFUTED = "refuted"          # evidence contradicts it
    INCONCLUSIVE = "inconclusive"


@dataclass
class Hypothesis:
    """A falsifiable claim about a model's failure behaviour.

    Attributes:
        statement:              Natural-language claim (LLM-generated/readable).
        target_model:           Registered model name the claim is about.
        predicted_failure_mode: Tag/description of the expected failure.
        status:                 Lifecycle state (:class:`HypothesisStatus`).
        parent_id:              Hypothesis this one was mutated from, if any.
        id:                     Stable identifier.
        evidence:               Result/case ids accumulated while testing.
        metadata:               Free-form extras.
    """

    statement: str
    target_model: str
    predicted_failure_mode: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    parent_id: str | None = None
    id: str = ""
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class HypothesisGenerator:
    """Proposes new hypotheses — from scratch or by mutating prior ones.

    Stage 2: an LLM-backed generator that reads the store (past supported/refuted
    hypotheses and their evidence) and proposes the next ones to test.
    """

    def propose(self, context: Any = None) -> list[Hypothesis]:
        raise NotImplementedError("HypothesisGenerator is planned for Stage 2.")

    def mutate(self, hypothesis: Hypothesis, feedback: Any = None) -> list[Hypothesis]:
        """Derive refined/adjacent hypotheses from one that was tested."""
        raise NotImplementedError("HypothesisGenerator.mutate is planned for Stage 2.")
