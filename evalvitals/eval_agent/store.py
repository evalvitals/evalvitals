"""Store — the persistent memory that makes self-evolution possible.

Without memory, an agent re-explores the same ground forever. The store
accumulates the corpus the agent evolves over: cases it has constructed,
results it has produced, and hypotheses it has tested. The loop writes to it
each cycle and reads from it to decide what to try next.

This module defines the interface and an in-memory reference shape; a durable
backend (JSONL / SQLite / vector index for semantic recall) lands in Stage 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalvitals.core.case import FailureCase
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.hypothesis import Hypothesis


class Store(ABC):
    """Persistent memory interface for the self-evolving loop."""

    @abstractmethod
    def add_case(self, case: "FailureCase") -> None: ...

    @abstractmethod
    def add_result(self, result: "Result") -> None: ...

    @abstractmethod
    def add_hypothesis(self, hypothesis: "Hypothesis") -> None: ...

    @abstractmethod
    def query(self, **filters: Any) -> list[Any]:
        """Retrieve stored items matching *filters* (tags, label, status, …)."""

    @abstractmethod
    def summarize(self) -> dict[str, Any]:
        """Aggregate view the agent reads to spot gaps and patterns."""


class InMemoryStore(Store):
    """Minimal non-persistent reference store (Stage-2 will add durability)."""

    def __init__(self) -> None:
        self.cases: list[FailureCase] = []
        self.results: list[Result] = []
        self.hypotheses: list[Hypothesis] = []

    def add_case(self, case: "FailureCase") -> None:
        self.cases.append(case)

    def add_result(self, result: "Result") -> None:
        self.results.append(result)

    def add_hypothesis(self, hypothesis: "Hypothesis") -> None:
        self.hypotheses.append(hypothesis)

    def query(self, **filters: Any) -> list[Any]:
        raise NotImplementedError("InMemoryStore.query is planned for Stage 2.")

    def summarize(self) -> dict[str, Any]:
        return {
            "n_cases": len(self.cases),
            "n_results": len(self.results),
            "n_hypotheses": len(self.hypotheses),
        }
