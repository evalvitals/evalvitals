"""FailureCase — the central data object of EvalVitals.

Everything in the system speaks ``FailureCase``:
  - datasets *produce* batches of cases,
  - analyzers *attribute* failures over cases,
  - stats *test* significance across cases,
  - the agent *accumulates* and *evolves* a corpus of cases.

Analyzers accept ``str | FailureCase | list | CaseBatch`` and normalise via
:func:`as_casebatch`, so ``model.call_attention("a prompt")`` stays ergonomic
while the canonical unit of work remains the case.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator


class Label(str, Enum):
    """Outcome label for a case."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class Source(str, Enum):
    """Where a case came from — important for self-evolution provenance."""

    HUMAN = "human"
    DATASET = "dataset"
    AGENT = "agent"


@dataclass
class Provenance:
    """How a case was created."""

    source: Source = Source.HUMAN
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Inputs:
    """Model inputs for a case.  Text now; image is VLM-ready for Stage 2."""

    prompt: str
    image: Any = None  # PIL.Image | path | None — populated in Stage 2

    def __str__(self) -> str:
        return self.prompt


@dataclass
class FailureCase:
    """A single unit of failure analysis.

    Attributes:
        inputs:     What was fed to the model (:class:`Inputs`).
        expected:   Gold / expected behaviour, if known.
        observed:   What the model actually produced, if run.
        label:      :class:`Label` — pass / fail / unknown.
        tags:       Free-form failure-taxonomy tags (e.g. ``{"hallucination"}``).
        provenance: How this case was created (:class:`Provenance`).
        id:         Stable identifier (auto-generated if omitted).
        metadata:   Free-form extra fields.
    """

    inputs: Inputs
    expected: Any = None
    observed: Any = None
    label: Label = Label.UNKNOWN
    tags: set[str] = field(default_factory=set)
    provenance: Provenance = field(default_factory=Provenance)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_prompt(cls, prompt: str, **kwargs) -> "FailureCase":
        """Build a case from a raw prompt string."""
        return cls(inputs=Inputs(prompt=prompt), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "inputs": {"prompt": self.inputs.prompt, "image": self.inputs.image},
            "expected": self.expected,
            "observed": self.observed,
            "label": self.label.value,
            "tags": sorted(self.tags),
            "provenance": {
                "source": self.provenance.source.value,
                "metadata": self.provenance.metadata,
            },
            "metadata": self.metadata,
        }


class CaseBatch:
    """An ordered collection of :class:`FailureCase` with convenience helpers."""

    def __init__(self, cases: Iterable[FailureCase] | None = None) -> None:
        self._cases: list[FailureCase] = list(cases) if cases else []

    # -- constructors --------------------------------------------------
    @classmethod
    def from_prompts(cls, prompts: Iterable[str], **kwargs) -> "CaseBatch":
        """Build a batch from raw prompt strings."""
        return cls(FailureCase.from_prompt(p, **kwargs) for p in prompts)

    # -- list-like behaviour -------------------------------------------
    def __iter__(self) -> Iterator[FailureCase]:
        return iter(self._cases)

    def __len__(self) -> int:
        return len(self._cases)

    def __getitem__(self, idx: int) -> FailureCase:
        return self._cases[idx]

    def append(self, case: FailureCase) -> None:
        self._cases.append(case)

    # -- querying (used by the agent / stats) --------------------------
    def filter(
        self,
        label: Label | None = None,
        tags: set[str] | None = None,
    ) -> "CaseBatch":
        """Return a new batch matching *label* and/or containing all *tags*."""
        out = [
            c
            for c in self._cases
            if (label is None or c.label == label)
            and (tags is None or tags.issubset(c.tags))
        ]
        return CaseBatch(out)

    def __repr__(self) -> str:
        return f"CaseBatch(n={len(self)})"


def as_casebatch(data: str | FailureCase | Inputs | Iterable | CaseBatch) -> CaseBatch:
    """Normalise common inputs into a :class:`CaseBatch`.

    Accepts:
      - a ``str``                → single-case batch from the prompt,
      - a :class:`FailureCase`   → single-case batch,
      - an :class:`Inputs`       → single-case batch,
      - a :class:`CaseBatch`     → returned unchanged,
      - any iterable of the above.
    """
    if isinstance(data, CaseBatch):
        return data
    if isinstance(data, str):
        return CaseBatch([FailureCase.from_prompt(data)])
    if isinstance(data, FailureCase):
        return CaseBatch([data])
    if isinstance(data, Inputs):
        return CaseBatch([FailureCase(inputs=data)])
    if isinstance(data, Iterable):
        batch = CaseBatch()
        for item in data:
            # flatten via single-item normalisation
            for case in as_casebatch(item):
                batch.append(case)
        return batch
    raise TypeError(
        f"Cannot interpret {type(data).__name__} as cases. "
        "Pass a str, FailureCase, Inputs, CaseBatch, or an iterable of these."
    )
