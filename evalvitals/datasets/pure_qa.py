"""Pure QA dataset loaders (VL QA and Language QA).

Planned for Stage 2. Loaders return a
:class:`~evalvitals.core.case.CaseBatch` of :class:`FailureCase` — the unit
analyzers consume and the agent accumulates.
"""

from __future__ import annotations

from evalvitals.core.case import CaseBatch


class PureQADataset:
    """Language and VL question-answering benchmark loader."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("PureQADataset is planned for Stage 2.")

    def load(self) -> CaseBatch:
        """Return the benchmark as a :class:`CaseBatch`."""
        raise NotImplementedError("PureQADataset is planned for Stage 2.")
