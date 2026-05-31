"""Web-search-augmented QA dataset loader.

Planned for Stage 3.
"""

from __future__ import annotations

from evalvitals.core.case import CaseBatch


class WebSearchQADataset:
    """QA benchmark with web-retrieved context."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("WebSearchQADataset is planned for Stage 3.")

    def load(self) -> CaseBatch:
        raise NotImplementedError("WebSearchQADataset is planned for Stage 3.")
