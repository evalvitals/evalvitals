"""GUI / OS-level task dataset loader.

Planned for Stage 3.
"""

from __future__ import annotations

from evalvitals.core.case import CaseBatch


class GUIOSDataset:
    """Dataset for GUI and OS-level agent evaluation tasks."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("GUIOSDataset is planned for Stage 3.")

    def load(self) -> CaseBatch:
        raise NotImplementedError("GUIOSDataset is planned for Stage 3.")
