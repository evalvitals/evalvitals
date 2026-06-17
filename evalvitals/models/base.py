"""Model base classes.

The abstract contract lives in :mod:`evalvitals.core.model` (:class:`Model`,
:class:`Trace`).  This module re-exports it and adds the agent base.
Deployment-specific bases live alongside their models:
  - :class:`~evalvitals.models.whitebox.base.WhiteboxModel` (local weights)
  - :class:`~evalvitals.models.blackbox.base.BlackboxModel` (API-only)
"""

from __future__ import annotations

from abc import abstractmethod

from evalvitals.core.model import Model, Trace

__all__ = ["Model", "Trace", "BaseAgent"]


class BaseAgent(Model):
    """Abstract base for agent-mode models that use tools over multiple steps."""

    @abstractmethod
    def step(self, observation: str, **kwargs) -> str:
        """Process one agent step and return the next action/response."""
