"""Analyzer — the sklearn-like estimator at the heart of EvalVitals.

Consistent contract, every analyzer:
  - is configured with hyper-parameters in ``__init__`` (stored, introspectable
    via ``get_params``/``set_params`` — exactly like a scikit-learn estimator),
  - declares the capabilities it ``requires``,
  - runs via ``run(model, data) -> Result``.

``run`` normalises ``data`` into a :class:`CaseBatch`, verifies the model
provides the required capabilities (clear :class:`CapabilityError` otherwise),
then delegates to the subclass's :meth:`_run`.  Subclasses implement only
``_run`` and never repeat the boilerplate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from evalvitals.core.capability import Capability, CapabilityError
from evalvitals.core.case import CaseBatch, as_casebatch

if TYPE_CHECKING:
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result


class Analyzer(ABC):
    """Abstract base for every analysis (sklearn-estimator style).

    Class attributes (set by subclasses):
        name:     Registered short name (e.g. ``"attention"``).
        requires: Capabilities the analysed model must provide.

    Example::

        result = AttentionAnalyzer(layer=-1).run(qwen, "The capital of France is")
    """

    name: str = "analyzer"
    requires: frozenset[Capability] = frozenset()

    def __init__(self, **params: Any) -> None:
        # Store hyper-parameters sklearn-style for introspection / reproduction.
        self._params: dict[str, Any] = dict(params)
        for key, value in params.items():
            setattr(self, key, value)

    # ------------------------------------------------------------------
    # sklearn-style introspection
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, Any]:
        """Return the analyzer's hyper-parameters."""
        return dict(self._params)

    def set_params(self, **params: Any) -> "Analyzer":
        """Update hyper-parameters in place and return self (chainable)."""
        self._params.update(params)
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, model: "Model", data: Any) -> "Result":
        """Validate capabilities, normalise *data*, and run the analysis.

        Args:
            model: Any :class:`~evalvitals.core.model.Model`.
            data:  ``str | FailureCase | Inputs | CaseBatch | iterable`` — normalised
                   via :func:`~evalvitals.core.case.as_casebatch`.

        Returns:
            A :class:`~evalvitals.core.result.Result` (subclass) instance.

        Raises:
            CapabilityError: if *model* lacks a required capability.
        """
        self._check_capabilities(model)
        cases = as_casebatch(data)
        return self._run(model, cases)

    @abstractmethod
    def _run(self, model: "Model", cases: CaseBatch) -> "Result":
        """Subclass hook: perform the analysis over an already-normalised batch."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_capabilities(self, model: "Model") -> None:
        missing = set(self.requires) - set(model.capabilities)
        if missing:
            raise CapabilityError(
                analyzer=self.name,
                model=repr(model),
                missing=missing,
            )

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return f"{type(self).__name__}({params})"
