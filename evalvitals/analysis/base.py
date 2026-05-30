"""Analysis base — re-exports the core estimator contract.

Every analyzer subclasses :class:`evalvitals.core.analyzer.Analyzer`:
configure with hyper-parameters in ``__init__``, declare ``requires``
capabilities, implement ``_run(model, cases) -> Result``.
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.result import Result

__all__ = ["Analyzer", "Result"]
