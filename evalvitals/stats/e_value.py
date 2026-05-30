"""E-value testing for multiple comparisons and FDR control.

Planned for Stage 2. Integrates with the team's prior implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalvitals.core.result import Result


def e_value_test(results: "list[Result]", **kwargs):
    """E-value-based test over a set of results. Planned for Stage 2."""
    raise NotImplementedError("e_value_test is planned for Stage 2.")
