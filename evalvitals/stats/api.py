"""Standalone statistical testing API (importable independently of models).

Planned for Stage 2. Consumes :class:`~evalvitals.core.result.Result` objects
(e.g. from two prompting strategies) and returns significance verdicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalvitals.core.result import Result


def ab_test(results_a: "list[Result]", results_b: "list[Result]", **kwargs):
    """A/B test two sets of results. Planned for Stage 2."""
    raise NotImplementedError("ab_test is planned for Stage 2.")
