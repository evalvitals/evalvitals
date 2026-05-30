"""RISE: Randomized Input Sampling for Explanation of Black-box Models.

Planned for Stage 2. References:
- Paper: https://arxiv.org/abs/1806.07421
- Code:  https://github.com/eclique/RISE
"""

from __future__ import annotations

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer


@register_analyzer("rise")
class RISEAnalyzer(Analyzer):
    """Black-box explainability via randomized input masking (RISE)."""

    name = "rise"
    requires = frozenset({Capability.GENERATE})

    def _run(self, model, cases):
        raise NotImplementedError("RISEAnalyzer is planned for Stage 2.")
