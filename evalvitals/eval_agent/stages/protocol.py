"""ExperimentProtocol — user-supplied description of what an experiment tests.

The protocol is the human prior that anchors the self-evolving loop:

- **M1** uses :meth:`ExperimentProtocol.probe_hints` to guide analyzer selection.
- **M2** uses the protocol to frame its statistical narrative.
- **M5** uses it to verify that a hypothesis is consistent with what the user
  actually set out to investigate.

Usage::

    protocol = ExperimentProtocol(
        description=(
            "We test QwenVL on spatial reasoning: given an image with two "
            "objects, the model often confuses their relative positions."
        ),
        task_domain="spatial reasoning",
        failure_patterns="left-right reversal, above-below confusion",
        target_modalities=frozenset({"text", "image"}),
    )
    hints = protocol.probe_hints()   # ["attention", "hallucination"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Keyword → failure-mode mapping used by probe_hints()
# ---------------------------------------------------------------------------

# Each entry: (tuple-of-keywords, failure-mode-tag).
# Keyword matching is case-insensitive substring search over
# description + failure_patterns.
_KEYWORD_TO_MODE: list[tuple[tuple[str, ...], str]] = [
    (("hallucin", "confabul", "invent", "made up", "fabricat"), "hallucination"),
    (("attention sink", "sink token"), "attention_sink"),
    (("attend", "attention", "look at", "focus", "visual focus", "ignor image",
      "spatial", "position", "left", "right", "above", "below", "location"),
     "attention"),
    (("loop", "repeat", "stuck", "cycle", "infinite"), "loop"),
    (("inconsist", "unstable", "vary", "unreliable", "differ"), "low_consistency"),
    (("confident", "certainty", "overconfid", "underconfid", "calibr"), "miscalibrated_confidence"),
    (("entropy", "uncertain", "hesitat"), "entropy"),
    (("tool", "action", "agent", "navigate", "execution"), "loop"),
]


@dataclass
class ExperimentProtocol:
    """Natural-language description of an evaluation experiment.

    This is the *human prior* passed into the loop so it can make
    informed, targeted decisions rather than running every analyzer
    blindly.

    Attributes:
        description:        What the experiment tests — free text (required).
        task_domain:        Short label, e.g. ``"spatial reasoning"``,
                            ``"GUI navigation"``.
        success_criteria:   What counts as a pass (used by M5 verifier).
        failure_patterns:   Known or suspected failure modes — feeds M1 hints
                            and M5 consistency checks.
        target_modalities:  ``{"text", "image"}`` for VLMs;
                            ``{"text"}`` for text-only LLMs.
        metadata:           Free-form extras (dataset names, hyperparams …).
    """

    description: str
    task_domain: str = ""
    success_criteria: str = ""
    failure_patterns: str = ""
    target_modalities: frozenset[str] = field(
        default_factory=lambda: frozenset({"text"})
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def probe_hints(self) -> list[str]:
        """Derive failure-mode hint tags for M1's :class:`~evalvitals.eval_agent.probe.StrategyProbe`.

        Scans ``description`` and ``failure_patterns`` for keywords and maps
        them to the failure-mode tags used by :data:`~evalvitals.eval_agent.probe._FAILURE_MODE_TO_ANALYZERS`.

        Returns a deduplicated list in keyword-match order.
        """
        text = (self.description + " " + self.failure_patterns).lower()
        seen: set[str] = set()
        hints: list[str] = []
        for keywords, mode in _KEYWORD_TO_MODE:
            if mode not in seen and any(kw in text for kw in keywords):
                hints.append(mode)
                seen.add(mode)
        return hints

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "task_domain": self.task_domain,
            "success_criteria": self.success_criteria,
            "failure_patterns": self.failure_patterns,
            "target_modalities": sorted(self.target_modalities),
            "metadata": self.metadata,
        }


@dataclass
class ProbingSchema:
    """What M1 decided to probe and why.

    Returned by :meth:`~evalvitals.eval_agent.probe_agent.ProbeAgent.probe_with_schema`
    alongside the raw ``{analyzer: Result}`` dict so callers can understand
    *why* those analyzers were chosen.

    Attributes:
        selected_analyzers: Analyzers to run, in priority order.
        rationale:          NL explanation of the selection.
        custom_params:      Per-analyzer parameter overrides (e.g.
                            ``{"attention": {"layer": -1}}``).
        protocol:           The protocol that shaped this schema, if any.
    """

    selected_analyzers: list[str]
    rationale: str = ""
    custom_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    protocol: ExperimentProtocol | None = None
