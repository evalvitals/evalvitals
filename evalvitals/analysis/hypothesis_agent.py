"""M3 — HypothesisAgent: propose falsifiable hypotheses from an exploratory
analysis report (an ``ExploratoryAnalysisAgent`` / M2 output).

This is a lightweight, standalone counterpart to the diagnosis loop's
``DiagnosisAgent`` (``evalvitals.eval_agent.stages.diagnosis``): that one is
hard-bound to the loop's own ``AnalysisReport`` type and only accepts a judge
``Model`` (Gemini by default) — it cannot read an ``ExploratoryAnalysisReport``
without raising or silently producing nothing (there is no adapter between the
two report shapes). This module reads the standalone explorer's own report
directly and supports the same ``judge``/``cli_config`` backend flexibility as
``ExploratoryAnalysisAgent``, so it can run with whatever ``--backend`` the
``evalvitals explore`` CLI was given.

Proposal only — no validation. A hypothesis here is a candidate explanation to
investigate further, not a conclusion; there is no confirm/test phase wired up
for it (see ``StatsAnalysisAgent``/``HypothesisTester`` in the diagnosis loop
for that, a separate, currently out-of-scope system).
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalvitals.core.model import Model
    from evalvitals.eval_agent.cli_agent import CliAgentConfig

logger = logging.getLogger(__name__)

_PROPOSE_PROMPT = """\
You are an expert data analyst. Based on the exploratory analysis below,
propose specific, falsifiable hypotheses that could explain the patterns
found. A hypothesis is a candidate explanation or mechanism — not a
restatement of a finding, and not a claim you are asked to prove here.

Question investigated: {question}

Key takeaways from the exploratory analysis (title: analysis):
{takeaways_text}

Observations:
{observations_text}

Candidate signals already noted:
{signals_text}

Propose 1-3 hypotheses. For each write exactly three lines:
HYPOTHESIS: <one-sentence falsifiable claim explaining a pattern above>
BASIS: <which takeaway(s)/signal(s) above this is grounded in>
TEST: <what evidence/analysis would confirm or refute this claim>

Do not repeat a takeaway verbatim — propose a CAUSE or MECHANISM behind what
was observed. If the findings are too thin to support any falsifiable
hypothesis, respond with exactly: NO_HYPOTHESIS"""


@dataclass
class Hypothesis:
    """One M3-proposed, falsifiable candidate explanation. Proposed only —
    ``test_design`` names how it *could* be checked, not a verdict."""

    statement: str = ""
    basis: str = ""
    test_design: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"statement": self.statement, "basis": self.basis, "test_design": self.test_design}


class HypothesisAgent:
    """M3: propose hypotheses from an ``ExploratoryAnalysisReport`` (as a dict).

    Args:
        judge:      LLM-like object with ``generate(prompt) -> str``.
        cli_config: Optional CLI coding-agent backend (same as
                    ``ExploratoryAnalysisAgent``'s ``cli_config``); used
                    instead of ``judge`` when its provider isn't ``"llm"``.
        timeout_sec: Timeout for a single generation call.
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        cli_config: "CliAgentConfig | None" = None,
        timeout_sec: int = 90,
    ) -> None:
        self._judge = judge
        self._cli_config = cli_config
        self._timeout_sec = timeout_sec

    @property
    def available(self) -> bool:
        return self._judge is not None or (
            self._cli_config is not None and self._cli_config.provider != "llm"
        )

    def propose(self, report: dict[str, Any], *, max_items: int = 8) -> list[Hypothesis]:
        """Read *report* (an ``ExploratoryAnalysisReport.to_dict()``-shaped
        dict) and return 0-3 proposed hypotheses. Never raises — a backend
        failure just yields no hypotheses."""
        if not self.available:
            return []

        takeaways = [t for t in report.get("takeaways") or [] if isinstance(t, dict)][:max_items]
        takeaways_text = "\n".join(
            f"- {t.get('title', '')}: {t.get('analysis', '')}" for t in takeaways
        ) or "(none recorded)"
        observations = [str(o) for o in report.get("observations") or []][:max_items]
        observations_text = "\n".join(f"- {o}" for o in observations) or "(none recorded)"
        signals = [s for s in report.get("candidate_signals") or [] if isinstance(s, dict)][:max_items]
        signals_text = "\n".join(
            f"- {s.get('display_name') or s.get('name')}: {s.get('rationale', '')}" for s in signals
        ) or "(none recorded)"

        if not takeaways and not observations and not signals:
            return []

        prompt = _PROPOSE_PROMPT.format(
            question=report.get("question") or "Explore this dataset.",
            takeaways_text=takeaways_text,
            observations_text=observations_text,
            signals_text=signals_text,
        )
        try:
            raw = self._generate(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HypothesisAgent: generation failed: %s", exc)
            return []
        return _parse_hypotheses(raw)

    def _generate(self, prompt: str) -> str:
        if self._cli_config is not None and self._cli_config.provider != "llm":
            from evalvitals.eval_agent.cli_agent import create_cli_agent

            agent = create_cli_agent(self._cli_config)
            with tempfile.TemporaryDirectory(prefix="evalvitals_m3_") as tmp:
                res = agent.run(prompt, workdir=Path(tmp), timeout_sec=self._timeout_sec)
                return res.raw_output or ""
        return str(self._judge.generate(prompt))  # type: ignore[union-attr]


def _parse_hypotheses(raw: str) -> list[Hypothesis]:
    """Parse ``HYPOTHESIS:``/``BASIS:``/``TEST:`` triples out of *raw*.

    For the CLI-agent backend, *raw* is the full rendered tool-call
    trajectory, not just a final answer (``CliAgentResult.raw_output``'s
    docstring: "assistant text, every Bash/Edit/Write/Read tool call +
    result") — an agent that narrates its plan before giving a final answer
    can restate the same hypothesis twice in one trajectory. Dedupe by
    statement (case-insensitive), keeping the order hypotheses first
    appeared but the most recent basis/test for a repeated statement, since
    later restatements tend to be the more refined ones.
    """
    if "HYPOTHESIS:" not in raw.upper():
        return []
    by_statement: dict[str, dict[str, str]] = {}
    order: list[str] = []
    cur: dict[str, str] | None = None
    cur_key: str = ""

    def _flush() -> None:
        if cur and cur.get("statement"):
            if cur_key not in by_statement:
                order.append(cur_key)
            by_statement[cur_key] = cur

    for line in raw.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("HYPOTHESIS:"):
            _flush()
            statement = line.split(":", 1)[1].strip()
            cur = {"statement": statement, "basis": "", "test_design": ""}
            cur_key = statement.lower()
        elif upper.startswith("BASIS:") and cur is not None:
            cur["basis"] = line.split(":", 1)[1].strip()
        elif upper.startswith("TEST:") and cur is not None:
            cur["test_design"] = line.split(":", 1)[1].strip()
    _flush()
    return [Hypothesis(**by_statement[key]) for key in order]
