"""M3 — DiagnosisAgent: synthesize analysis findings into falsifiable hypotheses.

The agent reads the structured :class:`~evalvitals.eval_agent.analysis.AnalysisReport`
produced by M2, formats it as a prompt, sends it to an LLM judge (Gemini by
default), and parses the response into :class:`~evalvitals.eval_agent.hypothesis.Hypothesis`
objects.

The judge model defaults to **Gemini** when ``GEMINI_API_KEY`` is set in the
environment and no explicit judge is passed.

Usage::

    # Gemini default (set GEMINI_API_KEY in env)
    agent = DiagnosisAgent()
    diag  = agent.diagnose(analysis_report)

    # Explicit judge (any Model with Capability.GENERATE)
    agent = DiagnosisAgent(judge=my_model)
    diag  = agent.diagnose(analysis_report)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalvitals.eval_agent.hypothesis import Hypothesis

if TYPE_CHECKING:
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.analysis import AnalysisReport


_DIAGNOSE_PROMPT = """\
You are an expert ML diagnostician. Based on the analysis report below, propose
specific, falsifiable hypotheses about the root cause of the model's failures.
{prior_section}
Model: {model_name}
Overall severity: {severity}

Analysis narrative:
{narrative}

Raw findings (JSON):
{findings_json}

Propose 1-3 hypotheses. For each write exactly two lines:
HYPOTHESIS: <one-sentence falsifiable claim about the failure mode>
FAILURE_MODE: <short tag, e.g. attention_sink / hallucination / loop / low_consistency>

Only include hypotheses clearly supported by the analysis.
Do NOT repeat hypotheses already listed in the prior cycles above.
If the model looks healthy, respond with: NO_ISSUE"""

_VALIDATE_PROMPT = """\
You are an adversarial ML reviewer. Your job is to find reasons to REJECT each
hypothesis below. Only approve a hypothesis if you cannot find a significant flaw.

Check each for:
1. Unsupported claim — does the cited evidence actually imply this failure mode?
2. Circular reasoning — does the hypothesis merely restate the symptom?
3. Overgeneralisation — does it make a claim far broader than the evidence supports?
4. Confounded alternative — is there a simpler explanation the hypothesis ignores?

Findings summary (the evidence the hypotheses were drawn from):
{findings_json}

Hypotheses to review:
{hypotheses_text}

For each hypothesis output exactly two lines:
KEEP: <hypothesis statement>  or  REJECT: <hypothesis statement>
REASON: <specific flaw, or "evidence directly supports this claim" if keeping>"""


def _format_prior_section(prior_cycles: list[dict]) -> str:
    """Render prior cycle history as a context block for the diagnosis prompt."""
    if not prior_cycles:
        return ""
    lines = [
        "\nPrior investigation cycles — do NOT repeat these hypotheses; propose "
        "distinct or refined ones that address what remains unexplained:",
    ]
    for c in prior_cycles:
        lines.append(f"\nCycle {c['cycle']} (severity={c['severity']}):")
        for h in c.get("hypotheses", []):
            lines.append(
                f"  [{h['status'].upper()}] {h['statement']} "
                f"(failure_mode: {h['failure_mode']})"
            )
    return "\n".join(lines) + "\n"


@dataclass
class DiagnosisResult:
    """Output of :class:`DiagnosisAgent`.

    Attributes:
        model_name:       ``repr()`` of the analysed model.
        hypotheses:       Proposed :class:`~evalvitals.eval_agent.hypothesis.Hypothesis`
                          objects for M4.
        findings_summary: The findings dict forwarded to the judge.
        raw_judge_output: Verbatim LLM response (useful for debugging).
    """

    model_name: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    findings_summary: dict[str, Any] = field(default_factory=dict)
    raw_judge_output: str = ""


_HYPOTHESIS_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["hypothesis", "failure_mode"],
        "properties": {
            "hypothesis":   {"type": "string", "minLength": 10},
            "failure_mode": {"type": "string", "minLength": 2},
        },
    },
}


def _validate_json_schema(data: object, schema: dict) -> list[str]:
    """Minimal JSON schema validator (no external deps).

    Returns a list of error strings; empty means valid.
    Only handles type/required/minLength — enough to catch the common
    LLM mistakes (missing field, empty string, wrong top-level type).
    """
    errors: list[str] = []

    def _check(node: object, s: dict, path: str) -> None:
        expected_type = s.get("type")
        if expected_type == "array":
            if not isinstance(node, list):
                errors.append(f"{path}: expected array, got {type(node).__name__}")
                return
            item_schema = s.get("items", {})
            for i, item in enumerate(node):
                _check(item, item_schema, f"{path}[{i}]")
        elif expected_type == "object":
            if not isinstance(node, dict):
                errors.append(f"{path}: expected object, got {type(node).__name__}")
                return
            for req in s.get("required", []):
                if req not in node:
                    errors.append(f"{path}: missing required field '{req}'")
            for prop, prop_schema in s.get("properties", {}).items():
                if prop in node:
                    _check(node[prop], prop_schema, f"{path}.{prop}")
        elif expected_type == "string":
            if not isinstance(node, str):
                errors.append(f"{path}: expected string, got {type(node).__name__}")
            elif "minLength" in s and len(node) < s["minLength"]:
                errors.append(
                    f"{path}: string too short ({len(node)} < {s['minLength']})"
                )

    _check(data, schema, "$")
    return errors


def _parse_hypotheses_json(raw: str, model_name: str) -> list[Hypothesis] | None:
    """Try to parse *raw* as a JSON array of hypothesis objects.

    Returns ``None`` when the response is not JSON or fails schema validation
    so the caller can fall back to the text parser.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        import re as _re
        text = _re.sub(r"^```\w*\n?", "", text)
        text = _re.sub(r"\n?```\s*$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    errors = _validate_json_schema(data, _HYPOTHESIS_SCHEMA)
    if errors:
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "DiagnosisAgent JSON schema validation failed: %s", errors
        )
        return None

    return [
        Hypothesis(
            statement=item["hypothesis"],
            target_model=model_name,
            predicted_failure_mode=item["failure_mode"],
        )
        for item in data
        if item.get("hypothesis") and item.get("failure_mode")
    ]


def _parse_hypotheses(raw: str, model_name: str) -> list[Hypothesis]:
    """Parse hypothesis objects from LLM output.

    Tries JSON structured format first (more reliable), then falls back to the
    line-oriented ``HYPOTHESIS:`` / ``FAILURE_MODE:`` text format.  This makes
    the parser resilient to both output modes without requiring a hard migration.
    """
    json_result = _parse_hypotheses_json(raw, model_name)
    if json_result is not None:
        return json_result

    # Text-format fallback
    hypotheses: list[Hypothesis] = []
    statement: str | None = None
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("HYPOTHESIS:"):
            statement = line[len("HYPOTHESIS:"):].strip()
        elif line.upper().startswith("FAILURE_MODE:") and statement:
            mode = line[len("FAILURE_MODE:"):].strip()
            hypotheses.append(
                Hypothesis(
                    statement=statement,
                    target_model=model_name,
                    predicted_failure_mode=mode,
                )
            )
            statement = None
    return hypotheses


def _validate_hypotheses(
    hypotheses: list[Hypothesis],
    findings_json: str,
    judge: "Model",
) -> list[Hypothesis]:
    """Adversarially filter *hypotheses* using a critic call at temperature=0.

    The same judge is called a second time with a prompt designed to find
    reasons to *reject* each hypothesis.  This breaks the confirmation bias
    loop where a model that generated a hypothesis then gives it a free pass.
    Temperature is forced to 0 for deterministic, reproducible verdicts.

    Returns the subset of hypotheses that survive the critic.  Falls back to
    the original list if the validation call fails so the loop is never
    completely blocked by a transient error.
    """
    if not hypotheses:
        return hypotheses

    hyp_lines = "\n".join(
        f"- HYPOTHESIS: {h.statement}  (failure_mode: {h.predicted_failure_mode})"
        for h in hypotheses
    )
    prompt = _VALIDATE_PROMPT.format(
        findings_json=findings_json,
        hypotheses_text=hyp_lines,
    )

    try:
        # Use temperature=0 so the critic is deterministic and strict.
        # Models that don't accept a temperature kwarg are called without it.
        import inspect
        sig = inspect.signature(judge.generate)
        if "temperature" in sig.parameters:
            raw = judge.generate(prompt, temperature=0)
        else:
            raw = judge.generate(prompt)
    except Exception:
        return hypotheses  # validation failed — keep originals

    kept: set[str] = set()
    for line in str(raw).splitlines():
        line = line.strip()
        if line.upper().startswith("KEEP:"):
            stmt = line[len("KEEP:"):].strip().lower()
            for h in hypotheses:
                if h.statement.lower()[:60] in stmt or stmt in h.statement.lower():
                    kept.add(h.statement)

    if not kept:
        # Critic rejected everything or parse failed — return originals so the
        # loop doesn't deadlock, but log the event
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "DiagnosisAgent validation: critic rejected all %d hypothesis(es) "
            "or could not be parsed — keeping originals",
            len(hypotheses),
        )
        return hypotheses

    return [h for h in hypotheses if h.statement in kept]


def _default_judge() -> "Model":
    """Return a GeminiModel when GEMINI_API_KEY is available, else raise."""
    import os

    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError(
            "DiagnosisAgent requires a judge model. "
            "Either pass judge= explicitly, or set GEMINI_API_KEY to use the "
            "Gemini default (install with: pip install 'evalvitals[gemini]')."
        )
    from evalvitals.models.blackbox.gemini import GeminiModel

    return GeminiModel()


class DiagnosisAgent:
    """Proposes hypotheses from an :class:`~evalvitals.eval_agent.analysis.AnalysisReport`.

    Args:
        judge:    Any :class:`~evalvitals.core.model.Model` with
                  ``Capability.GENERATE``.  Defaults to a ``GeminiModel``
                  when ``GEMINI_API_KEY`` is in the environment.
        api_key:  Gemini API key — used only when *judge* is ``None`` and no
                  ``GEMINI_API_KEY`` env var is set.
        model_id: Gemini model identifier (default: ``"gemini-2.0-flash"``).
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        api_key: str | None = None,
        model_id: str = "gemini-2.0-flash",
    ) -> None:
        if judge is None and api_key is not None:
            from evalvitals.models.blackbox.gemini import GeminiModel

            judge = GeminiModel(model_id=model_id, api_key=api_key)
        self._judge = judge  # None → resolved lazily on first call

    @property
    def judge(self) -> "Model":
        if self._judge is None:
            self._judge = _default_judge()
        return self._judge

    @judge.setter
    def judge(self, value: "Model") -> None:
        self._judge = value

    def diagnose(
        self,
        analysis: "AnalysisReport | dict[str, Result]",
        model_name: str = "",
        prior_cycles: list[dict] | None = None,
    ) -> DiagnosisResult:
        """Synthesize *analysis* into a set of falsifiable hypotheses.

        Args:
            analysis:     An :class:`~evalvitals.eval_agent.analysis.AnalysisReport`
                          from M2, or (for backward compatibility) a plain
                          ``{analyzer_name: Result}`` dict.
            model_name:   Ignored when *analysis* is an ``AnalysisReport``
                          (the name is taken from the report).
            prior_cycles: Summary of previous M1→M4 cycles produced by
                          :class:`~evalvitals.eval_agent.loop.AutoDiagnoseLoop`.
                          Each entry is ``{"cycle": int, "severity": str,
                          "hypotheses": [{"statement", "failure_mode", "status"}]}``.
                          Injected into the prompt so the judge avoids re-proposing
                          already-tested hypotheses.

        Returns:
            :class:`DiagnosisResult` with zero or more hypotheses.
        """
        from evalvitals.eval_agent.analysis import AnalysisModule, AnalysisReport

        if not isinstance(analysis, AnalysisReport):
            # Backward compat: wrap raw results in a minimal AnalysisReport.
            analysis = AnalysisModule().analyze(analysis, model_name)

        summary = {name: r.findings for name, r in analysis.raw_results.items()}
        prompt = _DIAGNOSE_PROMPT.format(
            prior_section=_format_prior_section(prior_cycles or []),
            model_name=analysis.model_name or model_name,
            severity=analysis.severity,
            narrative=analysis.narrative,
            findings_json=json.dumps(summary, indent=2, default=str),
        )
        raw = self.judge.generate(prompt)
        hypotheses = _parse_hypotheses(str(raw), analysis.model_name or model_name)

        # Adversarial validation: run a second critic call at temperature=0 to
        # prune hypotheses the generator produced without sufficient evidence.
        # This prevents the self-evaluation loop where the same model that
        # proposed a hypothesis then approves it uncritically.
        if hypotheses:
            findings_json_str = json.dumps(summary, indent=2, default=str)
            hypotheses = _validate_hypotheses(hypotheses, findings_json_str, self.judge)

        # Fallback: if the judge returned NO_ISSUE but M2 has medium/high findings,
        # auto-generate one hypothesis per finding so M4 can still run.
        # This prevents self-diagnosis bias when the judge is the model under test.
        if not hypotheses and analysis.findings:
            from evalvitals.eval_agent.analysis import _SEVERITY_ORDER
            for finding in analysis.findings:
                if _SEVERITY_ORDER.get(finding.severity, 0) >= 2:  # medium or high
                    hypotheses.append(Hypothesis(
                        statement=(
                            f"The model {finding.message} "
                            f"({finding.analyzer}.{finding.metric}={finding.value:.3g})"
                        ),
                        target_model=analysis.model_name or model_name,
                        predicted_failure_mode=finding.analyzer,
                    ))

        return DiagnosisResult(
            model_name=analysis.model_name or model_name,
            hypotheses=hypotheses,
            findings_summary=summary,
            raw_judge_output=str(raw),
        )
