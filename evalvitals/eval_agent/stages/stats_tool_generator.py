"""M2 tier (b) — StatsToolGenerator: synthesise a new statistical tool on demand.

When no catalog tool in :mod:`~evalvitals.eval_agent.stages.stats_tools` fits the
data, this generator asks an LLM / CLI coding agent to *write* a small statistics
script, runs it inside an :class:`~evalvitals.eval_agent.sandbox.ExperimentSandbox`
(subprocess + resource limits + path-traversal protection), and parses a strict
``STATS_RESULT_JSON=`` line from its stdout back into a
:class:`~evalvitals.eval_agent.stages.stats_tools.StatsToolResult`.

Design (mirrors M4's ExperimentWriter):

- **Generated code never touches the repo source.**  It lives only in the
  sandbox workdir and is executed in a child process; the host never imports it.
- The generated script reads ``m2_stats_input.json`` (the serialised
  :class:`StatsInput`) from its working directory and may ``import
  evalvitals.stats`` to compose existing primitives or write novel statistics
  with numpy.
- **Tier (c) reuse**: a script that runs and parses cleanly is returned as a
  :class:`GeneratedStatsTool`; the caller can re-run it on later cycles via
  :meth:`run_cached` with fresh data — no second LLM call.

Backends (first available wins):

1. ``cli_config`` with ``provider != "llm"`` → a CLI coding agent
   (agy / codex / claude_code …) writes files into the workdir.
2. ``judge`` model → single-pass ``generate()`` returning a fenced code block.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evalvitals.eval_agent.sandbox import ExperimentSandbox
from evalvitals.eval_agent.stages.stats_tools import StatsInput, StatsToolResult, describe_data

if TYPE_CHECKING:
    from evalvitals.core.model import Model
    from evalvitals.eval_agent.cli_agent import CliAgentConfig

logger = logging.getLogger(__name__)

_INPUT_FILENAME = "m2_stats_input.json"
_RESULT_MARKER = "STATS_RESULT_JSON="

_GENERATE_PROMPT = """\
You are writing a self-contained Python statistics script for a model-failure analysis.

GOAL (what statistic to compute):
{need}

DATA: a JSON file named "{input_filename}" sits in the current working directory with:
{{
  "labels":   {{case_id: is_fail_bool}},          # PASS/FAIL labels
  "per_case": {{"analyzer.metric": {{case_id: value}}}},  # per-case signals
  "scalars":  {{"analyzer.metric": value}},         # aggregate metrics
  "groups":   {{strategy: {{case_id: success}}}} or null  # strategy comparison
}}

Data shape available for THIS run:
{data_shape}

REQUIREMENTS:
- Read "{input_filename}" from the current directory. Do NOT hardcode the data.
- You MAY `import evalvitals.stats` (compare, compare_multiple, mcnemar,
  evalue_bernoulli, ebh, kendall_tau) and `import numpy`. No network, no file
  writes, no other I/O.
- Every verdict must carry an effect size; never decide on a bare p-value.
- The LAST line of stdout MUST be exactly one line of the form:
  {marker}{{"summary": "<one sentence>", "effect": <number or null>, "ci": [lo, hi] or null, "reject": <true/false/null>, "e_value": <number or null>, "p_value": <number or null>, "underpowered": <true/false>, "details": {{}}}}
- Print NOTHING after that line. Keep the script under ~60 lines.

Return ONLY the Python code{fences_hint}."""


@dataclass
class GeneratedStatsTool:
    """A validated, sandbox-executed statistical tool that can be re-run.

    Attributes:
        name:   Short identifier (becomes ``generated:<name>``).
        code:   The Python source that was written and executed.
        need:   The natural-language goal it was generated for.
        source: Which backend wrote it (``"cli:<provider>"`` or ``"llm"``).
    """

    name: str
    code: str
    need: str = ""
    source: str = ""


class StatsToolGenerator:
    """Generate, run, and cache bespoke statistical tools in a sandbox.

    Args:
        judge:       LLM used for the single-pass code-writing path.
        cli_config:  :class:`~evalvitals.eval_agent.cli_agent.CliAgentConfig`
                     with ``provider != "llm"`` to use a CLI coding agent
                     instead of (or before) the judge.
        sandbox:     Execution sandbox.  A fresh temp-dir sandbox is created
                     when ``None``.
        timeout_sec: Hard wall-clock limit per sandbox run — the primary guard.
        max_cpu_seconds / max_memory_bytes: optional resource caps for the
                     default sandbox.  Default ``None`` (no cap) to match M4:
                     generated tools may ``import evalvitals.stats`` which pulls
                     in torch, whose large virtual-memory reservation a
                     ``RLIMIT_AS`` cap would crash.  Ignored when *sandbox* is
                     supplied.
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        cli_config: "CliAgentConfig | None" = None,
        sandbox: "ExperimentSandbox | None" = None,
        timeout_sec: int = 60,
        max_cpu_seconds: int | None = None,
        max_memory_bytes: int | None = None,
        run_logger: "Any | None" = None,
    ) -> None:
        self._judge = judge
        self._cli_config = cli_config
        self._timeout_sec = timeout_sec
        self._sandbox = sandbox or ExperimentSandbox(
            max_cpu_seconds=max_cpu_seconds,
            max_memory_bytes=max_memory_bytes,
        )
        # Optional RunLogger — records each stats-tool code-writing attempt.
        self.run_logger = run_logger
        self._last_prompt: str = ""
        self._last_raw: str = ""

    @property
    def available(self) -> bool:
        """True when at least one code-writing backend is configured."""
        return self._judge is not None or (
            self._cli_config is not None and self._cli_config.provider != "llm"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        need: str,
        inp: StatsInput,
        name: str = "custom",
    ) -> "tuple[StatsToolResult, GeneratedStatsTool | None]":
        """Write a new stats tool for *need*, run it on *inp*, and return both.

        Returns ``(result, tool)``.  ``tool`` is ``None`` when generation or the
        first execution failed (the result then carries ``ok=False`` + error).
        """
        if not self.available:
            return (
                StatsToolResult(
                    tool=f"generated:{name}", ok=False,
                    error="no code-writing backend (judge or cli_config) configured",
                    summary="codegen unavailable",
                ),
                None,
            )

        self._write_input(inp)
        self._last_prompt = ""
        self._last_raw = ""
        try:
            code, source = self._write_code(need, inp)
        except Exception as exc:
            logger.warning("StatsToolGenerator: code writing failed: %s", exc)
            self._emit_codegen(name, need, "", "", ok=False, error=f"code writing failed: {exc}")
            return (
                StatsToolResult(
                    tool=f"generated:{name}", ok=False,
                    error=f"code writing failed: {exc}", summary="codegen failed",
                ),
                None,
            )

        if not code.strip():
            self._emit_codegen(name, need, source, "", ok=False, error="backend produced no code")
            return (
                StatsToolResult(
                    tool=f"generated:{name}", ok=False,
                    error="backend produced no code", summary="codegen empty",
                ),
                None,
            )

        result = self._run_code(code, name)
        self._emit_codegen(
            name, need, source, code, ok=result.ok,
            error="" if result.ok else (result.error or "tool run failed"),
        )
        tool = (
            GeneratedStatsTool(name=name, code=code, need=need, source=source)
            if result.ok else None
        )
        return result, tool

    def _emit_codegen(
        self, name: str, need: str, source: str, code: str, *, ok: bool, error: str = ""
    ) -> None:
        """Record one stats-tool code-writing attempt to the RunLogger."""
        if self.run_logger is None:
            return
        try:
            self.run_logger.log_tool_codegen(
                module="m2_stats", name=name, need=need, source=source, ok=ok,
                code=code, prompt=self._last_prompt, raw_output=self._last_raw, error=error,
            )
        except Exception as exc:  # logging must never break generation
            logger.debug("StatsToolGenerator: log_tool_codegen failed: %s", exc)

    def run_cached(self, tool: GeneratedStatsTool, inp: StatsInput) -> StatsToolResult:
        """Re-run an already-generated tool on fresh *inp* (no LLM call)."""
        self._write_input(inp)
        return self._run_code(tool.code, tool.name)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_input(self, inp: StatsInput) -> None:
        payload = {
            "labels": inp.labels,
            "per_case": inp.per_case,
            "scalars": inp.scalars,
            "groups": inp.groups,
        }
        path = Path(self._sandbox.workdir) / _INPUT_FILENAME
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_code(self, need: str, inp: StatsInput) -> tuple[str, str]:
        """Return ``(code, source)`` from the CLI agent or the judge."""
        if self._cli_config is not None and self._cli_config.provider != "llm":
            code = self._write_code_cli(need, inp)
            if code:
                return code, f"cli:{self._cli_config.provider}"
        # Single-pass LLM fallback
        prompt = self._build_prompt(need, inp, fenced=True)
        self._last_prompt = prompt
        raw = self._judge.generate(prompt)  # type: ignore[union-attr]
        self._last_raw = str(raw)
        return _extract_code(str(raw)), "llm"

    def _write_code_cli(self, need: str, inp: StatsInput) -> str:
        from evalvitals.eval_agent.cli_agent import create_cli_agent

        prompt = self._build_prompt(need, inp, fenced=False)
        self._last_prompt = prompt
        agent = create_cli_agent(self._cli_config)  # type: ignore[arg-type]
        res = agent.run(prompt, workdir=Path(self._sandbox.workdir), timeout_sec=self._timeout_sec)
        self._last_raw = res.raw_output
        if not res.ok:
            logger.debug("CLI codegen produced no files (%s)", res.error)
            return ""
        # Prefer a stats_tool.py; otherwise the largest .py the agent wrote.
        py_files = {n: c for n, c in res.files.items() if n.endswith(".py")}
        if not py_files:
            return ""
        if "stats_tool.py" in py_files:
            return py_files["stats_tool.py"]
        return max(py_files.values(), key=len)

    def _build_prompt(self, need: str, inp: StatsInput, *, fenced: bool) -> str:
        return _GENERATE_PROMPT.format(
            need=need.strip() or "Test whether the analyzer signals predict case FAIL.",
            input_filename=_INPUT_FILENAME,
            data_shape=json.dumps(describe_data(inp), indent=2),
            marker=_RESULT_MARKER,
            fences_hint=" inside a ```python code block" if fenced else
                        ", written to a file named stats_tool.py",
        )

    def _run_code(self, code: str, name: str) -> StatsToolResult:
        sandbox_result = self._sandbox.run(code, timeout_sec=self._timeout_sec)
        if not sandbox_result.ok:
            err = (sandbox_result.stderr or "").strip()[:300] or "non-zero exit"
            return StatsToolResult(
                tool=f"generated:{name}", ok=False,
                error=f"sandbox run failed: {err}",
                summary=f"generated:{name} failed to run",
                details={"returncode": sandbox_result.returncode,
                         "timed_out": sandbox_result.timed_out},
            )
        return _parse_result(sandbox_result.stdout, name)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _extract_code(raw: str) -> str:
    """Pull the Python source out of an LLM response (fenced or bare)."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", cleaned, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return cleaned.strip()


def _parse_result(stdout: str, name: str) -> StatsToolResult:
    """Parse the last ``STATS_RESULT_JSON=`` line of *stdout*."""
    marker_line = None
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith(_RESULT_MARKER):
            marker_line = s[len(_RESULT_MARKER):]
    if marker_line is None:
        return StatsToolResult(
            tool=f"generated:{name}", ok=False,
            error="no STATS_RESULT_JSON line in output",
            summary=f"generated:{name} produced no result line",
            details={"stdout_tail": stdout.strip()[-300:]},
        )
    try:
        data = json.loads(marker_line)
    except json.JSONDecodeError as exc:
        return StatsToolResult(
            tool=f"generated:{name}", ok=False,
            error=f"unparseable STATS_RESULT_JSON: {exc}",
            summary=f"generated:{name} bad result json",
        )

    ci = data.get("ci")
    ci_tuple = tuple(ci) if isinstance(ci, (list, tuple)) and len(ci) == 2 else None
    return StatsToolResult(
        tool=f"generated:{name}",
        ok=True,
        summary=str(data.get("summary", "")) or f"generated:{name}",
        effect=_as_float(data.get("effect")),
        ci=ci_tuple,
        reject=data.get("reject"),
        e_value=_as_float(data.get("e_value")),
        p_value=_as_float(data.get("p_value")),
        underpowered=bool(data.get("underpowered", False)),
        details=data.get("details") if isinstance(data.get("details"), dict) else {},
    )


def _as_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
