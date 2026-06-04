"""ExperimentWriter — LLM or CLI agent writes and executes a diagnostic script.

Mirrors the multi-phase pattern in researchclaw/pipeline/code_agent.py.

Two execution paths selected by ``ExperimentWriterConfig.cli_agent.provider``:

LLM path (default, ``provider="llm"``)
    Phase 1 · Write    LLM generates a self-contained Python script.
    Phase 2 · Validate AST-parse; flag critical syntax errors.
    Phase 3 · Exec-fix Run in sandbox; feed stderr back to LLM for repair;
                        retry up to ``exec_fix_max_iterations`` times.

CLI agent path (``provider="claude_code"`` / ``"codex"`` / ``"opencode"`` / …)
    Phase 1 · Write    CLI agent (agentic, has bash/file tools) generates the
                        script autonomously from ``cases.json`` + ``hypothesis.md``
                        written to the workdir.  No exec-fix loop — the agent
                        self-repairs during its own run.
    Phase 2 · Execute  One ``sandbox.run()`` call to parse metrics + verdict.

The script must print metrics in ARC-compatible format (parsed by
:func:`~evalvitals.eval_agent.sandbox.parse_metrics`), plus a final verdict::

    mean_consistency: 0.23
    n_cases: 3
    verdict: 1.0          # 1 = hypothesis SUPPORTED, 0 = REFUTED

After the exec-fix loop :class:`ExperimentWriterResult` carries the parsed
metrics and a ``verdict`` float that :class:`~evalvitals.eval_agent.surgery.SurgeryAgent`
uses to decide SUPPORTED / REFUTED / INCONCLUSIVE.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalvitals.eval_agent.sandbox import ExperimentSandbox, SandboxResult

if TYPE_CHECKING:
    from evalvitals.core.model import Model
    from evalvitals.eval_agent.hypothesis import Hypothesis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration  (mirrors CodeAgentConfig)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentWriterConfig:
    """Toggleable phases and limits for :class:`ExperimentWriter`.

    All defaults give a good quality/cost balance.  Disable phases to reduce
    LLM calls; lower ``exec_fix_max_iterations`` to reduce sandbox usage.

    To use a CLI coding agent for M4 instead of the single-pass LLM::

        from evalvitals.eval_agent.cli_agent import CliAgentConfig
        cfg = ExperimentWriterConfig(
            cli_agent=CliAgentConfig(provider="claude_code", model="sonnet"),
        )
    """

    # Phase 2: AST validation before execution
    hard_validation: bool = True
    hard_validation_max_repairs: int = 3

    # Phase 3: exec-fix loop (LLM path only; CLI agents self-repair)
    exec_fix_max_iterations: int = 3
    exec_fix_timeout_sec: int = 60

    # CLI agent dispatch — None means "use the LLM path" (backward-compatible default).
    # Pass CliAgentConfig(provider="claude_code") to route through a CLI agent instead.
    cli_agent: Any = None


# ---------------------------------------------------------------------------
# Result  (mirrors CodeAgentResult)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentWriterResult:
    """Output of :class:`ExperimentWriter.write_and_run`.

    Attributes:
        code:              Final (possibly repaired) script source.
        metrics:           ``{name: float}`` parsed from stdout.
        verdict:           ``1.0`` = SUPPORTED, ``0.0`` = REFUTED,
                           ``None`` = no verdict printed (INCONCLUSIVE).
        stdout / stderr:   Raw subprocess output from the last run.
        returncode:        Exit code of the last sandbox run.
        timed_out:         ``True`` when the process was killed.
        total_llm_calls:   Number of LLM calls made (write + repairs).
        total_sandbox_runs: Number of sandbox runs (initial + retries).
        validation_log:    Ordered list of events for debugging.
    """

    code: str
    metrics: dict[str, float] = field(default_factory=dict)
    verdict: float | None = None
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1
    timed_out: bool = False
    total_llm_calls: int = 0
    total_sandbox_runs: int = 0
    validation_log: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_WRITE_SYSTEM = """\
You are an expert ML debugging engineer. Write a concise, self-contained Python
diagnostic script that gathers evidence for or against a specific hypothesis
about a model's behaviour.

RULES:
- Use ONLY the evalvitals APIs shown in the user message.
- The script must be completely self-contained — no external files.
- Print ALL metrics in EXACTLY this format (one per line):
    metric_name: float_value
- Print a final verdict line as the LAST output:
    verdict: 1.0    # hypothesis is SUPPORTED by the evidence
    verdict: 0.0    # hypothesis is REFUTED
- Never print anything after the verdict line.
- Catch exceptions and print error metrics rather than crashing silently.
- Stay under {timeout_sec} seconds total runtime.\
"""

_WRITE_USER = """\
## Hypothesis to test
Statement  : {statement}
Failure mode: {failure_mode}

## Model setup (copy this verbatim)
```python
{import_expr}
model = {load_expr}
```

Available capabilities: {capabilities}

## EvalVitals API reference
```python
from evalvitals.core.case import CaseBatch, FailureCase, Inputs
from evalvitals.core.capability import Capability

# Text generation (always available)
output: str = model.generate(case.inputs)

# Token logprobs  (only if 'LOGPROBS' in capabilities)
lps = model.logprobs(case.inputs)   # list[TokenLogprob]; each has .token, .logprob

# Internals capture (only if 'ATTENTION' or 'HIDDEN_STATES' in capabilities)
trace = model.forward(case.inputs, capture={{Capability.ATTENTION}})
# trace.attentions     list[Tensor]  one per layer, shape [heads, seq, seq]
# trace.hidden_states  list[Tensor]  one per layer, shape [seq, hidden]
# trace.tokens         list[str]
```

## Failure cases (JSON — deserialize inside your script)
```json
{cases_json}
```

## Required output (EXACTLY this format)
```
metric_a: 0.72
metric_b: 3.0
verdict: 1.0
```

Write the complete Python script now:\
"""

_REPAIR_SYSTEM = """\
You are a Python debugging expert. The script below crashed.
Return the COMPLETE corrected script — no explanations, just the fixed code.
Preserve all logic; fix only the error.\
"""

_REPAIR_USER = """\
## Error
```
{error}
```

## Script to fix
```python
{code}
```\
"""


# ---------------------------------------------------------------------------
# Helper: build model context from a live model object
# ---------------------------------------------------------------------------

def build_model_context(model: "Model") -> dict[str, Any]:
    """Extract the information needed to reconstruct *model* inside a subprocess.

    Returns a dict with:
        ``import_expr``   — Python import line(s)
        ``load_expr``     — expression that produces the model object
        ``capabilities``  — sorted list of capability name strings
    """
    caps = sorted(str(c.name) for c in getattr(model, "capabilities", frozenset()))

    # Curated spec path (HFLocalModel, APIModel loaded via evalvitals.load)
    spec = getattr(model, "spec", None)
    if spec is not None and getattr(spec, "key", None):
        return {
            "import_expr": "import evalvitals",
            "load_expr": f"evalvitals.load({spec.key!r})",
            "capabilities": caps,
        }

    # GeminiModel (black-box, re-instantiates from env var)
    if type(model).__name__ == "GeminiModel":
        model_id = getattr(model, "model_id", "gemini-2.5-flash")
        return {
            "import_expr": "from evalvitals.models.blackbox.gemini import GeminiModel",
            "load_expr": f"GeminiModel(model_id={model_id!r})",
            "capabilities": caps,
        }

    # Generic APIModel — use generate_fn pattern (no easy reconstruction)
    return {
        "import_expr": "import evalvitals",
        "load_expr": f"# WARNING: cannot reconstruct {type(model).__name__!r} automatically",
        "capabilities": caps,
    }


# ---------------------------------------------------------------------------
# ExperimentWriter
# ---------------------------------------------------------------------------

class ExperimentWriter:
    """LLM or CLI agent writes and executes a targeted diagnostic script.

    Mirrors ``researchclaw.pipeline.code_agent.CodeAgent``.

    When ``config.cli_agent`` is ``None`` or has ``provider="llm"`` (default):
        Phase 1 — LLM writes the script (``_phase1_write``).
        Phase 2 — AST validation + repair (``_hard_validate_and_repair``).
        Phase 3 — exec-fix loop: run → stderr → LLM repair → retry.

    When ``config.cli_agent.provider`` is a CLI provider (e.g. ``"claude_code"``):
        Phase 1 — CLI agent writes ``experiment.py`` in the workdir autonomously.
        Phase 2 — One ``sandbox.run()`` to parse metrics + verdict (no exec-fix
                   loop; the agent self-repairs during its own run).

    Args:
        judge:    Any :class:`~evalvitals.core.model.Model` with
                  ``Capability.GENERATE``.  Used on the LLM path; ignored on
                  the CLI path (the CLI agent uses its own model).
        config:   :class:`ExperimentWriterConfig` controlling phases and limits.
    """

    def __init__(
        self,
        judge: "Model",
        config: ExperimentWriterConfig = ExperimentWriterConfig(),
    ) -> None:
        self._judge = judge
        self._cfg = config
        self._calls = 0
        self._runs = 0
        self._log: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def write_and_run(
        self,
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        cases_json: str,
        sandbox: ExperimentSandbox,
    ) -> ExperimentWriterResult:
        """Generate, validate, and execute a diagnostic script.

        Args:
            hypothesis:    The claim to test.
            model_context: Output of :func:`build_model_context`.
            cases_json:    JSON-serialised failure cases for the script.
            sandbox:       :class:`ExperimentSandbox` used for execution.

        Returns:
            :class:`ExperimentWriterResult` with metrics and verdict.
        """
        self._calls = 0
        self._runs = 0
        self._log = []
        self._log_event("ExperimentWriter.write_and_run() started")

        # ── CLI agent dispatch ──────────────────────────────────────────────
        cli_cfg = self._cfg.cli_agent
        if cli_cfg is None:
            from evalvitals.eval_agent.cli_agent import CliAgentConfig
            cli_cfg = CliAgentConfig()
        if cli_cfg.provider != "llm":
            return self._cli_write_and_run(
                hypothesis, model_context, cases_json, sandbox, cli_cfg
            )

        # ── LLM path (unchanged below) ─────────────────────────────────────
        # Phase 1: write
        code = self._phase1_write(hypothesis, model_context, cases_json)
        if not code.strip():
            self._log_event("Phase 1 produced empty script — aborting")
            return ExperimentWriterResult(
                code="",
                validation_log=list(self._log),
                total_llm_calls=self._calls,
                total_sandbox_runs=self._runs,
            )

        # Phase 2: hard validation (AST)
        if self._cfg.hard_validation:
            code = self._hard_validate_and_repair(code, hypothesis, model_context, cases_json)

        # Phase 3: exec-fix loop
        code, last_result = self._exec_fix_loop(code, sandbox, hypothesis, model_context, cases_json)

        verdict = last_result.metrics.get("verdict")
        self._log_event(
            f"write_and_run() done — {self._calls} LLM calls, "
            f"{self._runs} sandbox runs, verdict={verdict}"
        )

        return ExperimentWriterResult(
            code=code,
            metrics=last_result.metrics,
            verdict=verdict,
            stdout=last_result.stdout,
            stderr=last_result.stderr,
            returncode=last_result.returncode,
            timed_out=last_result.timed_out,
            total_llm_calls=self._calls,
            total_sandbox_runs=self._runs,
            validation_log=list(self._log),
        )

    # ------------------------------------------------------------------
    # CLI agent path  (phases 1+2 combined)
    # ------------------------------------------------------------------

    def _cli_write_and_run(
        self,
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        cases_json: str,
        sandbox: ExperimentSandbox,
        cli_cfg: Any,
    ) -> "ExperimentWriterResult":
        """CLI agent path: agent generates + self-repairs; we collect and execute.

        Mirrors the agentic sandbox pattern in
        ``researchclaw.experiment.collider_agent_sandbox``.
        """
        from evalvitals.eval_agent.cli_agent import create_cli_agent

        self._log_event(f"CLI path: provider={cli_cfg.provider!r}")
        workdir = sandbox.workdir

        # Write context files the agent reads from its workdir
        (workdir / "cases.json").write_text(cases_json, encoding="utf-8")
        (workdir / "hypothesis.md").write_text(
            f"# Hypothesis\n\n"
            f"**Statement:** {hypothesis.statement}\n\n"
            f"**Failure mode:** {hypothesis.predicted_failure_mode}\n\n"
            f"**Target model:** {hypothesis.target_model}\n",
            encoding="utf-8",
        )

        prompt = self._build_cli_prompt(
            hypothesis, model_context, self._cfg.exec_fix_timeout_sec
        )

        cli = create_cli_agent(cli_cfg)
        self._log_event(f"  invoking {cli._provider_name!r}")
        cli_result = cli.run(
            prompt=prompt,
            workdir=workdir,
            timeout_sec=cli_cfg.timeout_sec,
        )
        self._log_event(
            f"  CLI finished: ok={cli_result.ok}, "
            f"files={list(cli_result.files)}, elapsed={cli_result.elapsed_sec:.1f}s"
        )
        if cli_result.error:
            self._log_event(f"  CLI error: {cli_result.error}")

        if not cli_result.files:
            self._log_event("CLI agent produced no .py files — aborting")
            return ExperimentWriterResult(
                code="",
                validation_log=list(self._log),
                total_llm_calls=0,
                total_sandbox_runs=0,
            )

        # Prefer experiment.py; fallback to first file alphabetically
        code = cli_result.files.get("experiment.py") or next(
            iter(cli_result.files.values())
        )
        self._log_event(f"  collected script: {len(code)} chars")

        # Optional AST validation (warn only — don't abort; agent may have been creative)
        if self._cfg.hard_validation:
            errors = self._validate_ast(code)
            if errors:
                self._log_event(f"  AST warnings: {'; '.join(errors)}")

        # Single sandbox run to parse metrics + verdict
        result = sandbox.run(code, timeout_sec=self._cfg.exec_fix_timeout_sec)
        self._runs += 1
        self._log_event(
            f"  sandbox run: rc={result.returncode}, "
            f"timed_out={result.timed_out}, metrics={list(result.metrics)}"
        )

        verdict = result.metrics.get("verdict")
        self._log_event(
            f"_cli_write_and_run() done — {self._runs} sandbox run(s), verdict={verdict}"
        )
        return ExperimentWriterResult(
            code=code,
            metrics=result.metrics,
            verdict=verdict,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            timed_out=result.timed_out,
            total_llm_calls=0,
            total_sandbox_runs=self._runs,
            validation_log=list(self._log),
        )

    @staticmethod
    def _build_cli_prompt(
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        timeout_sec: int,
    ) -> str:
        """Build the prompt given to the CLI agent.

        References ``cases.json`` by filename — the agent reads it from its
        workdir rather than having it embedded inline.
        """
        caps = ", ".join(model_context.get("capabilities", [])) or "GENERATE"
        return (
            "You are an ML debugging engineer. Write a self-contained Python "
            "diagnostic script that tests a specific hypothesis about a model.\n\n"
            "## Hypothesis\n"
            f"Statement   : {hypothesis.statement}\n"
            f"Failure mode: {hypothesis.predicted_failure_mode}\n\n"
            "## Model setup (copy verbatim)\n"
            "```python\n"
            f"{model_context.get('import_expr', 'import evalvitals')}\n"
            f"model = {model_context.get('load_expr', '# model')}\n"
            "```\n\n"
            f"Available capabilities: {caps}\n\n"
            "## Input data\n"
            "Read `cases.json` from the current directory. "
            "Each record has: prompt, label (PASS/FAIL), id, metadata.\n\n"
            "## Required output (print to stdout, EXACTLY this format)\n"
            "```\n"
            "metric_a: 0.72\n"
            "metric_b: 3.0\n"
            "verdict: 1.0    # 1.0=SUPPORTED  0.0=REFUTED\n"
            "```\n\n"
            "## Rules\n"
            f"- Stay under {timeout_sec} seconds total runtime.\n"
            "- Save the script as `experiment.py` in the current directory.\n"
            "- Print all metrics as `name: float_value` lines.\n"
            "- The last printed line must be `verdict: 1.0` or `verdict: 0.0`.\n"
            "- Use only the evalvitals APIs shown above.\n"
            "- Do NOT make external network calls or read files other than "
            "`cases.json`.\n"
        )

    # ------------------------------------------------------------------
    # Phase 1: write (LLM path)
    # ------------------------------------------------------------------

    def _phase1_write(
        self,
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        cases_json: str,
    ) -> str:
        self._log_event("Phase 1: writing diagnostic script")
        system = _WRITE_SYSTEM.format(timeout_sec=self._cfg.exec_fix_timeout_sec)
        user = _WRITE_USER.format(
            statement=hypothesis.statement,
            failure_mode=hypothesis.predicted_failure_mode,
            import_expr=model_context.get("import_expr", "import evalvitals"),
            load_expr=model_context.get("load_expr", "# model"),
            capabilities=", ".join(model_context.get("capabilities", [])) or "GENERATE",
            cases_json=cases_json,
        )
        raw = self._llm_call(system, user)
        code = self._extract_code_block(raw)
        self._log_event(f"  Script: {len(code)} chars")
        return code

    # ------------------------------------------------------------------
    # Phase 2: hard validation (AST)
    # ------------------------------------------------------------------

    def _hard_validate_and_repair(
        self,
        code: str,
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        cases_json: str,
    ) -> str:
        for attempt in range(self._cfg.hard_validation_max_repairs + 1):
            errors = self._validate_ast(code)
            if not errors:
                self._log_event(f"  AST validation OK (attempt {attempt})")
                return code
            self._log_event(
                f"  AST errors (attempt {attempt}/{self._cfg.hard_validation_max_repairs}): "
                + "; ".join(errors)
            )
            if attempt >= self._cfg.hard_validation_max_repairs:
                self._log_event("  Max AST repairs reached — proceeding with warnings")
                return code
            code = self._repair_code(code, "\n".join(errors))
        return code

    def _validate_ast(self, code: str) -> list[str]:
        try:
            ast.parse(code)
            return []
        except SyntaxError as exc:
            return [f"SyntaxError at line {exc.lineno}: {exc.msg}"]

    # ------------------------------------------------------------------
    # Phase 3: exec-fix loop
    # ------------------------------------------------------------------

    def _exec_fix_loop(
        self,
        code: str,
        sandbox: ExperimentSandbox,
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        cases_json: str,
    ) -> tuple[str, SandboxResult]:
        last_result: SandboxResult | None = None

        for i in range(self._cfg.exec_fix_max_iterations):
            result = sandbox.run(code, timeout_sec=self._cfg.exec_fix_timeout_sec)
            self._runs += 1
            last_result = result

            if result.ok:
                self._log_event(f"  Exec-fix iter {i}: OK ({result.elapsed_sec:.1f}s)")
                break

            if result.timed_out:
                self._log_event(f"  Exec-fix iter {i}: TIMEOUT — no further repair")
                break

            self._log_event(
                f"  Exec-fix iter {i}: crashed (rc={result.returncode}), "
                f"stderr={len(result.stderr)} chars"
            )
            if i < self._cfg.exec_fix_max_iterations - 1:
                code = self._repair_code(code, result.stderr[-2000:])  # tail of stderr

        # Fallback: if sandbox never ran (shouldn't happen)
        if last_result is None:
            from evalvitals.eval_agent.sandbox import SandboxResult as SR
            last_result = SR(returncode=-1, stdout="", stderr="no runs", elapsed_sec=0.0)

        return code, last_result

    # ------------------------------------------------------------------
    # Repair helpers
    # ------------------------------------------------------------------

    def _repair_code(self, code: str, error: str) -> str:
        self._log_event(f"  Repairing: {error[:120]}…")
        raw = self._llm_call(
            _REPAIR_SYSTEM,
            _REPAIR_USER.format(error=error, code=code),
        )
        repaired = self._extract_code_block(raw)
        return repaired if repaired.strip() else code

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _llm_call(self, system: str, user: str) -> str:
        self._calls += 1
        try:
            prompt = f"{system}\n\n{user}"
            return self._judge.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            self._log_event(f"  LLM call failed: {exc}")
            return ""

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Extract the first ```python … ``` block, or the raw text if none found."""
        import re
        m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Strip markdown fences if present
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _log_event(self, msg: str) -> None:
        logger.debug(msg)
        self._log.append(msg)
