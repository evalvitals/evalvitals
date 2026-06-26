"""Local exploratory analysis agent for standalone M2 use.

This is the backend-only, LAMBDA-style path: a coding agent writes Python,
EvalVitals runs it locally in a sandbox, and the result is parsed into a
structured exploratory report.  Findings from this module are candidates; use
``StatsAnalysisAgent`` for confirmatory effect/CI/e-value/FDR verdicts.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evalvitals.eval_agent.sandbox import ExperimentSandbox, SandboxResult

if TYPE_CHECKING:
    from evalvitals.core.model import Model
    from evalvitals.eval_agent.cli_agent import CliAgentConfig

logger = logging.getLogger(__name__)

_INPUT_FILENAME = "records.json"
_RESULT_MARKER = "EXPLORATORY_RESULT_JSON="

_GENERATE_PROMPT = """\
You are an exploratory data-analysis agent for model failure analysis.

Question:
{question}

A JSON file named "{input_filename}" is in the current working directory.
It contains a list of row dictionaries. Data profile:
{data_profile}

Write a self-contained Python script that:
- reads "{input_filename}" from the current working directory
- explores patterns relevant to the question
- may use only local Python packages; no network and no repo mutation
- may write plots under a local "figures/" directory
- does NOT claim causal/statistical confirmation; this is exploratory only
- prints the final result as the LAST stdout line exactly like:
  {marker}{{"observations": ["..."], "candidate_signals": [{{"name": "...", "rationale": "...", "suggested_test": "..."}}], "plots": ["figures/name.png"], "tables": {{}}, "caveats": ["..."], "recommended_confirmatory_tests": ["..."]}}

Return ONLY the Python code{fences_hint}."""

_REPAIR_PROMPT = """\
The exploratory analysis script failed or produced an invalid result.

Question:
{question}

Data profile:
{data_profile}

Previous code:
```python
{code}
```

Sandbox stdout:
{stdout}

Sandbox stderr:
{stderr}

Parser/execution error:
{error}

Rewrite the script. It must read "{input_filename}" and print a final
{marker} JSON line with the required keys. Return ONLY Python code{fences_hint}."""


@dataclass
class CandidateSignal:
    """A signal worth testing later in confirmatory M2."""

    name: str
    rationale: str = ""
    suggested_test: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rationale": self.rationale,
            "suggested_test": self.suggested_test,
        }


@dataclass
class ExploratoryAnalysisReport:
    """Output of the standalone exploratory M2 backend."""

    question: str = ""
    ok: bool = False
    observations: list[str] = field(default_factory=list)
    candidate_signals: list[CandidateSignal] = field(default_factory=list)
    plots: list[str] = field(default_factory=list)
    tables: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    recommended_confirmatory_tests: list[str] = field(default_factory=list)
    data_profile: dict[str, Any] = field(default_factory=dict)
    code: str = ""
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    attempts: int = 0
    workdir: str = ""
    raw_outputs: list[str] = field(default_factory=list)

    @property
    def candidate_signal_names(self) -> list[str]:
        return [s.name for s in self.candidate_signals if s.name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ok": self.ok,
            "observations": self.observations,
            "candidate_signals": [s.to_dict() for s in self.candidate_signals],
            "plots": self.plots,
            "tables": self.tables,
            "caveats": self.caveats,
            "recommended_confirmatory_tests": self.recommended_confirmatory_tests,
            "data_profile": self.data_profile,
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "attempts": self.attempts,
            "workdir": self.workdir,
        }


class M2ExplorerAgent:
    """Backend-only exploratory analysis agent.

    Args:
        judge:      LLM-like object with ``generate(prompt) -> str`` used to
                    write the initial analysis script.
        inspector:  Optional LLM-like object used for repair prompts. When not
                    provided, ``judge`` repairs its own code.
        cli_config: Optional CLI coding-agent backend. If set to a non-``llm``
                    provider, it writes ``analysis.py`` locally.
        sandbox:    Local execution sandbox. A durable temp sandbox is created
                    when omitted so generated code and figures remain available.
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        inspector: "Model | None" = None,
        cli_config: "CliAgentConfig | None" = None,
        sandbox: ExperimentSandbox | None = None,
        timeout_sec: int = 60,
        max_attempts: int = 2,
    ) -> None:
        self._judge = judge
        self._inspector = inspector
        self._cli_config = cli_config
        self._sandbox = sandbox or ExperimentSandbox(cleanup=False)
        self._timeout_sec = timeout_sec
        self._max_attempts = max(1, max_attempts)

    @property
    def available(self) -> bool:
        return self._judge is not None or (
            self._cli_config is not None and self._cli_config.provider != "llm"
        )

    def explore_records(
        self,
        records: Any,
        *,
        question: str = "Explore patterns that distinguish failures from passes.",
    ) -> ExploratoryAnalysisReport:
        """Run local exploratory analysis over plain records."""
        rows = _records_to_rows(records)
        profile = _profile_rows(rows)
        self._write_input(rows)

        if not self.available:
            return ExploratoryAnalysisReport(
                question=question,
                ok=False,
                data_profile=profile,
                error="no code-writing backend configured (judge or cli_config)",
                workdir=str(self._sandbox.workdir),
            )

        raw_outputs: list[str] = []
        code = ""
        last_result: SandboxResult | None = None
        last_error = ""

        for attempt in range(1, self._max_attempts + 1):
            try:
                if attempt == 1:
                    code, raw = self._write_code(question, profile)
                else:
                    code, raw = self._repair_code(
                        question, profile, code, last_result, last_error
                    )
                raw_outputs.append(raw)
            except Exception as exc:  # noqa: BLE001
                last_error = f"code writing failed: {exc}"
                logger.warning("M2ExplorerAgent: %s", last_error)
                break

            if not code.strip():
                last_error = "backend produced no code"
                continue

            last_result = self._sandbox.run(code, timeout_sec=self._timeout_sec)
            report, last_error = _report_from_sandbox(
                question=question,
                profile=profile,
                code=code,
                result=last_result,
                attempts=attempt,
                workdir=Path(self._sandbox.workdir),
            )
            report.raw_outputs = raw_outputs
            if report.ok:
                return report

        stdout = last_result.stdout if last_result is not None else ""
        stderr = last_result.stderr if last_result is not None else ""
        return ExploratoryAnalysisReport(
            question=question,
            ok=False,
            data_profile=profile,
            code=code,
            stdout=stdout,
            stderr=stderr,
            error=last_error or "exploratory analysis failed",
            attempts=min(self._max_attempts, len(raw_outputs)),
            workdir=str(self._sandbox.workdir),
            raw_outputs=raw_outputs,
        )

    def _write_input(self, rows: list[dict[str, Any]]) -> None:
        path = Path(self._sandbox.workdir) / _INPUT_FILENAME
        path.write_text(json.dumps(rows, default=str), encoding="utf-8")

    def _write_code(self, question: str, profile: dict[str, Any]) -> tuple[str, str]:
        prompt = _GENERATE_PROMPT.format(
            question=question,
            input_filename=_INPUT_FILENAME,
            data_profile=json.dumps(profile, indent=2, default=str),
            marker=_RESULT_MARKER,
            fences_hint=_fences_hint(self._cli_config),
        )
        return self._run_writer(prompt, use_inspector=False)

    def _repair_code(
        self,
        question: str,
        profile: dict[str, Any],
        code: str,
        result: SandboxResult | None,
        error: str,
    ) -> tuple[str, str]:
        prompt = _REPAIR_PROMPT.format(
            question=question,
            data_profile=json.dumps(profile, indent=2, default=str),
            code=code,
            stdout=(result.stdout if result is not None else "")[-2000:],
            stderr=(result.stderr if result is not None else "")[-2000:],
            error=error,
            input_filename=_INPUT_FILENAME,
            marker=_RESULT_MARKER,
            fences_hint=_fences_hint(self._cli_config),
        )
        return self._run_writer(prompt, use_inspector=True)

    def _run_writer(self, prompt: str, *, use_inspector: bool) -> tuple[str, str]:
        if self._cli_config is not None and self._cli_config.provider != "llm":
            return self._run_cli_writer(prompt)
        model = self._inspector if use_inspector and self._inspector is not None else self._judge
        raw = model.generate(prompt)  # type: ignore[union-attr]
        raw_text = str(raw)
        return _extract_code(raw_text), raw_text

    def _run_cli_writer(self, prompt: str) -> tuple[str, str]:
        from evalvitals.eval_agent.cli_agent import create_cli_agent

        agent = create_cli_agent(self._cli_config)  # type: ignore[arg-type]
        res = agent.run(prompt, workdir=Path(self._sandbox.workdir), timeout_sec=self._timeout_sec)
        if not res.ok:
            return "", res.raw_output or (res.error or "")
        if "analysis.py" in res.files:
            return res.files["analysis.py"], res.raw_output
        py_files = {n: c for n, c in res.files.items() if n.endswith(".py")}
        if not py_files:
            return "", res.raw_output
        return max(py_files.values(), key=len), res.raw_output


def _records_to_rows(records: Any) -> list[dict[str, Any]]:
    if records is None:
        return []
    if hasattr(records, "to_dict"):
        try:
            data = records.to_dict(orient="records")
            if isinstance(data, list):
                return [dict(r) for r in data if isinstance(r, dict)]
        except TypeError:
            pass
    rows: list[dict[str, Any]] = []
    for row in list(records):
        if isinstance(row, dict):
            rows.append(dict(row))
        elif hasattr(row, "_asdict"):
            rows.append(dict(row._asdict()))
        elif hasattr(row, "__dict__"):
            rows.append(dict(vars(row)))
        else:
            rows.append({"value": row})
    return rows


def _profile_rows(rows: list[dict[str, Any]], *, sample_size: int = 5) -> dict[str, Any]:
    columns: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key, value in row.items():
            info = columns.setdefault(str(key), {"non_null": 0, "types": {}, "numeric": 0})
            if value is None:
                continue
            info["non_null"] += 1
            typ = type(value).__name__
            info["types"][typ] = info["types"].get(typ, 0) + 1
            if isinstance(value, (int, float, bool)):
                info["numeric"] += 1

    numeric_columns = [
        name for name, info in columns.items()
        if info["non_null"] > 0 and info["numeric"] == info["non_null"]
    ]
    label_like = [
        name for name in columns
        if name.lower() in {"label", "outcome", "status", "success", "pass", "fail"}
        or "label" in name.lower()
    ]
    return {
        "n_rows": len(rows),
        "columns": columns,
        "numeric_columns": numeric_columns,
        "label_like_columns": label_like,
        "sample_rows": rows[:sample_size],
    }


def _report_from_sandbox(
    *,
    question: str,
    profile: dict[str, Any],
    code: str,
    result: SandboxResult,
    attempts: int,
    workdir: Path,
) -> tuple[ExploratoryAnalysisReport, str]:
    if not result.ok:
        err = (result.stderr or "").strip() or "sandbox run failed"
        return (
            ExploratoryAnalysisReport(
                question=question,
                ok=False,
                data_profile=profile,
                code=code,
                stdout=result.stdout,
                stderr=result.stderr,
                error=err,
                attempts=attempts,
                workdir=str(workdir),
            ),
            err,
        )

    parsed, err = _parse_result_json(result.stdout)
    if err:
        return (
            ExploratoryAnalysisReport(
                question=question,
                ok=False,
                data_profile=profile,
                code=code,
                stdout=result.stdout,
                stderr=result.stderr,
                error=err,
                attempts=attempts,
                workdir=str(workdir),
            ),
            err,
        )

    plots = _normalize_plot_paths(parsed.get("plots", []), workdir)
    signals = [
        CandidateSignal(
            name=str(item.get("name", "")),
            rationale=str(item.get("rationale", "")),
            suggested_test=str(item.get("suggested_test", "")),
        )
        for item in parsed.get("candidate_signals", []) or []
        if isinstance(item, dict) and item.get("name")
    ]
    return (
        ExploratoryAnalysisReport(
            question=question,
            ok=True,
            observations=[str(x) for x in parsed.get("observations", []) or []],
            candidate_signals=signals,
            plots=plots,
            tables=dict(parsed.get("tables", {}) or {}),
            caveats=[str(x) for x in parsed.get("caveats", []) or []],
            recommended_confirmatory_tests=[
                str(x) for x in parsed.get("recommended_confirmatory_tests", []) or []
            ],
            data_profile=profile,
            code=code,
            stdout=result.stdout,
            stderr=result.stderr,
            attempts=attempts,
            workdir=str(workdir),
        ),
        "",
    )


def _parse_result_json(stdout: str) -> tuple[dict[str, Any], str]:
    marker_line = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(_RESULT_MARKER):
            marker_line = stripped[len(_RESULT_MARKER):]
    if marker_line is None:
        return {}, f"no {_RESULT_MARKER} line in output"
    try:
        parsed = json.loads(marker_line)
    except json.JSONDecodeError as exc:
        return {}, f"unparseable {_RESULT_MARKER} JSON: {exc}"
    if not isinstance(parsed, dict):
        return {}, f"{_RESULT_MARKER} payload must be a JSON object"
    return parsed, ""


def _normalize_plot_paths(raw: Any, workdir: Path) -> list[str]:
    paths: list[str] = []
    for item in raw or []:
        text = str(item)
        path = Path(text)
        if not path.is_absolute():
            candidate = workdir / path
            if candidate.exists():
                path = candidate
        paths.append(str(path))
    figures_dir = workdir / "figures"
    if figures_dir.exists():
        for png in sorted(figures_dir.glob("*.png")):
            p = str(png)
            if p not in paths:
                paths.append(p)
    return paths


def _extract_code(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", cleaned, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return cleaned.strip()


def _fences_hint(cli_config: "CliAgentConfig | None") -> str:
    if cli_config is not None and cli_config.provider != "llm":
        return ", written to a file named analysis.py"
    return " inside a ```python code block"
