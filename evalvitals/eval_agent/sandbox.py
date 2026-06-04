"""Experiment sandbox — safe subprocess execution + ARC-compatible metric parsing.

Mirrors researchclaw/experiment/sandbox.py.

Metric output formats understood by ``parse_metrics``:

    Plain:      ``metric: 0.85``
    Condition:  ``condition=arithmetic metric: 0.42``
    Summary:    ``SUMMARY condition=X metric=Y mean=M std=S``

The diagnostic script printed by ``ExperimentWriter`` uses these same formats,
so ``SandboxResult.metrics`` is immediately ready for the surgery agent to
interpret.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Metric parsing  (ARC-compatible formats)
# ---------------------------------------------------------------------------

_FLOAT_RE = r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?"

# Plain:      "metric: 0.85"
_PLAIN_RE = re.compile(rf"^(\w[\w./]*)\s*:\s*({_FLOAT_RE})\s*$")

# Condition:  "condition=X metric: 0.42"
_CONDITION_RE = re.compile(rf"^condition=(\S+)\s+(\w[\w./]*)\s*:\s*({_FLOAT_RE})\s*$")

# Summary:    "SUMMARY condition=X metric=Y mean=M std=S"
_SUMMARY_RE = re.compile(
    rf"^SUMMARY\s+condition=(\S+)\s+metric=(\S+)\s+mean=({_FLOAT_RE})\s+std=({_FLOAT_RE})"
)


def parse_metrics(stdout: str) -> dict[str, float]:
    """Extract ``name: float`` pairs from *stdout*.

    Matching priority: SUMMARY lines first, then condition-prefixed, then plain.
    Non-finite values are silently dropped.
    """
    metrics: dict[str, float] = {}

    for line in stdout.splitlines():
        stripped = line.strip()

        # SUMMARY condition=X metric=Y mean=M std=S
        m = _SUMMARY_RE.match(stripped)
        if m:
            cond, metric, mean_s, std_s = m.groups()
            try:
                mean_v, std_v = float(mean_s), float(std_s)
            except ValueError:
                continue
            if math.isfinite(mean_v):
                metrics[f"{cond}/{metric}"]      = mean_v
                metrics[f"{cond}/{metric}_mean"] = mean_v
                metrics[f"{cond}/{metric}_std"]  = std_v
                metrics[metric]                  = mean_v
            continue

        # condition=X metric: value
        m = _CONDITION_RE.match(stripped)
        if m:
            cond, name, val_s = m.groups()
            try:
                val = float(val_s)
            except ValueError:
                continue
            if math.isfinite(val):
                metrics[f"{cond}/{name}"] = val
                metrics[name]             = val
            continue

        # metric: value
        m = _PLAIN_RE.match(stripped)
        if m:
            name, val_s = m.groups()
            try:
                val = float(val_s)
            except ValueError:
                continue
            if math.isfinite(val):
                metrics[name] = val


    return metrics


# ---------------------------------------------------------------------------
# Result / sandbox
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """Output of one :class:`ExperimentSandbox` run.

    Attributes:
        returncode:  Process exit code (0 = success, −1 = internal/timeout).
        stdout:      Captured standard output.
        stderr:      Captured standard error.
        elapsed_sec: Wall-clock seconds.
        metrics:     Parsed ``{name: float}`` from *stdout*.
        timed_out:   ``True`` when the process was killed for exceeding the budget.
    """

    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    metrics: dict[str, float] = field(default_factory=dict)
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class ExperimentSandbox:
    """Run Python code in a subprocess and parse metrics from stdout.

    Mirrors ``researchclaw.experiment.sandbox.ExperimentSandbox``.

    Args:
        workdir:  Directory where scripts are written.  A temporary directory
                  is created when ``None``.
    """

    def __init__(self, workdir: Path | str | None = None) -> None:
        if workdir is None:
            workdir = Path(tempfile.mkdtemp(prefix="evalvitals_sandbox_"))
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._run_counter = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, code: str, *, timeout_sec: int = 60) -> SandboxResult:
        """Write *code* to a script file and execute it in a subprocess.

        Args:
            code:        Python source to execute.
            timeout_sec: Hard wall-clock limit; the process is killed on breach.

        Returns:
            :class:`SandboxResult` with parsed metrics.
        """
        self._run_counter += 1
        script_path = self.workdir / f"exp_{self._run_counter:04d}.py"
        script_path.write_text(code, encoding="utf-8")
        return self._run_script(script_path, timeout_sec=timeout_sec)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_script(self, script_path: Path, *, timeout_sec: int) -> SandboxResult:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        start = time.monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, "-u", str(script_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                cwd=str(self.workdir),
                env=env,
                check=False,
            )
            elapsed = time.monotonic() - start
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            return SandboxResult(
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                elapsed_sec=elapsed,
                metrics=parse_metrics(stdout),
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - start
            stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if exc.stdout else ""
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if exc.stderr else ""
            return SandboxResult(
                returncode=-1,
                stdout=stdout,
                stderr=f"[TIMEOUT after {timeout_sec}s]\n{stderr}",
                elapsed_sec=elapsed,
                metrics=parse_metrics(stdout),
                timed_out=True,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            return SandboxResult(
                returncode=-1,
                stdout="",
                stderr=f"[SANDBOX ERROR] {exc}",
                elapsed_sec=elapsed,
                metrics={},
            )
