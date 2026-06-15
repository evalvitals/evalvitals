"""RunLogger — structured per-cycle logging for AutoDiagnoseLoop.

Writes a JSONL event log (one line per M1/M2/M3/M4 event) and saves heavy
analyzer artifacts (attention tensors, hidden-state arrays) to a separate
``artifacts/`` directory, keyed by cycle number so they stay navigable.

Each JSON record always contains a ``ts`` (ISO-8601) and an ``event`` field.
The underlying file handler is a standard :class:`logging.FileHandler`, so
callers can attach additional handlers (e.g. a ``StreamHandler`` for console
output) by accessing :attr:`RunLogger.logger`.

Usage::

    from evalvitals.eval_agent import AutoDiagnoseLoop, RunLogger

    loop = AutoDiagnoseLoop(model=model, run_logger=RunLogger("runs/exp_01"))
    report = loop.run(cases)
    # runs/exp_01/run_log.jsonl          ← one JSON line per event, grep/jq friendly
    # runs/exp_01/artifacts/c0_attention_attn_weights.npy  ← attention tensor
    # runs/exp_01/artifacts/c0_cka_layer_similarities.npy  ← CKA matrix

Stream events while running::

    tail -f runs/exp_01/run_log.jsonl | python -m json.tool

Filter by module::

    jq 'select(.event=="diagnosis")' runs/exp_01/run_log.jsonl

Auto-timestamped run dir (default when no path is given)::

    loop = AutoDiagnoseLoop(model=model, run_logger=RunLogger())
    # loop.run_logger.run_dir  → runs/20260603_142305/

Verbose console output (human-readable summary to stdout)::

    loop = AutoDiagnoseLoop(model=model, run_logger=RunLogger(verbose=True))

Custom handler — e.g. redirect verbose output to a file instead::

    import logging, sys
    rl = RunLogger("runs/exp_01")
    rl.logger.addHandler(logging.StreamHandler(sys.stderr))
"""

from __future__ import annotations

import json
import logging
import re
import sys
import textwrap
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.hypothesis import Hypothesis
    from evalvitals.eval_agent.loop import AutoDiagnoseReport
    from evalvitals.eval_agent.stages.analysis import AnalysisReport
    from evalvitals.eval_agent.stages.diagnosis import DiagnosisResult
    from evalvitals.eval_agent.stages.surgery import InterventionResult


def _artifact_to_numpy(artifact: Any) -> "Any | None":
    """Convert *artifact* to a numpy array, or return None if not possible.

    Handles: torch.Tensor, list[torch.Tensor] (e.g. per-layer attentions),
    and numpy arrays.  A list of tensors is stacked along a new first axis so
    that ``attentions`` (list of ``(heads, seq, seq)``) becomes
    ``(layers, heads, seq, seq)`` — a single array that retains all the data.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    if hasattr(artifact, "detach"):  # torch.Tensor
        return artifact.detach().cpu().float().numpy()
    if isinstance(artifact, np.ndarray):
        return artifact
    if isinstance(artifact, list) and artifact and hasattr(artifact[0], "detach"):
        try:
            import torch
            return torch.stack(artifact).detach().cpu().float().numpy()
        except Exception:  # noqa: BLE001
            return None
    return None


def _save_artifact_figure(artifact_dir: Path, stem: str, arr: Any) -> None:
    """Save a matplotlib figure of *arr* when the shape and stem are recognised.

    Dispatch table (first match wins):
    - 4-D + ``attn`` in stem → mean over (layers, heads) → 2-D heatmap
    - 3-D + ``attn`` in stem → mean over heads → 2-D heatmap
    - 2-D + heatmap keyword  → direct heatmap (viridis)
    - 1-D + curve keyword    → line plot
    Skips silently when matplotlib is unavailable or the shape is unrecognised.
    """
    try:
        import matplotlib.pyplot as plt
        plt.ioff()
    except ImportError:
        return

    key = stem.lower()
    # Skip logit arrays — (seq, vocab) shape is too large for a useful figure
    if "logit" in key:
        return

    _is_attn = any(k in key for k in ("attn", "attention"))

    fig = None
    try:
        ndim = arr.ndim
        if ndim == 4 and _is_attn:
            mat = arr.mean(axis=(0, 1))  # (layers, heads, seq, seq) → (seq, seq)
            n_layers, n_heads = arr.shape[0], arr.shape[1]
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0)
            ax.set_title(f"{stem}  (mean over {n_layers}L × {n_heads}H)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
        elif ndim == 3 and _is_attn:
            mat = arr.mean(axis=0)  # (heads, seq, seq) → (seq, seq)
            n_heads = arr.shape[0]
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0)
            ax.set_title(f"{stem}  (mean over {n_heads} heads)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
        elif ndim == 2 and "diff" in key:
            # Signed difference map (e.g. FAIL-mean minus PASS-mean attention):
            # diverging colormap with symmetric limits so the sign is readable.
            bound = float(max(abs(arr.min()), abs(arr.max()))) or 1.0
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(arr, cmap="coolwarm", aspect="auto", vmin=-bound, vmax=bound)
            ax.set_title(stem)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
        elif ndim == 2 and (_is_attn or any(k in key for k in ("rollout", "spatial", "map"))):
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(arr, cmap="viridis", aspect="auto")
            ax.set_title(stem)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
        elif ndim == 1 and any(k in key for k in ("entropy", "score", "prob", "weight", "rollout")):
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(arr)
            ax.set_xlabel("position")
            ax.set_ylabel(stem)
            ax.set_title(stem)
            plt.tight_layout()

        if fig is not None:
            fig.savefig(artifact_dir / f"{stem}.png", dpi=100, bbox_inches="tight")
    except Exception:  # noqa: BLE001
        pass
    finally:
        if fig is not None:
            plt.close(fig)


class _JsonFormatter(logging.Formatter):
    """Format each LogRecord as a single JSON line using the ``_payload`` extra."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(getattr(record, "_payload", {}), default=str)


class _VerboseFormatter(logging.Formatter):
    """Format each LogRecord as a human-readable stage summary.

    Reads the same ``_payload`` dict that ``_JsonFormatter`` serialises to
    JSON, dispatches on ``payload["event"]``, and returns a multi-line string
    with the most useful fields for interactive / Docker console output.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: PLR0911
        p = getattr(record, "_payload", {})
        event = p.get("event", "")
        cycle = p.get("cycle", "?")

        if event == "run_start":
            lines = ["\n[START] run configuration"]
            for k in (
                "model", "judge", "coder", "max_cycles", "depth",
                "allow_codegen", "n_cases", "evalvitals_version", "git_commit",
            ):
                if p.get(k) is not None:
                    lines.append(f"     {k:18s}: {p[k]}")
            return "\n".join(lines)

        if event == "probe":
            lines = [f"\n[M1] cycle={cycle}  analyzers={p.get('analyzers', [])}"]
            rationale = p.get("selection_rationale", "")
            if rationale:
                lines.append(f"     rationale  : {rationale}")
            for name, findings in (p.get("findings") or {}).items():
                scalars = {
                    k: round(v, 4)
                    for k, v in (findings or {}).items()
                    if isinstance(v, (int, float))
                }
                lines.append(f"     {name}: {dict(list(scalars.items())[:6])}")
            return "\n".join(lines)

        if event == "analysis":
            lines = [f"\n[M2] cycle={cycle}  severity={p.get('severity')}"]
            conclusion = p.get("conclusion")
            if conclusion:
                lines.append(
                    "     conclusion : "
                    + textwrap.fill(conclusion, 72, subsequent_indent="     ")
                )
            for step in (p.get("evidence_chain") or [])[:3]:
                lines.append(f"     evidence   : {step}")
            stats_plan = p.get("stats_plan") or []
            if stats_plan:
                lines.append(f"     stats_tools: {[s['tool'] for s in stats_plan]}")
            corrected = p.get("corrected_rejections") or {}
            if corrected.get("rejected_tools"):
                lines.append(f"     fdr_survive: {corrected['rejected_tools']}")
            for tool in (p.get("stats_tool_results") or [])[:2]:
                lines.append(
                    f"     stats_tool : {tool.get('name')} - {tool.get('conclusion', '')}"
                )
            for fig in p.get("figures") or []:
                lines.append(f"     figure     : {fig}")
            if not conclusion:
                lines.append(
                    "     "
                    + textwrap.fill(p.get("narrative", ""), 72, subsequent_indent="     ")
                )
            return "\n".join(lines)

        if event == "diagnosis":
            lines = [f"\n[M3] cycle={cycle}  {p.get('n_hypotheses', 0)} hypothesis/es"]
            for h in p.get("hypotheses") or []:
                lines.append(f"     hypothesis  : {h.get('statement', '')}")
                lines.append(f"     failure_mode: {h.get('failure_mode', '')}")
            return "\n".join(lines)

        if event == "surgery":
            module = p.get("module", "m4").upper()
            hyp = p.get("hypothesis", "")[:70]
            status = p.get("status", "?")
            lines = [f"\n[{module}] cycle={cycle}  '{hyp}'"]
            ev = p.get("evidence") or {}
            if module == "M5":
                lines.append(
                    f"     status={status}"
                    f"  effect={ev.get('m5_effect_size', '?')}"
                    f"  confidence={ev.get('m5_confidence', '?')}"
                )
                lines.append(
                    f"     protocol_consistent={ev.get('m5_protocol_consistent', '?')}"
                )
                lines.append(f"     verdict : {ev.get('m5_verdict', '')}")
            else:
                lines.append(f"     status={status}  fixed={p.get('fixed')}")
                if ev:
                    lines.append(f"     evidence: {dict(list(ev.items())[:4])}")
            return "\n".join(lines)

        if event == "experiment":
            module = p.get("module", "m4").upper()
            lines = [f"\n[{module}] cycle={cycle}  experiment run"]
            lines.append(f"     hypothesis : {p.get('hypothesis', '')[:70]}")
            lines.append(
                f"     status={p.get('status')}  verdict={p.get('verdict')}"
                f"  fixed={p.get('fixed')}  rc={p.get('returncode')}"
                f"  provider={p.get('provider')}"
            )
            code_paths = p.get("code_paths") or {}
            if code_paths:
                lines.append(f"     code       : {list(code_paths.values())}")
            ws = p.get("workspace_snapshot") or {}
            if ws.get("dir"):
                lines.append(
                    f"     workspace  : {ws['dir']} ({len(ws.get('files', []))} files)"
                )
            return "\n".join(lines)

        if event == "tool_codegen":
            ok = "OK" if p.get("ok") else "FAILED"
            lines = [
                f"\n[TOOL] cycle={cycle}  {p.get('module')}/{p.get('tool_name')}  "
                f"{ok}  source={p.get('source')}"
            ]
            if p.get("need"):
                lines.append(f"     need       : {p['need'][:72]}")
            if p.get("error"):
                lines.append(f"     error      : {p['error'][:72]}")
            paths = p.get("artifact_paths") or {}
            if paths.get("code"):
                lines.append(f"     code       : {paths['code']}")
            return "\n".join(lines)

        if event == "tool_registry":
            lines = [
                f"\n[TOOL] cycle={cycle}  {p.get('module')}  "
                f"{p.get('n_tools', 0)} active synthesised tool(s)"
            ]
            for t in p.get("tools") or []:
                lines.append(f"     tool       : {t.get('name')} (source={t.get('source')})")
            return "\n".join(lines)

        if event == "loop_end":
            stopped_by = p.get("stopped_by")
            if stopped_by is not None:
                return f"\n[DONE] cycles={p.get('cycles')}  stopped_by={stopped_by}"
            return f"\n[DONE] cycles={p.get('cycles')}  resolved={p.get('resolved')}"

        return json.dumps(p, default=str)


class RunLogger:
    """Structured JSONL logger + artifact sink for one :class:`AutoDiagnoseLoop` run.

    Each call to a ``log_*`` method appends one JSON object to ``run_log.jsonl``
    (always including a ``ts`` ISO-8601 timestamp and a ``cycle`` index).
    Heavy artifacts from M1 results are written to ``artifacts/`` as ``.npy``
    (numpy / torch tensors) or ``.json`` (dicts / lists).

    The underlying :attr:`logger` is a standard :class:`logging.Logger` named
    ``evalvitals.run.<run_dir_name>``.  It does **not** propagate to the root
    logger so the library stays silent by default.  Attach additional handlers
    to customise where and how events appear::

        rl = RunLogger("runs/exp_01")
        rl.logger.addHandler(logging.StreamHandler(sys.stderr))

    Args:
        run_dir:  Directory to write into.  Created if it does not exist.
                  Defaults to ``runs/<YYYYMMDD_HHMMSS>/`` relative to cwd.
        verbose:  When ``True``, attach a stdout :class:`logging.StreamHandler`
                  with human-readable formatting.  Equivalent to::

                      rl.logger.addHandler(
                          logging.StreamHandler(sys.stdout)
                          # formatted by _VerboseFormatter
                      )

    The logger is safe to use as a context manager::

        with RunLogger("runs/my_exp", verbose=True) as rl:
            loop = AutoDiagnoseLoop(model=model, run_logger=rl)
            loop.run(cases)
    """

    def __init__(
        self,
        run_dir: str | Path | None = None,
        *,
        verbose: bool = False,
        trace_id: str | None = None,
    ) -> None:
        if run_dir is None:
            run_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(run_dir)
        self.artifact_dir = self.run_dir / "artifacts"
        # Dedicated, human-navigable sinks for the heavier event payloads.
        #   experiments/  — M4 experiment scripts, run stdout/stderr, the agent's
        #                   intermediate thinking (CLI narration / LLM phase log)
        #   tools/        — code the agent synthesised for new probes / stats tools
        #   workspace/    — per-event snapshots of the sandbox working directory
        self.experiments_dir = self.run_dir / "experiments"
        self.tools_dir = self.run_dir / "tools"
        self.workspace_dir = self.run_dir / "workspace"
        # fixes/ — one self-contained record per tiered-repair attempt, plus an
        #          outcome.md summarising all candidates and the escalation
        #          recommendation.  Written by log_fix().
        self.fixes_dir = self.run_dir / "fixes"
        # prompts/ — the verbatim prompt and raw response of every LLM judge
        #            call (M1 analyzer selection, M2 analysis, M3 diagnosis), so
        #            each conclusion can be traced back to exactly what the judge
        #            was shown and what it returned.
        self.prompts_dir = self.run_dir / "prompts"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(exist_ok=True)
        self.log_path = self.run_dir / "run_log.jsonl"

        # The loop stamps this at the top of every cycle so generator-level
        # events (which have no cycle of their own) can be correlated with the
        # M1→M5 events around them.  -1 means "outside any cycle" (e.g. post-loop M4).
        self.current_cycle: int = -1
        # Monotonic counter so codegen artifacts written in the same cycle never
        # collide on filename.
        self._codegen_seq: int = 0

        # trace_id ties all events from a single AutoDiagnoseLoop.run() call
        # together — including events from any recursive or nested sub-loops.
        # Callers may supply their own ID (e.g. to correlate with an outer
        # pipeline) or let RunLogger generate a fresh UUID.
        self.trace_id: str = trace_id or str(uuid.uuid4())

        self.logger = logging.getLogger(f"evalvitals.run.{self.run_dir.name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        self._file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        self._file_handler.setFormatter(_JsonFormatter())
        self.logger.addHandler(self._file_handler)

        self._console_handler: logging.StreamHandler | None = None
        if verbose:
            self._console_handler = logging.StreamHandler(sys.stdout)
            self._console_handler.setFormatter(_VerboseFormatter())
            self.logger.addHandler(self._console_handler)

    # ------------------------------------------------------------------
    # Run provenance
    # ------------------------------------------------------------------

    def log_run_start(self, config: "dict[str, Any] | None" = None) -> None:
        """Record a ``run_start`` event with the settings that produced this run.

        *config* is whatever the caller knows (model, protocol, judge/coder
        provider+model, max_cycles, cases…).  This method auto-enriches it with
        the evalvitals + Python versions and the current git commit so a run can
        be reproduced from ``run_log.jsonl`` alone.  Always written first.
        """
        import platform

        entry: dict[str, Any] = {"event": "run_start"}
        if config:
            entry.update(config)
        entry.setdefault("python_version", platform.python_version())
        try:
            from evalvitals import __version__ as _ver  # type: ignore
            entry.setdefault("evalvitals_version", _ver)
        except Exception:  # noqa: BLE001
            pass
        commit = self._git_commit()
        if commit:
            entry.setdefault("git_commit", commit)
        self._log(entry, span_id="run_start")

    @staticmethod
    def _git_commit() -> "str | None":
        """Best-effort current git commit hash (short), or None outside a repo."""
        import subprocess
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=3, check=False,
            )
            return out.stdout.strip() or None
        except Exception:  # noqa: BLE001
            return None

    def _save_judge_io(
        self, stem: str, prompt: "str | None", raw: "str | None"
    ) -> "dict[str, Any] | None":
        """Persist a judge prompt + raw response under ``prompts/``; return summary.

        Returns ``{prompt_path, prompt_chars, raw_path, raw_chars}`` (paths
        relative to the run dir) or ``None`` when neither is present.
        """
        if not prompt and not raw:
            return None
        info: dict[str, Any] = {}
        if prompt:
            p = self._save_text(self.prompts_dir, f"{stem}.prompt.txt", str(prompt))
            if p is not None:
                info["prompt_path"] = p
                info["prompt_chars"] = len(str(prompt))
        if raw:
            p = self._save_text(self.prompts_dir, f"{stem}.response.txt", str(raw))
            if p is not None:
                info["raw_path"] = p
                info["raw_chars"] = len(str(raw))
        return info or None

    # ------------------------------------------------------------------
    # Event hooks — called by AutoDiagnoseLoop at each stage
    # ------------------------------------------------------------------

    def log_probe(
        self,
        cycle: int,
        results: dict[str, "Result"],
        schema: "Any | None" = None,
        *,
        judge_prompt: "str | None" = None,
        judge_raw: "str | None" = None,
        duration_sec: "float | None" = None,
    ) -> "list[Path]":
        """M1: log findings (JSON) and persist heavy artifacts to disk.

        Returns a list of PNG figure paths that were saved for this cycle
        (attention heatmaps, spatial maps, etc.) so callers can forward them
        to the judge as visual context.
        """
        artifact_paths = self._save_probe_artifacts(cycle, results)
        entry: dict[str, Any] = {
            "event": "probe",
            "cycle": cycle,
            "analyzers": list(results),
            "findings": {name: r.findings for name, r in results.items()},
            "artifact_paths": artifact_paths,
        }
        if schema is not None:
            entry["selection_rationale"] = getattr(schema, "rationale", "")
        judge_io = self._save_judge_io(f"c{cycle}_m1_selection", judge_prompt, judge_raw)
        if judge_io:
            entry["judge_io"] = judge_io
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._log(entry, span_id=f"c{cycle}.m1")

        # Collect PNG heatmap paths saved alongside the .npy arrays.
        png_figures: list[Path] = []
        for rel_npy in artifact_paths.values():
            if not rel_npy.endswith(".npy"):
                continue
            png = (self.run_dir / rel_npy).with_suffix(".png")
            if png.exists():
                png_figures.append(png)
        return png_figures

    def log_analysis(
        self,
        cycle: int,
        report: "AnalysisReport",
        *,
        duration_sec: "float | None" = None,
    ) -> None:
        """M2: log severity, flagged anomalies, and the narrative sent to M3."""
        entry: dict[str, Any] = {
            "event": "analysis",
            "cycle": cycle,
            "severity": report.severity,
            "n_findings": len(report.findings),
            "findings": [str(f) for f in report.findings],
            "narrative": report.narrative,
        }
        # StatsAnalysisReport extras (present when VLDiagnoseLoop is used)
        conclusion = getattr(report, "conclusion", None)
        if conclusion:
            entry["conclusion"] = conclusion
        evidence_chain = getattr(report, "evidence_chain", None)
        if evidence_chain:
            entry["evidence_chain"] = list(evidence_chain)
        stats_tool_results = getattr(report, "stats_tool_results", None)
        if stats_tool_results:
            entry["stats_tool_results"] = list(stats_tool_results)
        visualizations = getattr(report, "visualizations", None)
        if visualizations:
            entry["visualizations"] = list(visualizations)
        # Statistical-tool layer: which tools ran, their verdicts, FDR, figures.
        stats_plan = getattr(report, "stats_plan", None)
        if stats_plan:
            entry["stats_plan"] = stats_plan
        stats_results = getattr(report, "stats_results", None)
        if stats_results:
            entry["stats_results"] = [r.to_dict() for r in stats_results]
        corrected = getattr(report, "corrected_rejections", None)
        if corrected:
            entry["corrected_rejections"] = corrected
        figures = getattr(report, "figures", None)
        if figures:
            entry["figures"] = list(figures)
        # M2 LLM-guided judge I/O (present when StatsAnalysisAgent has a judge).
        judge_io = self._save_judge_io(
            f"c{cycle}_m2_analysis",
            getattr(report, "llm_prompt", None),
            getattr(report, "llm_raw", None),
        )
        if judge_io:
            entry["judge_io"] = judge_io
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._log(entry, span_id=f"c{cycle}.m2")

    def log_diagnosis(
        self,
        cycle: int,
        diag: "DiagnosisResult",
        *,
        duration_sec: "float | None" = None,
    ) -> None:
        """M3: log raw LLM output, the prompt, and every parsed hypothesis."""
        entry: dict[str, Any] = {
            "event": "diagnosis",
            "cycle": cycle,
            "model_name": diag.model_name,
            "n_hypotheses": len(diag.hypotheses),
            "hypotheses": [
                {
                    "statement": h.statement,
                    "failure_mode": h.predicted_failure_mode,
                    "status": h.status.value if h.status else None,
                }
                for h in diag.hypotheses
            ],
            "raw_judge_output": diag.raw_judge_output,
        }
        judge_io = self._save_judge_io(
            f"c{cycle}_m3_diagnosis",
            getattr(diag, "prompt", None),
            diag.raw_judge_output,
        )
        if judge_io:
            entry["judge_io"] = judge_io
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._log(entry, span_id=f"c{cycle}.m3")

    def log_surgery(
        self,
        cycle: int,
        hypothesis: "Hypothesis",
        iv: "InterventionResult",
        *,
        duration_sec: "float | None" = None,
    ) -> None:
        """M4/M5: log intervention outcome for one hypothesis.

        M5 results are distinguished by the presence of ``m5_test_name`` in
        ``iv.evidence``; they get span_id ``c{cycle}.m5`` instead of ``.m4``.
        """
        is_m5 = "m5_test_name" in (iv.evidence or {})
        span_suffix = "m5" if is_m5 else "m4"
        entry: dict[str, Any] = {
            "event": "surgery",
            "cycle": cycle,
            "module": span_suffix,
            "hypothesis": hypothesis.statement,
            "failure_mode": hypothesis.predicted_failure_mode,
            "status": iv.status.value,
            "fixed": iv.fixed,
            "confidence_score": iv.confidence_score,
            "evidence_dimensions": iv.evidence_dimensions,
            "evidence": iv.evidence,
            "n_refocused_cases": len(iv.new_data) if iv.new_data else None,
        }
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._log(entry, span_id=f"c{cycle}.{span_suffix}")

    def log_fix(self, outcome: "Any") -> None:
        """Post-loop fix module: log the tiered repair attempt + recommendation.

        *outcome* is a :class:`~evalvitals.eval_agent.stages.fix_agent.FixOutcome`;
        its ``to_dict()`` carries every attempted candidate (tier, payload,
        paired-stats verdict, repaired/broken case ids) and the escalation
        recommendation when nothing validated.

        In addition to the lean JSONL ``fix`` event, this writes a
        self-contained human record under ``fixes/``: one
        ``NN_<tier>_<name>/record.md`` per attempted candidate plus a top-level
        ``outcome.md`` summarising all attempts and the recommendation — so each
        repair experiment can be read on its own without parsing the log.
        """
        d = outcome.to_dict()
        entry: dict[str, Any] = {"event": "fix", "cycle": -1, "module": "fix"}
        entry.update(d)
        record = self._write_fix_records(d)
        if record is not None:
            entry["record"] = record
        self._log(entry, span_id="fix")

    def _write_fix_records(self, d: "dict[str, Any]") -> "str | None":
        """Write per-candidate records + ``fixes/outcome.md``; return the outcome path."""
        attempts = d.get("attempted") or []

        def _eff(v: "Any") -> "Any":
            return round(v, 4) if isinstance(v, float) else v

        # One record per attempted candidate.
        for i, a in enumerate(attempts, start=1):
            tier = a.get("tier", "L?")
            name = a.get("name", "candidate")
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", f"{i:02d}_{tier}_{name}").strip("_")
            verdict = a.get("verdict") or ("FIXED" if a.get("fixed") else "did not fix")
            cov = a.get("coverage")
            lines = [
                f"# Fix attempt {i:02d} — {name}  [{tier}]",
                "",
                f"**Outcome:** {'FIXED' if a.get('fixed') else 'did not fix'} "
                f"(verdict: {verdict})",
                f"**Kind:** {a.get('kind')}    **Source:** {a.get('source')}",
                "",
                "## Validation (paired McNemar vs. unmodified baseline)",
                f"- pairs tested (applicable): {a.get('n_pairs')}",
                f"- cases fixed: {a.get('n_fixed')}",
                f"- cases broken: {a.get('n_broken')}",
                f"- coverage of failures: {'—' if cov is None else f'{cov:.0%}'}",
                f"- unstable cases dropped (noise): {a.get('n_unstable', 0)}",
                f"- effect: {_eff(a.get('effect'))}",
                f"- e-value: {_eff(a.get('e_value'))}",
                f"- statistically significant (rejects H0): {a.get('reject')}",
            ]
            if a.get("summary"):
                lines.append(f"- summary: {a['summary']}")
            lines.append("")
            if a.get("fixed_cases"):
                lines.append("## Cases fixed")
                lines += [f"- {c}" for c in a["fixed_cases"]]
                lines.append("")
            if a.get("broken_cases"):
                lines.append("## Cases broken")
                lines += [f"- {c}" for c in a["broken_cases"]]
                lines.append("")
            lines.append("## What was applied")
            lines.append("```json")
            lines.append(json.dumps(a.get("payload") or {}, indent=2, default=str))
            lines.append("```")
            self._save_text(self.fixes_dir / slug, "record.md", "\n".join(lines))

        # Top-level summary across all attempts.
        fixed = d.get("fixed")
        head = [
            "# Fix outcome",
            "",
            f"**Result:** {'FIXED' if fixed else 'NOT FIXED'}",
            f"**Max tier allowed:** {d.get('max_tier')}",
            f"**Best candidate:** {d.get('best') or '—'}",
        ]
        rec = d.get("recommendation")
        if rec:
            tier = rec.get("recommend_tier")
            if tier:
                head.append(
                    f"**Recommendation:** escalate to {tier} — {rec.get('reason', '')}"
                )
            else:
                action = rec.get("action", "no fix")
                head.append(
                    f"**Recommendation:** {action} — {rec.get('reason', '')}"
                )
        refine = d.get("refine_signal")
        if refine:
            head.append(f"**Re-diagnose:** {refine.get('message', '')}")
        head += ["", f"## Attempts ({len(attempts)})", ""]
        if attempts:
            head.append("| # | tier | candidate | verdict | n_fixed | n_broken "
                        "| coverage | effect | sig |")
            head.append("|---|------|-----------|---------|---------|----------"
                        "|----------|--------|-----|")
            for i, a in enumerate(attempts, start=1):
                cov = a.get("coverage")
                cov_s = "—" if cov is None else f"{cov:.0%}"
                head.append(
                    f"| {i:02d} | {a.get('tier')} | {a.get('name')} | "
                    f"{a.get('verdict') or ('fixed' if a.get('fixed') else 'no')} | "
                    f"{a.get('n_fixed')} | {a.get('n_broken')} | {cov_s} | "
                    f"{_eff(a.get('effect'))} | "
                    f"{'yes' if a.get('reject') else 'no'} |"
                )
            head += ["", "Each attempt's full record is in its own folder above "
                     "(`NN_<tier>_<name>/record.md`)."]
        return self._save_text(self.fixes_dir, "outcome.md", "\n".join(head))

    def log_loop_end(
        self,
        report: "AutoDiagnoseReport",
        *,
        tokens_used: "int | None" = None,
        timings: "dict[str, float] | None" = None,
    ) -> None:
        """Final summary entry for the diagnosis loop — does **not** close the log.

        ``loop_end`` marks the end of the M1→M5 diagnosis loop, not the end of
        logging: the post-loop experiments (M4 mechanism verification via
        :meth:`AutoDiagnoseLoop.run_m4`, tiered repair via ``run_fix``) run
        *after* ``loop.run()`` returns and must still be recorded.  The logger's
        lifecycle therefore belongs to whoever created it — use it as a context
        manager or call :meth:`close` explicitly when all work is done.  (Each
        event is flushed to disk as it is written, so an unclosed logger never
        loses data.)

        Accepts both :class:`AutoDiagnoseReport` (``resolved``,
        ``final_hypotheses``) and :class:`VLDiagnoseReport` (``stopped_by``,
        ``verified_hypotheses``, ``all_hypotheses``) via duck typing.

        *tokens_used* and *timings* (per-stage wall-clock totals in seconds)
        record the run's cost/latency profile when the loop supplies them.
        """
        entry: dict[str, Any] = {
            "event": "loop_end",
            "cycles": report.cycles,
        }
        if tokens_used is not None:
            entry["tokens_used"] = tokens_used
        if timings:
            entry["timings_sec"] = {k: round(v, 3) for k, v in timings.items()}
            entry["total_duration_sec"] = round(sum(timings.values()), 3)
        # AutoDiagnoseReport shape
        if hasattr(report, "resolved"):
            entry["resolved"] = report.resolved
            hyps = getattr(report, "final_hypotheses", [])
            entry["n_hypotheses"] = len(hyps)
            entry["final_hypotheses"] = [
                {
                    "statement": h.statement,
                    "failure_mode": h.predicted_failure_mode,
                    "status": h.status.value if h.status else None,
                }
                for h in hyps
            ]
        # VLDiagnoseReport shape
        if hasattr(report, "stopped_by"):
            entry["stopped_by"] = report.stopped_by
            all_hyps = getattr(report, "all_hypotheses", [])
            verified = getattr(report, "verified_hypotheses", [])
            entry["n_hypotheses"] = len(all_hyps)
            entry["n_verified"] = len(verified)
            entry["verified_hypotheses"] = [
                {
                    "statement": tr.hypothesis.statement,
                    "failure_mode": tr.hypothesis.predicted_failure_mode,
                    "status": tr.status.value,
                    "confidence": tr.confidence,
                    "protocol_consistent": tr.is_consistent_with_protocol,
                }
                for tr in verified
            ]
        self._log(entry)

    # ------------------------------------------------------------------
    # Experiment log (M4) + workspace snapshot
    # ------------------------------------------------------------------

    def log_experiment(
        self,
        cycle: int,
        hypothesis: "Hypothesis",
        iv: "InterventionResult",
        *,
        module: str = "m4",
    ) -> None:
        """M4: log the *experiment* the agent wrote and ran to test *hypothesis*.

        Consumes the rich ``iv.experiment`` payload attached by
        :class:`~evalvitals.eval_agent.stages.surgery.SurgeryAgent` (the
        generated script(s), the run's stdout/stderr, the verdict, and the
        agent's intermediate thinking — the CLI agent's narration or the
        multi-phase LLM ``validation_log``).  Heavy text is written under
        ``experiments/`` and the sandbox working directory is snapshotted under
        ``workspace/``; the JSONL event records the verdict, metrics and the
        paths to those artifacts so ``run_log.jsonl`` stays lean.

        Falls back gracefully (logs only the scalar evidence) when
        ``iv.experiment`` is absent, so passive / label-correlation
        interventions still produce an experiment event.
        """
        exp = getattr(iv, "experiment", None) or {}
        prefix = f"c{cycle}" if cycle >= 0 else "post"
        stem = f"{prefix}_{module}"

        # 1. Generated source files (the "changes").
        file_paths: dict[str, str] = {}
        files = exp.get("files") or {}
        if not files and exp.get("code"):
            files = {"main.py": exp["code"]}
        for fname, src in files.items():
            p = self._save_text(self.experiments_dir, f"{stem}_{fname}", str(src))
            if p is not None:
                file_paths[fname] = p

        # 2. Output + the agent's intermediate thinking.
        text_paths: dict[str, str] = {}
        for key, suffix in (
            ("stdout", "stdout.txt"),
            ("stderr", "stderr.txt"),
            ("blueprint", "blueprint.yaml"),
            ("cli_raw_output", "agent_thinking.txt"),
        ):
            val = exp.get(key)
            if val:
                p = self._save_text(self.experiments_dir, f"{stem}_{suffix}", str(val))
                if p is not None:
                    text_paths[key] = p
        vlog = exp.get("validation_log")
        if vlog:
            p = self._save_text(
                self.experiments_dir, f"{stem}_phase_log.txt",
                "\n".join(str(x) for x in vlog),
            )
            if p is not None:
                text_paths["validation_log"] = p

        # 3. Snapshot the workspace the agent operated in (the "workspace snapshot").
        workspace = None
        workdir = exp.get("workdir")
        if workdir:
            workspace = self._snapshot_workspace(stem, workdir)

        entry: dict[str, Any] = {
            "event": "experiment",
            "cycle": cycle,
            "module": module,
            "hypothesis": hypothesis.statement,
            "failure_mode": hypothesis.predicted_failure_mode,
            "status": iv.status.value if iv.status else None,
            "fixed": iv.fixed,
            "provider": exp.get("provider"),
            "verdict": exp.get("verdict"),
            "metrics": exp.get("metrics"),
            "returncode": exp.get("returncode"),
            "timed_out": exp.get("timed_out"),
            "cli_usage": exp.get("cli_usage"),
            "llm_calls": exp.get("llm_calls"),
            "sandbox_runs": exp.get("sandbox_runs"),
            "code_paths": file_paths,
            "output_paths": text_paths,
            "workspace_snapshot": workspace,
        }
        # Human-readable, self-contained record: open this one file to
        # understand the whole experiment without parsing JSONL.
        record = self._write_experiment_record(stem, entry)
        if record is not None:
            entry["record"] = record
        self._log(entry, span_id=f"{prefix}.{module}")

    def _write_experiment_record(
        self, stem: str, entry: "dict[str, Any]"
    ) -> "str | None":
        """Write a one-page Markdown summary of an M4 experiment to ``experiments/``."""
        status = (entry.get("status") or "unknown").upper()
        lines = [
            f"# Experiment — {entry.get('module', 'm4').upper()}  ({status})",
            "",
            f"**Hypothesis:** {entry.get('hypothesis', '')}",
            f"**Failure mode:** {entry.get('failure_mode', '—')}",
            "",
            f"**Verdict:** {entry.get('verdict')}    "
            f"**Fixed:** {entry.get('fixed')}",
            "",
        ]
        metrics = entry.get("metrics") or {}
        if metrics:
            lines.append("## Metrics")
            for k, v in metrics.items():
                lines.append(f"- {k}: {v}")
            lines.append("")
        lines.append("## How it ran")
        for label, key in (
            ("provider", "provider"), ("return code", "returncode"),
            ("timed out", "timed_out"), ("LLM calls", "llm_calls"),
            ("sandbox runs", "sandbox_runs"),
        ):
            if entry.get(key) is not None:
                lines.append(f"- {label}: {entry[key]}")
        lines.append("")
        files = {**(entry.get("code_paths") or {}), **(entry.get("output_paths") or {})}
        if files:
            lines.append("## Files")
            for name, path in files.items():
                lines.append(f"- `{path}`  — {name}")
            lines.append("")
        return self._save_text(self.experiments_dir, f"{stem}_record.md", "\n".join(lines))

    # ------------------------------------------------------------------
    # Tool synthesis — agent generates new probes / stats tools on demand
    # ------------------------------------------------------------------

    def log_tool_codegen(
        self,
        *,
        module: str,
        name: str,
        need: str,
        source: str,
        ok: bool,
        code: str = "",
        prompt: str = "",
        raw_output: str = "",
        error: str = "",
        stdout: str = "",
        cycle: "int | None" = None,
        extra: "dict[str, Any] | None" = None,
    ) -> None:
        """Log one tool-synthesis *attempt* (success OR failure).

        Called from inside a generator (ProbeGenerator / WhiteboxProbeGenerator
        / StatsToolGenerator) the moment it writes code, so the prompt, the raw
        code produced, the backend used (``cli:<provider>`` vs ``llm``) and the
        validation outcome are captured even when the attempt fails to compile
        or run — exactly the cases that vanish today.  The code/prompt/agent
        output are written under ``tools/``; the JSONL event records the paths
        plus the pass/fail outcome.

        ``module`` is e.g. ``"m1_probe"``, ``"m1_whitebox"`` or ``"m2_stats"``.
        """
        cyc = self.current_cycle if cycle is None else cycle
        self._codegen_seq += 1
        prefix = f"c{cyc}" if cyc >= 0 else "post"
        stem = f"{prefix}_{module}_{name}_{self._codegen_seq:02d}"

        paths: dict[str, str] = {}
        for key, content, suffix in (
            ("code", code, "code.py"),
            ("prompt", prompt, "prompt.txt"),
            ("raw_output", raw_output, "agent_thinking.txt"),
            ("stdout", stdout, "stdout.txt"),
        ):
            if content:
                p = self._save_text(self.tools_dir, f"{stem}_{suffix}", str(content))
                if p is not None:
                    paths[key] = p

        entry: dict[str, Any] = {
            "event": "tool_codegen",
            "cycle": cyc,
            "module": module,
            "tool_name": name,
            "need": need,
            "source": source,
            "ok": ok,
            "error": error or None,
            "artifact_paths": paths,
        }
        if extra:
            entry.update(extra)
        self._log(entry, span_id=f"{prefix}.{module}.codegen")

    def log_tool_registry(
        self,
        cycle: int,
        module: str,
        generated: "list[Any]",
    ) -> None:
        """Snapshot which synthesised tools are registered/active for *cycle*.

        *generated* is a list of objects carrying ``name``/``code``/``need``/
        ``source`` attributes (``GeneratedProbe`` / ``GeneratedStatsTool``).
        Records the active tool registry for the cycle and persists each tool's
        source under ``tools/`` (idempotent by name).
        """
        if not generated:
            return
        prefix = f"c{cycle}" if cycle >= 0 else "post"
        tools: list[dict[str, Any]] = []
        for g in generated:
            name = getattr(g, "name", "tool")
            code = getattr(g, "code", "")
            code_path = None
            if code:
                code_path = self._save_text(
                    self.tools_dir, f"{module}_{name}.code.py", str(code)
                )
            tools.append({
                "name": name,
                "need": getattr(g, "need", ""),
                "source": getattr(g, "source", ""),
                "code_path": code_path,
            })
        self._log(
            {
                "event": "tool_registry",
                "cycle": cycle,
                "module": module,
                "n_tools": len(tools),
                "tools": tools,
            },
            span_id=f"{prefix}.{module}.registry",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all log handlers."""
        for handler in (self._file_handler, self._console_handler):
            if handler is not None:
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"RunLogger(run_dir={str(self.run_dir)!r})"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log(self, entry: dict[str, Any], *, span_id: str | None = None) -> None:
        entry["ts"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        entry["trace_id"] = self.trace_id
        if span_id is not None:
            entry["span_id"] = span_id
        self.logger.info("run_event", extra={"_payload": entry})

    def _save_text(self, directory: Path, stem: str, text: str) -> "str | None":
        """Write *text* to ``directory/stem`` (creating *directory*); return rel path.

        ``stem`` already carries the extension (e.g. ``c0_m4_main.py``).  Returns
        the path relative to :attr:`run_dir` for embedding in the JSONL event, or
        ``None`` if writing fails.
        """
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / stem
            path.write_text(text, encoding="utf-8")
            return str(path.relative_to(self.run_dir))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"RunLogger: could not save text artifact {stem!r}: {exc}")
            return None

    # Files worth keeping in a workspace snapshot — code, data, prose, logs.
    # Heavy binaries (weights, tensors, images) are skipped to keep snapshots
    # small; the .npy/.png analyzer artifacts are already saved under artifacts/.
    _SNAPSHOT_SUFFIXES = frozenset(
        {".py", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".csv", ".log", ".toml"}
    )
    _SNAPSHOT_MAX_BYTES = 2_000_000  # skip any single file larger than 2 MB

    def _snapshot_workspace(self, stem: str, workdir: "str | Path") -> "dict[str, Any] | None":
        """Copy text/code/data files from *workdir* into ``workspace/<stem>/``.

        Returns a manifest ``{"dir": <rel path>, "files": [...], "skipped": n}``
        or ``None`` when *workdir* does not exist.  The sandbox deletes its
        working directory on success, so this is best-effort: callers should
        snapshot promptly after the run.
        """
        import shutil

        src = Path(workdir)
        if not src.exists() or not src.is_dir():
            return None
        dest = self.workspace_dir / stem
        kept: list[str] = []
        skipped = 0
        try:
            dest.mkdir(parents=True, exist_ok=True)
            for f in sorted(src.rglob("*")):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in self._SNAPSHOT_SUFFIXES:
                    skipped += 1
                    continue
                try:
                    if f.stat().st_size > self._SNAPSHOT_MAX_BYTES:
                        skipped += 1
                        continue
                    rel = f.relative_to(src)
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, target)
                    kept.append(str(rel))
                except Exception:  # noqa: BLE001
                    skipped += 1
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"RunLogger: workspace snapshot failed for {stem!r}: {exc}")
            return None
        return {
            "dir": str(dest.relative_to(self.run_dir)),
            "files": kept,
            "skipped": skipped,
        }

    def _save_probe_artifacts(
        self,
        cycle: int,
        results: dict[str, "Result"],
    ) -> dict[str, str]:
        """Persist heavy artifacts from all M1 results; return {key: path} map."""
        paths: dict[str, str] = {}
        for analyzer_name, result in results.items():
            for art_name, artifact in result.artifacts.items():
                stem = f"c{cycle}_{analyzer_name}_{art_name}"
                path = self._save_artifact(stem, artifact)
                if path is not None:
                    paths[f"{analyzer_name}/{art_name}"] = str(path.relative_to(self.run_dir))
        return paths

    def _save_artifact(self, stem: str, artifact: Any) -> Path | None:
        """Write one artifact to ``artifacts/<stem>.<ext>``; return path or None.

        For numeric artifacts (tensors, arrays, list-of-tensors):
          - Saves raw data as ``<stem>.npy``.
          - Also saves a ``<stem>.png`` figure when the shape and stem keyword
            are recognised (attention → heatmap, entropy/rollout → line/heatmap).
            Silently skipped when matplotlib is not installed.

        For dict/list artifacts: saves ``<stem>.json``.
        """
        try:
            import numpy as np

            arr = _artifact_to_numpy(artifact)
            if arr is not None:
                path = self.artifact_dir / f"{stem}.npy"
                np.save(path, arr)
                _save_artifact_figure(self.artifact_dir, stem, arr)
                return path
            if isinstance(artifact, (dict, list)):
                path = self.artifact_dir / f"{stem}.json"
                path.write_text(json.dumps(artifact, default=str), encoding="utf-8")
                return path
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"RunLogger: could not save artifact {stem!r}: {exc}")
        return None