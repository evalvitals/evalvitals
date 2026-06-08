"""RunLogger — structured per-cycle logging for AutoDiagnoseLoop.

Writes a JSONL event log (one line per M1/M2/M3/M4 event) and saves heavy
analyzer artifacts (attention tensors, hidden-state arrays) to a separate
``artifacts/`` directory, keyed by cycle number so they stay navigable.

Each JSON record always contains a ``ts`` (ISO-8601) and an ``event`` field.
The underlying file handler is a standard :class:`logging.FileHandler`, so
callers can attach additional handlers (e.g. a ``StreamHandler`` for console
output) by accessing ``RunLogger.logger``.

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
    print(loop.run_logger.run_dir)   # → runs/20260603_142305/

Add a console handler to mirror events to stderr::

    import logging, sys
    rl = RunLogger("runs/exp_01")
    rl.logger.addHandler(logging.StreamHandler(sys.stderr))
"""

from __future__ import annotations

import json
import logging
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


class RunLogger:
    """Structured JSONL logger + artifact sink for one :class:`AutoDiagnoseLoop` run.

    Each call to a ``log_*`` method appends one JSON object to ``run_log.jsonl``
    (always including a ``ts`` ISO-8601 timestamp and a ``cycle`` index).
    Heavy artifacts from M1 results are written to ``artifacts/`` as ``.npy``
    (numpy / torch tensors) or ``.json`` (dicts / lists).

    The underlying :attr:`logger` is a standard :class:`logging.Logger` named
    ``evalvitals.run.<run_dir_name>``.  It does **not** propagate to the root
    logger so it stays silent unless you add handlers.

    Args:
        run_dir:  Directory to write into.  Created if it does not exist.
                  Defaults to ``runs/<YYYYMMDD_HHMMSS>/`` relative to cwd.
        verbose:  When ``True``, print a human-readable summary of each event
                  to stdout as it occurs.  Off by default so library use stays
                  silent; set ``True`` when running interactively or in Docker.

    The logger is safe to use as a context manager::

        with RunLogger("runs/my_exp", verbose=True) as logger:
            loop = AutoDiagnoseLoop(model=model, run_logger=logger)
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
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(exist_ok=True)
        self.log_path = self.run_dir / "run_log.jsonl"

        self._verbose = verbose

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

    # ------------------------------------------------------------------
    # Event hooks — called by AutoDiagnoseLoop at each stage
    # ------------------------------------------------------------------

    def log_probe(
        self,
        cycle: int,
        results: dict[str, "Result"],
        schema: "Any | None" = None,
    ) -> None:
        """M1: log findings (JSON) and persist heavy artifacts to disk."""
        if self._verbose:
            print(f"\n[M1] cycle={cycle}  analyzers={list(results.keys())}", flush=True)
            if schema is not None and getattr(schema, "rationale", ""):
                print(f"     rationale  : {schema.rationale}", flush=True)
            for name, r in results.items():
                scalars = {
                    k: round(v, 4)
                    for k, v in (getattr(r, "findings", {}) or {}).items()
                    if isinstance(v, (int, float))
                }
                print(f"     {name}: {dict(list(scalars.items())[:6])}", flush=True)
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
        self._log(entry, span_id=f"c{cycle}.m1")

    def log_analysis(self, cycle: int, report: "AnalysisReport") -> None:
        """M2: log severity, flagged anomalies, and the narrative sent to M3."""
        if self._verbose:
            print(f"\n[M2] cycle={cycle}  severity={report.severity}", flush=True)
            conclusion = getattr(report, "conclusion", None)
            if conclusion:
                print(
                    f"     conclusion : "
                    f"{textwrap.fill(conclusion, 72, subsequent_indent='     ')}",
                    flush=True,
                )
            for step in (getattr(report, "evidence_chain", []) or [])[:3]:
                print(f"     evidence   : {step}", flush=True)
            plan = getattr(report, "stats_plan", []) or []
            if plan:
                print(f"     stats_tools: {[p['tool'] for p in plan]}", flush=True)
            corrected = getattr(report, "corrected_rejections", {}) or {}
            if corrected.get("rejected_tools"):
                print(f"     fdr_survive: {corrected['rejected_tools']}", flush=True)
            for tool in (getattr(report, "stats_tool_results", []) or [])[:2]:
                print(
                    f"     stats_tool : {tool.get('name')} - {tool.get('conclusion', '')}",
                    flush=True,
                )
            for fig in getattr(report, "figures", []) or []:
                print(f"     figure     : {fig}", flush=True)
            if not conclusion:
                print(
                    f"     {textwrap.fill(report.narrative, 72, subsequent_indent='     ')}",
                    flush=True,
                )
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
        self._log(entry, span_id=f"c{cycle}.m2")

    def log_diagnosis(self, cycle: int, diag: "DiagnosisResult") -> None:
        """M3: log raw LLM output and every parsed hypothesis."""
        if self._verbose:
            print(f"\n[M3] cycle={cycle}  {len(diag.hypotheses)} hypothesis/es", flush=True)
            for h in diag.hypotheses:
                print(f"     hypothesis  : {h.statement}", flush=True)
                print(f"     failure_mode: {h.predicted_failure_mode}", flush=True)
        self._log(
            {
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
            },
            span_id=f"c{cycle}.m3",
        )

    def log_surgery(
        self,
        cycle: int,
        hypothesis: "Hypothesis",
        iv: "InterventionResult",
    ) -> None:
        """M4/M5: log intervention outcome for one hypothesis.

        M5 results are distinguished by the presence of ``m5_test_name`` in
        ``iv.evidence``; they get span_id ``c{cycle}.m5`` instead of ``.m4``.
        """
        is_m5 = "m5_test_name" in (iv.evidence or {})
        if self._verbose:
            tag = "M5" if is_m5 else "M4"
            status = getattr(getattr(iv, "status", None), "value", "?")
            print(f"\n[{tag}] cycle={cycle}  '{hypothesis.statement[:70]}'", flush=True)
            if is_m5:
                ev = iv.evidence or {}
                print(
                    f"     status={status}"
                    f"  effect={ev.get('m5_effect_size', '?')}"
                    f"  confidence={ev.get('m5_confidence', '?')}",
                    flush=True,
                )
                print(
                    f"     protocol_consistent={ev.get('m5_protocol_consistent', '?')}",
                    flush=True,
                )
                print(f"     verdict : {ev.get('m5_verdict', '')}", flush=True)
            else:
                print(f"     status={status}  fixed={iv.fixed}", flush=True)
                ev = iv.evidence or {}
                if ev:
                    print(f"     evidence: {dict(list(ev.items())[:4])}", flush=True)
        span_suffix = "m5" if is_m5 else "m4"
        self._log(
            {
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
            },
            span_id=f"c{cycle}.{span_suffix}",
        )

    def log_loop_end(self, report: "AutoDiagnoseReport") -> None:
        """Final summary entry — written and file closed when the loop exits.

        Accepts both :class:`AutoDiagnoseReport` (``resolved``,
        ``final_hypotheses``) and :class:`VLDiagnoseReport` (``stopped_by``,
        ``verified_hypotheses``, ``all_hypotheses``) via duck typing.
        """
        if self._verbose:
            stopped_by = getattr(report, "stopped_by", None)
            if stopped_by is not None:
                print(f"\n[DONE] cycles={report.cycles}  stopped_by={stopped_by}", flush=True)
            else:
                print(
                    f"\n[DONE] cycles={report.cycles}  resolved={getattr(report, 'resolved', None)}",
                    flush=True,
                )
        entry: dict[str, Any] = {
            "event": "loop_end",
            "cycles": report.cycles,
        }
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
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the log file handler."""
        self._file_handler.flush()
        self._file_handler.close()
        self.logger.removeHandler(self._file_handler)

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
