"""Loop controllers for the self-evolving failure-analysis agent.

Three controllers are provided:

``SelfEvolveLoop``
    Original Stage-1 skeleton — kept for backward compatibility.

``AutoDiagnoseLoop``
    Concrete M1→M2→M3→M4 implementation (original architecture):

    ┌───────────────────────────────────────────────────────────────────┐
    │ M1 · ProbeAgent      select analyzers + execute (direct / Docker) │
    │ M2 · AnalysisModule  interpret results → structured report        │
    │ M3 · DiagnosisAgent  Gemini reads report → hypotheses             │
    │ M4 · SurgeryAgent    operate + verify; stop or refocus data       │
    └───────────────────────────────────────────────────────────────────┘
                            ↑__________________________│  (repeat)

``VLDiagnoseLoop``
    New architecture (2026-06-05 meeting) focused on VL tasks:
    M4 is intentionally removed from the inner loop and kept as a
    separate post-loop fix-proposal step (Plan A).

    ┌──────────────────────────────────────────────────────────────────────┐
    │ M1 · ProbeAgent         protocol-guided analyzer selection + execute │
    │ M2 · StatsAnalysisAgent protocol-aware stats analysis                │
    │ M3 · DiagnosisAgent     "AI scientist" hypothesis generation         │
    │ M5 · HypothesisTester   stats test + protocol consistency check      │
    └──────────────────────────────────────────────────────────────────────┘
                            ↑_________________________________│
             stop when M5 finds a verified, protocol-consistent hypothesis

    M4 (SurgeryAgent) runs separately via ``VLDiagnoseLoop.run_m4()`` once
    the loop stops — propose a fix for the best verified hypothesis (Plan A),
    or propose + execute a fix (Plan B).

Run-directory infrastructure (mirrors AutoResearchClaw):

When ``run_dir`` is supplied:
  - ``run_dir/artifacts/{run_id}/``  — per-run artifact staging
  - ``run_dir/checkpoint.json``      — atomic resume state (temp+rename)
  - ``run_dir/heartbeat.json``       — per-cycle liveness signal
  - ``run_dir/evolution/``           — JSONL lesson store (auto-created)

Git integration:
  - When the repo is detected, each resolved run is committed on
    ``eval/{run_id}``; unresolved runs are discarded with git reset.

Usage::

    loop   = AutoDiagnoseLoop(model=my_model, run_dir=Path("./runs"))
    report = loop.run(failure_cases)
    print(report.resolved, report.final_hypotheses)

    # Resume an interrupted run
    report = AutoDiagnoseLoop.resume(Path("./runs"), model=my_model, data=cases)

    # New VL-focused loop (Plan A)
    from evalvitals.eval_agent import VLDiagnoseLoop
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    protocol = ExperimentProtocol(
        description="QwenVL often confuses left/right positions in spatial tasks.",
        task_domain="spatial reasoning",
        target_modalities=frozenset({"text", "image"}),
    )
    loop   = VLDiagnoseLoop(model=vlm, protocol=protocol)
    report = loop.run(failure_cases)
    fix    = loop.run_m4(report, failure_cases)   # separate fix-proposal step
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evalvitals.eval_agent.hypothesis import HypothesisStatus
from evalvitals.eval_agent.store import InMemoryStore, Store

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.evolution import EvolutionStore
    from evalvitals.eval_agent.git_manager import ExperimentGitManager
    from evalvitals.eval_agent.hypothesis import Hypothesis, HypothesisGenerator
    from evalvitals.eval_agent.run_logger import RunLogger
    from evalvitals.eval_agent.stages.analysis import AnalysisModule, AnalysisReport
    from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
    from evalvitals.eval_agent.stages.hypothesis_tester import (
        HypothesisTester,
        HypothesisTestResult,
    )
    from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol
    from evalvitals.eval_agent.stages.stats_agent import StatsAnalysisAgent, StatsAnalysisReport
    from evalvitals.eval_agent.stages.surgery import SurgeryAgent

logger = logging.getLogger(__name__)


class SelfEvolveLoop:
    """Original Stage-1 skeleton — kept for backward compatibility.

    Args:
        generator: Proposes and mutates hypotheses.
        store:     Persistent memory; defaults to an in-memory store.
    """

    def __init__(
        self,
        generator: "HypothesisGenerator | None" = None,
        store: Store | None = None,
        runner: Any = None,  # kept for signature compat
    ) -> None:
        from evalvitals.core.experiment import ExperimentRunner

        self.generator = generator
        self.store = store or InMemoryStore()
        self.runner = runner if runner is not None else ExperimentRunner()

    def step(self) -> list:
        if self.generator is None:
            raise ValueError("SelfEvolveLoop needs a generator (e.g. ManualHypothesisGenerator).")
        proposed = self.generator.propose(self.store.summarize())
        for h in proposed:
            self.store.add_hypothesis(h)
        return list(proposed)

    def run(self, max_cycles: int = 10) -> list:
        history: list = []
        for _ in range(max_cycles):
            proposed = self.step()
            history.append(proposed)
            if not proposed:
                break
        return history


# ──────────────────────────────────────────────────────────────────────────────
# AutoDiagnoseLoop — M1 → M2 → M3 → M4
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AutoDiagnoseReport:
    """Summary returned by :class:`AutoDiagnoseLoop.run`.

    Attributes:
        cycles:           Number of M1→M4 cycles executed.
        resolved:         ``True`` when M4 surgery confirmed the problem is fixed.
        final_hypotheses: All hypotheses proposed across every cycle.
        final_results:    Raw analyzer results from the last M1 probe.
        final_analysis:   Structured report from the last M2 pass.
        store:            Accumulated results and hypotheses.
    """

    cycles: int
    resolved: bool
    final_hypotheses: list["Hypothesis"] = field(default_factory=list)
    final_results: dict[str, "Result"] = field(default_factory=dict)
    final_analysis: "AnalysisReport | None" = None
    store: Store = field(default_factory=InMemoryStore)
    # Internal — set by AutoDiagnoseLoop for evolution/git integration
    _run_id: str = field(default="", repr=False)


class AutoDiagnoseLoop:
    """Automated M1→M2→M3→M4 diagnosis loop.

    Args:
        model:            The model under evaluation.
        probe_agent:      M1 — selects and executes analyzers (direct or Docker).
                          Defaults to ``ProbeAgent()``.
        analysis_module:  M2 — interprets raw Results into an AnalysisReport.
                          Defaults to ``AnalysisModule()``.
        diagnosis_agent:  M3 — reads the AnalysisReport and proposes hypotheses via
                          Gemini (default when ``GEMINI_API_KEY`` is set).
                          Pass ``None`` to run in *analysis-only* mode (M1+M2 only).
        surgery_agent:    M4 — operates on each hypothesis to verify or refute it.
                          Defaults to ``SurgeryAgent()``.
        store:            Persistent memory.  Defaults to ``InMemoryStore()``.
        max_cycles:       Hard cap on M1→M4 iterations.
        run_logger:       Optional :class:`~evalvitals.eval_agent.run_logger.RunLogger`
                          that writes a JSONL event log and saves analyzer artifacts.
        run_dir:          Optional root directory for run infrastructure.
                          When set, enables checkpoints, heartbeat, and the
                          ``EvolutionStore``.
        git_manager:      Optional :class:`~evalvitals.eval_agent.git_manager.ExperimentGitManager`.
                          ``None`` → auto-detect git repo when *run_dir* is given.
        evolution_store:  Optional :class:`~evalvitals.eval_agent.evolution.EvolutionStore`.
                          ``None`` → auto-create under ``run_dir/evolution/`` when
                          *run_dir* is given.
    """

    def __init__(
        self,
        model: "Model",
        probe_agent: "ProbeAgent | None" = None,
        analysis_module: "AnalysisModule | None" = None,
        diagnosis_agent: "DiagnosisAgent | None" = None,
        surgery_agent: "SurgeryAgent | None" = None,
        store: Store | None = None,
        max_cycles: int = 5,
        run_logger: "RunLogger | None" = None,
        # --- run-directory infrastructure (new) ---
        run_dir: "Path | None" = None,
        git_manager: "ExperimentGitManager | None" = None,
        evolution_store: "EvolutionStore | None" = None,
        # --- token / cost budget ---
        token_budget: int = 0,
        _run_id_override: str | None = None,
    ) -> None:
        from evalvitals.eval_agent.stages.analysis import AnalysisModule
        from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
        from evalvitals.eval_agent.stages.surgery import SurgeryAgent

        self.model = model
        self.probe_agent = probe_agent or ProbeAgent()
        self.analysis_module = analysis_module or AnalysisModule()
        self.diagnosis_agent = diagnosis_agent
        self.surgery_agent = surgery_agent or SurgeryAgent()
        self.store = store or InMemoryStore()
        self.max_cycles = max_cycles
        self.run_logger = run_logger
        self.token_budget = token_budget
        self._tokens_used: int = 0

        # --- run-directory setup ---
        self._run_dir: Path | None = None
        self._artifacts_dir: Path | None = None
        self._checkpoint_path: Path | None = None
        self._heartbeat_path: Path | None = None
        self._run_id: str = ""

        if run_dir is not None:
            run_dir = Path(run_dir)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._run_id = _run_id_override or ts
            self._run_dir = run_dir
            self._artifacts_dir = run_dir / "artifacts" / ts
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            self._checkpoint_path = run_dir / "checkpoint.json"
            self._heartbeat_path = run_dir / "heartbeat.json"

        # EvolutionStore: auto-create when run_dir is set
        self.evolution_store: "EvolutionStore | None" = evolution_store
        if self.evolution_store is None and run_dir is not None:
            try:
                from evalvitals.eval_agent.evolution import EvolutionStore
                self.evolution_store = EvolutionStore(run_dir / "evolution")
            except Exception as exc:
                logger.debug("Could not create EvolutionStore: %s", exc)

        # ExperimentGitManager: auto-detect when run_dir is set
        self.git_manager: "ExperimentGitManager | None" = git_manager
        if self.git_manager is None and run_dir is not None:
            try:
                from evalvitals.eval_agent.git_manager import ExperimentGitManager
                _gm = ExperimentGitManager(run_dir)
                if _gm.is_git_repo():
                    self.git_manager = _gm
            except Exception as exc:
                logger.debug("Git manager auto-detect failed: %s", exc)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def run(self, data: "CaseBatch") -> AutoDiagnoseReport:
        """Drive the M1→M2→M3→M4 loop until resolved or *max_cycles* reached.

        Args:
            data: Cases to analyse.  Refocused to unexplained cases when M4
                  returns a SUPPORTED finding with ``new_data``.

        Returns:
            :class:`AutoDiagnoseReport` with the final state.
        """
        all_hypotheses: list[Any] = []
        final_results: dict[str, Any] = {}
        final_analysis = None

        # Determine start cycle from checkpoint (resume support).
        # Only honor the checkpoint when run_id matches — this means the
        # caller explicitly requested a resume of a specific prior run
        # (via AutoDiagnoseLoop.resume() or _run_id_override).  A fresh
        # AutoDiagnoseLoop always gets a new run_id and starts from 0.
        start_cycle = 0
        if self._checkpoint_path is not None:
            cp = self._read_checkpoint()
            if cp is not None and cp.get("run_id") == self._run_id:
                start_cycle = cp.get("last_completed_cycle", -1) + 1
                if start_cycle > 0:
                    logger.info(
                        "Resuming from checkpoint: skipping cycles 0–%d",
                        start_cycle - 1,
                    )

        prior_cycles: list[dict[str, Any]] = []
        outstanding_modes: list[str] = []
        completed_cycles = 0
        # Per-cycle max confidence scores: detect flat progress early
        _confidence_history: list[float] = []
        self._tokens_used = 0

        for cycle in range(start_cycle, self.max_cycles):
            # Budget guard: stop before consuming more tokens than allowed
            if self.token_budget > 0 and self._tokens_used >= self.token_budget:
                logger.warning(
                    "Token budget %d exhausted (%d used) — stopping loop after %d cycles",
                    self.token_budget, self._tokens_used, completed_cycles,
                )
                break
            completed_cycles = cycle + 1

            # ── M1: probe ───────────────────────────────────────────────
            probe_results = self.probe_agent.probe(
                self.model, data,
                hint_failure_modes=outstanding_modes or None,
            )
            if self.run_logger:
                self.run_logger.log_probe(
                    cycle, probe_results, schema=self.probe_agent.last_schema
                )
            if not probe_results:
                break
            final_results = probe_results
            for r in probe_results.values():
                self.store.add_result(r)

            # ── M2: analyze ─────────────────────────────────────────────
            analysis = self.analysis_module.analyze(probe_results, repr(self.model))
            final_analysis = analysis
            if self.run_logger:
                self.run_logger.log_analysis(cycle, analysis)

            if self.diagnosis_agent is None:
                self._on_cycle_complete(cycle, all_hypotheses)
                break  # analysis-only mode

            # ── M3: diagnose ─────────────────────────────────────────────
            diag = self.diagnosis_agent.diagnose(
                analysis, prior_cycles=prior_cycles or None
            )
            # Track token usage from the diagnosis LLM call.
            # Use response metadata when available; otherwise estimate from
            # output length (1 token ≈ 4 chars) so the budget is always counted.
            _diag_tokens = getattr(diag, "tokens_used", None)
            if _diag_tokens is None:
                _diag_tokens = max(1, len(diag.raw_judge_output) // 4)
            self._tokens_used += _diag_tokens
            if self.run_logger:
                self.run_logger.log_diagnosis(cycle, diag)
            if not diag.hypotheses:
                self._on_cycle_complete(cycle, all_hypotheses)
                break
            for h in diag.hypotheses:
                self.store.add_hypothesis(h)
            all_hypotheses.extend(diag.hypotheses)

            # ── M4: surgery ──────────────────────────────────────────────
            outstanding_modes = []
            cycle_max_confidence = 0.0
            for h in diag.hypotheses:
                iv = self.surgery_agent.operate(h, self.model, probe_results, data)
                h.status = iv.status
                cycle_max_confidence = max(cycle_max_confidence, iv.confidence_score)
                if self.run_logger:
                    self.run_logger.log_surgery(cycle, h, iv)
                if iv.fixed:
                    report = AutoDiagnoseReport(
                        cycles=completed_cycles,
                        resolved=True,
                        final_hypotheses=all_hypotheses,
                        final_results=final_results,
                        final_analysis=final_analysis,
                        store=self.store,
                        _run_id=self._run_id,
                    )
                    self._on_cycle_complete(cycle, all_hypotheses)
                    self._on_loop_end(report)
                    return report
                if iv.status == HypothesisStatus.INCONCLUSIVE:
                    outstanding_modes.append(h.predicted_failure_mode)
                if iv.new_data is not None and len(iv.new_data) > 0:
                    data = iv.new_data

            # Track confidence per cycle and stop early when progress stalls:
            # if the last 2 cycles both show confidence < 0.1 the hypotheses
            # are not gaining traction — continuing burns budget without insight.
            _confidence_history.append(cycle_max_confidence)
            if len(_confidence_history) >= 2 and all(
                c < 0.1 for c in _confidence_history[-2:]
            ):
                logger.info(
                    "Stopping early: confidence flat at %.3f for 2 consecutive cycles",
                    cycle_max_confidence,
                )
                break

            # Record cycle for M3 context
            prior_cycles.append({
                "cycle": cycle,
                "severity": analysis.severity,
                "hypotheses": [
                    {
                        "statement": h.statement,
                        "failure_mode": h.predicted_failure_mode,
                        "status": h.status.value if h.status else "pending",
                    }
                    for h in diag.hypotheses
                ],
            })

            # Write checkpoint + heartbeat after each full cycle
            self._on_cycle_complete(cycle, all_hypotheses)

        report = AutoDiagnoseReport(
            cycles=completed_cycles,
            resolved=False,
            final_hypotheses=all_hypotheses,
            final_results=final_results,
            final_analysis=final_analysis,
            store=self.store,
            _run_id=self._run_id,
        )
        self._on_loop_end(report)
        return report

    @classmethod
    def resume(
        cls,
        run_dir: Path,
        model: "Model",
        data: "CaseBatch",
        **kwargs: Any,
    ) -> "AutoDiagnoseReport":
        """Resume a loop from a checkpoint.

        Reads ``run_dir/checkpoint.json``, skips already-completed cycles,
        and continues from ``last_completed_cycle + 1``.

        Args:
            run_dir: Directory that contains ``checkpoint.json``.
            model:   The model under evaluation.
            data:    Original :class:`CaseBatch` to continue from.
            **kwargs: Forwarded to :class:`AutoDiagnoseLoop.__init__`.
                      ``run_dir`` is set automatically — do not pass it.
        """
        # Read the stored run_id so the checkpoint match succeeds and
        # the loop correctly skips already-completed cycles.
        run_id_override: str | None = None
        cp_path = Path(run_dir) / "checkpoint.json"
        if cp_path.exists():
            try:
                cp = json.loads(cp_path.read_text(encoding="utf-8"))
                run_id_override = cp.get("run_id")
            except (json.JSONDecodeError, OSError):
                pass
        instance = cls(
            model=model, run_dir=run_dir, _run_id_override=run_id_override, **kwargs
        )
        return instance.run(data)

    # ──────────────────────────────────────────────────────────────────
    # Per-cycle and end-of-loop hooks
    # ──────────────────────────────────────────────────────────────────

    def _on_cycle_complete(
        self, cycle: int, hypotheses: list["Hypothesis"]
    ) -> None:
        """Write checkpoint and heartbeat after each completed cycle."""
        statuses = [
            h.status.value if h.status else "pending"
            for h in hypotheses
        ]
        if self._checkpoint_path is not None:
            self._write_checkpoint(cycle, statuses)
        if self._heartbeat_path is not None:
            self._write_heartbeat(cycle)

    def _on_loop_end(self, report: AutoDiagnoseReport) -> None:
        """Append lessons to EvolutionStore and commit/discard via git."""
        if self.run_logger:
            self.run_logger.log_loop_end(report)

        # EvolutionStore lesson extraction
        if self.evolution_store is not None:
            try:
                from evalvitals.eval_agent.evolution import extract_lessons
                lessons = extract_lessons(report)
                self.evolution_store.append_many(lessons)
                logger.debug(
                    "EvolutionStore: appended %d lesson(s) for run %s",
                    len(lessons), self._run_id,
                )
            except Exception as exc:
                logger.warning("Failed to record evolution lessons: %s", exc)

        # Git integration
        if self.git_manager is not None and self._run_id:
            hyp_statuses = {
                h.statement[:80]: (h.status.value if h.status else "pending")
                for h in report.final_hypotheses
            }
            try:
                if report.resolved:
                    self.git_manager.commit_experiment(
                        self._run_id,
                        report.cycles,
                        hyp_statuses,
                        "resolved",
                    )
                    logger.info(
                        "Git: committed resolved run %s on eval/%s",
                        self._run_id, self._run_id,
                    )
                else:
                    self.git_manager.discard_experiment(
                        self._run_id, "not resolved"
                    )
            except Exception as exc:
                logger.warning("Git integration failed: %s", exc)

    # ──────────────────────────────────────────────────────────────────
    # Checkpoint / heartbeat
    # ──────────────────────────────────────────────────────────────────

    def _write_checkpoint(self, cycle: int, hypothesis_statuses: list[str]) -> None:
        """Atomic checkpoint write (temp-file + rename)."""
        assert self._checkpoint_path is not None
        data = {
            "last_completed_cycle": cycle,
            "run_id": self._run_id,
            "hypothesis_statuses": hypothesis_statuses,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        target = self._checkpoint_path
        fd, tmp_path = tempfile.mkstemp(
            dir=target.parent, suffix=".tmp", prefix="checkpoint_"
        )
        os.close(fd)
        try:
            Path(tmp_path).write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
            Path(tmp_path).replace(target)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def _write_heartbeat(self, cycle: int) -> None:
        assert self._heartbeat_path is not None
        data = {
            "pid": os.getpid(),
            "last_cycle": cycle,
            "run_id": self._run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._heartbeat_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def _read_checkpoint(self) -> dict[str, Any] | None:
        if self._checkpoint_path is None or not self._checkpoint_path.exists():
            return None
        try:
            return json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None


# ──────────────────────────────────────────────────────────────────────────────
# VLDiagnoseLoop — M1 → M2 → M3 → M5  (Plan A, 2026-06-05 architecture)
# ──────────────────────────────────────────────────────────────────────────────

_STOPPED_BY_CRITERIA  = "criteria_met"
_STOPPED_BY_MAX       = "max_cycles"
_STOPPED_BY_BUDGET    = "budget"
_STOPPED_BY_NO_HYPS   = "no_hypotheses"
_STOPPED_BY_NO_PROBE  = "no_probe_results"


@dataclass
class VLDiagnoseReport:
    """Summary returned by :class:`VLDiagnoseLoop.run`.

    Attributes:
        cycles:               Number of M1→M5 cycles executed.
        stopped_by:           Why the loop stopped: ``"criteria_met"``,
                              ``"max_cycles"``, ``"budget"``,
                              ``"no_hypotheses"``, or ``"no_probe_results"``.
        verified_hypotheses:  Statistically supported, protocol-consistent
                              test results from M5 — sorted highest confidence
                              first.  Feed into :meth:`VLDiagnoseLoop.run_m4`.
        all_hypotheses:       All M3 proposals across every cycle.
        all_test_results:     All M5 test results across every cycle.
        final_stats_report:   M2 report from the last cycle.
        fix_proposal:         Populated by :meth:`VLDiagnoseLoop.run_m4`
                              when called after :meth:`VLDiagnoseLoop.run`.
        store:                Accumulated results and hypotheses.
    """

    cycles: int
    stopped_by: str
    verified_hypotheses: "list[HypothesisTestResult]" = field(default_factory=list)
    all_hypotheses: "list[Any]" = field(default_factory=list)
    all_test_results: "list[HypothesisTestResult]" = field(default_factory=list)
    final_stats_report: "StatsAnalysisReport | None" = None
    fix_proposal: "Any | None" = None
    store: Store = field(default_factory=InMemoryStore)
    _run_id: str = field(default="", repr=False)


class VLDiagnoseLoop:
    """M1→M2→M3→M5 failure-analysis loop for VL tasks (Plan A architecture).

    M4 (**SurgeryAgent**) is intentionally excluded from the inner loop.
    Call :meth:`run_m4` on the returned :class:`VLDiagnoseReport` to obtain
    a fix proposal based on the best verified hypothesis candidates.

    Inner loop::

        for cycle in range(max_cycles):
            probe_results  = M1.probe(model, data, protocol)   # guided by protocol
            stats_report   = M2.analyze(probe_results, protocol)
            diag           = M3.diagnose(stats_report)
            test_results   = M5.test(diag.hypotheses, stats_report, data, protocol)
            if M5.stopping_criteria_met(test_results, protocol): break

    Stopping criteria: at least one M5-verified hypothesis that is also
    consistent with the user's experiment protocol.

    Args:
        model:              The model under evaluation.
        protocol:           Experiment protocol — the human prior that guides
                            M1 analyzer selection, M2 narrative, and M5
                            consistency checks.
        probe_agent:        M1.  Defaults to ``ProbeAgent()``.
        stats_agent:        M2.  Defaults to ``StatsAnalysisAgent()``.
        diagnosis_agent:    M3.  ``None`` lazily resolves ``DiagnosisAgent()``
                            on first use.
        hypothesis_tester:  M5.  Defaults to ``HypothesisTester()``.
        surgery_agent:      M4 — used only by :meth:`run_m4`, never inside
                            the main loop.  Defaults to ``SurgeryAgent()``.
        store:              Persistent memory.
        max_cycles:         Hard cap on M1→M5 iterations.
        run_logger:         Optional :class:`~evalvitals.eval_agent.run_logger.RunLogger`.
        token_budget:       Stop early when accumulated token usage reaches
                            this limit (0 = unlimited).
        analysis_only:      Run only M1→M2 and stop before hypothesis generation.
    """

    def __init__(
        self,
        model: "Model",
        protocol: "ExperimentProtocol",
        probe_agent: "Any | None" = None,
        stats_agent: "StatsAnalysisAgent | None" = None,
        diagnosis_agent: "Any | None" = None,
        hypothesis_tester: "HypothesisTester | None" = None,
        surgery_agent: "Any | None" = None,
        store: Store | None = None,
        max_cycles: int = 5,
        run_logger: "Any | None" = None,
        token_budget: int = 0,
        analysis_only: bool = False,
    ) -> None:
        from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTester
        from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
        from evalvitals.eval_agent.stages.stats_agent import StatsAnalysisAgent
        from evalvitals.eval_agent.stages.surgery import SurgeryAgent

        self.model = model
        self.protocol = protocol
        self.probe_agent = probe_agent or ProbeAgent()
        self.stats_agent = stats_agent or StatsAnalysisAgent()
        self.diagnosis_agent = diagnosis_agent  # None = lazy default on first call
        self.hypothesis_tester = hypothesis_tester or HypothesisTester()
        self.surgery_agent = surgery_agent or SurgeryAgent()
        self.store = store or InMemoryStore()
        self.max_cycles = max_cycles
        self.run_logger = run_logger
        self.token_budget = token_budget
        self.analysis_only = analysis_only
        self._tokens_used: int = 0
        self._run_id: str = ""

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def run(self, data: "CaseBatch") -> VLDiagnoseReport:
        """Drive the M1→M2→M3→M5 loop to convergence.

        Args:
            data: Cases to analyse (should carry :class:`~evalvitals.core.case.Label`
                  values for the M5 statistical tests to work).

        Returns:
            :class:`VLDiagnoseReport` with the final state.
            Call :meth:`run_m4` on this to get a fix proposal (Plan A).
        """
        all_hypotheses: list[Any] = []
        all_test_results: list[Any] = []
        final_stats_report = None
        stopped_by = _STOPPED_BY_MAX
        prior_cycles: list[dict[str, Any]] = []
        self._tokens_used = 0

        # Resolve M3 lazily so the default Gemini fallback is consistent with
        # AutoDiagnoseLoop — a DiagnosisAgent() created here would raise
        # immediately if GEMINI_API_KEY is absent, even if the caller
        # passes diagnosis_agent=None intentionally.
        def _get_diagnosis_agent() -> "Any":
            if self.diagnosis_agent is not None:
                return self.diagnosis_agent
            from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
            return DiagnosisAgent()

        for cycle in range(self.max_cycles):
            if self.token_budget > 0 and self._tokens_used >= self.token_budget:
                logger.warning(
                    "Token budget %d exhausted after %d cycles", self.token_budget, cycle
                )
                stopped_by = _STOPPED_BY_BUDGET
                break

            # ── M1: protocol-guided probing ──────────────────────────
            # Extract failure modes from prior M3 hypotheses for the static
            # fallback path (used when ProbeAgent has no judge model).
            prior_modes = list(dict.fromkeys(
                h.predicted_failure_mode for h in all_hypotheses
                if getattr(h, "predicted_failure_mode", None)
            ))
            probe_results = self.probe_agent.probe(
                self.model,
                data,
                protocol=self.protocol,
                prior_hypotheses=all_hypotheses or None,
                hint_failure_modes=prior_modes or None,
            )
            artifact_pngs: list = []
            if self.run_logger:
                artifact_pngs = self.run_logger.log_probe(
                    cycle, probe_results, schema=self.probe_agent.last_schema
                ) or []
            if not probe_results:
                logger.info("M1 produced no probe results — stopping.")
                stopped_by = _STOPPED_BY_NO_PROBE
                break
            for r in probe_results.values():
                self.store.add_result(r)

            # ── M2: protocol-aware stats analysis ────────────────────
            stats_report = self.stats_agent.analyze(
                probe_results,
                model_name=repr(self.model),
                protocol=self.protocol,
                data=data,
                extra_figures=artifact_pngs,
            )
            final_stats_report = stats_report
            if self.run_logger:
                self.run_logger.log_analysis(cycle, stats_report)

            if self.analysis_only:
                stopped_by = _STOPPED_BY_NO_HYPS
                break

            # ── M3: hypothesis generation ("AI scientist") ────────────
            try:
                diag_agent = _get_diagnosis_agent()
            except Exception as exc:
                logger.warning("Could not resolve DiagnosisAgent: %s", exc)
                stopped_by = _STOPPED_BY_NO_HYPS
                break

            try:
                diag = diag_agent.diagnose(stats_report, prior_cycles=prior_cycles or None)
            except Exception as exc:  # judge timeout/quota must not kill the loop
                logger.warning(
                    "M3 diagnosis failed at cycle %d (%s) — stopping with the "
                    "evidence collected so far.", cycle, exc,
                )
                stopped_by = _STOPPED_BY_NO_HYPS
                break

            # Track token usage
            _tok = getattr(diag, "tokens_used", None)
            if _tok is None:
                _tok = max(1, len(diag.raw_judge_output) // 4)
            self._tokens_used += _tok

            if self.run_logger:
                self.run_logger.log_diagnosis(cycle, diag)

            if not diag.hypotheses:
                logger.info("M3 produced no hypotheses at cycle %d.", cycle)
                stopped_by = _STOPPED_BY_NO_HYPS
                break

            for h in diag.hypotheses:
                self.store.add_hypothesis(h)
            all_hypotheses.extend(diag.hypotheses)

            # ── M5: hypothesis testing (stats + protocol consistency) ─
            test_results = self.hypothesis_tester.test(
                diag.hypotheses,
                stats_report,
                data,
                protocol=self.protocol,
            )
            all_test_results.extend(test_results)
            for tr in test_results:
                tr.hypothesis.status = tr.status

            if self.run_logger:
                # Reuse the surgery log slot for M5 results (backward compat)
                for tr in test_results:
                    _iv = _make_intervention_result_from_test(tr)
                    self.run_logger.log_surgery(cycle, tr.hypothesis, _iv)

            # ── Stopping criteria ────────────────────────────────────
            if self.hypothesis_tester.stopping_criteria_met(test_results, self.protocol):
                logger.info(
                    "Stopping criteria met at cycle %d: verified, protocol-consistent "
                    "hypothesis found.",
                    cycle,
                )
                stopped_by = _STOPPED_BY_CRITERIA
                break

            # Build prior-cycles context for next M3 call
            prior_cycles.append({
                "cycle": cycle,
                "severity": stats_report.severity,
                "hypotheses": [
                    {
                        "statement": h.statement,
                        "failure_mode": h.predicted_failure_mode,
                        "status": h.status.value if h.status else "pending",
                    }
                    for h in diag.hypotheses
                ],
            })

        # Collect best verified hypotheses (sorted by confidence)
        verified = self.hypothesis_tester.best_hypotheses(all_test_results)

        report = VLDiagnoseReport(
            cycles=cycle + 1 if self.max_cycles > 0 else 0,  # type: ignore[possibly-undefined]
            stopped_by=stopped_by,
            verified_hypotheses=verified,
            all_hypotheses=all_hypotheses,
            all_test_results=all_test_results,
            final_stats_report=final_stats_report,
            store=self.store,
            _run_id=self._run_id,
        )
        if self.run_logger:
            self.run_logger.log_loop_end(report)
        return report

    def run_m4(
        self,
        report: VLDiagnoseReport,
        data: "CaseBatch",
    ) -> "Any | None":
        """Plan A: propose a fix for the best verified hypothesis (post-loop M4).

        Called *after* :meth:`run` to avoid polluting the inner loop with
        fix-execution noise.  Operates on the highest-confidence verified
        hypothesis from :attr:`VLDiagnoseReport.verified_hypotheses`.

        Args:
            report: Returned by :meth:`run`.
            data:   Original case batch (needed by the surgery agent).

        Returns:
            :class:`~evalvitals.eval_agent.surgery.InterventionResult` or
            ``None`` if there are no verified hypotheses to fix.
        """
        if not report.verified_hypotheses:
            logger.info("run_m4: no verified hypotheses to act on.")
            return None

        best_tr = report.verified_hypotheses[0]
        results: dict[str, Any] = (
            report.final_stats_report.raw_results
            if report.final_stats_report is not None
            else {}
        )
        iv = self.surgery_agent.operate(
            best_tr.hypothesis,
            self.model,
            results,
            data,
        )
        report.fix_proposal = iv
        return iv


# ---------------------------------------------------------------------------
# Internal adapter: convert HypothesisTestResult → InterventionResult-like
# object for RunLogger.log_surgery (which expects an InterventionResult).
# ---------------------------------------------------------------------------

def _make_intervention_result_from_test(tr: "HypothesisTestResult") -> Any:
    """Wrap a HypothesisTestResult as a minimal InterventionResult."""
    from evalvitals.eval_agent.stages.surgery import InterventionResult

    return InterventionResult(
        hypothesis=tr.hypothesis,
        status=tr.status,
        fixed=False,
        evidence={
            "m5_test_name": tr.test_name,
            "m5_effect_size": tr.effect_size,
            "m5_confidence": tr.confidence,
            "m5_protocol_consistent": tr.is_consistent_with_protocol,
            "m5_verdict": tr.verdict,
            "m5_evidence_grade": tr.evidence_grade,
            **tr.evidence,
        },
        confidence_score=tr.confidence,
    )
