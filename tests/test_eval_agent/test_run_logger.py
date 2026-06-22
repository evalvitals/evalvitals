"""RunLogger core event contract: every run_log.jsonl line carries schema_version.

Downstream parsers of run_log.jsonl need a way to detect breaking changes to
event shapes without guessing from evalvitals_version (which tracks the
package, not the log format). See RUN_LOG_SCHEMA_VERSION in run_logger.py.
"""

from __future__ import annotations

import json
import threading


def test_log_run_start_carries_schema_version(tmp_path):
    from evalvitals.eval_agent.run_logger import RUN_LOG_SCHEMA_VERSION, RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    logger.log_run_start({"model": "fake-model"})
    logger.close()

    lines = (tmp_path / "run1" / "run_log.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["schema_version"] == RUN_LOG_SCHEMA_VERSION
    assert isinstance(entry["schema_version"], int)


def test_every_log_method_stamps_schema_version(tmp_path):
    """Spot-check a few distinct log_* methods, not just log_run_start."""
    from evalvitals.eval_agent.run_logger import RUN_LOG_SCHEMA_VERSION, RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    logger.log_run_start()
    logger.log_tool_codegen(
        module="m1_probe", name="fake_tool", need="testing", source="llm",
        ok=True, code="print(1)",
    )
    logger.close()

    lines = (tmp_path / "run1" / "run_log.jsonl").read_text().splitlines()
    entries = [json.loads(line) for line in lines]
    assert len(entries) == 2
    assert all(e["schema_version"] == RUN_LOG_SCHEMA_VERSION for e in entries)


def _stats_report(stats_plan):
    from evalvitals.eval_agent.stages.stats_agent import StatsAnalysisReport

    return StatsAnalysisReport(
        model_name="vlm",
        findings=[],
        severity="none",
        narrative="No anomalies detected.",
        raw_results={},
        stats_plan=stats_plan,
    )


def test_small_stats_payload_stays_inline(tmp_path):
    """A typical small analysis cycle keeps stats_plan inline, not externalized."""
    from evalvitals.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    small_plan = [{"tool": "signal_label_assoc", "rationale": "correlates with FAIL"}]
    logger.log_analysis(0, _stats_report(small_plan))
    logger.close()

    lines = (tmp_path / "run1" / "run_log.jsonl").read_text().splitlines()
    entry = json.loads(lines[0])
    assert entry["stats_plan"] == small_plan
    assert not (tmp_path / "run1" / "artifacts").exists() or not list(
        (tmp_path / "run1" / "artifacts").glob("*m2_stats_plan*")
    )


def test_large_stats_payload_externalized(tmp_path):
    """A stats_plan over the inline threshold is written to artifacts/ instead."""
    from evalvitals.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    big_plan = [
        {"tool": f"tool_{i}", "rationale": "x" * 200, "config": {"k": "v" * 50}}
        for i in range(30)
    ]
    logger.log_analysis(2, _stats_report(big_plan))
    logger.close()

    lines = (tmp_path / "run1" / "run_log.jsonl").read_text().splitlines()
    entry = json.loads(lines[0])
    summary = entry["stats_plan"]
    assert isinstance(summary, dict)
    assert summary["n_items"] == len(big_plan)
    assert summary["bytes"] > 4096
    artifact_path = tmp_path / "run1" / summary["path"]
    assert artifact_path.exists()
    assert json.loads(artifact_path.read_text()) == big_plan


def test_codegen_seq_increments_are_thread_safe(tmp_path):
    """Concurrent log_tool_codegen() calls must never collide on filename."""
    from evalvitals.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    n_threads = 20
    barrier = threading.Barrier(n_threads)

    def _call(i: int) -> None:
        barrier.wait()
        logger.log_tool_codegen(
            module="m1_probe", name=f"tool_{i}", need="testing", source="llm",
            ok=True, code="print(1)",
        )

    threads = [threading.Thread(target=_call, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logger.close()

    code_files = list((tmp_path / "run1" / "tools").glob("*_code.py"))
    assert len(code_files) == n_threads, "filename collision dropped a codegen artifact"
