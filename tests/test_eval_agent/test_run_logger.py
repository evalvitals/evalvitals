"""RunLogger core event contract: every run_log.jsonl line carries schema_version.

Downstream parsers of run_log.jsonl need a way to detect breaking changes to
event shapes without guessing from evalvitals_version (which tracks the
package, not the log format). See RUN_LOG_SCHEMA_VERSION in run_logger.py.
"""

from __future__ import annotations

import json


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
