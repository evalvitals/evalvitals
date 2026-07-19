"""Background workers for non-analysis turns in the local web workbench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalvitals.agent_runtime.cli_types import CliAgentConfig
from evalvitals.agent_runtime.codegen import CodegenRunner
from evalvitals.analysis.workbench import EventSink, ThreadStore


def answer_from_report(
    *,
    report_path: Path,
    question: str,
    provider: str,
    model: str,
    events_path: Path,
    thread_dir: Path,
    turn_dir: Path,
    thread_id: str,
    turn_id: str,
    timeout_sec: int,
) -> int:
    sink = EventSink(events_path, thread_id=thread_id, turn_id=turn_id)
    sink.emit("route", "started", "Answering from existing analysis artifacts")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        sink.emit("route", "failed", f"Could not read the selected report: {exc}")
        return 1
    prompt = """You are the follow-up assistant for an exploratory data-analysis report.
Answer the user's question using only the report below. Be concise, cite the
specific takeaway/hypothesis/metric used, and clearly say when the report does
not contain enough evidence. Do not write code or claim a new statistical
result. Return only the answer for the user.

User question:
{question}

Report:
{report}
""".format(question=question, report=json.dumps(report, default=str)[:100_000])
    try:
        result = CodegenRunner(CliAgentConfig(provider=provider, model=model, timeout_sec=timeout_sec)).run(
            prompt, workdir=turn_dir, timeout_sec=timeout_sec
        )
        if result.audit:
            (turn_dir / "agent_audit.json").write_text(
                json.dumps({"schema_version": 1, "attempts": [result.audit]}, indent=2),
                encoding="utf-8",
            )
        answer = (result.raw_output or result.error or "The provider returned no answer.").strip()
    except Exception as exc:  # noqa: BLE001 - a background job must persist a useful error
        sink.emit("route", "failed", f"Follow-up provider failed: {exc}")
        return 1
    answer_path = turn_dir / "answer.md"
    answer_path.write_text(answer, encoding="utf-8")
    ThreadStore.append_message(thread_dir, "assistant", answer, turn_id=turn_id)
    sink.emit("route", "completed", "Follow-up answer is ready", artifact_refs=[answer_path])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("answer", nargs="?")
    parser.add_argument("--report", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--events", required=True)
    parser.add_argument("--thread-dir", required=True)
    parser.add_argument("--turn-dir", required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--turn-id", required=True)
    parser.add_argument("--timeout-sec", type=int, default=1200)
    args = parser.parse_args(argv)
    return answer_from_report(
        report_path=Path(args.report), question=args.question, provider=args.provider,
        model=args.model, events_path=Path(args.events), thread_dir=Path(args.thread_dir),
        turn_dir=Path(args.turn_dir), thread_id=args.thread_id, turn_id=args.turn_id,
        timeout_sec=args.timeout_sec,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
