# ChestAgentBench M2 Chat

This example starts a Lambda/Codex-style M2 chat over an existing results
directory. You do not write analysis code; the local coding agent writes and
runs exploratory analysis scripts inside EvalVitals' sandbox.

Default data path:

```text
/tealab-data/rjin02/MedRAX/logs/202607/chestagentbench
```

Run from the repository root:

```bash
pip install -e .
bash examples/m2_statistics/chestagentbench_chat/run.sh
```

Or run directly:

```bash
evalvitals chat /tealab-data/rjin02/MedRAX/logs/202607/chestagentbench \
  --backend antigravity \
  --out /tealab-data/rjin02/MedRAX/logs/202607/chestagentbench_m2_chat \
  --max-rows 2000 \
  --max-files 20
```

Inside the chat, ask questions like:

```text
Which features distinguish incorrect cases from correct cases?
Compare failure patterns across model directories.
Does tool usage correlate with failures?
What candidate signals should I confirm with StatsAnalysisAgent?
```

Each turn writes:

```text
<out>/turn_001/exploratory_report.json
<out>/turn_001/analysis.py
<out>/turn_001/stdout.txt
<out>/turn_001/agent_raw_output.txt
```

By default, `tool_calls_*.json` files are skipped because they dominate the log
volume. Add `--include-tool-calls` when you specifically want tool-call-level
analysis.
