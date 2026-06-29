# ChestAgentBench M2 Explore

This example runs a Lambda/Codex-style single-shot M2 exploration over an
existing results directory. You do not write analysis code; the local coding
agent writes and runs one exploratory analysis script inside EvalVitals' sandbox.

For the general standalone M2 workflow, see
[`docs/m2_analysis.md`](../../../docs/m2_analysis.md).

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
evalvitals explore /tealab-data/rjin02/MedRAX/logs/202607/chestagentbench \
  --backend antigravity \
  -q "Which features distinguish incorrect cases from correct cases?" \
  --out /tealab-data/rjin02/MedRAX/logs/202607/chestagentbench_m2_chat \
  --max-rows 2000 \
  --max-files 20
```

Pass the analysis question with `-q`; example questions:

```text
Which features distinguish incorrect cases from correct cases?
Compare failure patterns across model directories.
Does tool usage correlate with failures?
What candidate signals should I confirm with StatsAnalysisAgent?
```

The run writes:

```text
<out>/exploratory_report.json
<out>/analysis.py
<out>/stdout.txt
<out>/agent_raw_output.txt
<out>/figures/   <out>/tables/
```

Open the optional Streamlit dashboard:

```bash
pip install -e ".[dashboard]"
evalvitals dashboard /tealab-data/rjin02/MedRAX/logs/202607/chestagentbench_m2_chat
```

By default, `tool_calls_*.json` files are skipped because they dominate the log
volume. Add `--include-tool-calls` when you specifically want tool-call-level
analysis.
