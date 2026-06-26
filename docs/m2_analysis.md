# M2 Statistical Analysis

M2 is EvalVitals' statistical analysis layer. It can run inside the full
M1-M5 diagnosis loop, or as a standalone analyst over an existing results
directory.

There are two standalone modes:

- `evalvitals chat`: no-code exploratory analysis over JSON/JSONL logs.
- `StatsAnalysisAgent`: confirmatory statistical tests over standardized records.

Use the chat mode first when you do not yet know which signals matter. Promote
the useful candidate signals into `StatsAnalysisAgent` when you need effect
sizes, confidence intervals, e-values, and FDR-aware decisions.

## Install

From the repository root:

```bash
pip install -e .
```

For the Streamlit dashboard:

```bash
pip install -e ".[dashboard]"
```

EvalVitals does not require changing your PyTorch install. For example, the
dashboard/chat dependencies can run in an existing `torch==2.6.*` environment.

## No-Code Chat

Point the chat CLI at a file or directory containing JSON/JSONL results:

```bash
evalvitals chat /path/to/results \
  --backend antigravity \
  --out m2_chat_output \
  --max-rows 2000 \
  --max-files 200
```

The input path may be a single `.json` or `.jsonl` file, or a directory tree.
EvalVitals recursively samples records across files and writes a normalized
`records.json` artifact for each turn.

Inside the chat, ask direct analysis questions:

```text
Compare accuracy across model directories.
Which features distinguish incorrect cases from correct cases?
Does tool usage correlate with failures?
Find candidate signals I should confirm with StatsAnalysisAgent.
```

Useful REPL commands:

```text
:help      show commands
:status    show loaded data and output directory
:history   show prior turns
:quit      exit
```

Each user turn asks the local coding agent to write an analysis script, executes
that script in EvalVitals' sandbox, and stores the artifacts under a numbered
turn directory.

## Chat Outputs

A typical session looks like this:

```text
m2_chat_output/
  chat_history.json
  turn_001/
    records.json
    analysis.py
    stdout.txt
    stderr.txt
    agent_raw_output.txt
    exploratory_report.json
    figures/
    tables/
```

Important files:

- `exploratory_report.json`: structured answer for the turn, including summary,
  metrics, candidate signals, charts, tables, limitations, and next questions.
- `analysis.py`: generated analysis code that was actually executed.
- `records.json`: sampled records given to the generated script.
- `figures/` and `tables/`: generated visual or tabular artifacts, if any.

The exploratory report is intentionally discovery-oriented. It is allowed to
surface hypotheses, patterns, suspicious correlations, and suggested follow-up
tests. It is not the final confirmatory statistics layer.

## Dashboard

Open a Streamlit dashboard over a chat output directory:

```bash
evalvitals dashboard m2_chat_output --port 8501
```

The dashboard reads the saved artifacts; it does not re-run the agent. Use it to
review multi-turn analyses, generated figures, tables, and the exact generated
code for each turn.

## One-Shot Explore

For a non-interactive run:

```bash
evalvitals-m2-explore /path/to/results \
  --question "Compare failure patterns across model directories." \
  --coder-provider antigravity \
  --out m2_explore_output
```

This uses the same exploratory backend as the chat CLI, but runs only one
question and exits.

## Confirmatory Statistics

Use `StatsAnalysisAgent` when your data already has a case id, pass/fail label,
and explicit signal columns:

```python
from evalvitals.analysis import StatsAnalysisAgent

rows = [
    {"case_id": "c0", "label": "fail", "low_img_attn": 1},
    {"case_id": "c1", "label": "pass", "low_img_attn": 0},
    {"case_id": "c2", "label": "fail", "low_img_attn": 1},
]

report = StatsAnalysisAgent().analyze_records(
    rows,
    id_col="case_id",
    label_col="label",
    signal_cols=["low_img_attn"],
)

print(report.conclusion)
for result in report.stats_results:
    print(result.summary)
```

`StatsAnalysisAgent` runs controlled statistical tools such as signal/label
association, McNemar/e-value tests, bootstrap differences, rank correlation, and
single-rate e-values. It applies e-BH FDR correction and returns a
`StatsAnalysisReport`.

## Standardization Boundary

Standalone chat mode does not require a standardized `StatsInput`. It accepts
messy logs and uses a local coding agent to inspect their shape.

The full diagnosis loop and confirmatory `StatsAnalysisAgent` do benefit from
standardized records because M2, M5, logging, and downstream tools need stable
contracts:

```text
M1 findings / user records
  -> standardized records
  -> controlled stats tools
  -> effect size + CI + e-value
  -> e-BH FDR correction
  -> StatsAnalysisReport
```

Use this rule of thumb:

- For exploratory questions, use `evalvitals chat` directly on result logs.
- For claims you want the loop to consume, standardize and run
  `StatsAnalysisAgent`.
- For M1-M5 automation, keep the standardized contract so M2/M5 can share
  evidence safely.

## ChestAgentBench Example

The repository includes a ready-to-run example:

```bash
RESULTS_DIR=/tealab-data/rjin02/MedRAX/logs/202607/chestagentbench \
OUT_DIR=/tealab-data/rjin02/MedRAX/logs/202607/chestagentbench_m2_chat \
bash examples/m2_statistics/chestagentbench_chat/run.sh
```

Then open:

```bash
evalvitals dashboard /tealab-data/rjin02/MedRAX/logs/202607/chestagentbench_m2_chat
```

See `examples/m2_statistics/chestagentbench_chat/README.md` for the example's
full command line.

## Backend Notes

`--backend antigravity` uses the local coding-agent path currently used by this
repository. The generated code runs in EvalVitals' experiment sandbox and writes
machine-readable JSON back to the M2 wrapper.

By default, `tool_calls_*.json` files are skipped because they can dominate log
volume. Add `--include-tool-calls` only when you specifically want
tool-call-level analysis.
