# M2 → M3: Exploratory Analysis & Hypothesis Generation

`evalvitals explore` runs two stages over a results directory, no code required:

- **M2 — `ExploratoryAnalysisAgent`**: a local coding agent profiles your data
  and writes/runs one analysis script, producing takeaways, charts, and
  candidate signals. **Purely descriptive** — no support/unsupported verdict,
  no hypothesis testing.
- **M3 — `HypothesisAgent`**: reads the M2 report and proposes 1-3 falsifiable
  hypotheses that could explain the patterns found. **Proposal only** —
  generating a hypothesis is not the same as testing one; nothing here is
  confirmed or refuted.

Confirmatory testing (turning a candidate signal into a validated claim) is a
separate, loop-internal system (`StatsAnalysisAgent`) — see
[Architecture](architecture.md#eval_agent-automated-diagnosis-pipeline).
This page only covers the standalone M2/M3 workflow.

## Install

```bash
pip install -e .                  # core
pip install -e ".[dashboard]"     # + Streamlit dashboard
```

## Quickstart

```bash
evalvitals explore /path/to/results \
  -q "Which features distinguish incorrect cases from correct cases?" \
  --out evalvitals_explore_output \
  --dashboard
```

`/path/to/results` is a single `.json`/`.jsonl` file or a directory tree;
EvalVitals recursively samples records across files. M3 hypotheses are
generated automatically after a successful M2 pass — pass `--no-hypotheses`
to skip that.

Other example questions:

```text
Compare accuracy across model directories.
Does tool usage correlate with failures?
What predicts yield / latency / cost? (any continuous outcome)
```

### Any outcome shape

The host profiles your data first and classifies the outcome column (if any)
as `binary`, `categorical`, `continuous`, or `none`, and adapts the analysis
and chart battery accordingly — you are not limited to pass/fail logs.
Outcome detection is name-heuristic (`label`, `outcome`, `target`,
`is_correct`, ...); pass `--outcome-col <name>` to point it at anything else
(e.g. `yield_pct`).

## Output layout

```text
evalvitals_explore_output/
  exploratory_report.json   # takeaways, observations, candidate signals,
                             # charts, tables, and M3 hypotheses
  analysis.py                # the generated code that was actually run
  records.json                # sampled input records
  figures/  tables/           # rendered charts + tabular artifacts
```

Each M3 hypothesis in `exploratory_report.json["hypotheses"]` has:

```json
{"statement": "...", "basis": "which M2 takeaway(s) this is grounded in",
 "test_design": "what evidence would confirm or refute it"}
```

## Dashboard

```bash
evalvitals dashboard evalvitals_explore_output --port 8501
```

Reads the saved artifacts (no re-run) across three tabs: **Problem Setting**,
**Exploratory Analysis** (M2 charts/takeaways), and **Hypotheses** (M3,
proposal-only — no verdict language).

## Python API

```python
from evalvitals.analysis import ExploratoryAnalysisAgent, HypothesisAgent
from evalvitals.eval_agent.cli_agent import CliAgentConfig

cli_config = CliAgentConfig(provider="claude_code")  # or antigravity/codex/...

m2 = ExploratoryAnalysisAgent(cli_config=cli_config)
report = m2.explore_path("/path/to/results", question="What predicts failure?")

m3 = HypothesisAgent(cli_config=cli_config)
hypotheses = m3.propose(report.to_dict())
for h in hypotheses:
    print(h.statement, "—", h.test_design)
```

## Backend notes

`--backend` selects the local coding-agent CLI: `antigravity` (default),
`claude_code`, `codex`, `opencode`, `gemini_cli`, `kimi_cli`. On `claude_code`
the bundled `nature-figure` Agent Skill styles agent-drawn figures
automatically (`--no-skills` to disable). `tool_calls_*.json` files are
skipped by default (`--include-tool-calls` to include them).

Run `evalvitals explore --help` for the full flag list.
