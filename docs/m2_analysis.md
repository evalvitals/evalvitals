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

### Arbitrary folders, any outcome shape

`explore` does not pre-parse your data. The raw file or directory is handed to
the coding agent as-is (`--path` may be one file or many, in any JSON shape —
a flat list, JSONL, or per-file metadata wrapping a nested list of records
under a key like `cases`/`results`/`rows`); the agent's own generated code
loads and organizes it into a tidy table, then classifies the outcome column
(if any) as `binary`, `categorical`, `continuous`, or `none` and adapts the
analysis and chart battery accordingly — you are not limited to pass/fail
logs, and no host-side loader needs to know your data's shape in advance.
Pass `--outcome-col <name>` to point the agent at an explicit target column
(e.g. `yield_pct`) instead of relying on its own name-based detection.

## Output layout

```text
evalvitals_explore_output/
  exploratory_report.json   # takeaways, observations, candidate signals,
                             # charts, tables, and M3 hypotheses
  analysis.py                # the generated code that was actually run,
                              # including its own data-loading step
  records.json                # the tidy table the agent built and analyzed
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

The one-call entry point runs M2 + host adjudication + M3 in a single step,
over a path or in-memory records:

```python
import evalvitals

result = evalvitals.explore(
    "/path/to/results",                 # or a list[dict] of in-memory records
    question="What predicts failure?",
    provider="claude_code",             # or antigravity/codex/...
    out="evalvitals_explore_output",    # omit to skip persisting artifacts
)
print(result.ok, result.hypotheses)
```

`evalvitals.explore` is a lazy re-export of `evalvitals.analysis.explore`;
`out=None` (the default) keeps everything in memory — pass a directory to also
persist `exploratory_report.json`, rendered figures, and tables (the same
artifacts the `evalvitals explore` CLI writes).

For direct control over each stage:

```python
from evalvitals.analysis import ExploratoryAnalysisAgent, HypothesisAgent
from evalvitals.agent_runtime import CliAgentConfig

cli_config = CliAgentConfig(provider="claude_code")  # or antigravity/codex/...

m2 = ExploratoryAnalysisAgent(cli_config=cli_config)
report = m2.explore_path("/path/to/results", question="What predicts failure?")

m3 = HypothesisAgent(cli_config=cli_config)
hypotheses = m3.propose(report.to_dict())
for h in hypotheses:
    print(h.statement, "—", h.test_design)
```

## Failure-Mode Clustering

`evalvitals.analysis.cluster_failures` groups FAIL cases into interpretable
clusters — pattern discovery over the raw failing cases themselves, rather
than the per-signal EDA above. No required dependency: a pure-numpy fallback
(hashing vectorizer + cosine-greedy grouping) always works; install the
`[cluster]` extra (`scikit-learn`, `hdbscan`) for TF-IDF + density-based
clustering.

```python
from evalvitals.analysis import cluster_failures

report = cluster_failures(records, min_cluster_size=3, max_clusters=8)
for cluster in report.clusters:
    print(cluster.name, cluster.size, cluster.top_terms)
```

*records* is a list of row dicts with an outcome column (`outcome_col`,
default `"label"`) and text/signal columns (auto-detected, or pass
`text_cols`/`signal_cols` explicitly). Pass `judge=` (any `Model` with
`Capability.GENERATE`) to have an LLM name/describe each cluster from its
exemplars instead of the deterministic top-terms naming. `report.method` tells
you which clustering backend actually ran (`"hdbscan"` / `"agglomerative"` /
`"cosine_greedy"` / `"single_cluster"`); `report.as_hypothesis_context()`
renders a compact section for feeding into M3 hypothesis generation — this is
exactly what `AgenticDiagnoseLoop`'s `cluster_failures` tool does
automatically (see [quickstart](quickstart.md#agenticdiagnoseloop--judge-decided-m1-m5-alternative-to-the-fixed-cycle)).

## Backend notes

`--backend` selects the local coding-agent CLI: `antigravity` (default),
`claude_code`, `codex`, `opencode`, `gemini_cli`, `kimi_cli`. On `claude_code`
the bundled `nature-figure` Agent Skill styles agent-drawn figures
automatically (`--no-skills` to disable). `tool_calls_*.json` files are
skipped by default (`--include-tool-calls` to include them).

Run `evalvitals explore --help` for the full flag list.
