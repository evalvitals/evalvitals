# Exploratory Analysis & Confirmatory Statistics

EvalVitals' data-analysis layer (`evalvitals.analysis`) has two distinct
tools, usable standalone or inside the full M1-M5 diagnosis loop:

- `evalvitals explore` (`ExploratoryAnalysisAgent`): no-code, general-purpose
  exploratory data analysis over JSON/JSONL logs — profile the data, find
  patterns, produce takeaways with charts. Purely descriptive: it does not
  generate or validate hypotheses.
- `StatsAnalysisAgent`: confirmatory statistical tests over standardized
  records — effect sizes, confidence intervals, e-values, FDR-aware decisions.

Use `explore` first when you do not yet know which signals matter. Promote
the useful candidate signals into `StatsAnalysisAgent` when you need a
confirmatory verdict.

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
dashboard/viz dependencies can run in an existing `torch==2.6.*` environment.

## No-Code Explore

Point the explore CLI at a file or directory containing JSON/JSONL results and
pass a single analysis question:

```bash
evalvitals explore /path/to/results \
  --backend antigravity \
  -q "Which features distinguish incorrect cases from correct cases?" \
  --out evalvitals_explore_output \
  --max-rows 2000 \
  --max-files 200 \
  --dashboard          # optional: open the dashboard when done
```

The input path may be a single `.json` or `.jsonl` file, or a directory tree.
EvalVitals recursively samples records across files and writes a normalized
`records.json` artifact. Example questions:

```text
Compare accuracy across model directories.
Which features distinguish incorrect cases from correct cases?
Does tool usage correlate with failures?
Find candidate signals I should confirm with StatsAnalysisAgent.
```

### Arbitrary data, not just pass/fail logs

`explore` is a general-purpose EDA agent, not a pass/fail-only tool. The host
profiles the data first and classifies the outcome column (if any) as
`binary`, `categorical`, `continuous`, or `none`, and the generated-code
prompt adapts its framing and standard chart battery accordingly:

- **binary** outcome (e.g. an M1 `label` column, or any 2-valued column) —
  the familiar FAIL-vs-PASS story: class balance, fail-rate curves, top
  discriminators.
- **categorical** outcome (3+ classes) — per-class distributions and
  cross-class separation, without collapsing classes into a binary split.
- **continuous** outcome — a correlation/regression-style story: scatter +
  trend lines, binned mean-outcome curves, ranked correlation magnitude.
- **none** — no recognizable outcome column at all. The agent runs
  unsupervised EDA (distributions, missingness, correlation structure, group
  contrasts) instead of inventing a label that isn't in the data.

Outcome detection is name-heuristic by default (`label`, `outcome`, `target`,
`is_correct`, ...). Pass `--outcome-col <name>` (CLI) or
`outcome_col="<name>"` (`ExploratoryAnalysisAgent.explore_records` /
`explore_path`) to point it at an arbitrarily-named target column, e.g. a
continuous metric like `revenue` or `yield_pct` that no heuristic would catch.

M1's diagnosis loop is one caller of this agent, not a special case: it works
because M1 records already carry a `label` column that the name heuristic
finds automatically.

The run asks the local coding agent to write one analysis script, executes it in
EvalVitals' sandbox, host-adjudicates any host-checkable candidate statistics,
renders the chart specs to PNG, and writes the artifacts. It is a single shot —
re-run with a new `-q` question for a new analysis. (The standalone console
script `evalvitals-explore` is equivalent.)

## Explore Outputs

A typical output directory looks like this:

```text
evalvitals_explore_output/
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

- `exploratory_report.json`: structured answer — a ranked list of
  `takeaways` (each paired with the chart/table that supports it), plus
  observations, candidate signals (with host-adjudicated verdicts when
  applicable), charts, and tables.
- `analysis.py`: generated analysis code that was actually executed.
- `records.json`: sampled records given to the generated script.
- `figures/` and `tables/`: rendered charts (host-side, from spec + CSV) and
  tabular artifacts, if any.

The exploratory report is intentionally discovery-oriented and purely
descriptive: patterns, correlations, and candidate signals, not hypotheses or
causal claims. It is not the confirmatory statistics layer.

### Figure styling (Agent Skills)

On the **claude** / **agy** backends, `explore` automatically applies the
package-bundled **nature-figure** Agent Skill so the figures the agent writes
under `figures/` come out publication-quality. The skill is vendored in the repo
(`evalvitals/agent_assets/skills/`), so it travels with `git clone` and `pip
install` — no per-machine `~/.claude` setup needed.

- `--no-skills` — skip bundled skills for a run.
- `--skill /path/to/other-skill` — add more skill dirs (repeatable).
- `--allow-skills` — also enable globally-installed `~/.claude/skills`.

Skills style only the agent-authored `figures/*.png`; the host-rendered chart
specs (`charts`) stay deterministic and are never touched by a skill.

## Dashboard

Open a Streamlit dashboard over an explore output directory:

```bash
evalvitals dashboard evalvitals_explore_output --port 8501
```

The dashboard reads the saved artifacts; it does not re-run the agent. Use it to
review the analysis, rendered figures, tables, and the exact generated code. The
same loader also renders a diagnostic loop run (`run_log.jsonl`) as an
M2 → M3 → M5 → Fix story.

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

Standalone explore mode does not require a standardized `StatsInput`. It accepts
messy logs and uses a local coding agent to inspect their shape.

## Generalized Analysis Core

Both tools share a reusable profiling core (`evalvitals.analysis.profile`):

```text
records
  -> DatasetProfile        # column types, roles, missingness, grain
  -> describe_outcome()    # binary / categorical / continuous / none — feeds
                            # ExploratoryAnalysisAgent's adaptive framing
```

The confirmatory side builds on the same `DatasetProfile` for its own
ranked, estimand-explicit test plan:

```text
records / StatsInput
  -> DatasetProfile
  -> AnalysisPlan          # ranked estimands, not first-N column order
  -> EvidenceResult        # effect, CI, p/e-value, correction family
  -> MultiplicityController# BH for p-values, e-BH for e-values
```

The diagnosis loop is one consumer of the confirmatory core. Loop data is
still adapted into `StatsInput`, but standalone callers can profile records
directly with `profile_records()` and can inspect the generated plan with
`plan_stats_input()`. Per-case signal tests now produce exact permutation
p-values and enter a BH family, while paired/e-value tests continue to use
e-BH. Reports preserve the legacy `rejected_tools` field for M1-M5 and add
precise `rejected_result_keys` plus per-family metadata for audits.

For p-value signal families, the raw effect/CI verdict remains available to the
M1-M5 loop while `fdr_corrected` marks whether the claim survived BH. Fused and
standalone analysis should use the FDR metadata for controlled claims; the
loop's hypothesis-testing stage can fall back to raw per-case comparison when
the confirmatory pass only produced descriptive/global results.

The full diagnosis loop and confirmatory `StatsAnalysisAgent` do benefit from
standardized records because the loop's stats, hypothesis-testing, logging,
and downstream tools need stable contracts:

```text
M1 findings / user records
  -> standardized records
  -> controlled stats tools
  -> effect size + CI + e-value
  -> e-BH FDR correction
  -> StatsAnalysisReport
```

Use this rule of thumb:

- For exploratory questions, use `evalvitals explore` directly on result logs.
- For claims you want the loop to consume, standardize and run
  `StatsAnalysisAgent`.
- For M1-M5 automation, keep the standardized contract so the loop's stages
  can share evidence safely.

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
machine-readable JSON back to the exploratory-analysis wrapper.

By default, `tool_calls_*.json` files are skipped because they can dominate log
volume. Add `--include-tool-calls` only when you specifically want
tool-call-level analysis.
