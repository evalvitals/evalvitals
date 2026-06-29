# Bundled Agent Skills

Agent Skills vendored into the package so they travel with the repo (both
`git clone` and `pip install`). There are two kinds here:

- **Agent-applied** skills are auto-discovered by `evalvitals explore` on the
  **claude** / **agy** backends and style the figures the exploratory coding
  agent writes under `figures/`. (`nature-figure`.)
- **Host-applied** skills are *not* pushed to the agent; the host code imports
  their asset directly. (`eval-chart-style`.)

Each `explore` run vendors the agent-applied skills into its sandbox at
`<workdir>/.claude/skills/<name>/`, which an `--add-dir <workdir>` Claude Code /
agy run discovers automatically.

## nature-figure

Submission-grade Nature/high-impact-journal matplotlib/ggplot figure workflow.

- **Source:** https://github.com/Yuan1z0825/nature-skills (`skills/nature-figure`)
- **License:** Apache-2.0 — see [`nature-figure/LICENSE`](nature-figure/LICENSE).
- **Trimmed:** only the functional core is vendored (SKILL.md, manifest.yaml,
  `static/`, `references/`, ~176 KB). The upstream `assets/` demo galleries
  (~30 MB of example PNGs and figures4papers scripts) are **omitted** to keep
  the repo lean; fetch them from the source repo if you want the demo galleries.

## eval-chart-style (host-applied)

Plotly chart-type + house-style standard for FAIL-vs-PASS eval analysis in the
interactive Streamlit dashboard, plus a matplotlib palette for the host-rendered
static PNGs.

- **Used by the host, not the agent.** Its `assets/eval_viz_theme.py` is imported
  directly by [`viz_theme.py`](../viz_theme.py): the dashboard
  (`dashboard_app.py`) renders its plotly builders (forest / violin / logistic /
  scatter / class-balance) via `st.plotly_chart`, and `render_chart_specs`
  (`charts.py`) adopts its semantic FAIL/PASS palette + matplotlib rcParams for
  the static PNGs.
- **Deliberately excluded from agent auto-apply** (`bundled_skill_paths()` skips
  it) so it does not change the `explore` agent's behaviour or collide with
  `nature-figure`. To apply it to an agent anyway, pass it explicitly:
  `evalvitals explore … --skill evalvitals/analysis/skills/eval-chart-style`.
- **Plotly** (the builders' backend) ships in the `[dashboard]` extra. The
  matplotlib palette path needs only `[viz]`.
- **Case-agnostic:** ships no domain-specific signal names; callers register
  display aliases at runtime via `viz.register_short_names({...})`.

## Disabling / overriding

- `evalvitals explore … --no-skills` skips bundled skills for a run.
- `evalvitals explore … --skill /path/to/other-skill` adds more skill dirs.
- `--allow-skills` also lets globally-installed `~/.claude/skills` be used.
