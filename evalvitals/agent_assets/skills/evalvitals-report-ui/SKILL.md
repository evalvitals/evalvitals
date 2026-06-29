---
name: evalvitals-report-ui
description: Generate EvalVitals exploratory results that read like diagnostic reports, not raw feature dumps.
---

# EvalVitals Report/UI Contract

Use this skill whenever you produce EvalVitals exploratory analysis JSON,
dashboard-ready artifacts, chart specs, or diagnostic report text.

## Audience

The reader is evaluating a model failure mode. They should not need to know
internal probe ids, generated column names, or implementation details to
understand the result.

## Required Output Discipline

- Every `candidate_signals` item must include:
  - `name`: stable machine id, snake_case, suitable for recipes and joins.
  - `display_name`: short human label, suitable for dashboard titles.
  - `rationale`: one sentence in domain language.
  - `suggested_test`: what would confirm or refute it.
- Every chart spec should include:
  - `name`: stable machine id.
  - `title` or `display_name`: human-readable, no raw probe/generated id unless
    no interpretation is known.
- Every `visual_plan` item should include a readable `question` and avoid raw
  field names in the user-facing wording.
- `chart_readings` and `claims` should cite what the chart means, not just which
  column changed.

## Label-Like And Probe-Derived Signals

Signals such as `generated_probe1_false_detection`, `probe1_false_detection`,
or anything that perfectly re-measures the FAIL label are audit plumbing. Treat
them as label checks, never as root causes.

Use labels like:

- `Label audit: probe false-detection flag`
- `Probe says object is present`
- `Attention focus share`
- `Maximum relative attention`
- `Mean relative attention`

Do not lead a report with a label-like signal even when it has the largest
effect size. Mention it as a sanity check, then rank non-leaky explanatory
signals first.

## Overall Result Shape

The final answer should be claim-first:

1. State the supported non-leaky finding in plain language.
2. Mention label-like audit signals only as demoted sanity checks.
3. Link each claim to evidence: confirmed signal, chart, or downstream test.
4. Say what not to infer: association is not causality unless intervention tests
   support it.

