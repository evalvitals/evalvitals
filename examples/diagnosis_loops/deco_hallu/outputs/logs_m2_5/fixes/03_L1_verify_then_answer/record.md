# Fix attempt 03 — verify_then_answer  [L1]

**Outcome:** did not fix (verdict: no_effect)
**Kind:** template    **Source:** judge

## Validation (paired McNemar vs. unmodified baseline)
- pairs tested (applicable): 40
- cases fixed: 1
- cases broken: 1
- coverage of failures: 100%
- unstable cases dropped (noise): 0
- effect: 0.0
- e-value: 0.6667
- statistically significant (rejects H0): False
- summary: [mcnemar + e-value (paired binary)] effect=+0.0000 (A=B) CI=-0.0750..+0.0750, e=0.67 -> inconclusive [no_effect, coverage=100%]

## Cases fixed
- 9906d4f70c33

## Cases broken
- d1edf34bb1bb

## What was applied
```json
{
  "prompt_template": "{prompt} Before answering, scan the entire image for the actual object itself, not for context that merely suggests it. Treat absence as the default and require direct visual confirmation to answer Yes."
}
```