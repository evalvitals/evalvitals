# Fix attempt 01 — evidence_grounded  [L1]

**Outcome:** did not fix (verdict: partial)
**Kind:** template    **Source:** judge

## Validation (paired McNemar vs. unmodified baseline)
- pairs tested (applicable): 40
- cases fixed: 5
- cases broken: 2
- coverage of failures: 100%
- unstable cases dropped (noise): 0
- effect: 0.075
- e-value: 0.7619
- statistically significant (rejects H0): False
- summary: [mcnemar + e-value (paired binary)] effect=+0.0750 (B>A) CI=-0.0500..+0.2000, e=0.76 -> inconclusive [partial, coverage=100%]

## Cases fixed
- 9906d4f70c33
- f4180dce7c58
- 8a6bd51a877b
- c8952e6d8856
- 18ffb4e4a446

## Cases broken
- d1edf34bb1bb
- 86e794df592c

## What was applied
```json
{
  "prompt_template": "{prompt} Answer Yes only if you can point to the specific visible region where the object actually appears. If you cannot locate it directly in the image, answer No. Do not infer its presence from other objects that are commonly found alongside it."
}
```