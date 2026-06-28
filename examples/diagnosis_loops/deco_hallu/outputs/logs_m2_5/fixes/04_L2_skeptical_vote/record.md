# Fix attempt 04 — skeptical_vote  [L2]

**Outcome:** did not fix (verdict: partial)
**Kind:** spec    **Source:** judge

## Validation (paired McNemar vs. unmodified baseline)
- pairs tested (applicable): 24
- cases fixed: 3
- cases broken: 0
- coverage of failures: 100%
- unstable cases dropped (noise): 0
- effect: 0.125
- e-value: 2.0
- statistically significant (rejects H0): False
- summary: [mcnemar + e-value (paired binary)] effect=+0.1250 (B>A) CI=+0.0000..+0.2500, e=2.00 -> inconclusive [partial, coverage=100%]

## Cases fixed
- 9906d4f70c33
- df9252ab7455
- 18ffb4e4a446

## What was applied
```json
{
  "name": "skeptical_vote",
  "image_ops": [],
  "prompt_template": "{prompt}\nBase your answer ONLY on what is actually visible. Do not assume an object is present just because it would be common in this kind of scene. If you cannot clearly see it, answer No.",
  "n_samples": 5
}
```