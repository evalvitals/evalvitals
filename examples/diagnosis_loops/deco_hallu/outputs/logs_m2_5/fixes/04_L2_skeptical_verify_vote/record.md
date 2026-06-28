# Fix attempt 04 — skeptical_verify_vote  [L2]

**Outcome:** did not fix (verdict: no_effect)
**Kind:** spec    **Source:** judge

## Validation (paired McNemar vs. unmodified baseline)
- pairs tested (applicable): 40
- cases fixed: 2
- cases broken: 2
- coverage of failures: 100%
- unstable cases dropped (noise): 0
- effect: 0.0
- e-value: 0.5333
- statistically significant (rejects H0): False
- summary: [mcnemar + e-value (paired binary)] effect=+0.0000 (A=B) CI=-0.1000..+0.1000, e=0.53 -> inconclusive [no_effect, coverage=100%]

## Cases fixed
- df9252ab7455
- 18ffb4e4a446

## Cases broken
- d1edf34bb1bb
- 86e794df592c

## What was applied
```json
{
  "name": "skeptical_verify_vote",
  "image_ops": [],
  "prompt_template": "Carefully examine the image. Do NOT assume an object is present just because it would be typical for this scene. Only answer Yes if you can directly and clearly see the object itself in the image. If you cannot point to it, answer No. {prompt}",
  "n_samples": 3
}
```