# Fix attempt 05 — salient_crop_skeptic  [L2]

**Outcome:** did not fix (verdict: partial)
**Kind:** spec    **Source:** judge

## Validation (paired McNemar vs. unmodified baseline)
- pairs tested (applicable): 40
- cases fixed: 6
- cases broken: 1
- coverage of failures: 100%
- unstable cases dropped (noise): 0
- effect: 0.125
- e-value: 2.2857
- statistically significant (rejects H0): False
- summary: [mcnemar + e-value (paired binary)] effect=+0.1250 (B>A) CI=+0.0000..+0.2500, e=2.29 -> inconclusive [partial, coverage=100%]

## Cases fixed
- 9906d4f70c33
- f4180dce7c58
- df9252ab7455
- 8a6bd51a877b
- 53f065ac6e65
- 18ffb4e4a446

## Cases broken
- d1edf34bb1bb

## What was applied
```json
{
  "name": "salient_crop_skeptic",
  "image_ops": [
    {
      "tool": "crop_salient_region",
      "params": {
        "padding": 0.05,
        "min_delta": 18
      }
    }
  ],
  "prompt_template": "Look only at what is actually visible in this image, not at what objects usually appear together. Answer Yes only if the named object is itself plainly visible; otherwise answer No. {prompt}",
  "n_samples": 3
}
```