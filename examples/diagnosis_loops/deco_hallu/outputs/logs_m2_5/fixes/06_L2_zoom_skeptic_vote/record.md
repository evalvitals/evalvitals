# Fix attempt 06 — zoom_skeptic_vote  [L2]

**Outcome:** did not fix (verdict: partial)
**Kind:** spec    **Source:** judge

## Validation (paired McNemar vs. unmodified baseline)
- pairs tested (applicable): 24
- cases fixed: 4
- cases broken: 0
- coverage of failures: 100%
- unstable cases dropped (noise): 0
- effect: 0.1667
- e-value: 3.2
- statistically significant (rejects H0): False
- summary: [mcnemar + e-value (paired binary)] effect=+0.1667 (B>A) CI=+0.0417..+0.3333, e=3.20 -> inconclusive [partial, coverage=100%]

## Cases fixed
- c560bf98c4ca
- 09d651295764
- df9252ab7455
- 18ffb4e4a446

## What was applied
```json
{
  "name": "zoom_skeptic_vote",
  "image_ops": [
    {
      "tool": "zoom_center",
      "params": {
        "factor": 1.5
      }
    },
    {
      "tool": "enhance_contrast",
      "params": {
        "factor": 1.5
      }
    }
  ],
  "prompt_template": "{prompt}\nThe presence of this object is not guaranteed. Verify it is genuinely visible before answering. When in doubt, answer No.",
  "n_samples": 5
}
```