# Fix attempt 05 — upscale_evidence_vote  [L2]

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
- 9906d4f70c33
- c9708d64631d
- 09d651295764
- df9252ab7455

## What was applied
```json
{
  "name": "upscale_evidence_vote",
  "image_ops": [
    {
      "tool": "upscale",
      "params": {
        "factor": 2.0
      }
    },
    {
      "tool": "sharpen",
      "params": {
        "factor": 2.0
      }
    }
  ],
  "prompt_template": "{prompt}\nLook carefully at the actual pixels for direct visual evidence of this specific object. Answer Yes only if you can point to where it appears; otherwise answer No.",
  "n_samples": 5
}
```