# Fix attempt 06 — upscale_evidence_vote  [L2]

**Outcome:** did not fix (verdict: partial)
**Kind:** spec    **Source:** judge

## Validation (paired McNemar vs. unmodified baseline)
- pairs tested (applicable): 40
- cases fixed: 7
- cases broken: 1
- coverage of failures: 100%
- unstable cases dropped (noise): 0
- effect: 0.15
- e-value: 3.5556
- statistically significant (rejects H0): False
- summary: [mcnemar + e-value (paired binary)] effect=+0.1500 (B>A) CI=+0.0250..+0.2750, e=3.56 -> inconclusive [partial, coverage=100%]

## Cases fixed
- 9906d4f70c33
- f4180dce7c58
- c9708d64631d
- 09d651295764
- df9252ab7455
- 8a6bd51a877b
- 53f065ac6e65

## Cases broken
- d1edf34bb1bb

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
  "prompt_template": "Base your answer strictly on visual evidence in this image. Ignore expectations about which objects commonly co-occur. If the object is not actually depicted, answer No. {prompt}",
  "n_samples": 5
}
```