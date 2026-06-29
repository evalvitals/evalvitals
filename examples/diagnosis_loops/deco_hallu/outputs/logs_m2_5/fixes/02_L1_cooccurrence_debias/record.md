# Fix attempt 02 — cooccurrence_debias  [L1]

**Outcome:** did not fix (verdict: partial)
**Kind:** template    **Source:** judge

## Validation (paired McNemar vs. unmodified baseline)
- pairs tested (applicable): 24
- cases fixed: 3
- cases broken: 1
- coverage of failures: 100%
- unstable cases dropped (noise): 0
- effect: 0.0833
- e-value: 0.8
- statistically significant (rejects H0): False
- summary: [mcnemar + e-value (paired binary)] effect=+0.0833 (B>A) CI=-0.0833..+0.2500, e=0.80 -> inconclusive [partial, coverage=100%]

## Cases fixed
- d1edf34bb1bb
- df9252ab7455
- 18ffb4e4a446

## Cases broken
- 86e794df592c

## What was applied
```json
{
  "prompt_template": "Do not assume an object is present just because it commonly appears alongside other things in the image. Only answer Yes if you can directly see the specific object being asked about. {prompt}"
}
```