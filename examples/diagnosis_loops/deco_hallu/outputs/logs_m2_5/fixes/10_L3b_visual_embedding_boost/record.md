# Fix attempt 10 — visual_embedding_boost  [L3b]

**Outcome:** did not fix (verdict: partial)
**Kind:** primitive    **Source:** judge

## Validation (paired McNemar vs. unmodified baseline)
- pairs tested (applicable): 24
- cases fixed: 1
- cases broken: 0
- coverage of failures: 100%
- unstable cases dropped (noise): 0
- effect: 0.0417
- e-value: 1.0
- statistically significant (rejects H0): False
- summary: [mcnemar + e-value (paired binary)] effect=+0.0417 (B>A) CI=+0.0000..+0.1250, e=1.00 -> inconclusive [partial, coverage=100%]

## Cases fixed
- 18ffb4e4a446

## What was applied
```json
{
  "primitive": "visual_embedding_boost",
  "params": {
    "gamma": 1.25
  }
}
```