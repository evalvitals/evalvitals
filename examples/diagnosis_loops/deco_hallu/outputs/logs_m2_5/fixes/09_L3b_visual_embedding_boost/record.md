# Fix attempt 09 — visual_embedding_boost  [L3b]

**Outcome:** did not fix (verdict: no_effect)
**Kind:** primitive    **Source:** judge

## Validation (paired McNemar vs. unmodified baseline)
- pairs tested (applicable): 24
- cases fixed: 1
- cases broken: 1
- coverage of failures: 100%
- unstable cases dropped (noise): 0
- effect: 0.0
- e-value: 0.6667
- statistically significant (rejects H0): False
- summary: [mcnemar + e-value (paired binary)] effect=+0.0000 (A=B) CI=-0.1250..+0.1250, e=0.67 -> inconclusive [no_effect, coverage=100%]

## Cases fixed
- 18ffb4e4a446

## Cases broken
- 8a6bd51a877b

## What was applied
```json
{
  "primitive": "visual_embedding_boost",
  "params": {
    "gamma": 2.0
  }
}
```