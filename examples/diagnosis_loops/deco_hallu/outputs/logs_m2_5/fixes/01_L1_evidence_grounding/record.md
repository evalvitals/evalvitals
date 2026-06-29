# Fix attempt 01 — evidence_grounding  [L1]

**Outcome:** did not fix (verdict: partial)
**Kind:** template    **Source:** judge

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
  "prompt_template": "Look carefully at the actual visual content of this image before answering. Base your answer only on what is literally visible, not on what objects you would typically expect to find in such a scene. {prompt}"
}
```