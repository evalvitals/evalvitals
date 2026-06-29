# Fix attempt 03 — skeptical_default_no  [L1]

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
- d1edf34bb1bb
- df9252ab7455
- 18ffb4e4a446

## What was applied
```json
{
  "prompt_template": "Answer No unless the queried object is clearly and unambiguously visible in the image. Avoid guessing based on context or scene type. {prompt}"
}
```