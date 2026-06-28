# Fix attempt 02 — anti_cooccurrence  [L1]

**Outcome:** did not fix (verdict: partial)
**Kind:** template    **Source:** judge

## Validation (paired McNemar vs. unmodified baseline)
- pairs tested (applicable): 40
- cases fixed: 6
- cases broken: 2
- coverage of failures: 100%
- unstable cases dropped (noise): 0
- effect: 0.1
- e-value: 1.0159
- statistically significant (rejects H0): False
- summary: [mcnemar + e-value (paired binary)] effect=+0.1000 (B>A) CI=-0.0250..+0.2250, e=1.02 -> inconclusive [partial, coverage=100%]

## Cases fixed
- 9906d4f70c33
- f4180dce7c58
- df9252ab7455
- 8a6bd51a877b
- 4efaf6638356
- 18ffb4e4a446

## Cases broken
- d1edf34bb1bb
- 86e794df592c

## What was applied
```json
{
  "prompt_template": "Look only at what is literally visible in this image. Ignore what objects are typically found together; the presence of related or commonly co-occurring items is not evidence. {prompt}"
}
```