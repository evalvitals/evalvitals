"""E-values — anytime-valid evidence (safe under optional stopping / peeking).

The closed loop keeps adding hypotheses and peeking at results; p-values break
under that, e-values don't.  ``evalue_bernoulli`` is the mixture (Bayes-factor
with a uniform prior) e-value for a Bernoulli mean vs ``p0`` — a valid e-value:
under H0, E[e] <= 1, so rejecting when ``e >= 1/alpha`` controls type-I error at
any stopping time.  For a paired A/B test, feed the discordant pairs that favour
B as ``successes`` out of ``n_discordant`` with ``p0=0.5`` (the McNemar null).

E-values and safe testing:
  "Safe Testing" — Grünwald, de Heide & Koolen (2022)
  J. Royal Statistical Society B — https://arxiv.org/abs/1906.07801

Mixture / Bayes-factor e-value for the Bernoulli mean:
  "Estimating means of bounded random variables by betting"
  Waudby-Smith & Ramdas (2023), JRSSB — https://arxiv.org/abs/2010.09686
"""

from __future__ import annotations

import math


def evalue_bernoulli(successes: int, n: int, p0: float = 0.5) -> float:
    """Mixture e-value for Bernoulli(p) vs H0: p = p0 (uniform prior over p)."""
    if n <= 0:
        return 1.0
    s = int(successes)
    if not (0.0 < p0 < 1.0):
        raise ValueError("p0 must be in (0, 1)")
    # numerator: log Beta(s+1, n-s+1) = mixture marginal (binomial coeff cancels with f0)
    log_num = math.lgamma(s + 1) + math.lgamma(n - s + 1) - math.lgamma(n + 2)
    log_den = s * math.log(p0) + (n - s) * math.log(1 - p0)
    return math.exp(log_num - log_den)


def e_value_test(successes: int, n: int, p0: float = 0.5, alpha: float = 0.05) -> dict:
    """Convenience wrapper: e-value + reject decision at level *alpha*."""
    e = evalue_bernoulli(successes, n, p0)
    return {"e_value": e, "reject": e >= 1.0 / alpha, "threshold": 1.0 / alpha, "alpha": alpha}
