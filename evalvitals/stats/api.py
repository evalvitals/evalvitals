"""stats/api.py — the single entry point. It NEVER returns a bare p-value.

``compare`` returns a :class:`StatResult` with an effect size + CI, an e-value, a
corrected reject decision, and an underpowered flag — so a caller can't ship
"p<0.05" with a CI hugging zero.  Defaults: paired binary -> McNemar + a paired
e-value (anytime-valid), clustered bootstrap CI (cluster at task).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from evalvitals.stats.bootstrap import clustered_bootstrap_diff
from evalvitals.stats.evalue import evalue_bernoulli
from evalvitals.stats.mcnemar import mcnemar


@dataclass
class StatResult:
    """A corrected, effect-sized verdict — the only thing stats hands back."""

    effect: float                       # mean(B) - mean(A)
    ci: tuple[float, float]
    reject: bool                        # corrected decision (e-value / CI), NOT a raw p
    method: str
    alpha: float
    e_value: Optional[float] = None
    underpowered: bool = False
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        arrow = "B>A" if self.effect > 0 else ("A>B" if self.effect < 0 else "A=B")
        verdict = "REJECT H0" if self.reject else "inconclusive"
        warn = "  ⚠ underpowered" if self.underpowered else ""
        e = f", e={self.e_value:.2f}" if self.e_value is not None else ""
        return (f"[{self.method}] effect={self.effect:+.4f} ({arrow}) "
                f"CI={self.ci[0]:+.4f}..{self.ci[1]:+.4f}{e} -> {verdict}{warn}")


def compare(
    success_a: Sequence,
    success_b: Sequence,
    *,
    paired: bool = True,
    alpha: float = 0.05,
    min_effect: float = 0.0,
    cluster_by: Optional[Sequence] = None,
    correction: str = "evalue",   # "evalue" (anytime-valid) | "p" (McNemar exact p)
    n_boot: int = 2000,
    seed: int = 0,
) -> StatResult:
    """Compare two per-example success vectors. Returns a :class:`StatResult`."""
    boot = clustered_bootstrap_diff(success_a, success_b, clusters=cluster_by,
                                    n_boot=n_boot, seed=seed, paired=paired)
    effect = boot["effect"]
    ci = (boot["ci_low"], boot["ci_high"])
    ci_includes_zero = ci[0] <= 0 <= ci[1]

    e_value = None
    if paired:
        mc = mcnemar(success_a, success_b)
        e_value = evalue_bernoulli(mc["b"], mc["n_discordant"], p0=0.5)
        if correction == "p":
            reject = mc["p_value"] < alpha
            method = "mcnemar-exact-p (paired binary)"
        else:
            reject = e_value >= 1.0 / alpha            # anytime-valid
            method = "mcnemar + e-value (paired binary)"
        details = {"p_value": mc["p_value"], "b": mc["b"], "c": mc["c"],
                   "n_discordant": mc["n_discordant"]}
    else:
        reject = not ci_includes_zero                  # unpaired: CI excludes 0
        method = "clustered bootstrap (unpaired)"
        details = {}

    # underpowered: not significant AND the CI is too wide to rule out a min_effect-sized effect
    width = ci[1] - ci[0]
    underpowered = (not reject) and (min_effect > 0) and (width > min_effect)

    return StatResult(effect=effect, ci=ci, reject=reject, method=method, alpha=alpha,
                      e_value=e_value, underpowered=underpowered, details=details)


def ab_test(success_a: Sequence, success_b: Sequence, **kwargs) -> StatResult:
    """Back-compat alias: pass per-example success vectors (bools/0-1)."""
    return compare(success_a, success_b, **kwargs)
