"""Statistical comparison example — compare() and compare_multiple().

Demonstrates the stats API for evaluating strategy A vs B (pairwise) and for
ranking 3+ strategies (Friedman + Nemenyi post-hoc).  No model or API key needed
— this example runs entirely on simulated outcome vectors.

Design principles (enforced by the API):
  • Never return a bare p-value: always report effect size + clustered CI.
  • Use McNemar (paired) by default — more powerful than unpaired proportion tests.
  • Use anytime-valid e-values so the eval loop can peek without inflating type-I error.
  • Reject only when both e >= 1/alpha AND CI excludes zero.

References:
  McNemar    — McNemar (1947) https://doi.org/10.1007/BF02295996
  Bootstrap  — Efron (1979), Cameron et al. (2008)
  E-values   — Grünwald et al. (2022) https://arxiv.org/abs/1906.07801
  e-BH (FDR) — Wang & Ramdas (2022)  https://arxiv.org/abs/2009.02824
  Friedman   — Demšar (2006) https://jmlr.org/papers/v7/demsarar06a.html

Usage (inside Docker):
    python run.py                 # uses config.yaml defaults
    python run.py --n-examples 500

Expected output:
    [A vs B] effect=+0.12 (B>A) CI=[+0.04, +0.20] e=18.4 reject=True underpowered=False
    [3-way]  Friedman p=0.003 → Nemenyi CDs: A-B differ, A-C differ, B-C do not
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from evalvitals.stats import compare, compare_multiple

CONFIG = Path(__file__).parent / "config.yaml"


def _simulate_outcomes(n: int, p_a: float, p_b: float, seed: int):
    """Generate paired binary outcomes for two strategies."""
    rng = random.Random(seed)
    a = [rng.random() < p_a for _ in range(n)]
    b = [rng.random() < p_b for _ in range(n)]
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--n-examples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n = args.n_examples or cfg.get("n_examples", 200)
    seed = cfg.get("seed", 42)
    alpha = cfg.get("alpha", 0.05)
    min_effect = cfg.get("min_effect", 0.03)

    # --- Pairwise A vs B ---
    # Strategy A: 65% success rate; Strategy B: 77% success rate
    a, b = _simulate_outcomes(n, p_a=0.65, p_b=0.77, seed=seed)
    task_ids = [f"task_{i % 20}" for i in range(n)]  # 20 task clusters

    r = compare(a, b, paired=True, alpha=alpha, min_effect=min_effect, cluster_by=task_ids)
    sign = "B>A" if r.effect > 0 else "A>B"
    print(f"[A vs B] effect={r.effect:+.3f} ({sign}) "
          f"CI=[{r.ci[0]:+.3f}, {r.ci[1]:+.3f}] "
          f"e={r.e_value:.1f} reject={r.reject} underpowered={r.underpowered}")
    print(f"  method={r.method}, alpha={r.alpha}")

    # --- 3-way Friedman + Nemenyi ---
    # Strategy C: 70% success
    _, c = _simulate_outcomes(n, p_a=0.65, p_b=0.70, seed=seed + 1)
    mr = compare_multiple({"A": a, "B": b, "C": c}, alpha=alpha)
    print(f"\n[3-way Friedman] p={mr.p_value:.4f} reject={mr.reject_global} "
          f"(CD={mr.critical_difference:.3f})")
    print(f"  avg_ranks: {mr.avg_ranks}")
    print(f"  significantly different pairs: {mr.significant_pairs}")


if __name__ == "__main__":
    main()
