"""Statistics — effect-sized, multiple-testing-aware verdicts for failure analysis.

``compare`` is the single entry point (never returns a bare p). Building blocks
are exported too: McNemar (paired binary), clustered bootstrap CI, e-values
(anytime-valid), e-BH (FDR under dependence), stratified subset sampling + τ.
"""

from evalvitals.stats.api import StatResult, ab_test, compare
from evalvitals.stats.bootstrap import clustered_bootstrap_diff
from evalvitals.stats.ebh import ebh
from evalvitals.stats.evalue import e_value_test, evalue_bernoulli
from evalvitals.stats.mcnemar import mcnemar
from evalvitals.stats.subset_sampling import kendall_tau, sample_subset, stratified_subset

__all__ = [
    "compare",
    "StatResult",
    "ab_test",
    "mcnemar",
    "clustered_bootstrap_diff",
    "evalue_bernoulli",
    "e_value_test",
    "ebh",
    "stratified_subset",
    "kendall_tau",
    "sample_subset",
]
