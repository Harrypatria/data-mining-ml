# Test selection

## Decision tree

```
What is compared?
├── One numeric outcome, two independent groups
│   ├── roughly normal or n >= 30 per group -> Welch t-test (ttest_ind, equal_var=False); effect: mean difference, Hedges g
│   └── skewed, ordinal or small n -> Mann-Whitney U; effect: Hodges-Lehmann shift or rank-biserial r
├── One numeric outcome, paired measurements
│   ├── differences roughly normal -> paired t-test (ttest_rel); effect: mean difference
│   └── otherwise -> Wilcoxon signed-rank
├── One numeric outcome, three or more groups
│   ├── normal, equal variance -> one-way ANOVA (f_oneway); effect: eta squared; post hoc Tukey HSD
│   ├── unequal variance -> Welch ANOVA; post hoc Games-Howell
│   └── non-normal -> Kruskal-Wallis; post hoc Dunn with Holm
├── Two categorical variables
│   ├── expected counts >= 5 -> chi-squared (chi2_contingency); effect: Cramer's V
│   └── small counts, 2x2 -> Fisher exact; effect: odds ratio with CI
├── Two proportions (A/B conversion)
│   └── two-proportion z-test or Newcombe CI for the difference; Bayesian Beta-Binomial alternative
├── Paired binary outcomes -> McNemar
├── Association of two numerics
│   ├── linear, roughly normal -> Pearson r with Fisher-z CI
│   └── monotonic or outliers -> Spearman rho or Kendall tau, bootstrap CI
├── Distribution comparison -> Kolmogorov-Smirnov (ks_2samp), or Anderson-Darling
└── Time to event -> log-rank test; Cox model for adjusted hazard ratios
```

## Assumption checks

| Assumption | Check | If violated |
|---|---|---|
| Independence | Design review (clusters, repeated measures) | Mixed models, cluster-robust errors, paired tests |
| Normality | Q-Q plot, Shapiro-Wilk (over-sensitive at large n) | Rank tests, bootstrap, transform |
| Equal variance | Levene, variance ratio | Welch versions |
| Sample size | Counts per group, expected cell counts | Exact tests, Bayesian |

## Effect size and interval code

```python
import numpy as np
from scipy import stats

res = stats.ttest_ind(b, a, equal_var=False)
ci = res.confidence_interval(0.95)
pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
print(f"difference {b.mean() - a.mean():.3f} (95% CI {ci.low:.3f} to {ci.high:.3f}), d {(b.mean() - a.mean()) / pooled:.2f}, p {res.pvalue:.4f}")

boot = stats.bootstrap((a, b), lambda x, y: np.median(y) - np.median(x), n_resamples=5000, random_state=SEED, vectorized=False)
```

## Multiple comparisons

```python
from statsmodels.stats.multitest import multipletests
reject, p_adj, _, _ = multipletests(pvalues, alpha=0.05, method="holm")      # family-wise
reject, q, _, _ = multipletests(pvalues, alpha=0.05, method="fdr_bh")          # false discovery rate
```

## Power and sample size

```python
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
n = NormalIndPower().solve_power(effect_size=proportion_effectsize(0.12, 0.10), alpha=0.05, power=0.8)
```

## R equivalents

`t.test(y ~ g, var.equal = FALSE)`, `wilcox.test`, `chisq.test`, `fisher.test`, `p.adjust(p, "holm")`, `pwr::pwr.2p.test`, `effectsize::cohens_d`.
