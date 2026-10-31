"""Clean A/B readout fixture: assumptions, effect size, interval and correction."""
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

SEED = 3
rng = np.random.default_rng(SEED)
control, variant = rng.normal(10, 2, 400), rng.normal(10.4, 2, 400)
print("normality p", stats.shapiro(control).pvalue, stats.shapiro(variant).pvalue)
print("equal variance p", stats.levene(control, variant).pvalue)
res = stats.ttest_ind(variant, control, equal_var=False)
diff = variant.mean() - control.mean()
pooled = np.sqrt((control.var(ddof=1) + variant.var(ddof=1)) / 2)
ci = res.confidence_interval(0.95)
print(f"difference {diff:.2f}, Cohen's d {diff / pooled:.2f}, 95% CI {ci.low:.2f} to {ci.high:.2f}")
print("adjusted p", multipletests([res.pvalue, 0.04, 0.2], method="holm")[1])
