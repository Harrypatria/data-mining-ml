---
name: statistical-analysis
description: Statistical Analysis. Hypothesis tests, A/B tests and experiments, confidence intervals, effect sizes, power and sample size, multiple comparisons, regression diagnostics, Bayesian alternatives, time series (stationarity, ARIMA, ETS) and survival analysis (Kaplan-Meier, Cox), with causal caution (randomization, difference-in-differences, matching, instrumental variables). Use for statistics, significance testing, t-test, chi-squared, ANOVA, Mann-Whitney, correlation, regression interpretation, experiment readouts, or any causal question.
---

# Statistical Analysis

Explore first (Tukey), then infer. A bare p-value is a defect.

## Before running any test, write

1. **Question** in plain words.
2. **Estimand**: the exact quantity (difference in means, ratio of proportions, hazard ratio).
3. **Null and alternative**, one- or two-sided, and alpha.
4. **Design**: randomized, observational, paired, clustered.
5. **Minimum effect worth acting on**, and the power to detect it.

## Choose the test from the data

Check independence, normality (Shapiro-Wilk with Q-Q plot, sample-size aware), equal variance (Levene), and sample size, then follow `references/test-selection.md`. Default to Welch's t-test over Student's. Use rank or exact tests when assumptions fail.

## Report

| Always | Example form |
|---|---|
| Effect size | difference, Cohen's d, odds ratio, Cramer's V |
| Interval | 95 percent CI (analytical or bootstrap) |
| p-value | exact, with the test named |
| Sample sizes | per group |
| Correction | Holm or Benjamini-Hochberg when several tests |

## Experiments and A/B tests

- Power analysis before launch (`statsmodels.stats.power`). Fix duration and sample size in advance.
- No peeking. If interim looks are needed, use a pre-registered sequential design (alpha spending).
- One primary metric; guardrails reported but not used to declare a win.
- Check sample ratio mismatch (chi-squared on assignment counts) before reading results.
- Clustered randomization needs cluster-robust errors.

## Causation

Correlation is not causation. A causal claim needs a design: randomization, difference-in-differences (parallel trends), matching or weighting (no unmeasured confounding), instrumental variables (exclusion restriction), regression discontinuity. List the untestable assumptions. Without a design, write "is associated with".

## Regression

Diagnostics before reading coefficients: residual plots, VIF, influence (Cook's distance), Breusch-Pagan, Q-Q plot. Use robust (HC3) standard errors when heteroscedastic.

## Bayesian alternative

Offer it for small samples or when priors matter: Beta-Binomial for conversion rates, normal models with weakly informative priors; report posterior intervals and the probability the effect exceeds the minimum meaningful effect.

## Time series and survival

Stationarity (ADF, KPSS), decomposition, ARIMA or ETS with residual diagnostics, forward-chaining validation; Kaplan-Meier, log-rank, Cox with proportional-hazards checks, censoring explained. `references/time-series-survival.md`.
