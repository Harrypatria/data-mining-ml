---
name: data-profiling
description: Data Profiling. Explore and profile a dataset before modeling - EDA (exploratory data analysis), distributions, skew, outliers, missingness (MCAR, MAR, MNAR), duplicates, cardinality, class balance, label quality, leakage risk, bias and representativeness, drift between periods. Use when someone says explore, profile, EDA, data quality, understand the data, check the dataset, or before any machine learning, data mining or statistical analysis. CRISP-DM Data Understanding.
---

# Data Profiling

Explore before inferring. Profiling ends with a short list of facts that change modeling choices, not a gallery of charts.

## Procedure

1. **Grain and keys.** State what one row is. Assert key uniqueness (`df[key].is_unique`). Count rows per entity; repeated entities force group-aware splits.
2. **Schema.** `df.shape`, `df.dtypes`, `df.head()`. Assert expected columns and types. Parse dates once. Note units per numeric column.
3. **Ranges.** `describe()` on numerics; flag impossible values (negative ages, future dates, totals that do not add up). Decide error versus signal before touching outliers.
4. **Missingness.** Share per column, share by target class, co-missing patterns. Classify each as MCAR, MAR or MNAR with a reason. Missing-by-target differences are signal and a leakage hint.
5. **Distributions.** Skew and tails, multimodality, zero inflation, bounded proportions. One chart per finding, axes labeled with units.
6. **Target.** Prevalence with the positive count, label definition, label delay, label noise sample check (read 20 raw records per class).
7. **Relationships.** Target rate by key categories, correlation matrix (Spearman for skew), VIF for linear models, Simpson's paradox check on the main segment.
8. **Leakage scan.** For each column: when is it recorded relative to the prediction time? Columns created after the event are removed now. Suspiciously strong single predictors get investigated.
9. **Representativeness and bias.** Compare sample with the population the model will score: time window, regions, channels, protected groups. Note selection and survivorship bias.
10. **Drift.** If data spans periods, compare early and late slices (KS test, PSI). Drift decides between random and forward-chaining validation.

## Output

A profile summary in the notebook, under findings headings, plus these lines for ANALYSIS_MANIFEST.md:

```
Grain: <one row = ...>; entities repeat: <yes/no, max rows per entity>
Target: <definition>; prevalence <p> (<k> positives of <n>)
Missingness: <column: rate, mechanism, plan>
Leakage removed: <columns and why>
Scenarios detected: <IDs from references/scenarios.md>
Split implication: <stratified | group | forward-chaining>
```

## Rules

- Print summaries, never full frames. Clear large outputs before saving.
- Every claim about the data cites the cell that computed it.
- Profile on training data only once a split exists; the test set is not explored.
- Data conditions and their remedies: `references/scenarios.md`.
