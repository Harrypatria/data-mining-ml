---
name: feature-engineering
description: Feature Engineering. Create, extract and select model inputs - feature extraction (PCA, SVD, TF-IDF, embeddings, signal features), feature engineering (date parts, cyclical encoding, lags, rolling windows, ratios, interactions, binning, target encoding, frequency encoding, geospatial), feature selection (mutual information, RFE, L1, stability selection), feature dictionary and ablation. Use for feature creation, feature construction, encoding categoricals, dimensionality reduction for modeling, or improving a machine learning model's inputs.
---

# Feature Engineering

## Two different ideas

- **Feature extraction** compresses raw or high-dimensional data into a compact representation (PCA, truncated SVD, NMF, TF-IDF, pretrained embeddings, FFT and rolling statistics). Always state what information is lost and how you inspected it: explained variance, reconstruction error, loadings.
- **Feature engineering** builds informative inputs from domain knowledge (ratios, lags, cyclical time, group aggregates, flags). Every feature needs a rationale.

## Rules

1. Each engineered feature has a one-line rationale and passes the leakage question: *would this value exist at prediction time?*
2. Features are computed inside the pipeline (`ColumnTransformer`, `FunctionTransformer` with a named importable function, or a custom transformer in `src/<package>/transformers.py`) so training and serving share code.
3. Anything that learns from data (encoders, binning edges, selectors, PCA, target encoding) is fitted on training folds only.
4. Lags and rolling windows use only past data: `shift(k)` with k above zero, trailing windows, grouped by entity, computed per snapshot.
5. Target encoding is out-of-fold: `sklearn.preprocessing.TargetEncoder` (cross-fitted in `fit_transform`) inside the pipeline.
6. Record a feature dictionary: name, source, transform, type, rationale, availability time. `mlkit.features.feature_dictionary()` writes it.

## Selection

| Family | Methods | Use when |
|---|---|---|
| Filter | variance threshold, mutual information, chi-squared, ANOVA F | Many features, quick screen |
| Wrapper | RFE, sequential selection | Moderate p, model-specific |
| Embedded | L1, elastic net, tree importance | Selection as part of fitting |
| Stability | Selection frequency across bootstrap resamples | Small n, need trustworthy sets |

Selection lives inside the pipeline and is evaluated with cross-validation. Remove leaky and near-duplicate features first.

## Ablation

Report the CV metric with and without each feature group (raw, ratios, lags, encodings) with fold spread. Keep groups whose gain exceeds the fold-to-fold noise.

## Output

- Pipeline code in `src/<package>/features.py`, imported by the notebook and the API.
- `artifacts/feature_dictionary.csv`.
- Ablation table under a findings heading.

Recipes with code: `references/feature-recipes.md`.
