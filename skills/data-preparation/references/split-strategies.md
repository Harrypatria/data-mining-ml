# Split strategies

| Data | Holdout | Cross-validation | Leakage it prevents | Code |
|---|---|---|---|---|
| Independent rows, balanced | Random 80/20 | `KFold(shuffle=True, random_state=SEED)` | None beyond chance | `train_test_split(X, y, test_size=0.2, random_state=SEED)` |
| Independent rows, classification | Stratified | `StratifiedKFold` | Unequal class ratios across folds | `train_test_split(..., stratify=y)` |
| Repeated entities (customers, patients, devices) | Group holdout | `GroupKFold`, `StratifiedGroupKFold` | Same entity in train and test | `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups)` |
| Time-ordered | Cutoff date (train before, test after) | `TimeSeriesSplit(n_splits=5, gap=h)` | Training on the future | `train = df[df.date < cutoff]` |
| Time-ordered with entities | Cutoff plus entity-aware features | Rolling origin by date | Future plus entity memory | Build features from data before each snapshot |
| Small n | Repeated stratified CV, no single holdout | `RepeatedStratifiedKFold(5, 10)` | Variance of a single split | Report mean and sd across repeats |
| Tuning plus honest estimate | Outer test or nested CV | Inner CV for search, outer CV for estimate | Optimistic search results | `cross_val_score(GridSearchCV(...), X, y, cv=outer)` |
| Spatial data | Region holdout | Spatial blocks | Autocorrelation between neighbors | Group by spatial block |

## Three-way split used by the templates

```python
X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=SEED)
```

Train: fit and CV. Validation: threshold, calibration, early stopping. Test: one final score.

## Time-series rules

- The gap between the last training timestamp and the first test timestamp is at least the label horizon.
- Lags and rolling windows use only past values (`shift(k)` with k above zero, trailing windows).
- Scale and impute per training window.
- Compare against seasonal naive and last-value baselines.

## Sanity checks

```python
assert set(train_ids).isdisjoint(test_ids)
assert y_train.mean() > 0 and y_test.sum() >= 30, "too few test positives for stable metrics"
print({name: (len(a), int(b.sum())) for name, a, b in [("train", X_train, y_train), ("test", X_test, y_test)]})
```
