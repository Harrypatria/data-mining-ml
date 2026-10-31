# Tuning reference

## Search spaces (pipeline step named `model`)

| Algorithm | Parameter | Range | Scale |
|---|---|---|---|
| Logistic regression | `C` | 1e-4 to 1e2 | log |
| | `penalty` | l2, l1 (solver liblinear or saga), elasticnet with `l1_ratio` 0 to 1 | categorical |
| Random forest | `n_estimators` | 200 to 1000 | linear |
| | `max_depth` | None, 4 to 32 | linear |
| | `min_samples_leaf` | 1 to 50 | log |
| | `max_features` | sqrt, 0.2 to 1.0 | linear |
| HistGradientBoosting | `learning_rate` | 0.01 to 0.3 | log |
| | `max_leaf_nodes` | 8 to 64 | log |
| | `min_samples_leaf` | 10 to 200 | log |
| | `l2_regularization` | 1e-4 to 10 | log |
| XGBoost | `learning_rate` (eta) | 0.01 to 0.3 | log |
| | `max_depth` | 3 to 10 | linear |
| | `min_child_weight` | 1 to 20 | log |
| | `subsample`, `colsample_bytree` | 0.5 to 1.0 | linear |
| | `reg_lambda`, `reg_alpha` | 1e-3 to 10 | log |
| LightGBM | `num_leaves` | 15 to 255 | log |
| | `min_child_samples` | 5 to 100 | log |
| | `feature_fraction`, `bagging_fraction` | 0.5 to 1.0 | linear |
| SVM (RBF) | `C` | 1e-2 to 1e3 | log |
| | `gamma` | 1e-4 to 1 | log |
| k-NN | `n_neighbors` | 3 to 75 | log |
| | `weights` | uniform, distance | categorical |
| MLP | `hidden_layer_sizes` | (32,), (64,), (64, 32), (128, 64) | categorical |
| | `alpha` | 1e-6 to 1e-2 | log |
| | `learning_rate_init` | 1e-4 to 1e-2 | log |

## Random search

```python
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(pipe, {"model__learning_rate": loguniform(0.01, 0.3), "model__max_leaf_nodes": randint(8, 64)},
                            n_iter=50, scoring="average_precision", cv=cv, random_state=SEED, n_jobs=-1)
search.fit(X_train, y_train)
res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
print(res[["mean_test_score", "std_test_score", "params"]].head())
```

## Optuna with pruning

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {"model__learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
              "model__max_leaf_nodes": trial.suggest_int("leaves", 8, 64, log=True)}
    scores = []
    for step, (tr, va) in enumerate(cv.split(X_train, y_train)):
        model = clone(pipe).set_params(**params).fit(X_train.iloc[tr], y_train.iloc[tr])
        scores.append(average_precision_score(y_train.iloc[va], model.predict_proba(X_train.iloc[va])[:, 1]))
        trial.report(np.mean(scores), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study.optimize(objective, n_trials=60, timeout=1800)
optuna.importance.get_param_importances(study)
```

## Nested CV

```python
outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
inner = StratifiedKFold(3, shuffle=True, random_state=SEED + 1)
nested = cross_val_score(RandomizedSearchCV(pipe, space, n_iter=30, cv=inner, scoring="average_precision", random_state=SEED),
                         X, y, cv=outer, scoring="average_precision")
print(f"nested CV PR-AUC {nested.mean():.3f} (sd {nested.std():.3f})")
```

## Early stopping

```python
X_fit, X_es, y_fit, y_es = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)
model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)])        # XGBoost or LightGBM; never the test set
HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=SEED)
```

## Reporting template

| Item | Value |
|---|---|
| Metric and CV | average precision, 5-fold stratified |
| Baseline | from the same CV |
| Default configuration | mean (sd) |
| Best configuration | mean (sd), parameters |
| Gain versus default | difference and whether it exceeds one sd |
| Budget | trials, wall time, hardware, seed |
