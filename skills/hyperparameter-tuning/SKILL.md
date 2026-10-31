---
name: hyperparameter-tuning
description: Hyperparameter Tuning. Tune machine learning models honestly - grid search, random search, successive halving, Hyperband, Bayesian optimization with Optuna and pruning, nested cross-validation, early stopping on validation folds, search spaces for logistic regression, random forest, XGBoost, LightGBM, SVM, k-NN and neural networks, compute budgets, reporting spread and parameter importance. Use for hyperparameter optimization, GridSearchCV, RandomizedSearchCV, Optuna, model tuning or improving a model's settings.
---

# Hyperparameter Tuning

## Order of work

1. Fix the metric (aligned with cost: PR-AUC, F-beta, log loss, MAE), the split strategy and the baseline.
2. Score the default configuration with the same CV. Tuning gains are measured against it.
3. Tune inside cross-validation on training data only. The test set never influences tuning, early stopping or model choice.
4. Tune the decision threshold afterwards, as a separate step on validation data.

## Method choice

| Situation | Method |
|---|---|
| Two or three parameters, cheap model | `GridSearchCV` |
| Default choice | `RandomizedSearchCV` with log-uniform ranges, 30 to 100 trials |
| Large space, cheap early signal | `HalvingRandomSearchCV`, Hyperband |
| Expensive models | Optuna (TPE) with median pruning, CV inside the objective |
| Small dataset, honest estimate needed | Nested CV: inner search, outer score |

## Rules

- Log-scale ranges for learning rates, regularization strengths and `C`.
- Early stopping on a validation fold carved from training data, never on test.
- Fix the seed of the search and of the model; record the budget (trials, wall time, hardware).
- With many trials, the best CV score is optimistic (validation overfitting). Confirm on the untouched test set once, or use nested CV.
- Differences smaller than the fold-to-fold standard deviation are not improvements.

## Report

Best mean CV score with its standard deviation across folds, the default-configuration score, the baseline score, the best parameters, a parameter importance or slice plot, the budget and the seed. Spaces and code: `references/tuning.md`.
