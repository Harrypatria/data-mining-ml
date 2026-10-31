---
name: model-explainability
description: Model Explainability. Explain machine learning models and export reusable knowledge - explainable AI (XAI), feature importance (permutation, impurity caveats), SHAP values, partial dependence, ALE, ICE, LIME, surrogate trees and rules, counterfactual explanations, feature dictionary, insights.json, model card, data sheet, metric glossary, slice and fairness analysis. Use for interpretability, why did the model predict this, what drives the model, model documentation, model cards or responsible AI reviews.
---

# Model Explainability

Explain the model on held-out data, state what the explanation cannot say, and export the knowledge so people and systems can reuse it.

## Global

1. Permutation importance on the test or validation set, scored with the headline metric, repeated (10 or more) with intervals.
2. Model-native importance only with its caveat: impurity importance favors high-cardinality and continuous features.
3. SHAP summary and dependence plots (TreeExplainer for tree ensembles, LinearExplainer for linear models, model-agnostic otherwise). `mlkit.explain.shapley_values` computes interventional Shapley values on raw predictors for any pipeline.
4. Partial dependence for independent features; ALE when features are correlated.
5. Surrogate decision tree or extracted rules when a rule-based representation is requested; report its fidelity to the model.

## Local

SHAP waterfall or bar for one case, ICE curves, LIME as a cross-check, counterfactuals (`mlkit.explain.counterfactuals`: smallest change inside the training range that flips the decision).

## Caveats in every output

- Importance is not causation; SHAP explains the model, not the world.
- Correlated features split or distort attribution; use grouped importance or ALE.
- Explanations on training data flatter the model; use held-out data.
- Report stability: importance intervals across repeats or resamples.

## Knowledge artifacts (`artifacts/`)

| File | Contents |
|---|---|
| `feature_dictionary.csv` | name, meaning, source, transform, type, unit, rationale, availability, importance with interval, direction |
| `insights.json` | finding, evidence metric with interval, segment, confidence, recommended action |
| `model_card.md` | intended use, data, metrics by slice, limits, fairness checks, monitoring plan |
| `data_sheet.md` | collection, bias risks, known quality issues |
| `rules.md` | surrogate or extracted rules with fidelity, when requested |

Add a metric glossary cell that defines every metric used in plain language.

## Fairness

For people-facing decisions: recall, precision and calibration by group with intervals; name the fairness definition chosen and why; record gaps and mitigations in the model card.

Method choice table and code: `references/explainability.md`.
