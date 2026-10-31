# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Should the retention team use a churn model to choose its weekly outreach list?
#
# **Decision.** Each week the team can call a limited number of customers. This notebook tests whether a model ranks
# likely cancellations better than the current rule (call everyone on the basic plan with three or more support calls),
# at a precision the team can afford. The computed answer is in the *Decision summary* cell near the end; the story
# report template moves it to the top for stakeholders.
#
# CRISP-DM spine: business understanding, data understanding, preparation, modeling, evaluation, deployment.
# Data: simulated by `mlkit.data.make_customer_churn` at run time, so every number below comes from this run.

# %% [markdown]
# ## Environment, seed and paths

# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
SAVE_TO_DRIVE = False  # Colab only: set True to keep models and artifacts after the runtime resets
MLKIT_SPEC = "git+https://github.com/Harrypatria/data-mining-ml@v1.0.0#subdirectory=templates"

if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", MLKIT_SPEC], check=True)
    ROOT = Path("/content")
    if SAVE_TO_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        ROOT = Path("/content/drive/MyDrive/data-mining-ml")
else:
    ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "mlkit").is_dir()), None)
    if ROOT is None:
        raise FileNotFoundError("src/mlkit not found in this folder or its parents; open the notebook inside the project.")
    sys.path.insert(0, str(ROOT / "src"))

OUT = Path(os.environ.get("DMML_OUTPUT_DIR", ROOT))
MODELS, ARTIFACTS = OUT / "models", OUT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# %matplotlib inline
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from mlkit import metrics as mx
from mlkit.persistence import library_versions

pd.set_option("display.max_columns", 20)
print(json.dumps(library_versions()))

# %% [markdown]
# ## Configuration
# Every constant that encodes a decision lives here, each with its reason.

# %%
SEED = 42                 # one seed for every splitter, estimator and resampler
N_CUSTOMERS = 8000        # simulated customers, one row each
PREVALENCE = 0.12         # expected share of customers who cancel within 30 days
COST_FP = 15.0            # cost of an unnecessary retention call, USD
COST_FN = 120.0           # margin lost when a cancelling customer is not called, USD
MIN_PRECISION = 0.30      # the team stops calling when fewer than 3 in 10 calls reach a real risk
VERSION = "1.0.0"

# %% [markdown]
# ## Data understanding: load and check the schema

# %%
from mlkit.data import make_customer_churn
from mlkit.features import CATEGORICAL, NUMERIC, PREDICTORS, TARGET

df = make_customer_churn(N_CUSTOMERS, PREVALENCE, SEED)
expected = {"customer_id", *PREDICTORS, TARGET}
assert set(df.columns) == expected, set(df.columns) ^ expected
assert df["customer_id"].is_unique, "one row per customer expected"
assert df["tenure_months"].between(1, 72).all() and (df["monthly_spend"] > 0).all()
print(df.shape)
print(df.dtypes.to_string())
df.head()

# %% [markdown]
# ## Data understanding: missingness, duplicates and target balance

# %%
profile = pd.DataFrame({
    "missing_share": df[PREDICTORS].isna().mean(),
    "missing_share_if_churned": df.loc[df[TARGET] == 1, PREDICTORS].isna().mean(),
})
print(profile.round(4).to_string())
print(f"duplicated rows: {int(df.duplicated(subset=PREDICTORS + [TARGET]).sum())}")
print(f"target prevalence: {df[TARGET].mean():.3f} ({int(df[TARGET].sum())} positives of {len(df)})")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, col in zip(axes, ["monthly_spend", "tenure_months"]):
    for label, part in df.groupby(TARGET):
        ax.hist(part[col], bins=40, alpha=0.6, density=True, label="cancelled" if label else "stayed")
    ax.set_xlabel(col.replace("_", " ") + (" (USD)" if col == "monthly_spend" else " (months)"))
    ax.set_ylabel("density")
    ax.legend(frameon=False)
axes[0].set_title("Spend is right-skewed; winsorize and scale inside the pipeline")
axes[1].set_title("Cancellations concentrate in short tenure")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Data preparation: leakage-safe split
# Stratified train, validation and test splits (60, 20, 20). One row per customer, so a random stratified split is
# group-safe. The test set is scored once, in the evaluation section.

# %%
from mlkit.features import split_xy
from mlkit.workflow import split_three

X, y = split_xy(df)
X_train, X_val, X_test, y_train, y_val, y_test = split_three(X, y, SEED)
for name, part in [("train", y_train), ("validation", y_val), ("test", y_test)]:
    print(f"{name:<10} rows {len(part):>5}  positives {int(part.sum()):>4}  prevalence {part.mean():.3f}")
assert len(X_train) + len(X_val) + len(X_test) == len(df)

# %% [markdown]
# ## Modeling: baselines first
# The current business rule and a prior-probability dummy set the bar. Scores are cross-validated PR-AUC (average
# precision) on the training data, because positives are rare and costly to miss.

# %%
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from mlkit.features import build_pipeline

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rule_score = ((X_val["plan"] == "basic") & (X_val["support_calls"] >= 3)).astype(float)
rule = mx.classification_metrics(y_val, rule_score, threshold=1.0)
print(f"current rule on validation: precision {rule['precision']:.3f}, recall {rule['recall']:.3f}, flagged {rule['tp'] + rule['fp']}")

candidates = {
    "dummy_prior": DummyClassifier(strategy="prior"),
    "logistic": build_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
    "gradient_boosting": build_pipeline(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED)),
}
cv_rows = []
for name, est in candidates.items():
    s = cross_val_score(est, X_train, y_train, cv=cv, scoring="average_precision")
    cv_rows.append({"model": name, "cv_pr_auc_mean": s.mean(), "cv_pr_auc_sd": s.std()})
cv_table = pd.DataFrame(cv_rows).sort_values("cv_pr_auc_mean", ascending=False)
print(cv_table.round(4).to_string(index=False))

# %% [markdown]
# ## Modeling: tune the best family inside cross-validation
# Random search, scored on average precision, 20 trials, seed fixed. The test set plays no part.

# %%
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV

spaces = {
    "logistic": {"model__C": loguniform(1e-3, 10.0)},
    "gradient_boosting": {"model__learning_rate": loguniform(0.02, 0.3), "model__max_leaf_nodes": randint(8, 48),
                          "model__min_samples_leaf": randint(10, 80), "model__l2_regularization": loguniform(1e-4, 1.0)},
}
family = cv_table[cv_table["model"] != "dummy_prior"].iloc[0]["model"]
search = RandomizedSearchCV(candidates[family], spaces[family], n_iter=20, scoring="average_precision", cv=cv,
                            random_state=SEED, n_jobs=-1)
search.fit(X_train, y_train)
trials = pd.DataFrame(search.cv_results_)
best = trials.loc[search.best_index_]
default_score = cv_table.set_index("model").loc[family, "cv_pr_auc_mean"]
print(f"tuned family: {family}")
print(f"best CV PR-AUC {best.mean_test_score:.4f} (sd {best.std_test_score:.4f}); default configuration {default_score:.4f}")
print({k.replace("model__", ""): (round(v, 4) if isinstance(v, float) else v) for k, v in search.best_params_.items()})
gain = best.mean_test_score - default_score
print("gain is within fold noise" if abs(gain) < best.std_test_score else "gain exceeds one fold standard deviation")

# %% [markdown]
# ## Evaluation: threshold on validation, calibration, then the test set once

# %%
from sklearn.calibration import calibration_curve

model = search.best_estimator_
p_val = model.predict_proba(X_val)[:, 1]
choice = mx.choose_threshold(y_val, p_val, min_precision=MIN_PRECISION)
cost_choice = mx.choose_threshold(y_val, p_val, cost_fp=COST_FP, cost_fn=COST_FN)
THRESHOLD = choice["threshold"]
print(f"threshold {THRESHOLD:.2f} by rule '{choice['rule']}'; cost-optimal threshold would be {cost_choice['threshold']:.2f}")

frac_pos, mean_pred = calibration_curve(y_val, p_val, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
ax.plot(mean_pred, frac_pos, marker="o", label="model")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("observed cancellation rate")
ax.set_title(f"Calibration on validation (Brier {mx.classification_metrics(y_val, p_val, THRESHOLD)['brier']:.3f})")
ax.legend(frameon=False)
plt.show()

# %%
p_test = model.predict_proba(X_test)[:, 1]
test_table = mx.metric_with_intervals(y_test, p_test, THRESHOLD, n_boot=500, seed=SEED)
print(test_table.round(4).to_string(index=False))
rule_score_test = ((X_test["plan"] == "basic") & (X_test["support_calls"] >= 3)).astype(float)
rule_test = mx.classification_metrics(y_test, rule_score_test, 1.0)
model_test = mx.classification_metrics(y_test, p_test, THRESHOLD)
print(f"test positives {model_test['positives']}; model flags {model_test['tp'] + model_test['fp']}, rule flags {rule_test['tp'] + rule_test['fp']}")
print(f"expected cost per customer: model {mx.expected_cost(y_test, p_test, THRESHOLD, COST_FP, COST_FN):.2f} USD, "
      f"rule {mx.expected_cost(y_test, rule_score_test, 1.0, COST_FP, COST_FN):.2f} USD")

# %% [markdown]
# ## Evaluation: slices
# Same threshold, test set. Wide intervals on small slices are expected; the positive count is shown.

# %%
from mlkit.workflow import slice_metrics

slices = pd.concat([slice_metrics(y_test, p_test, THRESHOLD, X_test["plan"]),
                    slice_metrics(y_test, p_test, THRESHOLD, X_test["region"])], ignore_index=True)
print(slices.round(3).to_string(index=False))

# %% [markdown]
# ## Deployment prep: final refit and persistence with a reload parity check
# Policy: refit the chosen configuration on train plus validation, keep the test result above as the reported
# estimate, and record that the shipped model saw more data than the tested one.

# %%
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from mlkit.features import FEATURE_INFO, feature_dictionary
from mlkit.persistence import parity_check, save_bundle, verify_bundle
from mlkit.workflow import AUDIENCE, LIMITATIONS, MODEL_PURPOSE, TARGET_LABEL, render_model_card

perm = permutation_importance(model, X_test, y_test, scoring="average_precision", n_repeats=10, random_state=SEED)
importance = pd.DataFrame({"feature": X_test.columns, "importance_mean": perm.importances_mean,
                           "importance_low": np.percentile(perm.importances, 2.5, axis=1),
                           "importance_high": np.percentile(perm.importances, 97.5, axis=1)}).sort_values("importance_mean", ascending=False)
X_trval, y_trval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
final_model = clone(model).fit(X_trval, y_trval)
card = render_model_card({"name": "churn-risk", "version": VERSION, "purpose": MODEL_PURPOSE, "audience": AUDIENCE,
                          "rows": len(X_trval), "prevalence": float(y_trval.mean()), "source": "simulated",
                          "threshold": THRESHOLD, "rule": choice["rule"], "limitations": LIMITATIONS}, slices, test_table)
bundle_dir = save_bundle(
    final_model, MODELS, X_trval, y_trval, version=VERSION, task="binary_classification", target=TARGET, classes=[0, 1],
    threshold=THRESHOLD, metrics={r.metric: {"value": r.value, "low": r.low, "high": r.high} for r in test_table.itertuples()},
    seed=SEED, cost_matrix={"fp": COST_FP, "fn": COST_FN}, feature_info=FEATURE_INFO, model_purpose=MODEL_PURPOSE,
    audience=AUDIENCE, calibration_note="Gradient boosting probabilities; see the validation calibration curve.",
    model_card=card, feature_dictionary=feature_dictionary(importance),
    extra={"target_label": TARGET_LABEL, "limitations": LIMITATIONS, "threshold_rule": choice["rule"],
           "global_importance": [{"feature": r.feature, "importance": float(r.importance_mean)} for r in importance.itertuples()]})
print(bundle_dir, verify_bundle(bundle_dir) or "bundle verified")
parity = parity_check(bundle_dir, X_test.head(200), final_model)
assert parity["passed"], parity
print(parity)

# %% [markdown]
# ## Explainability: global and local, on held-out data
# Permutation importance on the test set with 95 percent intervals over repeats. Importance describes what the model
# relies on, not what moves customers; correlated predictors share credit.

# %%
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.barh(importance["feature"], importance["importance_mean"],
        xerr=[importance["importance_mean"] - importance["importance_low"], importance["importance_high"] - importance["importance_mean"]])
ax.invert_yaxis()
ax.set_xlabel("drop in average precision when shuffled")
ax.set_ylabel("predictor")
ax.set_title(f"{importance.iloc[0]['feature']} carries the most ranking signal")
plt.tight_layout()
plt.show()

# %%
from mlkit.explain import local_explanation

background = X_trval.sample(100, random_state=SEED)
highest = X_test.assign(p=p_test).sort_values("p", ascending=False).head(1)[X_test.columns]
local = local_explanation(lambda d: final_model.predict_proba(d)[:, 1], highest, background)
print(f"highest-risk test profile: predicted {local['prediction']:.3f}, background mean {local['base_value']:.3f}")
print(pd.DataFrame(local["contributions"]).round(4).to_string(index=False))

# %% [markdown]
# ## Decision summary

# %%
beats_rule = model_test["recall"] > rule_test["recall"] and model_test["precision"] >= MIN_PRECISION
display(Markdown(
    f"**{'Use the model' if beats_rule else 'Keep the current rule'} for the weekly list.** On the untouched test set, at threshold "
    f"{THRESHOLD:.2f}, the model finds {model_test['recall']:.0%} of cancelling customers at {model_test['precision']:.0%} precision, "
    f"against {rule_test['recall']:.0%} recall and {rule_test['precision']:.0%} precision for the current rule "
    f"({model_test['positives']} positives in the test set)."))

# %% [markdown]
# ## Limitations and next steps
# - Simulated data: the effect sizes are built into the generator, so real data will score differently.
# - Associations, not causes: a predictor that raises the risk score is not proven to cause cancellations.
# - Thresholds move with prevalence; re-choose the threshold on fresh validation data after each retrain.
# - Next: run on real snapshots with a forward-chaining split, add a monitoring job (PSI per predictor), and run an
#   outreach experiment to measure whether calls change outcomes.

# %% [markdown]
# ## Artifact bundle export

# %%
insights = [
    {"finding": f"Model recall at the chosen threshold is {model_test['recall']:.3f}",
     "evidence": {"metric": "recall", **test_table.set_index("metric").loc["recall", ["value", "low", "high"]].round(4).to_dict()},
     "segment": "all test customers", "confidence": "interval from Wilson score",
     "recommended_action": "use the model for the weekly list" if beats_rule else "keep the current rule"},
    {"finding": f"{importance.iloc[0]['feature']} is the strongest ranking signal",
     "evidence": {"metric": "permutation importance", "value": round(float(importance.iloc[0]["importance_mean"]), 4)},
     "segment": "all test customers", "confidence": "10 permutation repeats",
     "recommended_action": "check data quality for this predictor first"},
]
(ARTIFACTS / "insights.json").write_text(json.dumps(insights, indent=2), encoding="utf-8")
feature_dictionary(importance).to_csv(ARTIFACTS / "feature_dictionary.csv", index=False)
(ARTIFACTS / "data_sheet.md").write_text(
    "# Data sheet\n\n- Collection: simulated by mlkit.data.make_customer_churn with a fixed seed.\n"
    "- Bias risks: no protected attributes simulated; region is a business segment, not a demographic.\n"
    "- Known quality issues: usage_hours is missing at random, more often on the basic plan.\n", encoding="utf-8")
print(sorted(p.name for p in ARTIFACTS.iterdir()), sorted(p.name for p in bundle_dir.iterdir()))
