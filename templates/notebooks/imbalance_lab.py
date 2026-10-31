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
# # Imbalance lab: as positives get rarer, accuracy rises while the model becomes less useful
#
# Every number in this notebook is simulated and computed when it runs. The lab shows, for one classifier and one
# data generator, how each metric behaves as prevalence falls, why the threshold is a decision, which remedies change
# what, and why resampled models need recalibration.
#
# **Confusion matrix vocabulary.** TP true positives, FP false positives, FN false negatives, TN true negatives.
#
# | Metric | Formula | Reads as |
# |---|---|---|
# | Precision | TP / (TP + FP) | share of flagged cases that are real |
# | Recall (sensitivity) | TP / (TP + FN) | share of real cases that are flagged |
# | Specificity | TN / (TN + FP) | share of negatives left alone |
# | F1 | 2 x precision x recall / (precision + recall) | harmonic mean, ignores TN |
# | F-beta | (1 + b^2) x P x R / (b^2 x P + R) | b above 1 weights recall |
# | Balanced accuracy | (recall + specificity) / 2 | accuracy with equal class weight |
# | MCC | (TP x TN - FP x FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | correlation of predictions and labels |
# | Cohen's kappa | agreement beyond chance | |

# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
MLKIT_SPEC = "git+https://github.com/Harrypatria/data-mining-ml@v1.0.0#subdirectory=templates"
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", MLKIT_SPEC, "imbalanced-learn==0.14.2"], check=True)
else:
    ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "mlkit").is_dir()), None)
    if ROOT is None:
        raise FileNotFoundError("src/mlkit not found in this folder or its parents; open the notebook inside the project.")
    sys.path.insert(0, str(ROOT / "src"))

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlkit import metrics as mx
from mlkit.data import make_imbalanced

SEED = 7
PREVALENCES = [0.5, 0.2, 0.05, 0.01, 0.001]
N_ROWS = 60000          # same size at every prevalence, so the rarest level has few positives on purpose
FALSE_ALARM_BUDGET = 0.05  # operating point: the threshold that flags 5 percent of negatives on validation
COST_FP, COST_FN = 1.0, 20.0

def split(X, y):
    assert len(np.unique(X, axis=0)) == len(X), "duplicated rows would leak across splits"
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.3, stratify=y_tr, random_state=SEED)
    return X_tr, X_va, X_te, y_tr, y_va, y_te

def threshold_at_fpr(y_val, p_val, fpr_target):
    fpr, _, thr = roc_curve(y_val, p_val)
    return float(thr[np.searchsorted(fpr, fpr_target, side="right") - 1])

# %% [markdown]
# ## The accuracy paradox: a model that never flags anyone

# %%
paradox = []
for prev in PREVALENCES:
    X, y = make_imbalanced(N_ROWS, prev, SEED)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(X, y)
    majority = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    m = mx.classification_metrics(y_te, majority.predict_proba(X_te)[:, 1], threshold=1.0)
    paradox.append({"prevalence": prev, "accuracy": m["accuracy"], "recall": m["recall"], "positives_in_test": m["positives"]})
paradox = pd.DataFrame(paradox)
print(paradox.round(4).to_string(index=False))
rare = paradox.iloc[-2]
display(Markdown(f"At {rare.prevalence:.1%} prevalence the never-flag model scores {rare.accuracy:.1%} accuracy and "
                 f"{rare.recall:.0%} recall: accuracy measures the class ratio, not the model."))

# %% [markdown]
# ## One classifier, five prevalence levels, every metric side by side
# Logistic regression, threshold set on validation data to keep false alarms at the same budget at every level.

# %%
rows, store = [], {}
for prev in PREVALENCES:
    X, y = make_imbalanced(N_ROWS, prev, SEED)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(X, y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
    t = threshold_at_fpr(y_va, model.predict_proba(X_va)[:, 1], FALSE_ALARM_BUDGET)
    p_te = model.predict_proba(X_te)[:, 1]
    m = mx.classification_metrics(y_te, p_te, t)
    rows.append({"prevalence": prev, "positives": m["positives"], "threshold": t, "accuracy": m["accuracy"],
                 "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "roc_auc": m["roc_auc"],
                 "pr_auc": m["pr_auc"], "pr_auc_no_skill": prev, "mcc": m["mcc"], "balanced_accuracy": m["balanced_accuracy"],
                 "brier": m["brier"]})
    store[prev] = (y_te, p_te, t)
sweep = pd.DataFrame(rows)
print(sweep.round(4).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
for col, style in [("accuracy", "-o"), ("recall", "-s"), ("precision", "-^"), ("roc_auc", "--o"), ("pr_auc", "--s"), ("f1", ":^")]:
    ax.plot(sweep["prevalence"], sweep[col], style, label=col.replace("_", "-").upper() if "auc" in col else col)
ax.plot(sweep["prevalence"], sweep["pr_auc_no_skill"], color="grey", linewidth=1, label="PR-AUC of a random model")
ax.set_xscale("log")
ax.set_xlabel("prevalence (share of positives, log scale)")
ax.set_ylabel("metric value on the test set")
ax.set_ylim(0, 1.02)
ax.set_title("Accuracy and ROC-AUC stay high as positives get rare; precision and PR-AUC fall")
ax.legend(frameon=False, ncol=2, fontsize=9)
plt.tight_layout()
plt.show()

# %%
hi, lo = sweep.iloc[0], sweep.iloc[-2]  # compare 50 percent with 1 percent; the 0.1 percent row is too small to read alone
def moved(a, b):
    return "rose" if b > a + 0.01 else "fell" if b < a - 0.01 else "barely moved"
findings = [
    f"- **Accuracy** {moved(hi.accuracy, lo.accuracy)} from {hi.accuracy:.3f} to {lo.accuracy:.3f} between {hi.prevalence:.0%} and {lo.prevalence:.0%} prevalence.",
    f"- **Recall** {moved(hi.recall, lo.recall)} ({hi.recall:.3f} to {lo.recall:.3f}); it is prevalence-independent in expectation, noisy with {int(lo.positives)} test positives, and blind to false alarms.",
    f"- **Precision** {moved(hi.precision, lo.precision)} from {hi.precision:.3f} to {lo.precision:.3f} at the same false-alarm rate: the base-rate effect.",
    f"- **ROC-AUC** {moved(hi.roc_auc, lo.roc_auc)} ({hi.roc_auc:.3f} to {lo.roc_auc:.3f}) while **PR-AUC** {moved(hi.pr_auc, lo.pr_auc)} "
    f"({hi.pr_auc:.3f} to {lo.pr_auc:.3f}, no-skill floor {lo.pr_auc_no_skill}). "
    + ("ROC-AUC hides the difficulty because its false-positive rate divides by a large negative class."
       if abs(lo.roc_auc - hi.roc_auc) < abs(lo.pr_auc - hi.pr_auc) else "In this run both moved; compare each with its no-skill floor."),
    f"- **F1** {moved(hi.f1, lo.f1)} ({hi.f1:.3f} to {lo.f1:.3f}); it ignores true negatives and depends on the threshold.",
    f"- At {sweep.iloc[-1].prevalence:.1%} prevalence there are {int(sweep.iloc[-1].positives)} test positives; every metric there is mostly noise.",
]
display(Markdown("\n".join(findings)))

# %% [markdown]
# ## Choose the headline metric by cost
#
# | Situation | Lead with | Because |
# |---|---|---|
# | Rare and costly misses (fraud, screening) | Recall at a fixed precision, PR-AUC, F-beta with beta above 1 | misses dominate cost |
# | Costly false alarms (manual review queue) | Precision at k | capacity is fixed |
# | Balanced costs | MCC or balanced accuracy | uses all four cells |
# | Ranking or targeting | Lift, gain, precision at k | only the top of the list is acted on |
# | Probabilities used directly | Log loss, Brier score, calibration curve | decisions use the probability itself |

# %% [markdown]
# ## The threshold is a decision: expected cost on validation data
# Prevalence of five percent; a miss costs twenty times a false alarm. The threshold is chosen on validation data and
# the test set only reports the result.

# %%
X, y = make_imbalanced(N_ROWS, 0.05, SEED)
X_tr, X_va, X_te, y_tr, y_va, y_te = split(X, y)
base_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
p_va, p_te = base_model.predict_proba(X_va)[:, 1], base_model.predict_proba(X_te)[:, 1]
table = mx.threshold_table(y_va, p_va, COST_FP, COST_FN)
chosen = mx.choose_threshold(y_va, p_va, cost_fp=COST_FP, cost_fn=COST_FN)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(table["recall"], table["precision"])
axes[0].scatter([chosen["recall"]], [chosen["precision"]], color="black", zorder=3, label=f"chosen threshold {chosen['threshold']:.2f}")
axes[0].set_xlabel("recall")
axes[0].set_ylabel("precision")
axes[0].set_title("Precision-recall trade-off (validation)")
axes[0].legend(frameon=False)
axes[1].plot(table["threshold"], table["expected_cost"])
axes[1].axvline(chosen["threshold"], color="black", linestyle="--")
axes[1].set_xlabel("decision threshold")
axes[1].set_ylabel("expected cost per case (cost units)")
axes[1].set_title("Expected cost is lowest well below the middle of the scale")
plt.tight_layout()
plt.show()
test_cost = mx.expected_cost(y_te, p_te, chosen["threshold"], COST_FP, COST_FN)
print(f"chosen on validation: {chosen['threshold']:.2f} ({chosen['rule']}); test expected cost {test_cost:.3f} per case")

# %% [markdown]
# ## Remedies, least to most invasive
# All resampling sits inside an imbalanced-learn pipeline, so it runs on training folds only. Scores: CV PR-AUC on
# the training data, then test recall and precision at a threshold chosen on validation (max recall with precision
# at least 0.3), Brier score, and the mean predicted probability compared with the true prevalence.

# %%
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import HistGradientBoostingClassifier

remedies = {
    "no remedy (threshold moved)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "class weights": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "random under-sampling": make_imb_pipeline(StandardScaler(), RandomUnderSampler(random_state=SEED), LogisticRegression(max_iter=2000)),
    "SMOTE": make_imb_pipeline(StandardScaler(), SMOTE(random_state=SEED), LogisticRegression(max_iter=2000)),
    "SMOTE-Tomek": make_imb_pipeline(StandardScaler(), SMOTETomek(random_state=SEED), LogisticRegression(max_iter=2000)),
    "balanced random forest": BalancedRandomForestClassifier(n_estimators=150, sampling_strategy="all", replacement=True,
                                                             bootstrap=False, random_state=SEED, n_jobs=-1),
    "boosting with class weights": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rem_rows, fitted = [], {}
for name, est in remedies.items():
    cv_scores = cross_val_score(est, X_tr, y_tr, cv=cv, scoring="average_precision")
    est.fit(X_tr, y_tr)
    fitted[name] = est
    t = mx.choose_threshold(y_va, est.predict_proba(X_va)[:, 1], min_precision=0.3)["threshold"]
    m = mx.classification_metrics(y_te, est.predict_proba(X_te)[:, 1], t)
    rem_rows.append({"remedy": name, "cv_pr_auc": cv_scores.mean(), "cv_sd": cv_scores.std(), "threshold": t,
                     "test_recall": m["recall"], "test_precision": m["precision"], "brier": m["brier"],
                     "mean_predicted": est.predict_proba(X_te)[:, 1].mean(), "true_prevalence": y_te.mean()})
remedy_table = pd.DataFrame(rem_rows)
print(remedy_table.round(4).to_string(index=False))
ref = remedy_table.iloc[0]
best_rem = remedy_table.loc[remedy_table["cv_pr_auc"].idxmax()]
resampled = remedy_table[remedy_table["remedy"].isin(["random under-sampling", "SMOTE", "SMOTE-Tomek"])]
display(Markdown(
    f"Best ranking in this run: **{best_rem.remedy}** (CV PR-AUC {best_rem.cv_pr_auc:.3f}, sd {best_rem.cv_sd:.3f}). "
    f"Resampling the logistic model {'improved' if resampled.cv_pr_auc.max() > ref.cv_pr_auc + ref.cv_sd else 'did not improve'} "
    f"ranking over simply moving the threshold ({ref.cv_pr_auc:.3f}), and it raised the mean predicted probability to "
    f"{resampled.mean_predicted.mean():.3f} against a true prevalence of {ref.true_prevalence:.3f}."))

# %% [markdown]
# What each remedy changes: class weights change the loss and keep the data; threshold moving changes the decision rule
# and keeps the model; resampling changes the training data inside each fold; balanced ensembles resample per tree.
# Resampling and reweighting push the mean predicted probability far above the true prevalence, which the next
# section repairs.

# %% [markdown]
# ## Recalibrate after resampling, on untouched validation data

# %%
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator

smote_model = fitted["SMOTE"]
calibrated = CalibratedClassifierCV(FrozenEstimator(smote_model), method="isotonic").fit(X_va, y_va)
curves = {}
for label, est in [("SMOTE, raw", smote_model), ("SMOTE, isotonic recalibrated", calibrated)]:
    p = est.predict_proba(X_te)[:, 1]
    curves[label] = (calibration_curve(y_te, p, n_bins=10, strategy="quantile"), mx.classification_metrics(y_te, p, chosen["threshold"])["brier"])
brier_raw, brier_cal = (b for _, b in curves.values())
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
for label, ((frac, mean_pred), brier) in curves.items():
    ax.plot(mean_pred, frac, marker="o", label=f"{label} (Brier {brier:.4f})")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("observed positive rate")
ax.set_title("Recalibration on validation lowers the Brier score" if brier_cal < brier_raw else "Recalibration did not lower the Brier score here")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Uncertainty on rare classes: recall intervals widen as positives disappear

# %%
unc = []
for prev, (y_te_p, p_te_p, t) in store.items():
    m = mx.classification_metrics(y_te_p, p_te_p, t)
    low, high = mx.wilson_interval(m["tp"], m["tp"] + m["fn"])
    unc.append({"prevalence": prev, "test_positives": m["positives"], "recall": m["recall"], "recall_low": low,
                "recall_high": high, "interval_width": high - low})
print(pd.DataFrame(unc).round(4).to_string(index=False))

# %% [markdown]
# ## Multi-class imbalance: which average hides minority failure

# %%
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, recall_score

Xm, ym = make_classification(n_samples=15000, n_features=12, n_informative=6, n_classes=3, n_clusters_per_class=1,
                             weights=[0.85, 0.12, 0.03], class_sep=0.8, random_state=SEED)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3, stratify=ym, random_state=SEED)
pred = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xm_tr, ym_tr).predict(Xm_te)
per_class = recall_score(ym_te, pred, average=None)
avgs = {avg: (recall_score(ym_te, pred, average=avg), f1_score(ym_te, pred, average=avg)) for avg in ("micro", "macro", "weighted")}
print("per-class recall:", {f"class {k} (n={int((ym_te == k).sum())})": round(float(v), 3) for k, v in enumerate(per_class)})
print(pd.DataFrame(avgs, index=["recall", "f1"]).T.round(3).to_string())
worst = int(np.argmin(per_class))
display(Markdown(f"Micro recall is {avgs['micro'][0]:.2f} and weighted recall {avgs['weighted'][0]:.2f}, close to the majority class; "
                 f"macro recall is {avgs['macro'][0]:.2f} because class {worst} is recalled only {per_class[worst]:.2f} of the time. "
                 "Report per-class recall whenever classes are imbalanced."))

# %% [markdown]
# ## Anti-pattern demonstration: SMOTE before the split inflates the score
# Deliberately wrong, to show the size of the error. Never copy this cell into real work.

# %%
X_leak, y_leak = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)  # ml-ignore L03 deliberate leakage demonstration
leaky = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X_leak, y_leak, cv=cv, scoring="average_precision")
honest = remedy_table.set_index("remedy").loc["SMOTE", "cv_pr_auc"]
print(f"CV PR-AUC with SMOTE before splitting: {leaky.mean():.3f}; with SMOTE inside the pipeline: {honest:.3f}")

# %% [markdown]
# Other anti-patterns the scanner catches: resampling the test set (L03), reporting accuracy or ROC-AUC alone (M02,
# M03), tuning the threshold on the test set (L08), and comparing models at different thresholds without saying so.
#
# ## Limitations
# - One generator and one model family; real data has label noise, drift and overlapping classes.
# - The rarest level has few positives by design, so its estimates are unstable; read the intervals, not the points.
# - Costs are illustrative; replace them with the cost matrix from ANALYSIS_MANIFEST.md.
