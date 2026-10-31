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
# # Train, test once, refit and save: a pipeline the API can load without drifting from training
#
# Protocol: one stratified split into train, validation and test; one scikit-learn Pipeline holding every fitted step;
# tuning by cross-validation on train; threshold on validation; the test set scored once; a documented final refit;
# a pickle bundle with checksum, metadata, schema and background sample; and a reload parity check in a fresh process.

# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
MLKIT_SPEC = "git+https://github.com/Harrypatria/data-mining-ml@v1.0.0#subdirectory=templates"
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", MLKIT_SPEC], check=True)
    ROOT = Path("/content")
else:
    ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "mlkit").is_dir()), None)
    if ROOT is None:
        raise FileNotFoundError("src/mlkit not found in this folder or its parents; open the notebook inside the project.")
    sys.path.insert(0, str(ROOT / "src"))
OUT = Path(os.environ.get("DMML_OUTPUT_DIR", ROOT))
MODELS = OUT / "models"

# %matplotlib inline
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve

from mlkit import metrics as mx
from mlkit.data import make_customer_churn
from mlkit.features import FEATURE_INFO, TARGET, build_pipeline, feature_dictionary, split_xy
from mlkit.persistence import (check_versions, library_versions, load_bundle, parity_check, save_bundle,
                               verify_bundle)
from mlkit.workflow import AUDIENCE, LIMITATIONS, MODEL_PURPOSE, TARGET_LABEL, split_three

SEED = 2024
VERSION = "1.1.0"
MIN_PRECISION = 0.30   # review-capacity constraint from the manifest
print(json.dumps(library_versions()))

# %% [markdown]
# ## 1. Split once, seed recorded

# %%
df = make_customer_churn(8000, 0.12, SEED)
assert df["customer_id"].is_unique and not df.duplicated(subset=list(df.columns[1:])).any()
X, y = split_xy(df)
X_train, X_val, X_test, y_train, y_val, y_test = split_three(X, y, SEED)
print({name: (len(a), int(b.sum())) for name, a, b in [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]})

# %% [markdown]
# ## 2. One pipeline holds preprocessing, engineered ratios and the estimator
# The fitted pipeline is the only object saved, so serving runs the same code as training.

# %%
pipe = build_pipeline(LogisticRegression(max_iter=2000))
pipe

# %% [markdown]
# ## 3. Cross-validate on train against a baseline, then choose the threshold on validation

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_model = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision")
cv_base = cross_val_score(DummyClassifier(strategy="prior"), X_train, y_train, cv=cv, scoring="average_precision")
print(f"CV PR-AUC model {cv_model.mean():.4f} (sd {cv_model.std():.4f}) vs prior baseline {cv_base.mean():.4f}")
pipe.fit(X_train, y_train)
choice = mx.choose_threshold(y_val, pipe.predict_proba(X_val)[:, 1], min_precision=MIN_PRECISION)
THRESHOLD = choice["threshold"]
print(f"threshold {THRESHOLD:.2f} ({choice['rule']}), validation recall {choice['recall']:.3f}, precision {choice['precision']:.3f}")

# %% [markdown]
# ## 4. Score the test set once
# No tuning after this cell. If a change is made after seeing these numbers, say so and collect a fresh holdout.

# %%
p_test = pipe.predict_proba(X_test)[:, 1]
test_table = mx.metric_with_intervals(y_test, p_test, THRESHOLD, n_boot=500, seed=SEED)
print(test_table.round(4).to_string(index=False))

# %% [markdown]
# ## 5. Learning curve: would more data or a simpler model help?

# %%
sizes, train_scores, val_scores = learning_curve(pipe, X_train, y_train, cv=cv, scoring="average_precision",
                                                 train_sizes=np.linspace(0.1, 1.0, 6), shuffle=True, random_state=SEED)
gap = train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(sizes, train_scores.mean(axis=1), marker="o", label="training folds")
ax.plot(sizes, val_scores.mean(axis=1), marker="s", label="validation folds")
ax.fill_between(sizes, val_scores.mean(axis=1) - val_scores.std(axis=1), val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
ax.set_xlabel("training rows")
ax.set_ylabel("PR-AUC (average precision)")
ax.set_title(f"Train-validation gap at full size: {gap:.3f}")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
slope = val_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-2]
print("validation score still rising: more data may help" if slope > val_scores.std(axis=1)[-1] else
      "validation score has flattened: more data alone is unlikely to help")

# %% [markdown]
# ## 6. Final refit policy
# Refit the same configuration on train plus validation. The reported estimate stays the test result above; the
# shipped model saw more data than the tested one, and the metadata says so.

# %%
X_trval, y_trval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
final = clone(pipe).fit(X_trval, y_trval)

# %% [markdown]
# ## 7. Save the bundle
# `model.pkl` by default; `filename="model.sav"` writes the same pickle format under the other common name.
# `joblib.dump` is an alternative for very large NumPy-heavy models; keep the same bundle files around it.

# %%
bundle_dir = save_bundle(
    final, MODELS, X_trval, y_trval, version=VERSION, task="binary_classification", target=TARGET, classes=[0, 1],
    threshold=THRESHOLD, seed=SEED,
    metrics={r.metric: {"value": r.value, "low": r.low, "high": r.high} for r in test_table.itertuples()},
    feature_info=FEATURE_INFO, model_purpose=MODEL_PURPOSE, audience=AUDIENCE,
    calibration_note="Logistic regression without class weights; probabilities track observed rates on validation data.",
    feature_dictionary=feature_dictionary(),
    extra={"target_label": TARGET_LABEL, "limitations": LIMITATIONS, "threshold_rule": choice["rule"],
           "tested_model_note": "Test metrics come from the model fitted on train only; the shipped model was refitted on train plus validation."})
print(sorted(p.name for p in bundle_dir.iterdir()))
print("verify:", verify_bundle(bundle_dir) or "passed")
print(json.loads((bundle_dir / "schema.json").read_text())["features"][0])

# %% [markdown]
# ## 8. Reload parity in a fresh process, and the version guard

# %%
parity = parity_check(bundle_dir, X_test.sample(300, random_state=SEED), final)
assert parity["passed"], parity
print(parity)
bundle = load_bundle(MODELS, mode="development")
print("loaded", bundle.path.name, "threshold", bundle.metadata["threshold"])
pretend = {**bundle.metadata["libraries"], "scikit-learn": "0.24.2"}
print("version guard in development mode:", check_versions(pretend, mode="development"))

# %% [markdown]
# Security: unpickling runs arbitrary code. Load only bundles you produced, from storage you control, after the
# checksum check that `load_bundle` performs. Never load a model path that came from a request or an upload.
#
# ## Limitations
# - Simulated data and one model family; the protocol, not the numbers, is the point of this template.
# - The version guard compares major and minor versions; patch releases can still change numerical results slightly.
