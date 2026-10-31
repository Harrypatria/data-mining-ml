# %% [markdown]
# # Logistic pipeline beats the prior baseline on simulated retention data
# Decision: whether the pipeline is good enough to rank customers for outreach.
#
# Limitations: simulated data, one model family, intervals from bootstrap resamples.

# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 7
rng = np.random.default_rng(SEED)
n = 2000
frame = pd.DataFrame({"spend": rng.lognormal(3, 0.5, n), "plan": rng.choice(["basic", "plus"], n)})
frame["target"] = (rng.random(n) < 0.1 + 0.1 * (frame["plan"] == "basic")).astype(int)
print(frame.shape, int(frame.isna().sum().sum()), int(frame.duplicated().sum()))
assert frame["spend"].between(0, 1e6).all()

# %%
X, y = frame[["spend", "plan"]], frame["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=SEED)
pre = ColumnTransformer([("num", StandardScaler(), ["spend"]), ("cat", OneHotEncoder(handle_unknown="ignore"), ["plan"])])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
baseline = DummyClassifier(strategy="prior")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for name, est in {"baseline": baseline, "model": model}.items():
    scores = cross_val_score(est, X_train, y_train, cv=cv, scoring="average_precision")
    print(f"{name}: CV average precision {scores.mean():.3f} (sd {scores.std():.3f})")

# %%
model.fit(X_train, y_train)
p_test = model.predict_proba(X_test)[:, 1]
boot = []
for _ in range(200):
    idx = rng.integers(0, len(y_test), len(y_test))
    boot.append(average_precision_score(y_test.iloc[idx], p_test[idx]))
low, high = np.percentile(boot, [2.5, 97.5])
print(f"Test average precision {average_precision_score(y_test, p_test):.3f}, 95% bootstrap interval {low:.3f} to {high:.3f}")
print(f"Positives in test: {int(y_test.sum())}")
