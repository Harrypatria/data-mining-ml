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
# *Story report template. The first output below is the takeaway, computed from the saved model bundle and a fresh
# labelled sample; every heading after it is a finding written by code from numbers that ran in this session.*

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

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from mlkit import metrics as mx
from mlkit.data import make_customer_churn
from mlkit.explainer import predict_frame
from mlkit.features import split_xy
from mlkit.workflow import load_or_train

SEED = 42
WEEKLY_CAPACITY = 0.10   # the team can call one customer in ten each week
bundle = load_or_train(OUT / "models", seed=SEED)
meta = bundle.metadata
fresh = make_customer_churn(6000, 0.15, SEED + 7)
X, y = split_xy(fresh)
p = predict_frame(bundle, X)
rule = ((X["plan"] == "basic") & (X["support_calls"] >= 3)).to_numpy()

k = int(WEEKLY_CAPACITY * len(X))
top = np.argsort(-p)[:k]
captured_model = y.to_numpy()[top].sum() / y.sum()
rule_top = np.flatnonzero(rule)[:k]
captured_rule = y.to_numpy()[rule_top].sum() / y.sum()
prec_model, prec_rule = mx.precision_at_k(y, p, k), y.to_numpy()[rule_top].mean()
test = meta["metrics"]
display(Markdown(
    f"# Calling the model's top {WEEKLY_CAPACITY:.0%} reaches {captured_model:.0%} of cancelling customers, "
    f"against {captured_rule:.0%} with the current rule"))

# %% [markdown]
# ## Situation, complication, resolution, action

# %%
better = captured_model > captured_rule
display(Markdown("\n".join([
    f"**Situation.** {meta['model_purpose'].capitalize()}. The retention team calls about {k} of {len(X)} customers a week.",
    f"**Complication.** {y.mean():.0%} of customers cancel within 30 days, so most calls under the current rule "
    f"reach customers who would have stayed: precision {prec_rule:.0%} in the top {k}.",
    f"**Resolution.** Ranking by the model {'raises' if prec_model > prec_rule else 'does not raise'} precision in the top {k} "
    f"(model {prec_model:.0%}, rule {prec_rule:.0%}). On the original test set the model's PR-AUC was "
    f"{test['pr_auc']['value']:.3f} (95% interval {test['pr_auc']['low']:.3f} to {test['pr_auc']['high']:.3f}).",
    (f"**Action.** Replace the rule with the model ranking for a four-week pilot, measure retention against a randomly "
     f"held-out control group, and keep the stored threshold of {meta['threshold']:.2f} for alerts." if better else
     "**Action.** Keep the current rule; the model does not reach more cancelling customers at this capacity."),
])))

# %% [markdown]
# ## Evidence

# %%
order = np.argsort(-p)
gain = np.cumsum(y.to_numpy()[order]) / y.sum()
contacted = np.arange(1, len(p) + 1) / len(p)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(contacted, gain, label="model ranking")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random calls")
ax.scatter([WEEKLY_CAPACITY], [captured_rule], color="black", zorder=3, label="current rule at capacity")
ax.axvline(WEEKLY_CAPACITY, color="grey", linewidth=0.8)
ax.set_xlabel("share of customers called")
ax.set_ylabel("share of cancelling customers reached")
ax.set_title(f"At {WEEKLY_CAPACITY:.0%} capacity the model reaches {captured_model:.0%} of cancellations")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
imp = pd.DataFrame(meta.get("global_importance", [])).sort_values("importance")
display(Markdown(f"## {imp.iloc[-1]['feature'].replace('_', ' ').capitalize()} carries the most ranking signal"))
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.barh(imp["feature"], imp["importance"])
ax.set_xlabel("drop in average precision when the predictor is shuffled (test set)")
ax.set_ylabel("predictor")
ax.set_title("What the model relies on (associations, not causes)")
plt.tight_layout()
plt.show()

# %%
by_plan = pd.DataFrame([{"plan": g, "customers": int(m.sum()), "cancel_rate": float(y[m].mean()),
                         "mean_score": float(p[m.to_numpy()].mean())} for g, m in ((g, X["plan"] == g) for g in sorted(X["plan"].unique()))])
worst = by_plan.sort_values("cancel_rate").iloc[-1]
display(Markdown(f"## Customers on the {worst.plan} plan cancel most often ({worst.cancel_rate:.0%}), and the model scores them highest"))
print(by_plan.round(3).to_string(index=False))

# %% [markdown]
# ## Metric glossary
# - **Precision**: of the customers we call, the share who were about to cancel.
# - **Recall**: of the customers who cancel, the share we called.
# - **PR-AUC (average precision)**: how well the ranking puts cancelling customers first, across all list lengths.
# - **Threshold**: the score above which a customer is flagged for a call.
# - **Interval**: the range the true value probably lies in, given the sample size.

# %% [markdown]
# ## Recommended actions, risks and open questions
# - **Do**: pilot the model ranking with a randomized control group so the effect of calls can be measured.
# - **Risk**: the model ranks risk; it does not show that a call prevents cancellation.
# - **Risk**: prevalence or behavior can shift; monitor PSI per predictor and recall on labelled outcomes monthly.
# - **Open question**: what is the cost of a call compared with the margin of a retained customer? It sets the threshold.
# - **Next step**: rerun this report on real snapshots with a forward-chaining split.
#
# ## Limitations
# Results come from simulated data generated with known effects; real customers will differ. Explanations describe
# associations the model learned, not causes.
