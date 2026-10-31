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
# # What-if dashboard: move the threshold, the costs and the customer profile, and see the model respond
#
# Seven core-ipywidgets panels that work in Jupyter, JupyterLab, VS Code and Colab: threshold explorer, prevalence
# explorer, what-if analyzer, cost-matrix editor, cohort filter, scenario comparison and sensitivity view. Every
# panel is a thin wrapper around a pure function; the last cell is the static fallback that prints the same views for
# nbconvert, GitHub previews and PDF exports. Callbacks never refit a model.

# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
MLKIT_SPEC = "git+https://github.com/Harrypatria/data-mining-ml@v1.0.0#subdirectory=templates"
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", MLKIT_SPEC, "imbalanced-learn==0.14.2"], check=True)
    ROOT = Path("/content")
else:
    ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "mlkit").is_dir()), None)
    if ROOT is None:
        raise FileNotFoundError("src/mlkit not found in this folder or its parents; open the notebook inside the project.")
    sys.path.insert(0, str(ROOT / "src"))
OUT = Path(os.environ.get("DMML_OUTPUT_DIR", ROOT))

# %matplotlib inline
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from mlkit import metrics as mx
from mlkit.data import make_customer_churn, make_imbalanced
from mlkit.explain import local_explanation, sensitivity_curve
from mlkit.features import FEATURE_INFO, split_xy
from mlkit.persistence import coerce_frame
from mlkit.workflow import load_or_train

SEED = 42
bundle = load_or_train(OUT / "models", seed=SEED)
SCHEMA, META = bundle.schema, bundle.metadata
FEATURES = {f["name"]: f for f in SCHEMA["features"]}
BACKGROUND = bundle.background.head(100)

def predict(frame: pd.DataFrame) -> np.ndarray:
    return bundle.pipeline.predict_proba(coerce_frame(frame, SCHEMA))[:, 1]

# A fresh simulated sample the model never saw, labelled, standing in for recent validation data.
fresh = make_customer_churn(4000, 0.15, SEED + 1)
X_fresh, y_fresh = split_xy(fresh)
P_FRESH = predict(X_fresh)
MEDIAN_PROFILE = {f["name"]: f["default"] for f in SCHEMA["features"]}
STATE = {"threshold": float(META["threshold"]), "cost_fp": 15.0, "cost_fn": 120.0,
         "profile": dict(MEDIAN_PROFILE), "scenarios": {}, "last_feature": "tenure_months"}
print(f"model v{META['model_version']} ({META.get('model_family', 'pipeline')}), stored threshold {STATE['threshold']:.2f}, "
      f"fresh sample {len(X_fresh)} rows with {int(y_fresh.sum())} positives")

# %% [markdown]
# ## 1. Threshold explorer

# %%
def threshold_view(threshold: float) -> dict:
    m = mx.classification_metrics(y_fresh, P_FRESH, threshold)
    cm = pd.DataFrame([[m["tp"], m["fn"]], [m["fp"], m["tn"]]], index=["actual cancel", "actual stay"],
                      columns=["flagged", "not flagged"])
    summary = {k: round(m[k], 4) for k in ("precision", "recall", "f1")}
    summary["expected_cost_usd"] = round(mx.expected_cost(y_fresh, P_FRESH, threshold, STATE["cost_fp"], STATE["cost_fn"]), 2)
    return {"confusion": cm, "summary": summary, "curve": mx.threshold_table(y_fresh, P_FRESH), "metrics": m}

def draw_threshold(threshold: float):
    view = threshold_view(threshold)
    STATE["threshold"] = threshold
    display(view["confusion"])
    print(view["summary"])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(view["curve"]["recall"], view["curve"]["precision"])
    ax.scatter([view["metrics"]["recall"]], [view["metrics"]["precision"]], color="black", zorder=3)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Threshold {threshold:.2f}: precision {view['metrics']['precision']:.2f}, recall {view['metrics']['recall']:.2f}")
    plt.show()

t_slider = widgets.FloatSlider(value=STATE["threshold"], min=0.01, max=0.99, step=0.01, continuous_update=False,
                               description="Threshold", tooltip="Decision threshold applied to the predicted probability",
                               layout=widgets.Layout(width="420px"))
display(widgets.VBox([t_slider, widgets.interactive_output(draw_threshold, {"threshold": t_slider})]))

# %% [markdown]
# ## 2. Prevalence and imbalance explorer
# Results for a grid of prevalence levels and remedies are computed once here; the controls only select from them.

# %%
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GRID_PREV = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
STRATEGIES = {
    "none": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "class weights": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "SMOTE in pipeline": lambda: make_imb_pipeline(StandardScaler(), SMOTE(random_state=SEED), LogisticRegression(max_iter=2000)),
}
prev_rows = []
for prev in GRID_PREV:
    Xg, yg = make_imbalanced(12000, prev, SEED)
    assert len(np.unique(Xg, axis=0)) == len(Xg), "duplicated rows would leak across the split"
    Xa, Xb, ya, yb = train_test_split(Xg, yg, test_size=0.4, stratify=yg, random_state=SEED)
    Xb_val, Xb_test, yb_val, yb_test = train_test_split(Xb, yb, test_size=0.5, stratify=yb, random_state=SEED)
    for name, make in STRATEGIES.items():
        fitted = make().fit(Xa, ya)  # ml-ignore M08 explorer shows single-split point estimates; intervals are in the imbalance lab
        t = mx.choose_threshold(yb_val, fitted.predict_proba(Xb_val)[:, 1])["threshold"]
        m = mx.classification_metrics(yb_test, fitted.predict_proba(Xb_test)[:, 1], t)
        prev_rows.append({"prevalence": prev, "strategy": name, "positives": m["positives"], **{k: m[k] for k in ("accuracy", "recall", "precision", "pr_auc", "roc_auc", "brier")}})
PREV_TABLE = pd.DataFrame(prev_rows)

def prevalence_view(prevalence: float, strategy: str) -> pd.DataFrame:
    return PREV_TABLE[(PREV_TABLE.prevalence == prevalence) & (PREV_TABLE.strategy == strategy)].round(4)

def draw_prevalence(prevalence: float, strategy: str):
    display(prevalence_view(prevalence, strategy))
    part = PREV_TABLE[PREV_TABLE.strategy == strategy]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for col in ("accuracy", "recall", "precision", "pr_auc", "roc_auc"):
        ax.plot(part.prevalence, part[col], marker="o", label=col)
    ax.axvline(prevalence, color="black", linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("prevalence (log scale)")
    ax.set_ylabel("metric on held-out half")
    ax.set_title(f"{strategy}: metrics across prevalence")
    ax.legend(frameon=False, fontsize=8)
    plt.show()

p_select = widgets.SelectionSlider(options=GRID_PREV, value=0.05, description="Prevalence", continuous_update=False,
                                   tooltip="Share of positives in the simulated data", layout=widgets.Layout(width="420px"))
s_select = widgets.Dropdown(options=list(STRATEGIES), value="none", description="Remedy", tooltip="Imbalance remedy")
display(widgets.VBox([widgets.HBox([p_select, s_select]), widgets.interactive_output(draw_prevalence, {"prevalence": p_select, "strategy": s_select})]))

# %% [markdown]
# ## 3. What-if scenario analyzer
# Sliders are clamped to the training range. Tick "Allow extrapolation" to widen them; values outside the range are
# flagged because the model has never seen them.

# %%
def what_if(profile: dict, baseline: dict | None = None) -> dict:
    baseline = baseline or MEDIAN_PROFILE
    frames = pd.DataFrame([baseline, profile])
    p_base, p_scen = predict(frames)
    flags = {k: bool(FEATURES[k]["type"] in ("number", "integer") and not FEATURES[k]["min"] <= v <= FEATURES[k]["max"])
             for k, v in profile.items()}
    expl = local_explanation(predict, pd.DataFrame([profile]), BACKGROUND)
    compare = pd.DataFrame({"baseline": baseline, "scenario": profile})
    compare.loc["probability"] = [round(float(p_base), 4), round(float(p_scen), 4)]
    compare.loc["label"] = ["flag" if p >= STATE["threshold"] else "no flag" for p in (p_base, p_scen)]
    return {"p_base": float(p_base), "p_scen": float(p_scen), "flags": flags, "explanation": expl, "compare": compare}

def control_for(name: str, wide: bool = False):
    f = FEATURES[name]
    tip = FEATURE_INFO.get(name, {}).get("description", name)
    label = f"{f['label']}" + (f" ({f['unit']})" if f.get("unit") else "")
    if f["type"] in ("number", "integer"):
        span = f["max"] - f["min"]
        lo, hi = (f["min"] - 0.5 * span, f["max"] + 0.5 * span) if wide else (f["min"], f["max"])
        cls = widgets.IntSlider if f["type"] == "integer" else widgets.FloatSlider
        return cls(value=f["default"], min=max(lo, 0), max=hi, step=f["step"], description=label, tooltip=tip,
                   continuous_update=False, style={"description_width": "150px"}, layout=widgets.Layout(width="460px"))
    if f["type"] == "boolean":
        return widgets.Checkbox(value=f["default"], description=label, tooltip=tip)
    return widgets.Dropdown(options=f["allowed"], value=f["default"], description=label, tooltip=tip,
                            style={"description_width": "150px"})

controls = {name: control_for(name) for name in SCHEMA["columns"]}
extrapolate = widgets.Checkbox(value=False, description="Allow extrapolation", tooltip="Widen numeric sliders beyond the training range")
reset = widgets.Button(description="Reset to median profile", tooltip="Set every predictor to its training median or mode")
out_whatif = widgets.Output()

def render_whatif(*_):
    STATE["profile"] = {k: (bool(c.value) if FEATURES[k]["type"] == "boolean" else c.value) for k, c in controls.items()}
    res = what_if(STATE["profile"])
    out_whatif.clear_output(wait=True)
    with out_whatif:
        display(Markdown(f"**Scenario probability {res['p_scen']:.1%}** versus {res['p_base']:.1%} for the median profile "
                         f"(delta {res['p_scen'] - res['p_base']:+.1%}). {META.get('calibration_note', '')}"))
        flagged = [k for k, v in res["flags"].items() if v]
        if flagged:
            display(Markdown(f"Outside the training range: {', '.join(flagged)}. Treat this prediction as an extrapolation."))
        display(res["compare"])
        contrib = pd.DataFrame(res["explanation"]["contributions"])
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.barh(contrib["feature"], contrib["contribution"], color=["#b2182b" if v > 0 else "#2166ac" for v in contrib["contribution"]])
        ax.invert_yaxis()
        ax.set_xlabel("contribution to predicted probability")
        ax.set_ylabel("predictor")
        ax.set_title("Local Shapley contributions (associations, not causes)")
        plt.show()

def rebuild_controls(change):
    for name in controls:
        old = controls[name].value
        controls[name] = control_for(name, wide=change["new"])
        controls[name].value = old
        controls[name].observe(on_change, names="value")
    box.children = [widgets.HBox([reset, extrapolate]), *controls.values(), out_whatif]
    render_whatif()

def on_change(change):
    STATE["last_feature"] = next((k for k, c in controls.items() if c is change["owner"]), STATE["last_feature"])
    render_whatif()

def on_reset(_):
    for name, c in controls.items():
        c.value = MEDIAN_PROFILE[name]
    render_whatif()

for c in controls.values():
    c.observe(on_change, names="value")
reset.on_click(on_reset)
extrapolate.observe(rebuild_controls, names="value")
box = widgets.VBox([widgets.HBox([reset, extrapolate]), *controls.values(), out_whatif])
display(box)
render_whatif()

# %% [markdown]
# ## 4. Cost-matrix editor

# %%
def cost_view(cost_fp: float, cost_fn: float) -> dict:
    best = mx.choose_threshold(y_fresh, P_FRESH, cost_fp=cost_fp, cost_fn=cost_fn)
    current = mx.expected_cost(y_fresh, P_FRESH, STATE["threshold"], cost_fp, cost_fn)
    nobody = float(y_fresh.mean() * cost_fn)
    return {"best_threshold": best["threshold"], "cost_at_best": best["expected_cost"], "cost_at_current": current,
            "cost_without_model": nobody, "savings_vs_no_model_per_customer": nobody - best["expected_cost"]}

def draw_cost(cost_fp: float, cost_fn: float):
    STATE["cost_fp"], STATE["cost_fn"] = cost_fp, cost_fn
    print({k: round(v, 3) for k, v in cost_view(cost_fp, cost_fn).items()})

fp_in = widgets.BoundedFloatText(value=STATE["cost_fp"], min=0, max=10000, step=1, description="Cost of FP (USD)", tooltip="Cost of one unnecessary call", style={"description_width": "130px"})
fn_in = widgets.BoundedFloatText(value=STATE["cost_fn"], min=0, max=100000, step=1, description="Cost of FN (USD)", tooltip="Margin lost per missed cancellation", style={"description_width": "130px"})
display(widgets.VBox([widgets.HBox([fp_in, fn_in]), widgets.interactive_output(draw_cost, {"cost_fp": fp_in, "cost_fn": fn_in})]))

# %% [markdown]
# ## 5. Cohort filter

# %%
def cohort_view(plan: str, region: str, tenure: tuple[int, int]) -> dict:
    mask = X_fresh["tenure_months"].between(*tenure)
    if plan != "all":
        mask &= X_fresh["plan"] == plan
    if region != "all":
        mask &= X_fresh["region"] == region
    mask = mask.to_numpy()
    if mask.sum() == 0 or y_fresh[mask].nunique() < 2:
        return {"n": int(mask.sum()), "metrics": None, "proba": P_FRESH[mask]}
    return {"n": int(mask.sum()), "metrics": mx.classification_metrics(y_fresh[mask], P_FRESH[mask], STATE["threshold"]), "proba": P_FRESH[mask]}

def draw_cohort(plan: str, region: str, tenure: tuple[int, int]):
    view = cohort_view(plan, region, tenure)
    if view["metrics"] is None:
        print(f"{view['n']} customers in this cohort: too few, or only one outcome, to compute metrics.")
        return
    m = view["metrics"]
    print(f"n {m['n']}, positives {m['positives']}, recall {m['recall']:.3f}, precision {m['precision']:.3f}, PR-AUC {m['pr_auc']:.3f}")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(view["proba"], bins=30)
    ax.axvline(STATE["threshold"], color="black", linestyle="--")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("customers")
    ax.set_title("Score distribution in the cohort")
    plt.show()

plan_dd = widgets.Dropdown(options=["all", *FEATURES["plan"]["allowed"]], value="all", description="Plan", tooltip="Filter by plan")
region_dd = widgets.Dropdown(options=["all", *FEATURES["region"]["allowed"]], value="all", description="Region", tooltip="Filter by region")
tenure_rs = widgets.IntRangeSlider(value=(1, 72), min=1, max=72, description="Tenure (months)", continuous_update=False,
                                   tooltip="Tenure range", style={"description_width": "120px"}, layout=widgets.Layout(width="460px"))
display(widgets.VBox([widgets.HBox([plan_dd, region_dd]), tenure_rs,
                      widgets.interactive_output(draw_cohort, {"plan": plan_dd, "region": region_dd, "tenure": tenure_rs})]))

# %% [markdown]
# ## 6. Scenario comparison

# %%
def save_scenario(name: str, profile: dict) -> pd.DataFrame:
    STATE["scenarios"][name] = {**profile, "probability": round(float(predict(pd.DataFrame([profile]))[0]), 4)}
    return pd.DataFrame(STATE["scenarios"]).T

name_in = widgets.Text(value="scenario 1", description="Name", tooltip="Name for the current what-if profile")
save_btn = widgets.Button(description="Save scenario", tooltip="Store the current what-if profile for comparison")
out_scen = widgets.Output()

def on_save(_):
    table = save_scenario(name_in.value, STATE["profile"])
    out_scen.clear_output(wait=True)
    with out_scen:
        display(table)
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.barh(table.index, table["probability"].astype(float))
        ax.set_xlim(0, 1)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("scenario")
        ax.set_title("Saved scenarios")
        plt.show()

save_btn.on_click(on_save)
display(widgets.VBox([widgets.HBox([name_in, save_btn]), out_scen]))

# %% [markdown]
# ## 7. Sensitivity view
# One predictor varied across and beyond its training range for the current profile; the shaded band is the range the
# model learned from.

# %%
def sensitivity_view(feature: str, profile: dict) -> pd.DataFrame:
    f = FEATURES[feature]
    if f["type"] in ("number", "integer"):
        span = f["max"] - f["min"]
        grid = np.linspace(max(f["min"] - 0.3 * span, 0), f["max"] + 0.3 * span, 40)
        grid = np.unique(np.round(grid)).astype(int) if f["type"] == "integer" else grid
    else:
        grid = f["allowed"] if f["type"] == "category" else [False, True]
    return sensitivity_curve(predict, pd.DataFrame([profile]), feature, list(grid))

def draw_sensitivity(feature: str):
    curve = sensitivity_view(feature, STATE["profile"])
    f = FEATURES[feature]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    if f["type"] in ("number", "integer"):
        ax.plot(curve["value"], curve["prediction"])
        ax.axvspan(f["min"], f["max"], alpha=0.15, label="training range")
        ax.axvline(STATE["profile"][feature], color="black", linestyle="--", label="current value")
        ax.legend(frameon=False)
    else:
        ax.bar([str(v) for v in curve["value"]], curve["prediction"])
    ax.set_xlabel(f"{f['label']}" + (f" ({f['unit']})" if f.get("unit") else ""))
    ax.set_ylabel("predicted probability")
    ax.set_title(f"Sensitivity of the prediction to {f['label'].lower()}")
    plt.show()

feat_dd = widgets.Dropdown(options=SCHEMA["columns"], value=STATE["last_feature"], description="Predictor", tooltip="Predictor to vary")
display(widgets.VBox([feat_dd, widgets.interactive_output(draw_sensitivity, {"feature": feat_dd})]))

# %% [markdown]
# ## Static fallback
# Widgets do not render in nbconvert HTML, GitHub previews or PDF. This cell runs every view function with fixed inputs
# and prints the results, which also proves each callback's logic runs without error.

# %%
print("1. threshold", threshold_view(STATE["threshold"])["summary"])
print("2. prevalence\n", PREV_TABLE.pivot(index="prevalence", columns="strategy", values="pr_auc").round(3).to_string())
fallback = {**MEDIAN_PROFILE, "tenure_months": FEATURES["tenure_months"]["min"], "support_calls": FEATURES["support_calls"]["max"]}
wi = what_if(fallback)
print(f"3. what-if: short tenure and many support calls -> {wi['p_scen']:.3f} vs median profile {wi['p_base']:.3f}")
print(pd.DataFrame(wi["explanation"]["contributions"]).round(4).to_string(index=False))
print("4. costs", {k: round(v, 3) for k, v in cost_view(15.0, 120.0).items()})
cv_basic = cohort_view("basic", "all", (1, 72))
print(f"5. cohort basic plan: n {cv_basic['n']}, recall {cv_basic['metrics']['recall']:.3f}")
print("6. scenarios\n", pd.concat([save_scenario("median", MEDIAN_PROFILE), save_scenario("at risk", fallback)]).drop_duplicates().to_string())
sens = sensitivity_view("tenure_months", MEDIAN_PROFILE)
print(f"7. sensitivity of tenure: prediction from {sens.prediction.iloc[0]:.3f} to {sens.prediction.iloc[-1]:.3f} across the grid")

# %% [markdown]
# ## Export
# `jupyter nbconvert --to html whatif_dashboard.ipynb` keeps tables and static charts; widgets need a live kernel.
# `voila whatif_dashboard.ipynb` serves the panels as a dashboard. In Colab the core widgets work as is; enable the
# custom widget manager only for third-party widgets.
#
# ## Limitations
# - Predictions describe associations in simulated training data; moving a slider is not an intervention.
# - The fresh sample is simulated with the same generator, so it shows no drift; real monitoring data will.
