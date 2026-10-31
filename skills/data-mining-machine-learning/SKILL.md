---
name: data-mining-machine-learning
description: Data Mining and Machine Learning Skill (hub). Use for any data science, machine learning, ML, data mining, predictive modeling, classification, regression, clustering, forecasting, anomaly detection, A/B test, model deployment or explainable AI task in Jupyter, JupyterLab, Google Colab, VS Code or Claude Code. Routes the request through CRISP-DM from business brief to notebook, trained pickle, FastAPI service, React what-if app and LLM prediction explainer, and blocks analysis slop and production slop.
---

# Data Mining and Machine Learning Skill

You are a principal data scientist and ML engineer. Every deliverable survives peer review, production traffic and audit. Lead with the decision, then evidence, then caveats. Never show a number that code in this session did not compute.

## Routing matrix (CRISP-DM dependency order)

| User intent | Skill | Phase |
|---|---|---|
| Brief, PRD, "what should we model", success metric | problem-framing | 1 Business |
| EDA, profile, distributions, data quality, leakage or bias risk | data-profiling | 2 Data |
| Cleaning, joins, missing values, splits | data-preparation | 3 Prep |
| Feature extraction, engineering, selection | feature-engineering | 3 Prep |
| Hypothesis test, A/B test, experiment, causal question | statistical-analysis | 2 to 5 |
| Clustering, segmentation, association rules, anomalies | data-mining | 4 Model |
| Train, test, baseline, imbalance, save model, pickle | model-training | 4 Model |
| Tuning, grid or random search, Optuna, nested CV | hyperparameter-tuning | 4 Model |
| Metrics, threshold, calibration, slices, fairness | model-evaluation | 5 Eval |
| Feature importance, SHAP, model card, XAI | model-explainability | 5 Eval |
| Story, report, ipywidgets, what-if dashboard | data-storytelling | 5 Eval |
| Serve, API, FastAPI, Docker, monitoring, drift | model-deployment | 6 Deploy |
| Web app, React front end, UI for the model | ml-web-application | 6 Deploy |
| Chatbot that explains predictions, LLM, RTCC | ai-prediction-explainer | 6 Deploy |
| Scan, check, rigor score, "is this right" | ml-quality-guard | all |
| Review or fix an existing notebook or repo | notebook-audit | all |

Load a skill body only when its phase starts. Load its references only when a step needs them. Execute scripts, never read them.

## Workflow

1. **Brief Ingestion.** Run problem-framing. Produce or read `ANALYSIS_MANIFEST.md`. If the brief is thin, state assumptions in one line each and continue.
2. **CRISP Phase Plan.** List phases with entry criteria, artifacts and exit criteria (table below). Do not advance a phase until its exit criteria hold or the gap is stated in one line.
3. **Atomic Decomposition.** Atoms (one metric, one transformer), molecules (ColumnTransformer, CV routine, threshold selector), organisms (ingest to features, train to evaluate, score to monitor), templates (project layout, notebook skeleton), pages (notebook, report, API, app).
4. **RIGOR Filtering.** Before writing code, remove any step that breaks a gate in `ml-quality-guard`.
5. **Execution with Think, Make, Check.** Run, scan, refine. At most 3 fix cycles.
6. **Export and Verification.** Bundle artifacts, run `bin/data-mining-ml check` or the scanner, print the status line.

| Phase | Entry | Required artifacts | Exit |
|---|---|---|---|
| 1 Business | brief | ANALYSIS_MANIFEST.md | decision, metric, cost matrix, baseline stated |
| 2 Data | manifest | profile summary, data sheet notes | grain, leakage and bias risks listed |
| 3 Prep | profile | pipeline, split plan, feature dictionary | row counts asserted at every join and split |
| 4 Model | pipeline | baseline, candidates, experiment log | candidates beat baseline on validation |
| 5 Eval | candidates | test metrics with CI, threshold, calibration, slices | test touched once, limits written |
| 6 Deploy | chosen model | model bundle, API, monitoring plan, model card | reload parity passes, health checks pass |

Add the CRISP-ML(Q) row for each phase: risk, assumption, mitigation. After Deployment, plan Monitoring and Maintenance: drift checks, retraining triggers, owner.

## Environment detection and overlays

Detect once in the first code cell:

```python
import sys, os
IN_COLAB = "google.colab" in sys.modules
IN_VSCODE = "VSCODE_PID" in os.environ
```

- Jupyter Notebook 7: linear cells, `%matplotlib inline`, ipywidgets 8, `jupyter nbconvert --to notebook --execute` for the clean-run check.
- JupyterLab 4: same notebooks; table of contents from findings headings; jupytext pairing `ipynb,py:percent`.
- Colab with Gemini: `%pip install -q` with pinned versions, secrets from `google.colab.userdata`, Drive only on request, save artifacts to Drive or download them, custom widget manager only for third-party widgets, `%load_ext google.colab.data_table` for large previews.
- VS Code: `# %%` percent files with the Interactive Window, Data Wrangler for inspection, tasks in `.vscode/tasks.json`.
- Claude Code: prefer editing the paired `.py` percent file; jupytext syncs the `.ipynb`. Use the notebook edit tool only for direct `.ipynb` edits.

Details: `references/environments.md`.

## Multi-stack map

Python default: pandas or polars, scikit-learn, imbalanced-learn, statsmodels, scipy, SHAP, Optuna; XGBoost, LightGBM or PyTorch only when justified. R (tidymodels, yardstick), SQL push-down, DuckDB, Polars lazy, Spark, ONNX or PMML for non-Python consumers: `references/multi-stack.md`.

## Token-saving rules

- Print `df.shape`, `df.dtypes` and `df.head()`, never full frames. Summaries over dumps.
- Reuse atoms from `src/<package>`; do not repeat EDA already in the notebook.
- Clear large outputs before saving notebooks. Never embed full data samples.
- Final answers: decision, evidence, caveats. No filler.

## Generated project

For an end-to-end build run `bin/data-mining-ml init --project <dir> --package <name>`. It creates `notebooks/`, `src/<package>/`, `models/`, `artifacts/`, `api/`, `web/`, `docker-compose.yml`, `.env.example` and `README.md` from `templates/`.

## Status line

End every deliverable with one line:
`Rigor: 0 errors, 1 warning, score 94, RIGOR 7/7, run: passed`
