# RIGOR gates in detail

Each gate is 0 or 1. Evidence is a cell, a test or a file, never an intention.

## 1. Reproducibility
- One `SEED` constant passed to every splitter, estimator and resampler.
- Library versions printed in the environment cell and recorded in `metadata.json`.
- Paths relative to the project root; no `C:\Users` or `/home/<name>`.
- `scripts/notebook_run.py` executes the notebook from a clean kernel without error.
- `%pip install` lines pin versions.

## 2. Integrity
- Scanner reports no L-series errors.
- Every imputer, scaler, encoder, selector, resampler and target encoder lives in the pipeline.
- The label definition is written and checked against a sample of raw records.
- Duplicate and group checks run before splitting.
- Time series use forward-chaining; features use trailing windows only.

## 3. Generalization
- Baseline metric reported next to the model metric.
- Cross-validation spread or bootstrap intervals on every headline metric.
- Threshold chosen on validation data; the test set scored once.
- Slice metrics for important segments; fairness metrics for people-facing decisions.
- Learning curve or train and validation gap reported.

## 4. Explainability
- Permutation importance on held-out data with intervals.
- Local explanation (SHAP or counterfactual) for representative cases.
- `feature_dictionary.csv` with rationale and availability time.
- Caveats stated: importance is not causation; correlated features share attribution.

## 5. Observability
- Logging in the service with request id and no raw PII.
- Data checks: schema, ranges, missingness at load and at serving.
- Monitoring plan: drift metric (PSI or KS), threshold, owner, retraining trigger.
- `model_card.md` exported.

## 6. Communication
- Title states the takeaway; headings are findings.
- Every chart has a takeaway title, labeled axes with units, colorblind-safe colors.
- Limitations, risks, open questions and next steps written.
- What-if widgets have a static fallback cell.

## 7. Deployability
- Saved pipeline reloads in a fresh process and reproduces predictions (`numpy.allclose`).
- `verify-model` accepts the bundle: checksum, metadata, schema, versions.
- API validates inputs against `schema.json`, rejects unknown fields, flags extrapolation.
- Health endpoints and API tests pass.
- No secrets in code, notebooks or the front-end bundle.
- The explainer passes the grounding test and falls back to a template when the LLM is unavailable.

## Score formula for the scanner

`score = max(0, 100 - 20 x errors - 6 x distinct warning rules)`. Repeated warnings of one rule cost once, so the score tracks breadth of problems; every error costs fully.
