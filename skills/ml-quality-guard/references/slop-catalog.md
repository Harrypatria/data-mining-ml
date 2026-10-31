# Analysis and production slop catalog

85 rules. Detection: `auto` rules are implemented in `scripts/ml_scan.py` (79 rules); `manual` rules are checked by the ml-critic agent and the notebook-audit rubric. Severity: `error` rules are certain defects and fail the scan; `warning` rules are heuristics and need a fix or a stated reason.

Escape hatch: put `# ml-ignore <ID> <reason>` on the flagged line or the line above (in Markdown cells: `<!-- ml-ignore <ID> <reason> -->`). A file-level rule is silenced by the same comment anywhere in the file. An ignore without a reason is not honored.

## L: Leakage

| ID | Sev | Det | Rule | Why | Fix |
|---|---|---|---|---|---|
| L01 | error | auto | Preprocessor fitted before the train and test split | Scaler, imputer, encoder, selector or PCA learns statistics from test rows | Put the transformer inside a Pipeline and fit after splitting |
| L02 | error | auto | Transformer, model or search fitted on test or validation data | The held-out set stops being held out | Fit on training data only; call `transform` or `predict` on held-out data |
| L03 | error | auto | Resampling outside the training fold (before the split, or on test or validation data) | Synthetic or duplicated rows leak into evaluation and distort prevalence | Use an imbalanced-learn Pipeline so resampling runs inside each training fold |
| L04 | error | manual | Target-derived feature | A feature computed from the label gives perfect scores that vanish in production | Remove it; ask "would this value exist at prediction time?" |
| L05 | warning | auto | Future data in time-series features (negative `shift`, centered `rolling`) | Features see values after the prediction time | Use positive shifts and trailing windows only |
| L06 | warning | auto | Target statistics computed without out-of-fold fitting (`groupby(...)[target]`) | Each row's own label leaks into its encoding | Use `sklearn.preprocessing.TargetEncoder` (cross-fitted) inside the pipeline |
| L07 | error | auto | Early stopping evaluated on the test set (`eval_set` with test data) | The test set selects the number of iterations | Early stop on a validation fold carved from training data |
| L08 | warning | auto | Decision threshold chosen on the test set | The reported test metric is optimistic | Choose the threshold on validation data, then score the test set once |
| L09 | warning | auto | No duplicate check before splitting | Repeated rows land in train and test (leakage by repetition) | Check `duplicated()` and use group-aware splits for repeated entities |
| L10 | warning | auto | Entity identifiers present but split is random, not group-aware | The same customer or patient appears on both sides | Use `GroupKFold`, `StratifiedGroupKFold` or `GroupShuffleSplit` |

## S: Statistics

| ID | Sev | Det | Rule | Why | Fix |
|---|---|---|---|---|---|
| S01 | warning | auto | p-value without effect size or confidence interval | A bare p-value says nothing about magnitude | Report the effect size and its CI next to every p-value |
| S02 | warning | auto | Parametric test without assumption checks | Normality and equal variance violations change the answer | Check assumptions and pick the test with `references/test-selection.md` |
| S03 | warning | auto | Multiple tests without correction | False positives grow with every test | Apply Holm, Bonferroni or Benjamini-Hochberg (`multipletests`) |
| S04 | warning | manual | "Significant" used to mean "important" | Statistical significance is not practical relevance | Report the effect against a minimum meaningful effect |
| S05 | warning | auto | Causal language without a causal design | Observational associations are presented as causes | Use associative wording or name the design and its assumptions |
| S06 | warning | manual | Strong claim from a tiny sample | Intervals are wide and estimates unstable | Report intervals, use exact or Bayesian methods, soften the claim |
| S07 | warning | auto | Regression coefficients read without diagnostics | Collinearity and heteroscedasticity invalidate coefficient reading | Run VIF, residual plots and a heteroscedasticity test first |

## M: Modeling

| ID | Sev | Det | Rule | Why | Fix |
|---|---|---|---|---|---|
| M01 | warning | auto | No baseline model | There is no evidence the model adds value | Fit a Dummy or simple linear baseline first |
| M02 | error | auto | Accuracy is the only metric | Accuracy hides failure on the minority class | Add recall, precision, PR-AUC, MCC or a cost metric |
| M03 | error | auto | ROC-AUC without a precision-recall metric on imbalanced data | ROC-AUC stays optimistic when negatives dominate | Report PR-AUC (average precision) and precision at the operating threshold |
| M04 | warning | auto | Hardcoded 0.5 threshold | 0.5 is rarely the cost-optimal decision rule on imbalanced data | Tune the threshold on validation data with a cost matrix |
| M05 | warning | auto | Shuffled split on time-ordered data | The model trains on the future | Use `TimeSeriesSplit` or a cutoff date |
| M06 | warning | auto | Missing `random_state` on a stochastic splitter or estimator | Results cannot be reproduced | Pass a single `SEED` constant |
| M07 | warning | auto | Classification split without `stratify` | Class ratios differ across splits | Pass `stratify=y` |
| M08 | warning | auto | Single split with no uncertainty estimate | One number hides variance | Use cross-validation or bootstrap confidence intervals |
| M09 | error | auto | Resampler used outside an imbalanced-learn Pipeline | Resampling cannot be confined to training folds | Use `imblearn.pipeline.Pipeline` |
| M10 | warning | auto | Default hyperparameters presented as tuned | Readers assume a search happened | Run a search or say "default configuration" |
| M11 | warning | auto | Distance-based method on unscaled features | Large-unit features dominate distances | Scale inside the pipeline before k-means, k-NN, SVM, DBSCAN or PCA |
| M12 | warning | auto | Number of clusters chosen without validation | k is arbitrary | Compare elbow, silhouette, gap statistic and stability |
| M13 | warning | auto | No calibration check after reweighting or resampling | Resampled models over-predict the minority probability | Plot the calibration curve, report Brier score, recalibrate on validation data |

## F: Features and tuning

| ID | Sev | Det | Rule | Why | Fix |
|---|---|---|---|---|---|
| F01 | warning | auto | Engineered features without a feature dictionary | Nobody can audit availability time or rationale | Record name, source, transform, rationale and availability time |
| F02 | error | auto | Feature selection fitted outside the pipeline | Selection sees validation and test rows | Put the selector inside the Pipeline |
| F03 | warning | auto | Target encoder without internal cross-fitting | In-sample encodings leak the label | Use `sklearn.preprocessing.TargetEncoder` or out-of-fold encoding |
| F04 | warning | auto | Manual search loop without cross-validation (`ParameterGrid`, nested loops) | One validation split overfits the search | Use a `*SearchCV` or Optuna with CV |
| F05 | error | auto | Tuning objective reads the test set | Model choice is fitted to the test set | Score trials on CV folds of the training data |
| F06 | warning | auto | Best score reported without spread across folds | Differences may be smaller than fold noise | Report mean and standard deviation from `cv_results_` |
| F07 | warning | auto | Search scored on accuracy with imbalanced data | The search optimizes the wrong objective | Score on average precision, F-beta or log loss |

## X: Explainability

| ID | Sev | Det | Rule | Why | Fix |
|---|---|---|---|---|---|
| X01 | warning | auto | Explanations computed on training data only | Explanations on training data flatter the model | Compute permutation importance and SHAP on held-out data |
| X02 | warning | auto | Impurity importance used with no permutation check | Impurity importance is biased toward high-cardinality features | Add permutation importance on held-out data |
| X03 | warning | auto | SHAP used without a correlated-feature caveat | Correlated features split or distort attribution | State the caveat and check correlations or use grouped attributions |
| X04 | warning | auto | Importance described as causation | Attribution explains the model, not the world | Say "the model associates" |

## D: Data

| ID | Sev | Det | Rule | Why | Fix |
|---|---|---|---|---|---|
| D01 | warning | auto | Silent row drops (`dropna`, `drop_duplicates` with no count check) | Lost rows change the population unnoticed | Print or assert the row count before and after |
| D02 | warning | auto | Merge or join without `validate=` or a row-count check | Many-to-many joins duplicate rows | Pass `validate="one_to_one"` or `"many_to_one"` and assert counts |
| D03 | warning | manual | Outliers removed without inspection | Errors and rare signal get mixed | Inspect, decide error or signal, document the rule |
| D04 | warning | auto | Missingness never examined | MCAR, MAR and MNAR need different handling | Profile `isna()` by column and by target |
| D05 | warning | auto | Personal data printed or exported | PII leaks into outputs and logs | Drop or hash identifiers before display |
| D06 | warning | auto | No schema or range checks after loading | Silent type and unit errors propagate | Assert columns, dtypes and value ranges |

## E: Engineering

| ID | Sev | Det | Rule | Why | Fix |
|---|---|---|---|---|---|
| E01 | error | auto | Absolute local path | Breaks on every other machine | Use paths relative to the project root |
| E02 | error | auto | Hardcoded secret or API key | Secrets leak through version control and notebooks | Read from environment variables or `google.colab.userdata` |
| E03 | error | auto | Secret literal assigned to `os.environ` | Same leak, disguised as configuration | Set secrets outside code |
| E04 | error | auto | Wildcard import | Names shadow each other silently | Import names explicitly |
| E05 | warning | auto | Global warnings suppression | Hides convergence and deprecation problems | Filter a specific category and message |
| E06 | warning | auto | Out-of-order or missing execution counts | The notebook was not run top to bottom | Restart and Run All, or clear outputs |
| E07 | warning | auto | Oversized stored output | Bloats the notebook and may embed data | Clear or summarize outputs |
| E08 | error | auto | TODO, FIXME or stub function | Unfinished logic ships | Finish or remove it |
| E09 | warning | auto | Unpinned `%pip install` or `!pip install` | Results drift with library versions | Pin versions, for example `scikit-learn==1.9.0` |
| E10 | warning | auto | Printing full data frames | Floods context and may leak data | Print `shape`, `dtypes` and `head()` |
| E11 | warning | manual | Magic numbers | Unexplained constants hide decisions | Name constants in the config cell with a reason |

## C: Communication

| ID | Sev | Det | Rule | Why | Fix |
|---|---|---|---|---|---|
| C01 | error | auto | Placeholder or literal metric values | Numbers not computed in this run are fabricated | Print metrics from code that ran |
| C02 | warning | auto | Chart without axis labels | Readers cannot read units | Label both axes with units |
| C03 | warning | auto | Misleading chart geometry (truncated bar axis, 3D, dual axis) | Distorts magnitudes | Start bars at zero, use 2D, split dual axes |
| C04 | warning | auto | Notebook without a limitations section | Readers over-trust the result | Add limitations, risks and open questions |
| C05 | warning | auto | Widgets without a static fallback | Exports and previews show nothing | Add a cell that prints key scenarios as tables |
| C06 | warning | manual | Jargon without a decision | The reader cannot act | Lead with the decision in plain language |

## P: Production and serving

| ID | Sev | Det | Rule | Why | Fix |
|---|---|---|---|---|---|
| P01 | error | auto | Unpickling a path from user input or an upload | Unpickling runs arbitrary code | Load only an operator-set path; no upload endpoint |
| P02 | error | auto | Unpickling without a checksum check | A swapped file runs unchecked | Verify `model.sha256` before loading |
| P03 | error | auto | Model loaded inside a request handler | Latency spikes and repeated disk reads | Load once in the `lifespan` handler |
| P04 | warning | auto | Preprocessing saved separately from the model | Serving can drift from training | Save one fitted Pipeline |
| P05 | error | auto | Custom transformer defined in `__main__` of a file that pickles | The service cannot unpickle it | Move it to `src/<package>/transformers.py` |
| P06 | error | auto | Blocking model call inside an `async def` route | Blocks the event loop | Use a plain `def` route or `run_in_threadpool` |
| P07 | error | auto | CORS wildcard together with credentials | Any site can make credentialed calls | Use an explicit origin allowlist |
| P08 | error | auto | `--reload` in a container command | Dev server behaviour in production | Remove `--reload` |
| P09 | error | auto | Container runs as root | A compromise gets root in the container | Add a non-root `USER` |
| P10 | warning | auto | FastAPI app without a health endpoint | Orchestrators cannot probe readiness | Add `/health/live` and `/health/ready` |
| P11 | warning | auto | Model saved with no reload parity test | A broken pickle ships | Reload in a fresh process and assert equal predictions |
| P12 | error | auto | Model saved with no metadata | Versions, threshold and schema are lost | Write `metadata.json` and `schema.json` beside the pickle |
| P13 | warning | auto | Dockerfile without `HEALTHCHECK` | Unhealthy containers keep serving | Add a `HEALTHCHECK` on `/health/live` |

## G: Generative AI explainer

| ID | Sev | Det | Rule | Why | Fix |
|---|---|---|---|---|---|
| G01 | error | auto | Explainer prompt without a constraints block or causation guard | The model invents numbers and causes | Use the RTCC prompt with its CONSTRAINTS section |
| G02 | error | auto | Request model accepts client-supplied explanation values (SHAP, contributions) | A client can forge the explanation | Recompute everything on the server |
| G03 | warning | auto | LLM output used without a numeric-grounding check | Invented numbers reach users | Run `grounding.py` before display |
| G04 | error | auto | Provider key or LLM SDK in front-end code or a `VITE_` variable | The key ships to every browser | Call the LLM from the backend only |
| G05 | warning | auto | LLM call without timeout or token cap | Hung or runaway requests | Set `timeout` and `max_tokens` |
| G06 | warning | auto | LLM call without fallback | Provider outages break the feature | Catch errors and return the template explanation |
| G07 | error | auto | User text placed in the system prompt | Prompt injection rewrites the rules | Put user text in the user message only |
| G08 | warning | auto | Hardcoded LLM model name | Model changes need a redeploy | Read `LLM_MODEL` from the environment |
