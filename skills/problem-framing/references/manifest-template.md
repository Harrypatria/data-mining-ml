# ANALYSIS_MANIFEST.md template

Copy into the project root and fill every line. Keep it to one page.

```markdown
# <Decision-first title, for example: Flag customers likely to cancel in the next 30 days for the retention team>

## Objective
- Decision informed: <who decides what, how often>
- Unit of prediction: <entity and period>
- Prediction time: <when the score must exist; features must be available by then>
- Model purpose (for the explainer prompt): <one sentence>
- Explanation audience: <role of the people reading explanations>

## Success
- Headline metric: <for example recall at precision >= 0.40, or PR-AUC>
- Guardrail metric: <for example calibration (Brier score), latency p95>
- Baseline to beat: <current rule or naive model, with its measured value once computed>
- Expected prevalence: <rate or "unknown, measure in profiling">

## Cost matrix (per case, currency or effort units)
| | Predicted positive | Predicted negative |
|---|---|---|
| Actual positive | <value of TP> | <cost of FN> |
| Actual negative | <cost of FP> | 0 |

## Constraints
- Interpretability: <required level>
- Latency and volume: <requests per second or batch size>
- Regulation and fairness: <protected attributes, chosen fairness definition>
- Data residency and privacy: <PII handling>

## Data needs
| Source | Grain | History | Label or feature | Available at prediction time |
|---|---|---|---|---|

## Risks (CRISP-ML(Q))
| Phase | Risk | Assumption | Mitigation |
|---|---|---|---|
| Data | <for example label delay> | | |
| Modeling | <for example leakage through post-event fields> | | |
| Deployment | <for example drift after pricing change> | | |

## Phase plan
1. Data understanding: <exit criteria>
2. Preparation: <split strategy and why>
3. Modeling: <baseline and candidate families>
4. Evaluation: <threshold policy, slices, fairness>
5. Deployment: <batch or API, monitoring, retraining trigger>

## Deliverables
- notebooks/, models/v1.0.0/ bundle, artifacts/, api/, web/, model card, data sheet

## Assumptions
- Assumption: <one line each>
```
