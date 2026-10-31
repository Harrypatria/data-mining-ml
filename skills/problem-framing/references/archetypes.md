# Problem archetypes

Each archetype sets defaults. Override with a one-line reason.

| Archetype | Target and grain | Split strategy | Headline metric | Baseline | Typical traps |
|---|---|---|---|---|---|
| Churn | Cancels within N days, per customer per snapshot | Forward-chaining by snapshot date, group by customer | Recall at fixed precision, PR-AUC, lift in top decile | Recency rule or logistic regression | Features computed after the snapshot; same customer in train and test |
| Demand forecasting | Units per item per period | Forward-chaining (rolling origin) | MAE, WAPE, pinball loss for quantiles | Seasonal naive, last value | Random split; leaking future promotions; scaling across items |
| Fraud or anomaly detection | Fraud flag per transaction | Time-based split, group by account | Recall at precision constraint, precision at k, PR-AUC | Rule set in production | Prevalence below 1 percent; label delay; accuracy reported |
| A/B analysis | Metric per user per variant | Randomized assignment (no model split) | Effect size with CI, power | Control group | Peeking; multiple metrics without correction; novelty effect |
| Segmentation | Cluster per customer | Stability across bootstrap samples | Silhouette plus business profile, stability | RFM quantile segments | Unscaled features; silhouette on non-convex shapes; clusters with no action |
| Recommender | Relevance per user-item | Time-based holdout per user | Precision at k, recall at k, NDCG, coverage | Popularity | Cold start ignored; evaluating on items users already saw |
| NLP classification | Label per document | Stratified, group by author or thread | Macro F1, per-class recall | TF-IDF plus logistic regression | Duplicated texts across splits; label noise |
| Risk scoring | Default or adverse event within horizon | Out-of-time validation | PR-AUC or ROC-AUC with calibration (Brier), KS | Scorecard or logistic regression | Reject inference; regulatory explainability; drift |

## Prevalence bands and consequences

- Above 20 percent: standard stratified CV; accuracy still misleading next to a cost matrix.
- 1 to 20 percent: PR-AUC and recall at precision as headline; tune threshold on validation.
- Below 1 percent: precision at k, anomaly framing, wide intervals on recall; report the positive count.

## Questions per archetype

- Churn: what action is taken and how many customers can the team contact per week?
- Forecasting: what horizon and which aggregation level does planning use?
- Fraud: what is the review capacity per day and how long until labels arrive?
- A/B: what is the minimum detectable effect worth acting on?
- Segmentation: what will each segment receive differently?
