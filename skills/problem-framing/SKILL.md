---
name: problem-framing
description: Problem Framing. Turn a business brief, PRD, dataset description or stakeholder question into a one-page ANALYSIS_MANIFEST.md before any modeling. Use when someone asks what to predict, how to measure success, which metric or baseline to use, the cost of errors, or how to scope a machine learning, data science, data mining, predictive modeling, forecasting, segmentation, churn, fraud, risk scoring or A/B testing project. CRISP-DM Business Understanding.
---

# Problem Framing

Goal: one page that makes every later choice defensible. No data is touched until the manifest exists.

## Steps

1. **Extract** from the brief, in the stakeholder's words:
   - Decision the output informs, who makes it, how often, and what happens today (the baseline process).
   - Unit of prediction (customer per month, transaction, store per week) and prediction time (when the score must exist).
   - Success metric tied to the decision, plus a guardrail metric.
   - Cost of each error type. Fill a cost matrix even if rough: cost of a false positive, cost of a false negative, value of a true positive.
   - Constraints: latency, interpretability, regulation, fairness, data residency, budget.
   - Stakeholders and the audience for explanations.
   - Data touchpoints: sources, grain, history length, label definition and label delay.
   - Expected prevalence of the target if known.
2. **Classify the archetype** (see `references/archetypes.md`): churn, demand forecasting, fraud or anomaly detection, A/B analysis, segmentation, recommender, NLP classification, risk scoring. The archetype sets default split strategy, headline metric and traps.
3. **Choose the baseline** that the model must beat: current rule, majority class, last value, seasonal naive, or a logistic or linear regression.
4. **Pick the headline metric by cost** (`model-evaluation/references/imbalance-and-metrics.md`). Never accuracy alone on imbalanced targets.
5. **Write `ANALYSIS_MANIFEST.md`** from `references/manifest-template.md`. Keep it to one page. Mark each assumption with `Assumption:`.
6. **Plan phases** with CRISP-ML(Q) rows: for each phase one risk, its assumption and a mitigation.

## Questions that change the design

Ask the user only if the answer changes the architecture and cannot be assumed. Otherwise state the assumption in one line and continue.

- Is the decision made per entity in batch, or per request in real time?
- Is the label available at prediction time for past periods only (label delay)?
- Will a person act on each prediction (review queue capacity sets precision at k)?
- Are people affected by the decision (fairness analysis becomes mandatory)?

## Exit criteria

- Decision, metric, cost matrix, baseline and prevalence written.
- Leakage-relevant timing written: prediction time and feature availability.
- Deliverables listed: notebooks, model bundle, API, web app, explainer, report.
- RIGOR gates that will be hard to meet are named now, not at the end.

## Anti-patterns

- Starting with "which algorithm" before "which decision".
- A metric the business cannot act on (ROC-AUC for a review queue of 50 cases a day).
- A target defined with information from after the prediction time.
- Prevalence ignored, so the threshold defaults to 0.5.
