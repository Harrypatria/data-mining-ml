# Audit rubric

Questions the scanner cannot answer. Each "no" is a finding with a location.

## Problem and data
- Is the decision, metric and cost of each error written down?
- Is the target defined with information available only after the prediction time? (L04)
- Is the grain stated and do entities repeat across rows?
- Was missingness examined by target class, and its mechanism argued?
- Were outliers inspected before removal, with a rule and a count? (D03)
- Are units consistent within each column? Are magic numbers named and justified? (E11)
- Is the label quality checked on a sample?

## Splits and leakage
- Does the split match the structure (stratified, group, time)?
- Is any column recorded after the event?
- Is the test set used exactly once, after all choices?
- Are thresholds, calibration and early stopping fitted on validation data?

## Statistics
- Is "significant" used to mean "important"? (S04)
- Are strong claims made from small samples without intervals? (S06)
- Were hypotheses fixed before looking at results?
- Are causal claims backed by a design and its assumptions?

## Modeling and evaluation
- Is there a baseline, and does the model beat it by more than the fold spread?
- Is the headline metric tied to the cost matrix?
- Are models compared at the same operating point?
- Are intervals and positive counts reported?
- Are slices and, for people-facing use, fairness metrics reported?

## Explainability and communication
- Are explanations computed on held-out data with caveats?
- Do chart titles state the finding, with labeled axes and honest geometry?
- Is jargon translated into a decision? (C06)
- Are limitations, risks and next steps present?
- Are t-SNE or UMAP distances used for quantitative claims?

## Production
- Does the saved model reload with identical predictions?
- Are versions, schema and threshold stored with the model?
- Is input validated against training ranges?
- Are secrets absent from code, notebooks and the front end?
- Is the LLM explainer grounded and does it fall back safely?

## Scoring

RIGOR gates as in `ml-quality-guard/references/rigor-gates.md`. Status line:
`Rigor: <e> errors, <w> warnings, score <s>, RIGOR <p>/<n>, run: passed|failed|not run`
