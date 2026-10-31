---
name: data-storytelling
description: Data Storytelling. Turn analysis into a decision-first narrative and interactive what-if analytics - SCQA story structure, findings-first headings, chart design, ipywidgets dashboards (threshold explorer, prevalence explorer, what-if scenario analyzer, cost-matrix editor, cohort filter, scenario comparison, sensitivity view), static fallback, nbconvert HTML and Voila export. Use for reports, presentations of results, executive summaries, dashboards in notebooks, interactive widgets or explaining findings to stakeholders.
---

# Data Storytelling

## Narrative rules

1. Lead with the decision: the first cell or slide states the takeaway as a sentence with its number.
2. SCQA: situation, complication, question, resolution and action.
3. Every heading is a finding ("Recall drops 18 points below 2 percent prevalence"), not a topic ("Evaluation"). In notebooks, compute headings with `display(Markdown(f"## ..."))` so the numbers are real.
4. One message per chart. The title states the takeaway, axes carry units, direct labels beat legends, colorblind-safe palette (Okabe-Ito), bar axes start at zero, no dual axes or 3D without need.
5. End with recommended actions, risks, open questions and next steps, then limitations.
6. Conditional wording: when a sentence interprets a number, write it so it stays true if the number moves (`'rose' if b > a else 'fell'`).

## Widgets (core ipywidgets only)

Templates: `templates/notebooks/whatif_dashboard.ipynb` (seven panels) and `story_report.ipynb`.

| Panel | Controls | Shows |
|---|---|---|
| Threshold explorer | threshold slider | confusion matrix, precision, recall, F1, expected cost, point on the PR curve |
| Prevalence explorer | prevalence, remedy | accuracy, recall, precision, PR-AUC, ROC-AUC from a precomputed grid |
| What-if analyzer | sliders and dropdowns per predictor, reset | probability, label, local contributions, baseline versus scenario table |
| Cost-matrix editor | cost of FP and FN | optimal threshold, expected savings |
| Cohort filter | segment, time range | metrics and score distribution |
| Scenario comparison | name, save | table and chart of saved scenarios |
| Sensitivity view | predictor | ICE-style curve with the training range shaded |

## Widget engineering

- `interactive_output` with `Output` widgets, or `interact_manual` for heavy work. `continuous_update=False` on sliders that call a model. Never refit inside a callback; precompute grids.
- State in one dictionary or small class.
- Every control has a description and a tooltip; layouts use `HBox`, `VBox`, `GridBox` with set widths; keyboard reachable.
- Clamp to the training range; warn on extrapolation.
- Each panel wraps a pure function; a static fallback cell calls every function with fixed inputs and prints tables, which also proves the callbacks run.

## Export

`jupyter nbconvert --to html` keeps tables and static charts; `voila notebook.ipynb` serves widgets as a dashboard; widgets need a live kernel. In Colab, core widgets work; enable the custom widget manager only for third-party widgets.

Recipes: `references/widgets.md`.
