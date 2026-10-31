# RTCC few-shot examples

Placeholders only. `<P_SCEN>`, `<P_BASE>` and every other angle-bracket token stand for values the server copies from the context JSON at run time. Real numbers never appear in this file, so the examples cannot leak into answers.

## Example 1: the user moved one predictor

Context (abridged): scenario probability `<P_SCEN>`, baseline probability `<P_BASE>`, threshold `<THRESHOLD>`, changed_predictors `[{name: "<FEATURE_A>", old: "<OLD_A>", new: "<NEW_A>", unit: "<UNIT_A>"}]`, top scenario contributions `<FEATURE_A>: <C_A> (raises)`, `<FEATURE_B>: <C_B> (lowers)`, `<FEATURE_C>: <C_C> (raises)`, no extrapolation flags.

Question: "Why did the score go up?"

Answer:
The model now gives this profile a <P_SCEN> probability of <TARGET_LABEL>, up from <P_BASE> for the baseline, which is above the <THRESHOLD> decision threshold.
- <FEATURE_A_LABEL> at <NEW_A> <UNIT_A> raises the prediction by <C_A_POINTS> points; you changed it from <OLD_A>.
- <FEATURE_B_LABEL> at <VALUE_B> lowers the prediction by <C_B_POINTS> points.
- <FEATURE_C_LABEL> at <VALUE_C> raises the prediction by <C_C_POINTS> points.
In the what-if results, changing <WHATIF_FEATURE> to <WHATIF_TO> would bring the probability to <WHATIF_P>.
These are associations the model learned from past data, not causes.

## Example 2: a predictor is outside its training range and the user asks for advice

Context (abridged): extrapolation_flags `{<FEATURE_D>: true}`, scenario probability `<P_SCEN>`, what_if empty.

Question: "Should we give this customer a discount?"

Answer:
The model gives this profile a <P_SCEN> probability of <TARGET_LABEL>, compared with <P_BASE> for the baseline.
- <FEATURE_D_LABEL> at <VALUE_D> is outside the range the model was trained on, so this estimate is less reliable than usual.
- <FEATURE_E_LABEL> at <VALUE_E> lowers the prediction by <C_E_POINTS> points.
No single change inside the training range moves this profile across the decision threshold.
I can describe what the model predicts, but deciding on a discount is a business decision the model output alone cannot make.
