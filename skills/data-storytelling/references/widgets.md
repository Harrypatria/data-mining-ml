# Widget recipes

## Pattern: pure function plus thin widget

```python
import ipywidgets as widgets
from IPython.display import display

STATE = {"threshold": 0.3, "cost_fp": 15.0, "cost_fn": 120.0}

def threshold_view(threshold: float) -> dict:            # pure, testable, used by the static fallback
    m = mx.classification_metrics(y_val, p_val, threshold)
    return {"precision": m["precision"], "recall": m["recall"],
            "expected_cost": mx.expected_cost(y_val, p_val, threshold, STATE["cost_fp"], STATE["cost_fn"])}

def draw(threshold: float):
    STATE["threshold"] = threshold
    print({k: round(v, 3) for k, v in threshold_view(threshold).items()})

slider = widgets.FloatSlider(value=0.3, min=0.01, max=0.99, step=0.01, continuous_update=False,
                             description="Threshold", tooltip="Decision threshold", layout=widgets.Layout(width="420px"))
display(widgets.VBox([slider, widgets.interactive_output(draw, {"threshold": slider})]))
```

## Heavy computation

```python
@widgets.interact_manual(k=widgets.IntSlider(min=2, max=10, value=4, description="Clusters"))
def recluster(k):
    ...                                   # runs only when the user presses the button
```

Precompute grids (prevalence by remedy, thresholds) once, outside callbacks.

## Clamping and extrapolation

```python
slider = widgets.FloatSlider(min=feat["min"], max=feat["max"], step=feat["step"], value=feat["default"])
allow = widgets.Checkbox(description="Allow extrapolation")
# when allowed, rebuild sliders with a wider range and flag values outside [min, max]
```

## Layout and accessibility

- Group related controls in `VBox`; place paired inputs in `HBox`; use `GridBox` for comparison grids.
- `style={"description_width": "150px"}` so labels do not truncate.
- Describe every control (`description`, `tooltip`); buttons say what they do ("Reset to median profile").
- Keep plots small (about 5 by 3 inches) with labeled axes and takeaway titles.

## Static fallback cell

```python
print("threshold", threshold_view(0.3))
print("cost", cost_view(15, 120))
print(sensitivity_view("tenure_months", MEDIAN_PROFILE).round(3).to_string(index=False))
```

## Export notes

| Target | Interactive | How |
|---|---|---|
| Jupyter, JupyterLab, VS Code | Yes | Run the notebook |
| Colab | Yes (core widgets) | `output.enable_custom_widget_manager()` only for third-party widgets |
| Voila | Yes | `voila notebook.ipynb --strip_sources=True` |
| nbconvert HTML | Tables and static images only | `jupyter nbconvert --to html notebook.ipynb` |
| PDF, GitHub preview | Static only | Rely on the fallback cell |
