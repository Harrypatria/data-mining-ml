# Time series and survival analysis

## Time series workflow

1. Plot the series with its rolling mean and standard deviation; mark known events.
2. Decompose (`statsmodels.tsa.seasonal.STL`) into trend, seasonality and remainder.
3. Stationarity: ADF (null: unit root) and KPSS (null: stationary). Disagreement means trend-stationary or difference-stationary; difference or detrend accordingly.
4. Baselines: last value, seasonal naive, moving average. Every model must beat the seasonal naive.
5. Models: ETS (`ExponentialSmoothing`) for level, trend and seasonality; ARIMA or SARIMA via ACF and PACF or information criteria; gradient boosting on lag features for many related series.
6. Residual diagnostics: Ljung-Box for remaining autocorrelation, residual Q-Q plot, heteroscedasticity over time.
7. Validation: rolling origin (`TimeSeriesSplit` or expanding windows) with the forecast horizon as the test window and a gap equal to the horizon.
8. Metrics: MAE, WAPE, RMSE for level; MASE against seasonal naive; pinball loss and interval coverage for quantile forecasts.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

print(adfuller(y.dropna())[1], kpss(y.dropna(), nlags="auto")[1])
fit = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
print(acorr_ljungbox(fit.resid, lags=[12]))
```

Pitfalls: random splits, features computed with future data, scaling on the full series, comparing models on different horizons.

## Survival analysis

Use when the outcome is time until an event and some units have not had the event yet (censoring).

1. Define the origin (signup date), the event (cancellation) and censoring (still active at the data cutoff).
2. Kaplan-Meier curves by group with confidence bands; log-rank test for differences.
3. Cox proportional hazards for adjusted hazard ratios; check proportional hazards with Schoenfeld residuals; stratify or add time interactions when violated.
4. Report median survival with CI, hazard ratios with CI, number at risk under curves.
5. For prediction: concordance index (C-index), time-dependent AUC, integrated Brier score.

```python
from lifelines import CoxPHFitter, KaplanMeierFitter
km = KaplanMeierFitter().fit(df.duration, event_observed=df.event)
cph = CoxPHFitter().fit(df[["duration", "event", *covariates]], "duration", "event")
cph.check_assumptions(df[["duration", "event", *covariates]], p_value_threshold=0.05)
```

Pitfalls: dropping censored units (biases toward short durations), treating censoring as the event, immortal time bias from covariates measured after the origin.
