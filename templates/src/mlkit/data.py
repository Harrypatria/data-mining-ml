"""Simulated datasets. Every value is generated at run time from a seed; nothing is downloaded."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

PLANS = ("basic", "standard", "premium")
REGIONS = ("north", "south", "east", "west")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _intercept_for_prevalence(z: np.ndarray, prevalence: float) -> float:
    """Bisection on the intercept so the mean simulated probability equals the requested prevalence."""
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(mid + z).mean() > prevalence:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def make_customer_churn(n: int = 5000, prevalence: float = 0.15, seed: int = 42,
                        missing_rate: float = 0.03) -> pd.DataFrame:
    """Simulated subscription customers with a binary `churned` target.

    One row per customer (customer_id is unique). Effects are built in so models have
    signal to find: short tenure, many support calls, low usage, the basic plan and no
    autopay raise churn risk. `usage_hours` is missing at random, more often on the basic
    plan, so the pipeline needs an imputer.
    """
    rng = np.random.default_rng(seed)
    plan = rng.choice(PLANS, n, p=[0.5, 0.35, 0.15])
    plan_rank = np.array([PLANS.index(p) for p in plan])
    region = rng.choice(REGIONS, n)
    tenure = rng.integers(1, 73, n)
    monthly_spend = np.round(rng.lognormal(mean=3.4 + 0.35 * plan_rank, sigma=0.45, size=n), 2)
    support_calls = rng.poisson(1.0 + 0.9 * (plan == "basic"), n)
    usage_hours = np.round(np.clip(rng.gamma(4.0, 6.0, n) + 0.25 * tenure, 0, None), 1)
    autopay = rng.random(n) < 0.55
    z = (-0.045 * tenure + 0.42 * support_calls - 0.03 * usage_hours + 0.55 * (plan == "basic")
         - 0.65 * autopay + 0.006 * monthly_spend + 0.25 * (region == "south")
         + rng.normal(0, 0.5, n))
    p = _sigmoid(_intercept_for_prevalence(z, prevalence) + z)
    churned = (rng.random(n) < p).astype(int)
    usage = usage_hours.astype(float)
    missing = rng.random(n) < missing_rate * (1 + (plan == "basic"))
    usage[missing] = np.nan
    return pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(n)],
        "tenure_months": tenure,
        "monthly_spend": monthly_spend,
        "support_calls": support_calls,
        "usage_hours": usage,
        "plan": plan,
        "region": region,
        "autopay": autopay,
        "churned": churned,
    })


def make_imbalanced(n: int = 20000, prevalence: float = 0.05, seed: int = 42, class_sep: float = 1.0,
                    n_features: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Binary classification data at a chosen prevalence (fraction of positives)."""
    X, y = make_classification(n_samples=n, n_features=n_features, n_informative=6, n_redundant=2,
                               n_clusters_per_class=1, weights=[1 - prevalence], flip_y=0.0, class_sep=class_sep,
                               random_state=seed)
    return X, y
