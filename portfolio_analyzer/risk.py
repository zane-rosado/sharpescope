"""Value at Risk (VaR) and Conditional VaR (Expected Shortfall).

All functions return losses as POSITIVE fractions: 0.021 means "a 2.1% loss".
`level` is the confidence level (0.95 = "on 19 days out of 20 you won't lose more").
`horizon` is in days; multi-day figures use the square-root-of-time rule.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


def var_historical(returns: pd.Series, level: float = 0.95, horizon: int = 1) -> float:
    """Loss threshold taken straight from the past distribution of daily returns."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    q = np.quantile(r, 1 - level)
    return float(max(-q, 0.0) * np.sqrt(horizon))


def var_parametric(returns: pd.Series, level: float = 0.95, horizon: int = 1) -> float:
    """Loss threshold assuming normally distributed returns (mean-variance VaR)."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    z = NormalDist().inv_cdf(1 - level)
    mu, sd = r.mean() * horizon, r.std(ddof=1) * np.sqrt(horizon)
    return float(max(-(mu + z * sd), 0.0))


def cvar_historical(returns: pd.Series, level: float = 0.95, horizon: int = 1) -> float:
    """Average loss on the days that are worse than the VaR threshold."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    q = np.quantile(r, 1 - level)
    tail = r[r <= q]
    return float(max(-tail.mean(), 0.0) * np.sqrt(horizon))


def risk_table(returns: pd.DataFrame | pd.Series, level: float = 0.95, value: float = 10_000) -> pd.DataFrame:
    """1-day and 1-month (21 trading days) VaR/CVaR for every column, in % and $."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    rows = {}
    for col in returns.columns:
        r = returns[col]
        v1 = var_historical(r, level)
        rows[col] = {
            "1-Day VaR": v1,
            "1-Day CVaR": cvar_historical(r, level),
            "1-Month VaR": var_historical(r, level, 21),
            "Parametric 1-Day VaR": var_parametric(r, level),
            "1-Day VaR ($)": v1 * value,
        }
    return pd.DataFrame(rows).T
