"""Core portfolio metrics.

All functions take pandas objects of *prices* or *daily returns* and
return plain numbers or DataFrames, so they are easy to test and reuse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple daily percentage returns from a price table."""
    return prices.sort_index().pct_change().dropna(how="all")


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Daily returns of a portfolio rebalanced daily to fixed weights.

    If no weights are given, every ticker gets an equal weight.
    Weights are normalised so they always sum to 1.
    """
    if weights is None:
        weights = {t: 1.0 for t in returns.columns}
    missing = set(weights) - set(returns.columns)
    if missing:
        raise ValueError(f"No price data for: {', '.join(sorted(missing))}")
    w = pd.Series(weights, dtype=float)
    if (w < 0).any():
        raise ValueError("Weights must be non-negative (long-only).")
    if w.sum() == 0:
        raise ValueError("Weights must not all be zero.")
    w = w / w.sum()
    return returns[w.index].dot(w).rename("Portfolio")


def annualized_return(returns: pd.Series) -> float:
    """Compound annual growth rate (CAGR) from daily returns."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    growth = (1 + r).prod()
    years = len(r) / TRADING_DAYS
    return float(growth ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns."""
    return float(returns.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio.

    risk_free is an annual rate (e.g. 0.04 for 4%).
    """
    excess = returns.dropna() - risk_free / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio (penalises only downside volatility)."""
    excess = returns.dropna() - risk_free / TRADING_DAYS
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    if downside == 0 or np.isnan(downside):
        return float("nan")
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Percentage below the running peak, for every day."""
    wealth = (1 + returns.dropna()).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss (a negative number)."""
    dd = drawdown_series(returns)
    return float(dd.min()) if not dd.empty else float("nan")


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Sensitivity of returns to a benchmark (market beta)."""
    df = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(df) < 2:
        return float("nan")
    var = df.iloc[:, 1].var(ddof=1)
    if var == 0:
        return float("nan")
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / var)


def summary(
    returns: pd.DataFrame | pd.Series,
    risk_free: float = 0.0,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """One row of metrics per column of returns."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        row = {
            "Annual Return": annualized_return(r),
            "Volatility": annualized_volatility(r),
            "Sharpe": sharpe_ratio(r, risk_free),
            "Sortino": sortino_ratio(r, risk_free),
            "Max Drawdown": max_drawdown(r),
        }
        if benchmark is not None:
            row["Beta"] = beta(r, benchmark)
        rows[col] = row
    return pd.DataFrame(rows).T


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation of daily returns."""
    return returns.corr()
