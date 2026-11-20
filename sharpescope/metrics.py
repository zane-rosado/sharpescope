"""Core return and risk-adjusted performance metrics.

Every function works on *daily returns* (pandas Series/DataFrame) and takes
`periods`, the number of return observations per year:
  * 252 for stocks/ETFs (or any mix that includes them)
  * 365 for crypto-only portfolios (crypto trades every day)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
CRYPTO_DAYS = 365


def daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple daily percentage returns from a price table."""
    return prices.sort_index().pct_change().dropna(how="all")


def normalize_weights(weights: dict[str, float], columns) -> pd.Series:
    """Validate weights and scale them so they sum to 1."""
    missing = set(weights) - set(columns)
    if missing:
        raise ValueError(f"No price data for: {', '.join(sorted(missing))}")
    w = pd.Series(weights, dtype=float)
    if (w < 0).any():
        raise ValueError("Weights must be non-negative (long-only).")
    if w.sum() == 0:
        raise ValueError("Weights must not all be zero.")
    return w / w.sum()


REBALANCE_OPTIONS = ("daily", "monthly", "quarterly", "yearly", "none")
_FREQ = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float] | None = None,
                      rebalance: str = "daily") -> pd.Series:
    """Daily returns of a portfolio.

    rebalance:
      * "daily"     - weights reset to target every day (the textbook assumption)
      * "monthly" / "quarterly" / "yearly" - reset on the first trading day of each period
      * "none"      - buy and hold: buy once at the start, then let winners grow
                      (this is how a brokerage or Bloomberg "shares" portfolio behaves)
    With no weights, every column gets an equal weight.
    """
    if weights is None:
        weights = {t: 1.0 for t in returns.columns}
    if rebalance not in REBALANCE_OPTIONS:
        raise ValueError(f"rebalance must be one of {REBALANCE_OPTIONS}")
    w = normalize_weights(weights, returns.columns)
    r = returns[w.index]
    if rebalance == "daily":
        return r.dot(w).rename("Portfolio")

    growth = (1 + r.fillna(0)).values
    target = w.values
    if rebalance == "none":
        period = np.zeros(len(r), dtype=int)
    else:
        period = r.index.to_period(_FREQ[rebalance]).asi8
    holdings = target.copy()          # value of each position, starting from $1 total
    out = np.empty(len(r))
    for i in range(len(r)):
        if i > 0 and period[i] != period[i - 1]:
            holdings = holdings.sum() * target   # rebalance at the previous close
        before = holdings.sum()
        holdings = holdings * growth[i]
        out[i] = holdings.sum() / before - 1
    return pd.Series(out, index=r.index, name="Portfolio")


def weights_over_time(returns: pd.DataFrame, weights: dict[str, float], rebalance: str = "none") -> pd.DataFrame:
    """How each position's share of the portfolio drifts (useful for buy-and-hold)."""
    w = normalize_weights(weights, returns.columns)
    r = returns[w.index].fillna(0)
    if rebalance == "daily":
        return pd.DataFrame([w.values] * len(r), index=r.index, columns=w.index)
    period = np.zeros(len(r), dtype=int) if rebalance == "none" else r.index.to_period(_FREQ[rebalance]).asi8
    holdings, rows = w.values.copy(), []
    for i, g in enumerate((1 + r).values):
        if i > 0 and period[i] != period[i - 1]:
            holdings = holdings.sum() * w.values
        holdings = holdings * g
        rows.append(holdings / holdings.sum())
    return pd.DataFrame(rows, index=r.index, columns=w.index)


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Compound annual growth rate (CAGR)."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    growth = (1 + r).prod()
    years = len(r) / periods
    return float(growth ** (1 / years) - 1)


def total_return(returns: pd.Series) -> float:
    """Cumulative return over the whole period."""
    r = returns.dropna()
    return float((1 + r).prod() - 1) if not r.empty else float("nan")


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of daily returns."""
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio. `risk_free` is an annual rate (0.04 = 4%)."""
    excess = returns.dropna() - risk_free / periods
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised Sortino ratio (only downside volatility counts as risk)."""
    excess = returns.dropna() - risk_free / periods
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    if downside == 0 or np.isnan(downside):
        return float("nan")
    return float(excess.mean() / downside * np.sqrt(periods))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Percentage below the running peak, for every day."""
    wealth = (1 + returns.dropna()).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss (a negative number)."""
    dd = drawdown_series(returns)
    return float(dd.min()) if not dd.empty else float("nan")


def calmar_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annual return divided by the size of the worst drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return annualized_return(returns, periods) / abs(mdd)


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Sensitivity of returns to a benchmark (market beta)."""
    df = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(df) < 2:
        return float("nan")
    var = df.iloc[:, 1].var(ddof=1)
    if var == 0:
        return float("nan")
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / var)


def alpha(returns: pd.Series, benchmark: pd.Series, risk_free: float = 0.0,
          periods: int = TRADING_DAYS) -> float:
    """Jensen's alpha (annualised): return beyond what the benchmark exposure (beta) explains."""
    df = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(df) < 2:
        return float("nan")
    rf = risk_free / periods
    b = beta(df.iloc[:, 0], df.iloc[:, 1])
    return float(((df.iloc[:, 0] - rf).mean() - b * (df.iloc[:, 1] - rf).mean()) * periods)


def tracking_error(returns: pd.Series, benchmark: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualised volatility of the gap between you and the benchmark."""
    diff = (returns - benchmark).dropna()
    return float(diff.std(ddof=1) * np.sqrt(periods)) if len(diff) > 1 else float("nan")


def information_ratio(returns: pd.Series, benchmark: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Average outperformance per unit of tracking error (consistency of beating the benchmark)."""
    diff = (returns - benchmark).dropna()
    sd = diff.std(ddof=1)
    if len(diff) < 2 or sd == 0:
        return float("nan")
    return float(diff.mean() / sd * np.sqrt(periods))


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Compounded return for each calendar month: rows = years, columns = months, plus a Year column."""
    r = returns.dropna()
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    table = m.unstack()
    table = table.reindex(columns=range(1, 13))
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    table["Year"] = (1 + r).groupby(r.index.year).prod() - 1
    table.index.name = None
    return table


def rolling_sharpe(returns: pd.Series, window: int = 126, risk_free: float = 0.0,
                   periods: int = TRADING_DAYS) -> pd.Series:
    """Sharpe ratio over a moving window (default about 6 months)."""
    excess = returns - risk_free / periods
    return excess.rolling(window).mean() / excess.rolling(window).std(ddof=1) * np.sqrt(periods)


def rolling_volatility(returns: pd.Series, window: int = 63, periods: int = TRADING_DAYS) -> pd.Series:
    """Annualised volatility over a moving window (default about 3 months)."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods)


def summary(
    returns: pd.DataFrame | pd.Series,
    risk_free: float = 0.0,
    benchmark: pd.Series | None = None,
    periods: int = TRADING_DAYS,
) -> pd.DataFrame:
    """One row of metrics per column of returns."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        row = {
            "Total Return": total_return(r),
            "Annual Return": annualized_return(r, periods),
            "Volatility": annualized_volatility(r, periods),
            "Sharpe": sharpe_ratio(r, risk_free, periods),
            "Sortino": sortino_ratio(r, risk_free, periods),
            "Max Drawdown": max_drawdown(r),
        }
        if benchmark is not None:
            row["Beta"] = beta(r, benchmark)
            row["Alpha"] = alpha(r, benchmark, risk_free, periods)
            row["Info Ratio"] = information_ratio(r, benchmark, periods)
        rows[col] = row
    out = pd.DataFrame(rows).T
    out.index.name = None
    return out


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation of daily returns."""
    corr = returns.corr()
    corr.index.name = None
    corr.columns.name = None
    return corr
