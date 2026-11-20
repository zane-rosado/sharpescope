"""Stress tests: "what happens to my portfolio if...?"

Three tools:

1. Factor exposures. For every asset we estimate, from its own daily history,
   how much it moves when (a) the stock market (SPY) moves and (b) the 10-year
   Treasury yield moves. The rate sensitivity is an *empirical duration*: a bond
   fund with a duration of 6 loses about 6% when rates rise 1 percentage point.

2. Hypothetical scenarios. Combine those sensitivities with a shock, e.g.
   "rates +2%, stocks -15%". This is a linear estimate: good for intuition,
   less precise for very large moves.

3. Historical replays. Re-run real crises (2008, 2020, 2022...) on today's
   portfolio. Assets that did not exist yet are estimated from their beta to
   the S&P 500 and clearly marked as estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import is_crypto

MARKET = "SPY"
RATES = "^TNX"   # CBOE 10-year Treasury yield index, quoted in percent (4.25 = 4.25%)

HYPOTHETICAL: dict[str, dict] = {
    "Rates +1%": {"market": 0.0, "rates": 1.0,
                  "why": "The Fed or bond market pushes the 10-year yield up 1 percentage point."},
    "Rates +2%": {"market": 0.0, "rates": 2.0,
                  "why": "A 2022-style rate shock: yields jump 2 points."},
    "Rates -1%": {"market": 0.0, "rates": -1.0,
                  "why": "Rates fall 1 point, e.g. the Fed cuts in a slowdown. Bonds usually gain."},
    "Stocks -10% (correction)": {"market": -0.10, "rates": 0.0,
                                 "why": "A routine market correction."},
    "Stocks -20% (bear market)": {"market": -0.20, "rates": 0.0,
                                  "why": "The textbook definition of a bear market."},
    "Stagflation (stocks -15%, rates +1.5%)": {"market": -0.15, "rates": 1.5,
                                               "why": "Inflation forces rates up while growth stalls: "
                                                      "stocks and bonds fall together, like 2022."},
    "Recession (stocks -25%, rates -1%)": {"market": -0.25, "rates": -1.0,
                                           "why": "A classic recession: stocks fall, investors flee "
                                                  "to Treasuries, yields drop."},
    "Crypto winter (crypto -60%)": {"market": 0.0, "rates": 0.0, "crypto": -0.60,
                                    "why": "A crypto-only crash like 2018 or 2022."},
}

HISTORICAL: dict[str, dict] = {
    "Dot-com crash (2000-02)": {"start": "2000-03-24", "end": "2002-10-09",
                                "why": "Tech bubble bursts; the Nasdaq falls almost 80%."},
    "Global Financial Crisis (2007-09)": {"start": "2007-10-09", "end": "2009-03-09",
                                          "why": "Housing and banking collapse; the worst bear market since the 1930s."},
    "2018 Q4 sell-off": {"start": "2018-09-20", "end": "2018-12-24",
                         "why": "Fed rate hikes and trade-war fears."},
    "COVID crash (2020)": {"start": "2020-02-19", "end": "2020-03-23",
                           "why": "The fastest 30%+ drop in history, then a fast recovery."},
    "2022 inflation & rate shock": {"start": "2022-01-03", "end": "2022-10-12",
                                    "why": "Rates rise the fastest in 40 years; stocks AND bonds fall."},
}


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ coef
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return coef[1:], float(r2)


def factor_exposures(returns: pd.DataFrame, market_returns: pd.Series, yield_change: pd.Series) -> pd.DataFrame:
    """Regress each column of daily returns on market returns and daily yield changes (in points)."""
    df = pd.concat([returns, market_returns.rename("_m"), yield_change.rename("_y")], axis=1, join="inner").dropna()
    if len(df) < 30:
        raise ValueError("Need at least 30 overlapping days to estimate stress sensitivities.")
    X = df[["_m", "_y"]].values
    rows = {}
    for col in returns.columns:
        (b_m, b_r), r2 = _ols(df[col].values, X)
        rows[col] = {"Market beta": b_m, "Rate sensitivity": b_r,
                     "Empirical duration": -100 * b_r, "R²": r2}
    out = pd.DataFrame(rows).T
    out.index.name = None
    return out


def hypothetical(exposures: pd.DataFrame, weights: pd.Series, value: float,
                 scenarios: dict[str, dict] | None = None) -> pd.DataFrame:
    """Estimated % and $ impact of each scenario on each asset and on the portfolio."""
    scenarios = scenarios or HYPOTHETICAL
    assets = list(weights.index)
    rows = {}
    for name, s in scenarios.items():
        if "crypto" in s and not any(is_crypto(a) for a in assets):
            continue
        r = {}
        for a in assets:
            e = exposures.loc[a]
            move = e["Market beta"] * s.get("market", 0) + e["Rate sensitivity"] * s.get("rates", 0)
            if is_crypto(a) and "crypto" in s:
                move = s["crypto"]
            r[a] = max(move, -1.0)
        port = float(sum(weights[a] * r[a] for a in assets))
        rows[name] = {"Portfolio": port, "Portfolio ($)": port * value, **r}
    return pd.DataFrame(rows).T


def custom_shock(exposures: pd.DataFrame, weights: pd.Series, market: float, rates: float,
                 crypto: float | None = None) -> tuple[float, pd.Series]:
    s = {"market": market, "rates": rates}
    if crypto is not None:
        s["crypto"] = crypto
    t = hypothetical(exposures, weights, 1.0, {"Custom": s})
    if t.empty:
        return 0.0, pd.Series(dtype=float)
    row = t.loc["Custom"]
    return float(row["Portfolio"]), row.drop(["Portfolio", "Portfolio ($)"])


def _window_return(series: pd.Series, start: str, end: str) -> float | None:
    s = series.dropna()
    if s.empty or s.index.min() > pd.Timestamp(start) + pd.Timedelta(days=7) or s.index.max() < pd.Timestamp(end):
        return None
    p0 = s.asof(pd.Timestamp(start))
    p1 = s.asof(pd.Timestamp(end))
    if pd.isna(p0) or pd.isna(p1):
        return None
    return float(p1 / p0 - 1)


def historical(history: pd.DataFrame, weights: pd.Series, market_beta: pd.Series, value: float,
               scenarios: dict[str, dict] | None = None, market: str = MARKET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay past crises. Returns (summary table, per-asset detail with 'actual'/'estimated' flags)."""
    scenarios = scenarios or HISTORICAL
    summary, detail = {}, {}
    for name, s in scenarios.items():
        m = _window_return(history[market], s["start"], s["end"]) if market in history else None
        if m is None:
            continue
        rets, flags = {}, {}
        for a in weights.index:
            actual = _window_return(history[a], s["start"], s["end"]) if a in history else None
            if actual is not None:
                rets[a], flags[a] = actual, "actual"
            else:
                rets[a], flags[a] = max(float(market_beta.get(a, 1.0)) * m, -1.0), "estimated"
        port = float(sum(weights[a] * rets[a] for a in weights.index))
        est = sum(weights[a] for a in weights.index if flags[a] == "estimated")
        summary[name] = {"Portfolio": port, "Portfolio ($)": port * value, "S&P 500 (SPY)": m,
                         "Share of portfolio estimated": est}
        detail[name] = {a: f"{rets[a]:+.1%} ({flags[a]})" for a in weights.index}
    return pd.DataFrame(summary).T, pd.DataFrame(detail).T


def bond_price_change(duration: float, yield_change_pts: float, convexity: float = 0.0) -> float:
    """Classic bond math: %ΔP ≈ -D·Δy + ½·C·Δy²  (Δy in decimal)."""
    dy = yield_change_pts / 100
    return -duration * dy + 0.5 * convexity * dy ** 2


def sharpe_vs_risk_free(returns: pd.Series, rates=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08),
                        periods: int = 252) -> pd.Series:
    """How the Sharpe ratio changes if the risk-free rate (T-bill yield) is different."""
    from .metrics import sharpe_ratio

    return pd.Series({f"{rf:.0%}": sharpe_ratio(returns, rf, periods) for rf in rates}, name="Sharpe")
