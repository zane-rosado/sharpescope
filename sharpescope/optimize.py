"""Long-only portfolio optimisation: maximum Sharpe ratio and minimum volatility.

Uses scipy if it is installed (exact answer); otherwise falls back to a large
random search, which gets very close.

Important: optimisers are only as good as their inputs. These weights are what
*would have* worked best over the chosen period, not a forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS


def _stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple[float, float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    sharpe = (ret - rf) / vol if vol > 0 else float("nan")
    return ret, vol, sharpe


def _inputs(returns: pd.DataFrame, periods: int):
    r = returns.dropna()
    if r.shape[1] < 2:
        raise ValueError("Pick at least two assets to optimise.")
    return r.mean().values * periods, r.cov().values * periods


def random_portfolios(returns: pd.DataFrame, n: int = 5000, risk_free: float = 0.0,
                      periods: int = TRADING_DAYS, max_weight: float = 1.0,
                      seed: int = 0) -> pd.DataFrame:
    """Random long-only portfolios, used to draw the efficient-frontier cloud."""
    mu, cov = _inputs(returns, periods)
    k = len(mu)
    rng = np.random.default_rng(seed)
    # Mix of spread-out and concentrated portfolios so the cloud covers the edges.
    w = np.vstack([rng.dirichlet(np.ones(k), n // 2), rng.dirichlet(np.full(k, 0.3), n - n // 2)])
    if max_weight < 1:
        w = w[(w <= max_weight + 1e-9).all(axis=1)]
    rets = w @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    out = pd.DataFrame({"Return": rets, "Volatility": vols})
    out["Sharpe"] = (out["Return"] - risk_free) / out["Volatility"]
    for i, c in enumerate(returns.columns):
        out[c] = w[:, i]
    return out


def _solve(objective, k: int, max_weight: float, mu, cov, rf):
    max_weight = max(max_weight, 1.0 / k)  # weights must be able to reach 100%
    try:
        from scipy.optimize import minimize

        best = None
        for x0 in [np.full(k, 1 / k)] + [np.random.default_rng(i).dirichlet(np.ones(k)) for i in range(4)]:
            x0 = np.minimum(x0, max_weight)
            x0 = x0 / x0.sum()
            res = minimize(objective, x0, method="SLSQP", bounds=[(0, max_weight)] * k,
                           constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                           options={"maxiter": 500, "ftol": 1e-12})
            if res.success and (best is None or res.fun < best.fun):
                best = res
        if best is not None:
            w = np.clip(best.x, 0, None)
            return w / w.sum()
    except ImportError:
        pass
    # Fallback: random search
    rng = np.random.default_rng(42)
    w = np.vstack([rng.dirichlet(np.ones(k), 40_000), rng.dirichlet(np.full(k, 0.2), 40_000)])
    w = w[(w <= max_weight + 1e-9).all(axis=1)]
    scores = np.array([objective(x) for x in w])
    return w[int(np.nanargmin(scores))]


def max_sharpe(returns: pd.DataFrame, risk_free: float = 0.0, periods: int = TRADING_DAYS,
               max_weight: float = 1.0) -> pd.Series:
    """Weights (summing to 1, no shorting) with the highest historical Sharpe ratio."""
    mu, cov = _inputs(returns, periods)

    def neg_sharpe(w):
        _, vol, s = _stats(w, mu, cov, risk_free)
        return -s if vol > 0 else 1e6

    w = _solve(neg_sharpe, len(mu), max_weight, mu, cov, risk_free)
    return pd.Series(w, index=returns.columns, name="Max Sharpe").round(6)


def min_volatility(returns: pd.DataFrame, periods: int = TRADING_DAYS, max_weight: float = 1.0) -> pd.Series:
    """Weights (summing to 1, no shorting) with the lowest historical volatility."""
    mu, cov = _inputs(returns, periods)
    w = _solve(lambda w: float(w @ cov @ w), len(mu), max_weight, mu, cov, 0.0)
    return pd.Series(w, index=returns.columns, name="Min Volatility").round(6)


def portfolio_stats(weights: pd.Series, returns: pd.DataFrame, risk_free: float = 0.0,
                    periods: int = TRADING_DAYS) -> dict:
    """Expected return, volatility and Sharpe for a set of weights."""
    mu, cov = _inputs(returns[weights.index], periods)
    ret, vol, s = _stats(weights.values, mu, cov, risk_free)
    return {"Return": ret, "Volatility": vol, "Sharpe": s}


def risk_parity(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.Series:
    """Equal risk contribution: every asset adds the same amount of risk to the portfolio.

    Volatile assets (e.g. crypto) get small weights, steady ones (e.g. bonds) get large weights.
    """
    _, cov = _inputs(returns, periods)
    k = len(cov)
    inv = 1 / np.sqrt(np.diag(cov))
    x0 = inv / inv.sum()                     # inverse-volatility starting point
    try:
        from scipy.optimize import minimize

        def objective(w):
            port_var = w @ cov @ w
            contrib = w * (cov @ w) / port_var
            return float(((contrib - 1 / k) ** 2).sum())

        res = minimize(objective, x0, method="SLSQP", bounds=[(1e-6, 1)] * k,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       options={"maxiter": 1000, "ftol": 1e-14})
        w = np.clip(res.x, 0, None) if res.success else x0
    except ImportError:
        w = x0
    w = w / w.sum()
    return pd.Series(w, index=returns.columns, name="Risk Parity").round(6)


def risk_contributions(weights: pd.Series, returns: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.Series:
    """Share of total portfolio risk coming from each asset (sums to 100%)."""
    _, cov = _inputs(returns[weights.index], periods)
    w = weights.values
    return pd.Series(w * (cov @ w) / (w @ cov @ w), index=weights.index, name="Risk contribution")
