"""One function that runs the whole analysis.

The web app, the command line and the PDF report all call `analyze()`, so a new
metric only has to be added here once to show up everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import metrics, optimize, risk
from .data import align_prices


@dataclass
class Analysis:
    prices: pd.DataFrame
    returns: pd.DataFrame          # asset returns + "Portfolio"
    weights: pd.Series             # your weights (sum to 1)
    periods: int
    risk_free: float
    value: float
    var_level: float
    benchmark: str | None
    rebalance: str
    summary: pd.DataFrame          # return/risk table, one row per asset + Portfolio
    risk: pd.DataFrame             # VaR table
    correlation: pd.DataFrame
    optimal: pd.DataFrame          # columns: Your weights, Max Sharpe, Min Volatility
    optimal_stats: pd.DataFrame    # Return/Volatility/Sharpe for each weight set
    frontier: pd.DataFrame | None
    current_weights: pd.Series | None = None   # where weights drifted to (buy & hold / periodic)
    notes: list[str] = field(default_factory=list)

    @property
    def assets(self) -> list[str]:
        return list(self.weights.index)

    @property
    def start(self) -> pd.Timestamp:
        return self.returns.index.min()

    @property
    def end(self) -> pd.Timestamp:
        return self.returns.index.max()

    @property
    def years(self) -> float:
        return len(self.returns) / self.periods


def analyze(
    prices: pd.DataFrame,
    weights: dict[str, float],
    risk_free: float = 0.04,
    benchmark: str | None = None,
    value: float = 10_000,
    var_level: float = 0.95,
    max_weight: float = 1.0,
    frontier_points: int = 4000,
    rebalance: str = "daily",
    start: str | None = None,
    end: str | None = None,
) -> Analysis:
    """Run every calculation for a portfolio. `prices` must include the benchmark if given.

    `start`/`end` (YYYY-MM-DD, inclusive) trim the prices. Returns are measured from the
    close on the first day, so start on the day you bought.
    """
    prices = prices.sort_index()
    if start:
        prices = prices.loc[pd.Timestamp(start):]
    if end:
        prices = prices.loc[: pd.Timestamp(end)]
    keep = list(dict.fromkeys(list(weights) + ([benchmark] if benchmark else [])))
    missing = [t for t in keep if t not in prices.columns]
    if missing:
        raise ValueError(f"No price data for: {', '.join(missing)}")
    prices, periods, notes = align_prices(prices[keep])
    asset_rets = metrics.daily_returns(prices)
    w = metrics.normalize_weights(weights, asset_rets.columns)
    assets = list(w.index)

    port = metrics.portfolio_returns(asset_rets[assets], w.to_dict(), rebalance)
    bench = asset_rets[benchmark] if benchmark and benchmark in asset_rets else None
    cols = [asset_rets[assets], port]
    if bench is not None and benchmark not in assets:
        cols.append(bench.rename(f"{benchmark} (benchmark)"))
    rets = pd.concat(cols, axis=1)

    table = metrics.summary(rets, risk_free, bench, periods)
    table["Calmar"] = [metrics.calmar_ratio(rets[c], periods) for c in rets.columns]
    risk_tbl = risk.risk_table(rets, var_level, value)
    if rebalance != "daily":
        notes.append({"none": "Buy and hold: bought once at the start, no rebalancing (like a brokerage account).",
                      "monthly": "Rebalanced back to your weights at the start of every month.",
                      "quarterly": "Rebalanced back to your weights at the start of every quarter.",
                      "yearly": "Rebalanced back to your weights at the start of every year."}[rebalance])

    if len(assets) >= 2:
        ms = optimize.max_sharpe(asset_rets[assets], risk_free, periods, max_weight)
        mv = optimize.min_volatility(asset_rets[assets], periods, max_weight)
        optimal = pd.DataFrame({"Your weights": w, "Max Sharpe": ms, "Min Volatility": mv})
        stats = {c: optimize.portfolio_stats(optimal[c], asset_rets[assets], risk_free, periods)
                 for c in optimal.columns}
        optimal_stats = pd.DataFrame(stats).T.rename(columns={"Return": "Avg Return"})
        frontier = optimize.random_portfolios(asset_rets[assets], frontier_points, risk_free,
                                              periods, max_weight)
    else:
        optimal = pd.DataFrame({"Your weights": w})
        optimal_stats = pd.DataFrame()
        frontier = None
        notes.append("Add a second asset to see the optimiser.")

    return Analysis(
        prices=prices, returns=rets, weights=w, periods=periods, risk_free=risk_free,
        value=value, var_level=var_level, benchmark=benchmark if bench is not None else None,
        rebalance=rebalance,
        summary=table, risk=risk_tbl, correlation=metrics.correlation_matrix(asset_rets[assets]),
        optimal=optimal, optimal_stats=optimal_stats, frontier=frontier, notes=notes,
        current_weights=None if rebalance == "daily" else
        metrics.weights_over_time(asset_rets[assets], w.to_dict(), rebalance).iloc[-1].rename("Current weight"),
    )


def _sharpe_word(s: float) -> str:
    if s != s:  # NaN
        return "n/a"
    if s < 0:
        return "negative (you'd have done better in T-bills)"
    if s < 0.5:
        return "weak"
    if s < 1:
        return "decent"
    if s < 2:
        return "strong"
    return "exceptional (check the period isn't too short)"


def plain_english(a: Analysis) -> list[str]:
    """Short sentences that explain the results to a non-expert."""
    p = a.summary.loc["Portfolio"]
    r = a.risk.loc["Portfolio"]
    level = int(round(a.var_level * 100))
    odds = int(round(1 / (1 - a.var_level)))
    lines = [
        f"Over {a.years:.1f} years ({a.start:%b %d, %Y} to {a.end:%b %d, %Y}), your portfolio grew "
        f"{p['Annual Return']:.1%} a year, turning ${a.value:,.0f} into "
        f"${a.value * (1 + metrics.total_return(a.returns['Portfolio'])):,.0f}.",
        f"Its Sharpe ratio is {p['Sharpe']:.2f}, which is {_sharpe_word(p['Sharpe'])}.",
        f"Typical yearly swing (volatility): {p['Volatility']:.1%}. "
        f"Worst fall from a peak: {p['Max Drawdown']:.1%}.",
        f"Value at Risk ({level}%): on about 1 day in {odds}, expect to lose more than "
        f"{r['1-Day VaR']:.2%} (${r['1-Day VaR ($)']:,.0f} on ${a.value:,.0f}).",
    ]
    bench_row = f"{a.benchmark} (benchmark)" if a.benchmark and a.benchmark not in a.assets else a.benchmark
    if bench_row and bench_row in a.summary.index:
        b = a.summary.loc[bench_row]
        diff = p["Total Return"] - b["Total Return"]
        lines.insert(1, f"Total return {p['Total Return']:.2%} vs {b['Total Return']:.2%} for {a.benchmark}: "
                        f"you {'beat' if diff >= 0 else 'trailed'} the benchmark by {abs(diff) * 100:.2f} percentage points.")
    if "Max Sharpe" in a.optimal:
        best = a.optimal["Max Sharpe"]
        top = best[best > 0.005].sort_values(ascending=False)
        mix = ", ".join(f"{t} {x:.0%}" for t, x in top.items())
        lines.append(
            f"The max-Sharpe mix over this period would have been {mix}, with a Sharpe of "
            f"{a.optimal_stats.loc['Max Sharpe', 'Sharpe']:.2f} vs. "
            f"{a.optimal_stats.loc['Your weights', 'Sharpe']:.2f} for your weights, both held fixed with daily "
            f"rebalancing (based on past data, not a forecast)."
        )
    return lines
