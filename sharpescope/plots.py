"""Charts. Every function returns a matplotlib Figure (used by the app and the PDF)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # works without a screen; the CLI switches back when showing charts
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

from .metrics import drawdown_series, rolling_sharpe

PORTFOLIO_COLOR = "#d62728"
# tab10/tab20 without red, so no asset can be confused with the portfolio line
ASSET_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#c5b0d5", "#c49c94",
                "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"]
STYLE = {
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
}


def _lw(col: str) -> float:
    return 2.4 if col == "Portfolio" else 1.1


def growth_and_drawdown(returns: pd.DataFrame, value: float = 10_000, max_assets: int = 8):
    """Growth of $value and drawdowns. With many assets, only the portfolio and benchmark are drawn."""
    cols = list(returns.columns)
    special = [c for c in cols if c == "Portfolio" or c.endswith("(benchmark)")]
    if len(cols) - len(special) > max_assets:
        cols = special
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, height_ratios=[2, 1])
        growth = value * (1 + returns[cols].fillna(0)).cumprod()
        for i, col in enumerate(cols):
            if col == "Portfolio":
                kw = {"color": PORTFOLIO_COLOR}
            elif col.endswith("(benchmark)"):
                kw = {"color": "black", "linestyle": "--"}
            else:
                kw = {"color": ASSET_COLORS[i % len(ASSET_COLORS)]}
            ax1.plot(growth.index, growth[col], label=col, linewidth=_lw(col), **kw)
            ax2.plot(returns.index, drawdown_series(returns[col].fillna(0)), linewidth=_lw(col), **kw)
        ax1.set_title(f"Growth of ${value:,.0f}")
        ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax1.legend(frameon=False, ncol=min(len(cols), 5), loc="upper left")
        ax2.set_title("Drawdown (% below previous peak)")
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        fig.tight_layout()
    return fig


def rolling_sharpe_chart(returns: pd.Series, risk_free: float, periods: int, window: int = 126):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 3))
        rs = rolling_sharpe(returns, window, risk_free, periods)
        ax.plot(rs.index, rs, color=PORTFOLIO_COLOR, linewidth=1.6)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(f"Rolling Sharpe ratio ({window}-day window)")
        fig.tight_layout()
    return fig


def return_histogram(returns: pd.Series, var: float, cvar: float, level: float):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.hist(returns.dropna(), bins=60, color="#7f8c8d", alpha=0.8)
        ax.axvline(-var, color=PORTFOLIO_COLOR, linewidth=1.8, label=f"{level:.0%} VaR: -{var:.2%}")
        ax.axvline(-cvar, color="#8e44ad", linewidth=1.8, linestyle="--", label=f"CVaR: -{cvar:.2%}")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
        ax.set_title("Daily returns of your portfolio")
        ax.set_ylabel("Number of days")
        ax.legend(frameon=False)
        fig.tight_layout()
    return fig


def efficient_frontier(frontier: pd.DataFrame, optimal_stats: pd.DataFrame):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sc = ax.scatter(frontier["Volatility"], frontier["Return"], c=frontier["Sharpe"],
                        cmap="viridis", s=6, alpha=0.6)
        fig.colorbar(sc, ax=ax, label="Sharpe ratio")
        markers = {"Your weights": ("o", PORTFOLIO_COLOR), "Max Sharpe": ("*", "gold"),
                   "Min Volatility": ("D", "#1f77b4"), "Risk Parity": ("s", "#2ca02c")}
        for name, row in optimal_stats.iterrows():
            m, c = markers.get(name, ("o", "black"))
            ax.scatter(row["Volatility"], row.get("Avg Return", row.get("Return")), marker=m, s=220 if m == "*" else 90,
                       color=c, edgecolor="black", zorder=5, label=name)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Volatility (risk)")
        ax.set_ylabel("Expected annual return")
        ax.set_title("Every possible mix of your assets")
        ax.legend(frameon=True, loc="lower right")
        fig.tight_layout()
    return fig


def weights_chart(optimal: pd.DataFrame):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 0.5 * len(optimal) + 1.6))
        optimal.plot.barh(ax=ax, width=0.8, color=[PORTFOLIO_COLOR, "#f1c40f", "#1f77b4", "#2ca02c"][: optimal.shape[1]])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.invert_yaxis()
        ax.set_title("Weights: yours vs. optimised")
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
    return fig


def correlation_heatmap(corr: pd.DataFrame):
    with plt.rc_context(STYLE):
        n = len(corr)
        fig, ax = plt.subplots(figsize=(min(1 + 0.8 * n, 9), min(0.8 + 0.7 * n, 7)))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(n), corr.index)
        ax.grid(False)
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Correlation (1 = move together, 0 = unrelated)")
        fig.tight_layout()
    return fig


def save(fig, path: str) -> str:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def monte_carlo_fan(mc, title: str | None = None):
    """Fan chart of simulated future values (5-95% and 25-75% bands, median line)."""
    p = mc.percentiles
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.fill_between(p.index, p["5%"], p["95%"], color=PORTFOLIO_COLOR, alpha=0.12, label="90% of outcomes")
        ax.fill_between(p.index, p["25%"], p["75%"], color=PORTFOLIO_COLOR, alpha=0.25, label="Middle 50%")
        ax.plot(p.index, p["50%"], color=PORTFOLIO_COLOR, linewidth=2.2, label="Median")
        months = len(p) - 1
        invested = mc.start_value + (mc.invested - mc.start_value) * (p.index * 12 / max(months, 1))
        ax.plot(p.index, invested, color="black", linestyle="--", linewidth=1, label="Money put in")
        if mc.goal:
            ax.axhline(mc.goal, color="#2ca02c", linestyle=":", linewidth=1.5, label=f"Goal ${mc.goal:,.0f}")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.set_xlabel("Years from today")
        ax.set_title(title or f"Monte Carlo: {len(mc.final_values):,} possible futures over {mc.years} years")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
    return fig


def monthly_heatmap(table: pd.DataFrame):
    """Calendar of monthly returns: green = up month, red = down month."""
    import numpy as np

    data = table.drop(columns=["Year"])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 0.55 * len(data) + 1.4))
        lim = max(0.05, float(np.nanmax(np.abs(data.values)))) if data.size else 0.05
        im = ax.imshow(data.values.astype(float), cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(12), data.columns)
        ax.set_yticks(range(len(data)), [str(y) for y in data.index])
        ax.grid(False)
        for i in range(data.shape[0]):
            for j in range(12):
                v = data.values[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=7.5)
        ax.set_title("Monthly returns of your portfolio")
        fig.tight_layout()
    return fig


def scenario_bars(table: pd.DataFrame, column: str = "Portfolio", title: str = "Stress tests"):
    """Horizontal bars of the portfolio's estimated move in each scenario."""
    s = table[column].sort_values()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(s) + 1.2))
        colors = ["#c0392b" if v < 0 else "#27ae60" for v in s.values]
        ax.barh(s.index, s.values, color=colors)
        for i, v in enumerate(s.values):
            ax.text(v, i, f" {v:+.1%} ", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        lo, hi = min(s.min(), 0), max(s.max(), 0)
        pad = (hi - lo) * 0.25 + 0.01
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(title)
        fig.tight_layout()
    return fig


def sharpe_vs_rf_chart(series: pd.Series, current: float):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(series.index, series.values, marker="o", color=PORTFOLIO_COLOR)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Risk-free rate (T-bill yield)")
        ax.set_ylabel("Sharpe ratio")
        ax.set_title(f"How your Sharpe ratio changes with the risk-free rate (you chose {current:.2%})")
        fig.tight_layout()
    return fig
