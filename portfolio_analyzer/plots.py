"""Charts for growth of $10,000 and drawdowns."""

from __future__ import annotations

import pandas as pd

from .metrics import drawdown_series


def plot_growth_and_drawdown(returns: pd.DataFrame, save_path: str | None = None, show: bool = True):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[2, 1])
    growth = 10_000 * (1 + returns.fillna(0)).cumprod()
    for col in growth.columns:
        lw = 2.5 if col == "Portfolio" else 1.2
        ax1.plot(growth.index, growth[col], label=col, linewidth=lw)
        ax2.plot(returns.index, drawdown_series(returns[col].fillna(0)) * 100, linewidth=lw)
    ax1.set_title("Growth of $10,000")
    ax1.set_ylabel("Value ($)")
    ax1.legend(frameon=False)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("% below peak")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    elif show:
        plt.show()
    return fig
