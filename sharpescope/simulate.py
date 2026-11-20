"""Monte Carlo simulation of future portfolio value.

Method: circular block bootstrap. We build thousands of possible futures by stitching together
random ~1-month chunks of the portfolio's real daily returns. Keeping whole chunks
preserves fat tails and short-term momentum that a bell-curve model would miss.

Optionally the average return can be overridden (e.g. 7% a year) so that an unusually
good or bad history does not dominate the projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS


@dataclass
class MonteCarloResult:
    percentiles: pd.DataFrame   # index = years (0..N, monthly steps), columns = 5%, 25%, 50%, 75%, 95%
    final_values: np.ndarray    # ending value of every simulated path
    invested: float             # starting value + all contributions
    start_value: float
    years: int
    goal: float | None
    method: str

    def stats(self) -> dict:
        f = self.final_values
        out = {
            "Median ending value": float(np.median(f)),
            "Bad case (5th percentile)": float(np.percentile(f, 5)),
            "Good case (95th percentile)": float(np.percentile(f, 95)),
            "Total invested": self.invested,
            "Chance of ending below what you put in": float((f < self.invested).mean()),
            "Median annual growth": float((np.median(f) / self.invested) ** (1 / self.years) - 1)
            if self.invested > 0 else float("nan"),
        }
        if self.goal:
            out[f"Chance of reaching ${self.goal:,.0f}"] = float((f >= self.goal).mean())
        return out


def monte_carlo(
    returns: pd.Series,
    years: int = 10,
    n_paths: int = 2000,
    start_value: float = 10_000,
    monthly_contribution: float = 0.0,
    expected_return: float | None = None,
    periods: int = TRADING_DAYS,
    block: int = 21,
    goal: float | None = None,
    seed: int = 42,
) -> MonteCarloResult:
    r = returns.dropna().values
    if len(r) < block * 2:
        raise ValueError("Not enough history for a Monte Carlo simulation (need about 2 months).")
    if expected_return is not None:
        # shift every daily return so the average matches the chosen annual return,
        # keeping the real day-to-day swings
        r = r - r.mean() + (1 + expected_return) ** (1 / periods) - 1

    rng = np.random.default_rng(seed)
    steps = int(years * periods)
    n_blocks = int(np.ceil(steps / block))
    # circular blocks: wrap around the end so every historical day is equally likely to be drawn
    starts = rng.integers(0, len(r), size=(n_paths, n_blocks))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % len(r)).reshape(n_paths, -1)[:, :steps]
    growth = 1 + r[idx]                          # (paths, steps)

    month = max(int(round(periods / 12)), 1)
    value = np.full(n_paths, float(start_value))
    snapshots = [value.copy()]
    for t in range(steps):
        value *= growth[:, t]
        if (t + 1) % month == 0:
            value += monthly_contribution
            snapshots.append(value.copy())
    snap = np.array(snapshots).T                   # (paths, months+1)
    months = snap.shape[1] - 1
    invested = start_value + monthly_contribution * months

    pct = np.percentile(snap, [5, 25, 50, 75, 95], axis=0).T
    index = np.arange(snap.shape[1]) / 12
    table = pd.DataFrame(pct, index=index, columns=["5%", "25%", "50%", "75%", "95%"])
    table.index.name = "Years"
    method = "bootstrap" if expected_return is None else f"bootstrap, average return set to {expected_return:.1%}"
    return MonteCarloResult(table, value, invested, start_value, years, goal, method)
