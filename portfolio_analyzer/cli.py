"""Command-line interface.

Examples:
    py -m portfolio_analyzer.cli VOO VXUS BND --weights 0.6 0.3 0.1 --benchmark VOO
    py -m portfolio_analyzer.cli VOO BTC-USD ETH-USD --pdf report.pdf
    py -m portfolio_analyzer.cli --preset "Crypto majors"
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from .analysis import analyze, plain_english
from .data import fetch_prices, load_prices_csv
from .universe import PRESETS, clean_ticker


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze the risk and return of a portfolio.")
    p.add_argument("tickers", nargs="*", help="Tickers, e.g. VOO VXUS BND BTC-USD")
    p.add_argument("--weights", nargs="*", type=float, help="Weights in the same order as tickers")
    p.add_argument("--preset", choices=list(PRESETS), help="Use a ready-made portfolio")
    p.add_argument("--start", default="2021-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--rf", type=float, default=0.04, help="Annual risk-free rate, e.g. 0.04")
    p.add_argument("--benchmark", default=None, help="Benchmark ticker for beta, e.g. VOO")
    p.add_argument("--value", type=float, default=10_000, help="Portfolio value in $ (for VaR)")
    p.add_argument("--var-level", type=float, default=0.95, help="VaR confidence, e.g. 0.95 or 0.99")
    p.add_argument("--max-weight", type=float, default=1.0, help="Optimiser cap per asset, e.g. 0.4")
    p.add_argument("--csv", default=None, help="Load prices from a CSV instead of downloading")
    p.add_argument("--pdf", default=None, help="Save a PDF report to this file")
    p.add_argument("--plot", action="store_true", help="Save and open a growth/drawdown chart")
    return p


def _print(title: str, df: pd.DataFrame, pct=(), num=(), money=()):
    out = df.copy().astype(object)
    for c in df.columns:
        if c in pct:
            out[c] = df[c].map(lambda v: f"{v:7.2%}")
        elif c in money:
            out[c] = df[c].map(lambda v: f"${v:,.0f}")
        elif c in num:
            out[c] = df[c].map(lambda v: f"{v:6.2f}")
    print(f"\n{title}\n{'-' * len(title)}")
    print(out.to_string())


def run(args: argparse.Namespace):
    if args.preset:
        weights = PRESETS[args.preset]
        tickers = list(weights)
    else:
        tickers = [clean_ticker(t) for t in args.tickers]
        weights = None

    if args.csv:
        prices = load_prices_csv(args.csv)
        tickers = tickers or list(prices.columns)
    else:
        if not tickers:
            raise SystemExit("Give at least one ticker, use --preset, or use --csv.")
        bench = [clean_ticker(args.benchmark)] if args.benchmark else []
        prices = fetch_prices(list(dict.fromkeys(tickers + bench)), args.start, args.end)

    if weights is None:
        if args.weights:
            if len(args.weights) != len(tickers):
                raise SystemExit("Give one weight per ticker.")
            weights = dict(zip(tickers, args.weights))
        else:
            weights = {t: 1.0 for t in tickers}

    a = analyze(prices, weights, args.rf, clean_ticker(args.benchmark) if args.benchmark else None,
                args.value, args.var_level, args.max_weight)

    print("\nSUMMARY")
    print("=======")
    for line in plain_english(a) + a.notes:
        print(f"* {line}")

    _print("Return & risk", a.summary, pct=("Annual Return", "Volatility", "Max Drawdown"),
           num=("Sharpe", "Sortino", "Beta", "Calmar"))
    _print(f"Value at Risk ({a.var_level:.0%})", a.risk,
           pct=("1-Day VaR", "1-Day CVaR", "1-Month VaR", "Parametric 1-Day VaR"), money=("1-Day VaR ($)",))
    if "Max Sharpe" in a.optimal:
        _print("Weights: yours vs. optimised", a.optimal, pct=tuple(a.optimal.columns))
        _print("Optimised portfolio stats", a.optimal_stats, pct=("Avg Return", "Volatility"), num=("Sharpe",))
    _print("Correlation of daily returns", a.correlation, num=tuple(a.correlation.columns))

    if args.plot:
        from . import plots

        path = plots.save(plots.growth_and_drawdown(a.returns, a.value), "portfolio_chart.png")
        print(f"\nChart saved to {os.path.abspath(path)}")
        _open(path)
    if args.pdf:
        from .report import build_pdf

        with open(args.pdf, "wb") as f:
            f.write(build_pdf(a))
        print(f"\nPDF report saved to {os.path.abspath(args.pdf)}")
        _open(args.pdf)
    return a


def _open(path: str):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
    except OSError:
        pass


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
