"""Command-line interface.

Example:
    python -m portfolio_analyzer.cli VOO VXUS BND --weights 0.6 0.3 0.1 \
        --start 2021-01-01 --rf 0.04 --benchmark VOO --plot
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import metrics
from .data import fetch_prices, load_prices_csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze the risk and return of a portfolio.")
    p.add_argument("tickers", nargs="*", help="Tickers, e.g. VOO VXUS BND")
    p.add_argument("--weights", nargs="*", type=float, help="Weights in the same order as tickers")
    p.add_argument("--start", default="2021-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--rf", type=float, default=0.04, help="Annual risk-free rate, e.g. 0.04")
    p.add_argument("--benchmark", default=None, help="Benchmark ticker for beta, e.g. VOO")
    p.add_argument("--csv", default=None, help="Load prices from a CSV instead of downloading")
    p.add_argument("--plot", action="store_true", help="Show growth and drawdown charts")
    return p


def run(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        prices = load_prices_csv(args.csv)
        tickers = args.tickers or list(prices.columns)
    else:
        if not args.tickers:
            raise SystemExit("Give at least one ticker, or use --csv.")
        tickers = [t.upper() for t in args.tickers]
        download = list(dict.fromkeys(tickers + ([args.benchmark.upper()] if args.benchmark else [])))
        prices = fetch_prices(download, args.start, args.end)

    returns = metrics.daily_returns(prices)

    if args.weights:
        if len(args.weights) != len(tickers):
            raise SystemExit("Give one weight per ticker.")
        weights = dict(zip(tickers, args.weights))
    else:
        weights = None

    port = metrics.portfolio_returns(returns[tickers], weights)
    table_input = pd.concat([returns[tickers], port], axis=1)
    bench = returns[args.benchmark.upper()] if args.benchmark else None
    table = metrics.summary(table_input, args.rf, bench)

    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("\nRisk & return summary")
    print(f"Period: {returns.index.min():%Y-%m-%d} to {returns.index.max():%Y-%m-%d} | risk-free {args.rf:.2%}\n")
    print(table)
    print("\nCorrelation of daily returns\n")
    print(metrics.correlation_matrix(returns[tickers]))

    if args.plot:
        from .plots import plot_growth_and_drawdown

        plot_growth_and_drawdown(table_input)
    return table


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
