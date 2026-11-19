"""Loading price data, either from Yahoo Finance or from a CSV file."""

from __future__ import annotations

import pandas as pd


def fetch_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted daily closing prices from Yahoo Finance.

    Returns a DataFrame with one column per ticker, indexed by date.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required: pip install yfinance") from exc

    tickers = [t.upper() for t in tickers]
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError("No data returned. Check the tickers and dates.")
    prices = data["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    empty = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if empty:
        raise ValueError(f"No price data for: {', '.join(empty)}")
    return prices[tickers].dropna(how="all")


def load_prices_csv(path: str) -> pd.DataFrame:
    """Load prices from a CSV with a Date column and one column per ticker."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df.sort_index()
