"""Loading and aligning price data, from Yahoo Finance or a CSV file."""

from __future__ import annotations

import pandas as pd

from .metrics import CRYPTO_DAYS, TRADING_DAYS
from .universe import clean_ticker, is_crypto


def fetch_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download dividend-adjusted daily closing prices from Yahoo Finance.

    Returns a DataFrame with one column per ticker, indexed by date.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required: pip install yfinance") from exc

    tickers = list(dict.fromkeys(clean_ticker(t) for t in tickers))
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError("No data returned. Check the tickers, dates and your internet connection.")
    prices = data["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    empty = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if empty:
        raise ValueError(f"No price data for: {', '.join(empty)}. "
                         "Crypto needs the -USD suffix, e.g. BTC-USD.")
    prices = prices[tickers]
    prices.columns.name = None
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices


def load_prices_csv(path: str) -> pd.DataFrame:
    """Load prices from a CSV with a Date column and one column per ticker."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df.sort_index()


def align_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, int, list[str]]:
    """Put assets on a common calendar and pick the right annualisation.

    * Crypto trades 7 days a week, stocks 5. In a mixed portfolio we keep only
      stock-market days (weekend crypto moves still show up in Monday's return).
    * Assets that started trading later (e.g. a newer coin) shorten the period
      for everyone, so all numbers cover the same dates.

    Returns (aligned prices, periods per year, notes for the user).
    """
    notes: list[str] = []
    prices = prices.sort_index().dropna(how="all")
    crypto = [c for c in prices.columns if is_crypto(c)]
    other = [c for c in prices.columns if c not in crypto]

    # 1) Common start date: the day the youngest asset began trading.
    first_valid = prices.apply(lambda s: s.first_valid_index())
    if first_valid.isna().any():
        raise ValueError(f"No price data for: {', '.join(first_valid[first_valid.isna()].index)}")
    first_day = prices.index.min()
    start = first_valid.max()
    late = first_valid[first_valid > first_day + pd.Timedelta(days=5)]
    if len(late):
        names = ", ".join(f"{t} ({d:%Y-%m-%d})" for t, d in late.items())
        notes.append(f"Some assets have shorter histories, so the analysis starts {start:%Y-%m-%d}: {names}.")
    prices = prices.loc[start:]

    # 2) Common calendar.
    if other:
        prices = prices.dropna(subset=other, how="all")  # keep stock-market days only
        periods = TRADING_DAYS
        if crypto:
            notes.append("Mixed stocks and crypto: using stock-market trading days only "
                         "(weekend crypto moves are included in Monday's return).")
    else:
        periods = CRYPTO_DAYS

    # 3) Fill the odd missing day (e.g. a holiday on one exchange) with the last price.
    prices = prices.ffill().dropna()
    return prices, periods, notes
