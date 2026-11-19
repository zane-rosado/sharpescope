"""Ticker catalogue and ready-made portfolios.

To add a new asset to the picker, add one line to CATALOG.
To add a new preset portfolio, add one entry to PRESETS.
"""

from __future__ import annotations

# ticker: (display name, asset class)
CATALOG: dict[str, tuple[str, str]] = {
    # US stock ETFs
    "VOO": ("Vanguard S&P 500 ETF", "Stock ETF"),
    "VTI": ("Vanguard Total Stock Market ETF", "Stock ETF"),
    "QQQ": ("Invesco Nasdaq-100 ETF", "Stock ETF"),
    "VUG": ("Vanguard Growth ETF", "Stock ETF"),
    "VTV": ("Vanguard Value ETF", "Stock ETF"),
    "VB": ("Vanguard Small-Cap ETF", "Stock ETF"),
    "VIG": ("Vanguard Dividend Appreciation ETF", "Stock ETF"),
    "VYM": ("Vanguard High Dividend Yield ETF", "Stock ETF"),
    # International
    "VXUS": ("Vanguard Total International Stock ETF", "International ETF"),
    "VEA": ("Vanguard Developed Markets ETF", "International ETF"),
    "VWO": ("Vanguard Emerging Markets ETF", "International ETF"),
    # Bonds
    "BND": ("Vanguard Total Bond Market ETF", "Bond ETF"),
    "BNDX": ("Vanguard Total International Bond ETF", "Bond ETF"),
    "VGSH": ("Vanguard Short-Term Treasury ETF", "Bond ETF"),
    "VGLT": ("Vanguard Long-Term Treasury ETF", "Bond ETF"),
    "VTIP": ("Vanguard Short-Term Inflation-Protected ETF", "Bond ETF"),
    # Real assets
    "VNQ": ("Vanguard Real Estate ETF", "Real Estate ETF"),
    "GLD": ("SPDR Gold Shares", "Commodity ETF"),
    # Crypto (Yahoo Finance uses the -USD suffix)
    "BTC-USD": ("Bitcoin", "Crypto"),
    "ETH-USD": ("Ethereum", "Crypto"),
    "XRP-USD": ("XRP", "Crypto"),
    "SOL-USD": ("Solana", "Crypto"),
    "BNB-USD": ("BNB", "Crypto"),
    "ADA-USD": ("Cardano", "Crypto"),
    "DOGE-USD": ("Dogecoin", "Crypto"),
    "AVAX-USD": ("Avalanche", "Crypto"),
    "LINK-USD": ("Chainlink", "Crypto"),
    "LTC-USD": ("Litecoin", "Crypto"),
}

# name: {ticker: weight}
PRESETS: dict[str, dict[str, float]] = {
    "Vanguard core (60/30/10)": {"VOO": 0.6, "VXUS": 0.3, "BND": 0.1},
    "Classic 60/40": {"VTI": 0.6, "BND": 0.4},
    "Three-fund (40/20/40)": {"VTI": 0.4, "VXUS": 0.2, "BND": 0.4},
    "Crypto majors": {"BTC-USD": 0.5, "ETH-USD": 0.2, "SOL-USD": 0.1, "XRP-USD": 0.1, "BNB-USD": 0.1},
    "Stocks + 10% crypto": {"VOO": 0.5, "VXUS": 0.2, "BND": 0.2, "BTC-USD": 0.07, "ETH-USD": 0.03},
}

# Friendly names people type, mapped to Yahoo tickers. Short coin codes such as
# "BTC" are deliberately NOT mapped, because they are also stock/ETF tickers.
ALIASES: dict[str, str] = {
    "BITCOIN": "BTC-USD",
    "ETHEREUM": "ETH-USD",
    "SOLANA": "SOL-USD",
    "CARDANO": "ADA-USD",
    "DOGECOIN": "DOGE-USD",
    "AVALANCHE": "AVAX-USD",
    "CHAINLINK": "LINK-USD",
    "LITECOIN": "LTC-USD",
    "RIPPLE": "XRP-USD",
}


def clean_ticker(t: str) -> str:
    t = t.strip().upper()
    return ALIASES.get(t, t)


def is_crypto(ticker: str) -> bool:
    ticker = ticker.upper()
    return CATALOG.get(ticker, ("", ""))[1] == "Crypto" or ticker.endswith("-USD")


def name_of(ticker: str) -> str:
    return CATALOG.get(ticker, (ticker, ""))[0]


def label(ticker: str) -> str:
    """Text shown in the picker, e.g. 'SOL-USD - Solana (Crypto)'."""
    if ticker in CATALOG:
        n, c = CATALOG[ticker]
        return f"{ticker} - {n} ({c})"
    return ticker
