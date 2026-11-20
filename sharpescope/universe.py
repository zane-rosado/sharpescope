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
    "IVV": ("iShares Core S&P 500 ETF", "Stock ETF"),
    # Sector ETFs
    "XLV": ("Health Care Select Sector SPDR", "Sector ETF"),
    "XLI": ("Industrial Select Sector SPDR", "Sector ETF"),
    "XLE": ("Energy Select Sector SPDR", "Sector ETF"),
    "XLK": ("Technology Select Sector SPDR", "Sector ETF"),
    "XLF": ("Financial Select Sector SPDR", "Sector ETF"),
    # Individual stocks
    "CAH": ("Cardinal Health", "Stock"),
    "BMY": ("Bristol-Myers Squibb", "Stock"),
    "JNJ": ("Johnson & Johnson", "Stock"),
    "EW": ("Edwards Lifesciences", "Stock"),
    "VRTX": ("Vertex Pharmaceuticals", "Stock"),
    "GD": ("General Dynamics", "Stock"),
    "UBER": ("Uber Technologies", "Stock"),
    "FIX": ("Comfort Systems USA", "Stock"),
    "FAST": ("Fastenal", "Stock"),
    "CTAS": ("Cintas", "Stock"),
    "EOG": ("EOG Resources", "Stock"),
    "COP": ("ConocoPhillips", "Stock"),
    "CVX": ("Chevron", "Stock"),
    "VLO": ("Valero Energy", "Stock"),
    "MPC": ("Marathon Petroleum", "Stock"),
    "ADBE": ("Adobe", "Stock"),
    "SNDK": ("SanDisk", "Stock"),
    "MU": ("Micron Technology", "Stock"),
    "INTU": ("Intuit", "Stock"),
    "CSCO": ("Cisco Systems", "Stock"),
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

PRESETS["SMIF_ZR - Project 2 stocks"] = {
    **{t: 0.101707 for t in ("CAH", "BMY", "JNJ", "EW", "VRTX")},   # Health Care 50.85%
    **{t: 0.040295 for t in ("GD", "UBER", "FIX", "FAST", "CTAS")},  # Industrials 20.15%
    **{t: 0.036998 for t in ("EOG", "COP", "CVX", "VLO", "MPC")},    # Energy 18.50%
    **{t: 0.021000 for t in ("ADBE", "SNDK", "MU", "INTU", "CSCO")},  # Info Tech 10.50%
}
PRESETS["SMIF TARGET - sector policy"] = {"XLV": 0.5085, "XLI": 0.2015, "XLE": 0.1850, "XLK": 0.1050}

# Suggested settings that load with a preset (start date, rebalancing, benchmark)
PRESET_SETTINGS: dict[str, dict] = {
    "SMIF_ZR - Project 2 stocks": {"start": "2025-12-31", "rebalance": "none", "benchmark": "IVV"},
    "SMIF TARGET - sector policy": {"start": "2025-12-31", "rebalance": "daily", "benchmark": "IVV"},
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
