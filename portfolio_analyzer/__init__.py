"""Portfolio Analyzer: risk, return, VaR and optimisation for ETF, stock and crypto portfolios."""

from .analysis import Analysis, analyze, plain_english
from .metrics import (
    annualized_return,
    annualized_volatility,
    beta,
    correlation_matrix,
    daily_returns,
    drawdown_series,
    max_drawdown,
    portfolio_returns,
    sharpe_ratio,
    sortino_ratio,
    summary,
)

__version__ = "0.2.0"

__all__ = [
    "Analysis",
    "analyze",
    "plain_english",
    "annualized_return",
    "annualized_volatility",
    "beta",
    "correlation_matrix",
    "daily_returns",
    "drawdown_series",
    "max_drawdown",
    "portfolio_returns",
    "sharpe_ratio",
    "sortino_ratio",
    "summary",
]
