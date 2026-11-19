"""Portfolio Analyzer: risk and return metrics for ETF and stock portfolios."""

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

__version__ = "0.1.0"

__all__ = [
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
