import numpy as np
import pandas as pd
import pytest

from portfolio_analyzer import metrics, optimize, risk
from portfolio_analyzer.analysis import analyze, plain_english
from portfolio_analyzer.data import align_prices
from portfolio_analyzer.report import build_pdf
from portfolio_analyzer.universe import clean_ticker, is_crypto


def _prices(cols, index, seed=1, mu=0.0004, sd=0.01):
    rng = np.random.default_rng(seed)
    data = {c: 100 * np.cumprod(1 + rng.normal(mu * (i + 1), sd * (i + 1), len(index)))
            for i, c in enumerate(cols)}
    return pd.DataFrame(data, index=index)


@pytest.fixture
def returns():
    idx = pd.bdate_range("2022-01-03", periods=600)
    return metrics.daily_returns(_prices(["A", "B", "C"], idx))


# ---------- VaR ----------
def test_var_historical_known_distribution():
    r = pd.Series(np.linspace(-0.10, 0.09, 20))  # 20 evenly spaced returns
    assert risk.var_historical(r, 0.95) == pytest.approx(-np.quantile(r, 0.05))


def test_var_scales_with_sqrt_time(returns):
    one = risk.var_historical(returns["A"], 0.95, 1)
    assert risk.var_historical(returns["A"], 0.95, 21) == pytest.approx(one * np.sqrt(21))


def test_cvar_at_least_var(returns):
    for c in returns:
        assert risk.cvar_historical(returns[c]) >= risk.var_historical(returns[c])


def test_99_var_bigger_than_95(returns):
    assert risk.var_historical(returns["B"], 0.99) > risk.var_historical(returns["B"], 0.95)


def test_parametric_var_close_to_normal_formula():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 200_000))
    assert risk.var_parametric(r, 0.95) == pytest.approx(0.01645, abs=3e-4)


# ---------- Optimiser ----------
def test_max_sharpe_weights_valid(returns):
    w = optimize.max_sharpe(returns, 0.04)
    assert w.sum() == pytest.approx(1, abs=1e-5)
    assert (w >= -1e-9).all()


def test_max_sharpe_beats_random_portfolios(returns):
    w = optimize.max_sharpe(returns, 0.04)
    best = optimize.portfolio_stats(w, returns, 0.04)["Sharpe"]
    cloud = optimize.random_portfolios(returns, 3000, 0.04)
    assert best >= cloud["Sharpe"].max() - 1e-6


def test_min_vol_beats_random_portfolios(returns):
    w = optimize.min_volatility(returns)
    vol = optimize.portfolio_stats(w, returns)["Volatility"]
    cloud = optimize.random_portfolios(returns, 3000)
    assert vol <= cloud["Volatility"].min() + 1e-9


def test_max_weight_cap_respected(returns):
    w = optimize.max_sharpe(returns, 0.04, max_weight=0.4)
    assert w.max() <= 0.4 + 1e-6


def test_optimizer_sharpe_matches_metrics(returns):
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    port = metrics.portfolio_returns(returns, w.to_dict())
    assert optimize.portfolio_stats(w, returns, 0.04)["Sharpe"] == pytest.approx(
        metrics.sharpe_ratio(port, 0.04))


# ---------- Crypto handling ----------
def test_ticker_helpers():
    assert clean_ticker(" solana ") == "SOL-USD"
    assert clean_ticker("btc") == "BTC"  # ambiguous short code left alone
    assert is_crypto("XRP-USD") and not is_crypto("VOO")


def test_crypto_only_uses_365_days():
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    _, periods, _ = align_prices(_prices(["BTC-USD", "ETH-USD"], idx))
    assert periods == 365


def test_mixed_portfolio_uses_stock_days():
    days = pd.date_range("2024-01-01", periods=400, freq="D")
    crypto = _prices(["BTC-USD"], days)
    stocks = _prices(["VOO"], pd.bdate_range("2024-01-01", days[-1]))
    aligned, periods, notes = align_prices(pd.concat([stocks, crypto], axis=1, sort=True))
    assert periods == 252
    assert aligned.index.dayofweek.max() <= 4  # no weekends
    assert not aligned.isna().any().any()
    assert any("crypto" in n.lower() for n in notes)


def test_late_starting_asset_trims_period():
    idx = pd.bdate_range("2023-01-02", periods=300)
    p = _prices(["VOO", "NEW"], idx)
    p.loc[: idx[100], "NEW"] = np.nan
    aligned, _, notes = align_prices(p)
    assert aligned.index.min() == idx[101]
    assert any("shorter histories" in n for n in notes)


# ---------- Full pipeline ----------
def test_analyze_and_pdf():
    days = pd.date_range("2023-01-01", periods=600, freq="D")
    crypto = _prices(["BTC-USD", "SOL-USD"], days, seed=3)
    stocks = _prices(["VOO", "BND"], pd.bdate_range(days[0], days[-1]), seed=4)
    prices = pd.concat([stocks, crypto], axis=1, sort=True)
    a = analyze(prices, {"VOO": 0.6, "BND": 0.3, "BTC-USD": 0.05, "SOL-USD": 0.05},
                benchmark="VOO", max_weight=0.6, frontier_points=500)
    assert set(a.summary.index) == {"VOO", "BND", "BTC-USD", "SOL-USD", "Portfolio"}
    assert a.summary.loc["VOO", "Beta"] == pytest.approx(1.0)
    assert a.optimal["Max Sharpe"].max() <= 0.6 + 1e-6
    assert a.optimal_stats.loc["Max Sharpe", "Sharpe"] >= a.optimal_stats.loc["Your weights", "Sharpe"] - 1e-9
    lines = plain_english(a)
    assert len(lines) == 5 and "Sharpe" in lines[1]
    pdf = build_pdf(a)
    assert pdf[:4] == b"%PDF" and len(pdf) > 20_000


def test_single_asset_has_no_optimizer():
    idx = pd.bdate_range("2023-01-02", periods=300)
    a = analyze(_prices(["VOO"], idx), {"VOO": 1.0}, frontier_points=100)
    assert a.frontier is None
    assert build_pdf(a)[:4] == b"%PDF"
