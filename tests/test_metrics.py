import numpy as np
import pandas as pd
import pytest

from portfolio_analyzer import metrics


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2022-01-03", periods=504)
    return pd.DataFrame(
        {"A": rng.normal(0.0005, 0.01, 504), "B": rng.normal(0.0002, 0.005, 504)},
        index=idx,
    )


def test_daily_returns_from_prices():
    prices = pd.DataFrame({"X": [100, 110, 99]}, index=pd.bdate_range("2024-01-01", periods=3))
    r = metrics.daily_returns(prices)["X"].tolist()
    assert r == pytest.approx([0.10, -0.10])


def test_constant_growth_cagr():
    r = pd.Series([0.001] * 252)
    assert metrics.annualized_return(r) == pytest.approx(1.001**252 - 1)


def test_volatility_matches_numpy(returns):
    expected = returns["A"].std(ddof=1) * np.sqrt(252)
    assert metrics.annualized_volatility(returns["A"]) == pytest.approx(expected)


def test_sharpe_formula(returns):
    rf = 0.04
    ex = returns["A"] - rf / 252
    expected = ex.mean() / ex.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe_ratio(returns["A"], rf) == pytest.approx(expected)


def test_sharpe_zero_vol_is_nan():
    assert np.isnan(metrics.sharpe_ratio(pd.Series([0.0] * 10)))


def test_max_drawdown_simple():
    # up 10%, down 50%, up 20%  -> worst drop is -50% from the peak
    r = pd.Series([0.10, -0.50, 0.20])
    assert metrics.max_drawdown(r) == pytest.approx(-0.50)


def test_portfolio_weights_normalised(returns):
    p1 = metrics.portfolio_returns(returns, {"A": 3, "B": 1})
    p2 = metrics.portfolio_returns(returns, {"A": 0.75, "B": 0.25})
    pd.testing.assert_series_equal(p1, p2)


def test_equal_weight_default(returns):
    p = metrics.portfolio_returns(returns)
    assert p.iloc[0] == pytest.approx(returns.iloc[0].mean())


def test_negative_weight_rejected(returns):
    with pytest.raises(ValueError):
        metrics.portfolio_returns(returns, {"A": 1.2, "B": -0.2})


def test_unknown_ticker_rejected(returns):
    with pytest.raises(ValueError):
        metrics.portfolio_returns(returns, {"ZZZ": 1})


def test_beta_of_itself_is_one(returns):
    assert metrics.beta(returns["A"], returns["A"]) == pytest.approx(1.0)


def test_beta_of_scaled_series(returns):
    assert metrics.beta(2 * returns["A"], returns["A"]) == pytest.approx(2.0)


def test_summary_shape(returns):
    table = metrics.summary(returns, 0.04, benchmark=returns["A"])
    assert list(table.index) == ["A", "B"]
    assert {"Sharpe", "Max Drawdown", "Beta"} <= set(table.columns)
