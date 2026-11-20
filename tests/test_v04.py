import numpy as np
import pandas as pd
import pytest

from sharpescope import metrics, optimize, simulate, stress
from sharpescope.analysis import analyze, plain_english
from sharpescope.data import holdings_to_weights
from sharpescope.report import build_pdf


def _market(n=800, seed=0):
    """Synthetic world: SPY, 10y yield, a stock (beta 1.3), a bond fund (duration ~6)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    m = rng.normal(0.0004, 0.01, n)
    dy = rng.normal(0, 0.06, n)                 # daily yield change in percentage points
    stock = 1.3 * m + rng.normal(0, 0.005, n)
    bond = -0.06 * dy + rng.normal(0, 0.0005, n)  # -6% per +1pt => duration 6
    prices = pd.DataFrame({
        "SPY": 100 * np.cumprod(1 + m),
        "STK": 50 * np.cumprod(1 + stock),
        "BND": 80 * np.cumprod(1 + bond),
    }, index=idx)
    factors = pd.DataFrame({"SPY": prices["SPY"], "^TNX": 3 + np.cumsum(dy)}, index=idx)
    return prices, factors


# ---------- new metrics ----------
def test_alpha_zero_for_pure_beta():
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0.0005, 0.01, 1000))
    assert metrics.alpha(1.5 * b, b, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_alpha_detects_constant_excess():
    rng = np.random.default_rng(2)
    b = pd.Series(rng.normal(0.0005, 0.01, 1000))
    assert metrics.alpha(b + 0.0001, b, 0.0) == pytest.approx(0.0001 * 252, rel=1e-6)


def test_information_ratio_sign():
    rng = np.random.default_rng(3)
    b = pd.Series(rng.normal(0.0005, 0.01, 500))
    better = b + rng.normal(0.0003, 0.002, 500)
    assert metrics.information_ratio(better, b) > 0


def test_monthly_returns_compound_to_total():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.0005, 0.01, 400), index=pd.bdate_range("2023-01-02", periods=400))
    m = metrics.monthly_returns(r)
    assert (1 + m["Year"]).prod() - 1 == pytest.approx(metrics.total_return(r))


# ---------- risk parity ----------
def test_risk_parity_equalises_contributions():
    prices, _ = _market()
    r = metrics.daily_returns(prices[["STK", "BND", "SPY"]])
    w = optimize.risk_parity(r)
    rc = optimize.risk_contributions(w, r)
    assert w.sum() == pytest.approx(1, abs=1e-6)
    assert rc.max() - rc.min() < 0.01
    assert w["BND"] > w["STK"]   # the calmer asset gets more weight


# ---------- Monte Carlo ----------
def test_monte_carlo_shapes_and_order():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0004, 0.01, 750))
    mc = simulate.monte_carlo(r, years=5, n_paths=500, start_value=10_000, monthly_contribution=100, goal=20_000)
    p = mc.percentiles
    assert len(mc.final_values) == 500
    assert (p["5%"] <= p["50%"]).all() and (p["50%"] <= p["95%"]).all()
    assert mc.invested == pytest.approx(10_000 + 100 * 60)
    s = mc.stats()
    assert 0 <= s["Chance of ending below what you put in"] <= 1
    assert "Chance of reaching $20,000" in s


def test_monte_carlo_expected_return_override():
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0.003, 0.01, 750))  # unrealistically high history
    mc = simulate.monte_carlo(r, years=10, n_paths=2000, expected_return=0.07, seed=1)
    median_growth = (np.median(mc.final_values) / 10_000) ** (1 / 10) - 1
    assert 0.03 < median_growth < 0.08   # pulled back to ~7% minus volatility drag


def test_monte_carlo_needs_history():
    with pytest.raises(ValueError):
        simulate.monte_carlo(pd.Series([0.01] * 10))


# ---------- stress tests ----------
def test_factor_exposures_recover_true_values():
    prices, factors = _market()
    r = metrics.daily_returns(prices[["STK", "BND"]])
    e = stress.factor_exposures(r, factors["SPY"].pct_change(), factors["^TNX"].diff())
    assert e.loc["STK", "Market beta"] == pytest.approx(1.3, abs=0.05)
    assert e.loc["BND", "Empirical duration"] == pytest.approx(6.0, abs=0.3)


def test_rate_shock_hurts_bonds():
    prices, factors = _market()
    r = metrics.daily_returns(prices[["STK", "BND"]])
    e = stress.factor_exposures(r, factors["SPY"].pct_change(), factors["^TNX"].diff())
    t = stress.hypothetical(e, pd.Series({"STK": 0.5, "BND": 0.5}), 10_000)
    assert t.loc["Rates +1%", "BND"] == pytest.approx(-0.06, abs=0.005)
    assert t.loc["Rates -1%", "BND"] > 0
    assert t.loc["Stocks -20% (bear market)", "STK"] == pytest.approx(-0.26, abs=0.02)
    assert "Crypto winter (crypto -60%)" not in t.index   # no crypto held


def test_bond_math():
    assert stress.bond_price_change(6, 1) == pytest.approx(-0.06)
    assert stress.bond_price_change(6, 1, convexity=50) == pytest.approx(-0.06 + 0.5 * 50 * 0.0001)


def test_historical_replay_actual_and_estimated():
    idx = pd.bdate_range("2007-01-01", "2010-01-01")
    spy = pd.Series(np.linspace(100, 60, len(idx)), index=idx)
    old = pd.Series(np.linspace(50, 40, len(idx)), index=idx)
    new = pd.Series(np.nan, index=idx)          # did not exist yet
    hist = pd.DataFrame({"SPY": spy, "OLD": old, "NEW": new})
    w = pd.Series({"OLD": 0.5, "NEW": 0.5})
    summary, detail = stress.historical(hist, w, pd.Series({"OLD": 0.5, "NEW": 2.0}), 10_000)
    row = summary.loc["Global Financial Crisis (2007-09)"]
    m = row["S&P 500 (SPY)"]
    old_ret = old.asof(pd.Timestamp("2009-03-09")) / old.asof(pd.Timestamp("2007-10-09")) - 1
    assert row["Portfolio"] == pytest.approx(0.5 * old_ret + 0.5 * 2.0 * m)
    assert row["Share of portfolio estimated"] == pytest.approx(0.5)
    assert "estimated" in detail.loc["Global Financial Crisis (2007-09)", "NEW"]


def test_sharpe_falls_as_risk_free_rises():
    rng = np.random.default_rng(7)
    s = stress.sharpe_vs_risk_free(pd.Series(rng.normal(0.0005, 0.01, 500)))
    assert s.is_monotonic_decreasing


# ---------- holdings input ----------
def test_holdings_to_weights():
    idx = pd.bdate_range("2024-01-01", periods=5)
    p = pd.DataFrame({"A": [10.0] * 5, "B": [40.0] * 5}, index=idx)
    w, total = holdings_to_weights(p, {"A": 30, "B": 5}, "shares")
    assert total == pytest.approx(500) and w["A"] == pytest.approx(0.6)
    w, total = holdings_to_weights(p, {"A": 750, "B": 250}, "dollars")
    assert total == 1000 and w["B"] == pytest.approx(0.25)


# ---------- full pipeline ----------
def test_full_analysis_with_stress_and_pdf():
    prices, factors = _market()
    a = analyze(prices[["STK", "BND", "SPY"]], {"STK": 0.6, "BND": 0.4}, benchmark="SPY",
                factors=factors, history=prices, mc_paths=300, frontier_points=200, mc_goal=20_000)
    assert a.exposures is not None and a.scenarios is not None and a.montecarlo is not None
    assert "Risk Parity" in a.optimal
    assert {"Alpha", "Info Ratio"} <= set(a.summary.columns)
    lines = plain_english(a)
    assert any("Monte Carlo" in l for l in lines) and any("rates rose" in l for l in lines)
    pdf = build_pdf(a)
    assert pdf[:4] == b"%PDF"


def test_monte_carlo_is_unbiased_with_short_history():
    # good days early, a crash at the end: every day must be equally likely to be drawn
    r = pd.Series(np.r_[np.full(150, 0.004), np.full(50, -0.01)] + np.random.default_rng(8).normal(0, 0.005, 200))
    mc = simulate.monte_carlo(r, years=5, n_paths=4000, expected_return=0.06, seed=3)
    growth = (np.median(mc.final_values) / 10_000) ** (1 / 5) - 1
    theory = np.exp(np.log(1.06) - 252 * r.var() / 2) - 1
    assert growth == pytest.approx(theory, abs=0.01)
