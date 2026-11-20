"""Quant SharpeScope (QSS) web app.   Run with:   py -m streamlit run app.py"""

from __future__ import annotations

import datetime as dt
import os
import urllib.parse

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sharpescope import plots, stress
from sharpescope.analysis import analyze, plain_english
from sharpescope.data import fetch_factors, fetch_prices, holdings_to_weights
from sharpescope.report import build_pdf
from sharpescope.universe import CATALOG, PRESET_SETTINGS, PRESETS, clean_ticker, is_crypto, label, name_of

REPO = "zane-rosado/sharpescope"
FEEDBACK_URL = os.environ.get("QSS_FEEDBACK_URL", "")   # optional: a Google Form link
REBALANCE = {"daily": "Daily (fixed weights)", "monthly": "Monthly", "quarterly": "Quarterly",
             "yearly": "Yearly", "none": "Never (buy and hold)"}
PCT, NUM = "{:.2%}", "{:.2f}"


def USD(v: float) -> str:
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"

st.set_page_config(page_title="Quant SharpeScope", page_icon="📈", layout="wide")


# ================================================================ cached work
@st.cache_data(ttl=900, show_spinner=False)  # re-download prices at most every 15 minutes
def load_prices(tickers: tuple[str, ...], start: str, end: str | None = None) -> pd.DataFrame:
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end else None  # Yahoo end is exclusive
    return fetch_prices(list(tickers), start, end_excl)


@st.cache_data(ttl=3600, show_spinner=False)
def load_factors(start: str, end: str | None) -> pd.DataFrame:
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end else None
    return fetch_factors(start, end_excl)


@st.cache_data(ttl=86400, show_spinner=False)
def load_history(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Prices back to 2000 (where they exist) for crisis replays."""
    raw = __import__("yfinance").download(list(tickers), start="1999-12-01", auto_adjust=True, progress=False)
    px = raw["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px.columns.name = None
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px


@st.cache_data(show_spinner=False)
def run_analysis(params: dict):
    holdings = params["holdings"]
    bench = params["benchmark"]
    download = list(dict.fromkeys(list(holdings) + ([bench] if bench else [])))
    prices = load_prices(tuple(download), params["start"], params["end"])
    weights, total = holdings_to_weights(prices, holdings, params["kind"], params["start"])
    value = total or params["value"]
    factors = history = None
    if params["stress"]:
        try:
            factors = load_factors(params["start"], params["end"])
            history = load_history(tuple(dict.fromkeys(list(holdings) + ["SPY"])))
        except Exception:   # stress data is optional: never block the main analysis
            factors = history = None
    return analyze(prices, weights, params["rf"], bench, value, params["var_level"], params["max_weight"],
                   rebalance=params["rebalance"], start=params["start"], end=params["end"],
                   factors=factors, history=history, mc_years=params["mc_years"],
                   mc_contribution=params["mc_contribution"], mc_expected_return=params["mc_return"],
                   mc_goal=params["mc_goal"])


@st.cache_data(show_spinner=False)
def make_pdf(params: dict) -> bytes:
    return build_pdf(run_analysis(params))


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def md(text: str) -> str:
    """Escape $ so Streamlit doesn't read dollar amounts as math formulas."""
    return text.replace("$", "\\$")


# ================================================================ sidebar
with st.sidebar:
    st.header("1 · Pick your assets")
    preset = st.selectbox("Start from a ready-made portfolio", ["Custom"] + list(PRESETS), index=1)
    base = PRESETS.get(preset, PRESETS["Vanguard core (60/30/10)"])
    defaults = PRESET_SETTINGS.get(preset, {})
    chosen = st.multiselect("Assets (type to search: stocks, ETFs, bonds, crypto)",
                            options=list(dict.fromkeys(list(CATALOG) + list(base))),
                            default=list(base), format_func=label, key=f"assets::{preset}")
    extra = st.text_input("Any other Yahoo tickers (comma-separated)", placeholder="e.g. AAPL, MSFT, DOT-USD",
                          key=f"extra::{preset}")
    tickers = list(dict.fromkeys(chosen + [clean_ticker(t) for t in extra.split(",") if t.strip()]))

    st.header("2 · Enter your holdings")
    kind = st.radio("Enter as", ["weights", "dollars", "shares"], horizontal=True,
                    format_func={"weights": "% weights", "dollars": "$ amounts", "shares": "Shares"}.get,
                    help="Dollar amounts or share counts are turned into weights automatically. "
                         "Shares are valued at the close on your start date.")
    equal = st.checkbox("Equal weights", disabled=kind != "weights")
    col = {"weights": "Weight %", "dollars": "Amount $", "shares": "Shares"}[kind]
    if kind == "weights":
        start_vals = [100 / len(tickers) if equal else base.get(t, 0.10) * 100 for t in tickers] if tickers else []
    elif kind == "dollars":
        start_vals = [round(base.get(t, 0.10) * 10_000, 2) for t in tickers]
    else:
        start_vals = [10.0 for _ in tickers]
    edited = st.data_editor(
        pd.DataFrame({"Ticker": tickers, "Name": [name_of(t) for t in tickers], col: start_vals}),
        hide_index=True, disabled=["Ticker", "Name"],
        column_config={col: st.column_config.NumberColumn(min_value=0.0, step=1.0, format="%.2f")},
        key=f"hold::{preset}::{kind}::{'|'.join(tickers)}::{equal}")
    if kind == "weights" and len(edited):
        total = float(edited[col].fillna(0).sum())
        st.caption(f"Total: {total:.1f}%" + ("" if abs(total - 100) < 0.05 else " (will be scaled to 100%)"))

    st.header("3 · Settings")
    start = st.date_input("Start date (the day you bought)", dt.date.fromisoformat(defaults.get("start", "2021-01-01")),
                          min_value=dt.date(2000, 1, 1), max_value=dt.date.today() - dt.timedelta(days=30),
                          key=f"start::{preset}")
    end = st.date_input("End date", dt.date.today(), min_value=dt.date(2000, 2, 1), max_value=dt.date.today(),
                        key=f"end::{preset}")
    reb_keys = list(REBALANCE)
    rebalance = st.selectbox("Rebalancing", reb_keys, index=reb_keys.index(defaults.get("rebalance", "daily")),
                             format_func=REBALANCE.get, key=f"reb::{preset}",
                             help="'Never' = buy once and hold, like a brokerage account or a Bloomberg shares portfolio.")
    bench_opts = ["None"] + list(dict.fromkeys(["VOO", "IVV", "VTI", "QQQ", "BTC-USD"] + tickers))
    benchmark = st.selectbox("Benchmark (to compare against)", bench_opts,
                             index=bench_opts.index(defaults.get("benchmark", "VOO")), key=f"bench::{preset}")
    with st.expander("Risk settings"):
        rf = st.number_input("Risk-free rate (% per year)", 0.0, 20.0, 4.0, 0.25,
                             help="Use the current 3-month Treasury bill yield.") / 100
        value = st.number_input("Portfolio value ($)", 100.0, 1e9, 10_000.0, 1_000.0, format="%.0f",
                                disabled=kind != "weights", help="Taken from your $ amounts or shares if you used those.")
        var_level = st.select_slider("VaR confidence", options=[0.90, 0.95, 0.99], value=0.95,
                                     format_func=lambda x: f"{x:.0%}")
        max_weight = st.slider("Optimizer: max % in any one asset", 10, 100, 100, 5) / 100
    with st.expander("Monte Carlo & stress tests"):
        mc_years = st.slider("Years to simulate", 1, 40, 10)
        mc_contribution = st.number_input("Monthly contribution ($)", 0.0, 1e7, 0.0, 50.0)
        mc_goal = st.number_input("Goal ($, optional)", 0.0, 1e10, 0.0, 1_000.0, format="%.0f")
        use_hist = st.checkbox("Use my portfolio's past average return", value=True,
                               help="Untick to set your own long-run return, so a lucky or unlucky "
                                    "history doesn't dominate the forecast.")
        mc_return = None if use_hist else st.slider("Expected annual return (%)", -5.0, 20.0, 7.0, 0.5) / 100
        run_stress = st.checkbox("Run stress tests & crisis replays", value=True,
                                 help="Downloads S&P 500, 10-year Treasury yield and price history since 2000.")

    learn = st.toggle("📘 Learning mode", value=False, help="Opens every explanation box automatically.")
    go = st.button("Analyze", type="primary")

if go:
    vals = dict(zip(edited["Ticker"], edited[col].fillna(0)))
    holdings = {t: float(v) for t, v in vals.items() if v and v > 0}
    if not holdings:
        st.sidebar.error("Give at least one asset an amount above 0.")
    else:
        st.session_state.params = {
            "holdings": holdings, "kind": kind, "start": str(start), "end": str(end), "rebalance": rebalance,
            "rf": rf, "value": value, "var_level": var_level, "max_weight": max_weight,
            "benchmark": None if benchmark == "None" else benchmark, "stress": run_stress,
            "mc_years": mc_years, "mc_contribution": mc_contribution, "mc_return": mc_return,
            "mc_goal": mc_goal or None,
        }


def explain(title: str, body: str):
    with st.expander(f"📘 {title}", expanded=learn):
        st.markdown(md(body))


# ================================================================ main page
st.title("📈 Quant SharpeScope")
st.caption("**QSS** · Portfolio risk, stress tests, Monte Carlo and optimization for stocks, ETFs and crypto. "
           "Built for quants, made to teach. Data: Yahoo Finance. Educational use only, not investment advice.")

params = st.session_state.get("params")
if not params:
    st.info("**How to use:** 1) pick assets or a ready-made portfolio on the left, 2) enter weights, dollars or "
            "shares, 3) press **Analyze**. Turn on **📘 Learning mode** to see an explanation next to every number.")
    explain("New to this? Start here",
            "- **Return** is how much you made. **Risk** is how bumpy the ride was.\n"
            "- The **Sharpe ratio** puts them together: return earned per unit of risk. Above 1 is strong.\n"
            "- **Stress tests** ask *what if*: rates jump, stocks crash, or 2008 happens again.\n"
            "- **Monte Carlo** simulates thousands of possible futures to show a range, not one guess.")
    st.stop()

try:
    with st.spinner("Downloading prices and crunching numbers..."):
        a = run_analysis(params)
except Exception as e:  # friendly message instead of a crash
    st.error(f"Something went wrong: {e}")
    st.stop()

p, rk = a.summary.loc["Portfolio"], a.risk.loc["Portfolio"]

# ---------------------------------------------------------------- summary
st.header("Summary")
c = st.columns(5)
c[0].metric("Total return", f"{p['Total Return']:.1%}", f"{p['Annual Return']:.1%} a year", delta_color="off")
c[1].metric("Sharpe ratio", f"{p['Sharpe']:.2f}")
c[2].metric("Volatility", f"{p['Volatility']:.1%}")
c[3].metric("Max drawdown", f"{p['Max Drawdown']:.1%}")
c[4].metric(f"1-day VaR ({a.var_level:.0%})", f"${rk['1-Day VaR ($)']:,.0f}", f"-{rk['1-Day VaR']:.2%}",
            delta_color="off")
st.markdown("\n".join(f"- {md(line)}" for line in plain_english(a)))
for note in a.notes:
    st.caption(f"ℹ️ {md(note)}")

tabs = st.tabs(["📈 Performance", "⚠️ Risk (VaR)", "🔮 Monte Carlo", "🌪️ Stress tests", "🎯 Optimizer",
                "🔗 Correlation", "📋 Data", "📘 Glossary"])

# ---------------------------------------------------------------- performance
with tabs[0]:
    st.subheader("Return & risk by asset")
    fmt = {"Total Return": PCT, "Annual Return": PCT, "Volatility": PCT, "Max Drawdown": PCT, "Alpha": PCT,
           "Sharpe": NUM, "Sortino": NUM, "Beta": NUM, "Calmar": NUM, "Info Ratio": NUM}
    st.dataframe(a.summary.style.format(fmt, na_rep="-"))
    explain("What do these columns mean?",
            "- **Total / Annual return**: growth over the whole period, and the yearly compounded rate.\n"
            "- **Volatility**: how much returns swing (annualised standard deviation).\n"
            "- **Sharpe**: (return - risk-free rate) / volatility. Return per unit of total risk. Above 1 is strong.\n"
            "- **Sortino**: like Sharpe but only counts downside swings as risk.\n"
            "- **Max drawdown**: worst fall from a high point to a low point.\n"
            "- **Beta**: how much it moves with the benchmark (1 = in step, 0.5 = half as much).\n"
            "- **Alpha**: yearly return *beyond* what beta explains. Positive alpha = skill or luck.\n"
            "- **Info ratio**: how *consistently* you beat the benchmark (above 0.5 is good).\n"
            "- **Calmar**: annual return divided by max drawdown.")
    show(plots.growth_and_drawdown(a.returns, a.value))
    if a.monthly is not None and len(a.monthly):
        show(plots.monthly_heatmap(a.monthly))
    if a.current_weights is not None:
        st.subheader("How your weights drifted")
        drift = pd.DataFrame({"Starting weight": a.weights, "Weight now": a.current_weights})
        st.dataframe(drift.sort_values("Weight now", ascending=False).style.format(PCT))
        explain("Why do weights drift?",
                "With buy and hold, winners grow into a bigger slice of the portfolio and losers shrink. "
                "Rebalancing sells some winners and buys losers to get back to your targets.")
    show(plots.rolling_sharpe_chart(a.returns["Portfolio"], a.risk_free, a.periods))
    explain("Rolling Sharpe", "The Sharpe ratio measured over a moving 6-month window. It shows *when* the "
                              "portfolio was rewarding risk well and when it wasn't. The dashed line marks 1.0.")

# ---------------------------------------------------------------- risk
with tabs[1]:
    st.subheader(md(f"Value at Risk ({a.var_level:.0%} confidence, ${a.value:,.0f} portfolio)"))
    st.dataframe(a.risk.style.format({k: PCT for k in a.risk.columns if "($)" not in k} | {"1-Day VaR ($)": USD}))
    show(plots.return_histogram(a.returns["Portfolio"], rk["1-Day VaR"], rk["1-Day CVaR"], a.var_level))
    explain("What is VaR?",
            f"- **1-Day VaR**: on {a.var_level:.0%} of days you lose less than this. On the worst "
            f"{1 - a.var_level:.0%} of days you lose more.\n"
            "- **CVaR (Expected Shortfall)**: the *average* loss on those worst days. Regulators prefer it.\n"
            "- **1-Month VaR**: the 1-day figure scaled to about 21 trading days (square-root-of-time rule).\n"
            "- **Parametric VaR**: assumes a bell curve. If it is much lower than historical VaR, the asset has "
            "*fat tails*: more extreme days than a bell curve predicts.")
    if a.risk_contrib is not None:
        st.subheader("Where your risk comes from")
        rc = pd.DataFrame({"Weight": a.weights_now, "Share of risk": a.risk_contrib})
        st.dataframe(rc.sort_values("Share of risk", ascending=False).style.format(PCT))
        explain("Risk contribution",
                "A 10% position can contribute 30% of the risk if it is volatile (crypto, single tech stocks). "
                "Risk parity portfolios set every asset's share of risk equal.")

# ---------------------------------------------------------------- Monte Carlo
with tabs[2]:
    mc = a.montecarlo
    if mc is None:
        st.info("Not enough history for a Monte Carlo simulation (need about 2 months).")
    else:
        stt = mc.stats()
        cc = st.columns(4)
        cc[0].metric("Median outcome", f"${stt['Median ending value']:,.0f}")
        cc[1].metric("Bad case (5%)", f"${stt['Bad case (5th percentile)']:,.0f}")
        cc[2].metric("Good case (95%)", f"${stt['Good case (95th percentile)']:,.0f}")
        cc[3].metric("Chance of losing money", f"{stt['Chance of ending below what you put in']:.0%}")
        goal_key = next((k for k in stt if k.startswith("Chance of reaching")), None)
        if goal_key:
            st.metric(md(goal_key), f"{stt[goal_key]:.0%}")
        show(plots.monte_carlo_fan(mc))
        st.caption(md(f"{len(mc.final_values):,} simulations over {mc.years} years; method: {mc.method}; "
                      f"total put in ${mc.invested:,.0f}."))
        if a.years < 3 and params["mc_return"] is None:
            st.warning(f"Your history is only {a.years:.1f} years long, so the simulation repeats a short "
                       "stretch of the market. Untick *Use my portfolio's past average return* in the sidebar "
                       "and enter a long-run figure (e.g. 7%) for a more realistic forecast.")
        explain("How the Monte Carlo works",
                "We build thousands of possible futures by stitching together random **1-month chunks** of "
                "your portfolio's real daily returns (a *block bootstrap*). Keeping whole chunks preserves real "
                "crash days and streaks that a bell-curve model would smooth away.\n\n"
                "- The **dark band** holds the middle 50% of outcomes and the **light band** 90%.\n"
                "- The spread grows with time: uncertainty compounds.\n"
                "- It is a range of *possibilities* based on the past, not a prediction.")

# ---------------------------------------------------------------- stress tests
with tabs[3]:
    if a.exposures is None:
        st.info("Stress tests are off or the extra data could not be downloaded. Tick *Run stress tests* in "
                "the sidebar (under Monte Carlo & stress tests) and press Analyze.")
    else:
        st.subheader("1 · How sensitive are you?")
        st.dataframe(a.exposures.style.format({"Market beta": NUM, "Rate sensitivity": "{:.4f}",
                                               "Empirical duration": "{:.1f}", "R²": NUM}))
        explain("Market beta and empirical duration",
                "We measured, from each asset's own daily history, how it reacted to:\n"
                "- the **stock market** (SPY): *market beta*, and\n"
                "- the **10-year Treasury yield**: *empirical duration*. A duration of 6 means roughly "
                "**-6% for every +1 percentage point** rise in rates.\n\n"
                "Bonds lose value when rates rise because new bonds pay more, so old ones must get cheaper. "
                "Longer bonds (higher duration) are hit harder. **R²** shows how much of the movement these two "
                "factors explain (0 to 1).")

        st.subheader("2 · What-if scenarios")
        st.dataframe(a.scenarios.style.format({k: PCT for k in a.scenarios.columns if k != "Portfolio ($)"}
                                              | {"Portfolio ($)": USD}))
        show(plots.scenario_bars(a.scenarios, title="Estimated move in each scenario (today's weights)"))
        with st.expander("📘 What each scenario means", expanded=learn):
            for name, s in stress.HYPOTHETICAL.items():
                st.markdown(f"- **{name}**: {s['why']}")

        st.subheader("3 · Build your own shock")
        has_crypto = any(is_crypto(t) for t in a.assets)
        s1, s2, s3 = st.columns(3)
        shock_m = s1.slider("Stock market move (%)", -50, 30, -15) / 100
        shock_r = s2.slider("10-year yield change (percentage points)", -3.0, 3.0, 1.0, 0.25)
        shock_c = s3.slider("Crypto move (%)", -90, 100, -50) / 100 if has_crypto else None
        move, per_asset = stress.custom_shock(a.exposures, a.weights_now, shock_m, shock_r, shock_c)
        st.metric("Estimated portfolio move", f"{move:+.1%}", md(("-" if move < 0 else "+") + USD(abs(move * a.value))),
                  delta_color="normal")
        st.dataframe(per_asset.to_frame("Estimated move").style.format(PCT))
        explain("Is this exact?",
                "It is a *linear* estimate (sensitivity x shock). It works well for normal-sized moves. In "
                "real crashes correlations jump and losses can be bigger, which is why we also replay real "
                "crises below.")

        if a.crises is not None and len(a.crises):
            st.subheader("4 · Replaying real crises")
            st.dataframe(a.crises.style.format({"Portfolio": PCT, "S&P 500 (SPY)": PCT,
                                                "Share of portfolio estimated": PCT, "Portfolio ($)": USD}))
            show(plots.scenario_bars(a.crises, title="Historical crisis replays (today's weights)"))
            with st.expander("Per-asset detail (actual vs. estimated)"):
                st.dataframe(a.crisis_detail)
            with st.expander("📘 About these crises", expanded=learn):
                for name, s in stress.HISTORICAL.items():
                    st.markdown(f"- **{name}** ({s['start']} to {s['end']}): {s['why']}")
                st.markdown("Assets that didn't exist yet (e.g. most crypto before 2014) are **estimated** from "
                            "their beta to the S&P 500. That likely *understates* how badly they would have done.")

    st.subheader("5 · What if the risk-free rate changes?")
    show(plots.sharpe_vs_rf_chart(a.rf_sensitivity, a.risk_free))
    explain("Why the risk-free rate matters",
            "The Sharpe ratio subtracts the T-bill yield, the return you could get with zero risk. When "
            "rates are high (like 2023-25), a portfolio must earn more just to look good. Higher risk-free "
            "rates also raise the bar for stocks, and can push their prices down.")

    st.subheader("6 · Bond math calculator")
    b1, b2, b3 = st.columns(3)
    dur = b1.number_input("Duration (years)", 0.0, 30.0, 6.0, 0.5, help="BND is about 6, TLT about 16.")
    dy = b2.number_input("Yield change (percentage points)", -5.0, 5.0, 1.0, 0.25)
    conv = b3.number_input("Convexity (optional)", 0.0, 400.0, 0.0, 10.0)
    st.metric("Estimated bond price change", f"{stress.bond_price_change(dur, dy, conv):+.2%}")
    explain("The formula",
            "%ΔPrice ≈ -Duration x ΔYield + ½ x Convexity x ΔYield²\n\n"
            "Example: a fund with duration 6 when yields rise 1 point (0.01): -6 x 0.01 = **-6%**. Convexity is a "
            "small correction that softens losses and boosts gains for big moves.")

# ---------------------------------------------------------------- optimizer
with tabs[4]:
    if a.frontier is None:
        st.info("Add at least two assets to use the optimizer.")
    else:
        st.subheader("Your weights vs. the optimal mixes")
        st.dataframe(a.optimal.style.format(PCT))
        st.dataframe(a.optimal_stats.style.format({"Avg Return": PCT, "Volatility": PCT, "Sharpe": NUM}))
        show(plots.efficient_frontier(a.frontier, a.optimal_stats))
        show(plots.weights_chart(a.optimal))
        explain("How to read the optimizer",
                "- **Max Sharpe** ⭐: the mix with the best return per unit of risk *over this period*.\n"
                "- **Min Volatility** ◆: the smoothest ride.\n"
                "- **Risk Parity** ■: every asset contributes the same amount of risk; calm assets get more weight.\n"
                "- Each dot is one possible mix; the upper-left edge is the *efficient frontier*.\n\n"
                "Stats assume fixed weights rebalanced daily. Optimizers chase the past, so use the "
                "'max % in any one asset' setting to stay diversified.")

# ---------------------------------------------------------------- correlation
with tabs[5]:
    if len(a.assets) > 1:
        show(plots.correlation_heatmap(a.correlation))
        explain("Correlation", "1 = always move together, 0 = unrelated, -1 = move opposite. Mixing assets with "
                               "low correlation is what makes diversification work.")
    else:
        st.info("Add at least two assets to see correlations.")

# ---------------------------------------------------------------- data
with tabs[6]:
    st.caption(f"{len(a.prices)} days of dividend-adjusted closing prices from Yahoo Finance.")
    st.dataframe(a.prices.sort_index(ascending=False).style.format("{:,.2f}"))
    st.download_button("Download prices (CSV)", a.prices.to_csv(), "prices.csv", "text/csv")

# ---------------------------------------------------------------- glossary
with tabs[7]:
    st.markdown(md("""
**Alpha** · Return beyond what market exposure (beta) explains.
**Beta** · Sensitivity to the benchmark. 1.2 = moves 20% more than the market.
**Block bootstrap** · Monte Carlo method that resamples real chunks of history.
**Calmar ratio** · Annual return / max drawdown.
**Convexity** · Second-order bond effect; makes the price-yield curve bend.
**Correlation** · How two assets move together, from -1 to 1.
**CVaR / Expected Shortfall** · Average loss on the worst days beyond VaR.
**Drawdown** · Fall from a previous peak.
**Duration** · A bond's sensitivity to interest rates, in years.
**Efficient frontier** · The best possible return for each level of risk.
**Information ratio** · Average outperformance / tracking error.
**Max drawdown** · Worst peak-to-trough loss.
**Monte Carlo** · Simulating many random futures to see a range of outcomes.
**Rebalancing** · Trading back to target weights.
**Risk-free rate** · Return on T-bills; the zero-risk baseline.
**Risk parity** · Weights set so each asset contributes equal risk.
**Sharpe ratio** · (Return - risk-free rate) / volatility.
**Sortino ratio** · Like Sharpe, but only downside volatility counts.
**Stress test** · Estimating losses under a specific bad scenario.
**Tracking error** · Volatility of your return minus the benchmark's.
**VaR (Value at Risk)** · Loss you won't exceed on, e.g., 95% of days.
**Volatility** · Annualised standard deviation of returns.
""").replace("\n**", "  \n**"))

# ================================================================ report (bottom)
st.divider()
st.header("📄 Download your report")
st.write("A PDF with the summary, every table and chart, Monte Carlo and stress tests. Ready to print or email.")
d1, d2 = st.columns(2)
with d1:
    with st.spinner("Building PDF..."):
        pdf = make_pdf(params)
    st.download_button("Download PDF report", pdf, f"SharpeScope_report_{dt.date.today()}.pdf",
                       "application/pdf", type="primary")
with d2:
    st.download_button("Download summary table (CSV)", a.summary.join(a.risk).to_csv(),
                       "sharpescope_summary.csv", "text/csv")

# ================================================================ feedback
st.divider()
st.header("💬 Feedback & feature requests")
st.write("What should Quant SharpeScope add next? Tell us what you'd like to see.")
with st.form("feedback"):
    f_type = st.selectbox("Type", ["Feature request", "Bug report", "Question", "Other"])
    f_title = st.text_input("Short title", placeholder="e.g. Add options strategies")
    f_body = st.text_area("Details", placeholder="What would you like, and how would you use it?")
    sent = st.form_submit_button("Prepare feedback")
if sent:
    if not f_title.strip():
        st.warning("Please add a short title.")
    else:
        body = f"**Type:** {f_type}\n\n{f_body}\n\n---\n_Sent from the Quant SharpeScope app_"
        url = (f"https://github.com/{REPO}/issues/new?"
               + urllib.parse.urlencode({"title": f"[{f_type}] {f_title}", "body": body,
                                          "labels": "feedback"}))
        st.link_button("Send on GitHub (opens a pre-filled form)", url)
        st.caption("Requires a free GitHub account. Nothing is posted until you press *Submit* on GitHub.")
if FEEDBACK_URL:
    st.link_button("No GitHub account? Use the quick form", FEEDBACK_URL)
