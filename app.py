"""Portfolio Analyzer web app.   Run with:   py -m streamlit run app.py"""

from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from portfolio_analyzer import plots
from portfolio_analyzer.analysis import analyze, plain_english
from portfolio_analyzer.data import fetch_prices
from portfolio_analyzer.report import build_pdf
from portfolio_analyzer.universe import CATALOG, PRESETS, clean_ticker, label, name_of

st.set_page_config(page_title="Portfolio Analyzer", page_icon="📈", layout="wide")

PCT = "{:.2%}"
NUM = "{:.2f}"


# ----------------------------------------------------------------- cached work
@st.cache_data(ttl=900, show_spinner=False)  # re-download prices at most every 15 min
def load_prices(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    return fetch_prices(list(tickers), start)


@st.cache_data(show_spinner=False)
def run_analysis(params: dict):
    download = list(dict.fromkeys(list(params["weights"]) + ([params["benchmark"]] if params["benchmark"] else [])))
    prices = load_prices(tuple(download), params["start"])
    return analyze(prices, params["weights"], params["rf"], params["benchmark"],
                   params["value"], params["var_level"], params["max_weight"])


@st.cache_data(show_spinner=False)
def make_pdf(params: dict) -> bytes:
    return build_pdf(run_analysis(params))


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def md(text: str) -> str:
    """Escape $ so Streamlit doesn't treat dollar amounts as math formulas."""
    return text.replace("$", "\\$")


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Pick your assets")
    preset = st.selectbox("Start from a ready-made portfolio", ["Custom"] + list(PRESETS), index=1)
    base = PRESETS.get(preset, PRESETS["Vanguard core (60/30/10)"])

    chosen = st.multiselect(
        "Assets (type to search: stocks, bonds, crypto)",
        options=list(CATALOG),
        default=[t for t in base if t in CATALOG],
        format_func=label,
        key=f"assets::{preset}",
    )
    extra = st.text_input("Any other Yahoo tickers (comma-separated)", placeholder="e.g. AAPL, MSFT, DOT-USD",
                          key=f"extra::{preset}")
    tickers = list(dict.fromkeys(chosen + [clean_ticker(t) for t in extra.split(",") if t.strip()]))

    st.header("2 · Set weights")
    equal = st.checkbox("Use equal weights")
    default_w = [100 / len(tickers) if equal else base.get(t, 0.10) * 100 for t in tickers] if tickers else []
    edited = st.data_editor(
        pd.DataFrame({"Ticker": tickers, "Name": [name_of(t) for t in tickers], "Weight %": default_w}),
        hide_index=True,
        disabled=["Ticker", "Name"],
        column_config={"Weight %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0,
                                                                 format="%.1f")},
        key=f"weights::{preset}::{'|'.join(tickers)}::{equal}",
    )
    total = float(edited["Weight %"].fillna(0).sum()) if len(edited) else 0.0
    st.caption(f"Total: {total:.1f}%" + ("" if abs(total - 100) < 0.05 else " (will be scaled to 100%)"))

    st.header("3 · Settings")
    start = st.date_input("Start date", dt.date(2021, 1, 1), min_value=dt.date(2000, 1, 1),
                          max_value=dt.date.today() - dt.timedelta(days=30))
    bench_opts = ["None"] + list(dict.fromkeys(["VOO", "VTI", "QQQ", "BTC-USD"] + tickers))
    benchmark = st.selectbox("Benchmark (for beta)", bench_opts, index=1)
    with st.expander("More settings"):
        rf = st.number_input("Risk-free rate (% per year)", 0.0, 20.0, 4.0, 0.25,
                             help="Use the current 3-month Treasury bill yield.") / 100
        value = st.number_input("Portfolio value ($)", 100.0, 1e9, 10_000.0, 1_000.0, format="%.0f")
        var_level = st.select_slider("VaR confidence", options=[0.90, 0.95, 0.99], value=0.95,
                                     format_func=lambda x: f"{x:.0%}")
        max_weight = st.slider("Optimizer: max % in any one asset", 10, 100, 100, 5,
                               help="Stops the optimizer putting everything in one asset.") / 100

    go = st.button("Analyze", type="primary")

if go:
    weights = {t: float(w) for t, w in zip(edited["Ticker"], edited["Weight %"].fillna(0)) if w and w > 0}
    if not weights:
        st.sidebar.error("Give at least one asset a weight above 0.")
    else:
        st.session_state.params = {
            "weights": weights, "start": str(start), "rf": rf, "value": value, "var_level": var_level,
            "max_weight": max_weight, "benchmark": None if benchmark == "None" else benchmark,
        }

# ----------------------------------------------------------------- main page
st.title("📈 Portfolio Analyzer")
st.caption("Risk, return, Value at Risk and optimal weights for stock, ETF and crypto portfolios. "
           "Data: Yahoo Finance. Educational use only, not investment advice.")

params = st.session_state.get("params")
if not params:
    st.info("**How to use:** 1) pick assets or a ready-made portfolio on the left, 2) set weights, "
            "3) press **Analyze**. A summary appears here, and a PDF report is at the bottom of the page.")
    st.stop()

try:
    with st.spinner("Downloading prices and crunching numbers..."):
        a = run_analysis(params)
except Exception as e:  # show a friendly message instead of a crash
    st.error(f"Something went wrong: {e}")
    st.stop()

p, rk = a.summary.loc["Portfolio"], a.risk.loc["Portfolio"]

# ---- Summary (top of page)
st.header("Summary")
c = st.columns(5)
c[0].metric("Annual return", f"{p['Annual Return']:.1%}")
c[1].metric("Sharpe ratio", f"{p['Sharpe']:.2f}")
c[2].metric("Volatility", f"{p['Volatility']:.1%}")
c[3].metric("Max drawdown", f"{p['Max Drawdown']:.1%}")
c[4].metric(f"1-day VaR ({a.var_level:.0%})", f"${rk['1-Day VaR ($)']:,.0f}", f"-{rk['1-Day VaR']:.2%}",
            delta_color="off")
st.markdown("\n".join(f"- {md(line)}" for line in plain_english(a)))
for note in a.notes:
    st.caption(f"ℹ️ {md(note)}")

# ---- Detail tabs
tab_perf, tab_risk, tab_opt, tab_corr, tab_data = st.tabs(
    ["📈 Performance", "⚠️ Risk (VaR)", "🎯 Optimizer", "🔗 Correlation", "📋 Data"])

with tab_perf:
    st.subheader("Return & risk by asset")
    st.dataframe(a.summary.style.format({"Annual Return": PCT, "Volatility": PCT, "Max Drawdown": PCT,
                                         "Sharpe": NUM, "Sortino": NUM, "Beta": NUM, "Calmar": NUM}))
    show(plots.growth_and_drawdown(a.returns, a.value))
    show(plots.rolling_sharpe_chart(a.returns["Portfolio"], a.risk_free, a.periods))
    with st.expander("What do these mean?"):
        st.markdown(
            "- **Annual return**: average yearly growth (compounded).\n"
            "- **Volatility**: how much returns swing; higher = bumpier ride.\n"
            "- **Sharpe**: return above the risk-free rate per unit of volatility. Above 1 is strong.\n"
            "- **Sortino**: like Sharpe, but only counts downside swings as risk.\n"
            "- **Max drawdown**: the worst drop from a high point to a low point.\n"
            "- **Beta**: how much the asset moves with the benchmark (1 = moves in step).\n"
            "- **Calmar**: annual return divided by max drawdown.")

with tab_risk:
    st.subheader(f"Value at Risk ({a.var_level:.0%} confidence, {md(f'${a.value:,.0f}')} portfolio)")
    st.dataframe(a.risk.style.format({c: PCT for c in a.risk.columns if "($)" not in c}
                                     | {"1-Day VaR ($)": "${:,.0f}"}))
    show(plots.return_histogram(a.returns["Portfolio"], rk["1-Day VaR"], rk["1-Day CVaR"], a.var_level))
    with st.expander("What is VaR?"):
        st.markdown(md(
            f"- **1-Day VaR**: on {a.var_level:.0%} of days you lose less than this. On the worst "
            f"{1 - a.var_level:.0%} of days you lose more.\n"
            "- **CVaR (Expected Shortfall)**: the *average* loss on those worst days.\n"
            "- **1-Month VaR**: the 1-day figure scaled to about 21 trading days.\n"
            "- **Parametric VaR**: the same idea, assuming a bell-curve distribution. If it is much lower "
            "than historical VaR, the asset has fatter tails (more extreme days) than a bell curve."))

with tab_opt:
    if a.frontier is None:
        st.info("Add at least two assets to use the optimizer.")
    else:
        st.subheader("Your weights vs. the optimal mixes")
        st.dataframe(a.optimal.style.format(PCT))
        st.dataframe(a.optimal_stats.style.format({"Avg Return": PCT, "Volatility": PCT, "Sharpe": NUM}))
        show(plots.efficient_frontier(a.frontier, a.optimal_stats))
        show(plots.weights_chart(a.optimal))
        st.caption("⭐ Max Sharpe = best return per unit of risk over this period. ◆ Min Volatility = "
                   "smoothest ride. These are based on past prices, not a forecast. Use the "
                   "'max % in any one asset' setting to keep the mix diversified.")

with tab_corr:
    if len(a.assets) > 1:
        show(plots.correlation_heatmap(a.correlation))
        st.caption("Assets with low or negative correlation smooth out each other's swings.")
    else:
        st.info("Add at least two assets to see correlations.")

with tab_data:
    st.caption(f"{len(a.prices)} days of dividend-adjusted closing prices from Yahoo Finance.")
    st.dataframe(a.prices.sort_index(ascending=False).style.format("{:,.2f}"))
    st.download_button("Download prices (CSV)", a.prices.to_csv(), "prices.csv", "text/csv")

# ---- Report (bottom of page)
st.divider()
st.header("📄 Download your report")
st.write("A PDF with the summary, every table and every chart, ready to print or attach to an email.")
col1, col2 = st.columns(2)
with col1:
    with st.spinner("Building PDF..."):
        pdf = make_pdf(params)
    st.download_button("Download PDF report", pdf, f"portfolio_report_{dt.date.today()}.pdf",
                       "application/pdf", type="primary")
with col2:
    st.download_button("Download summary table (CSV)", a.summary.join(a.risk).to_csv(),
                       "portfolio_summary.csv", "text/csv")
