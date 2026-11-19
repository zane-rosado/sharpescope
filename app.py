"""Streamlit web app.  Run with:  streamlit run app.py"""

import datetime as dt

import pandas as pd
import streamlit as st

from portfolio_analyzer import metrics
from portfolio_analyzer.data import fetch_prices
from portfolio_analyzer.plots import plot_growth_and_drawdown

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.title("Portfolio Analyzer")
st.caption("Risk and return metrics for ETF and stock portfolios. Educational use only, not investment advice.")

with st.sidebar:
    tickers_text = st.text_input("Tickers (comma-separated)", "VOO, VXUS, BND")
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    st.write("Weights")
    weights = {t: st.number_input(t, 0.0, 1.0, round(1 / len(tickers), 2), 0.05) for t in tickers}
    start = st.date_input("Start date", dt.date(2021, 1, 1))
    rf = st.number_input("Risk-free rate (annual)", 0.0, 0.2, 0.04, 0.005, format="%.3f")
    benchmark = st.text_input("Benchmark (for beta)", "VOO").strip().upper()
    go = st.button("Analyze", type="primary")

if go and tickers:
    try:
        download = list(dict.fromkeys(tickers + ([benchmark] if benchmark else [])))
        prices = fetch_prices(download, str(start))
        returns = metrics.daily_returns(prices)
        port = metrics.portfolio_returns(returns[tickers], weights)
        table_input = pd.concat([returns[tickers], port], axis=1)
        bench = returns[benchmark] if benchmark else None
        table = metrics.summary(table_input, rf, bench)

        p = table.loc["Portfolio"]
        c = st.columns(4)
        c[0].metric("Annual return", f"{p['Annual Return']:.1%}")
        c[1].metric("Volatility", f"{p['Volatility']:.1%}")
        c[2].metric("Sharpe", f"{p['Sharpe']:.2f}")
        c[3].metric("Max drawdown", f"{p['Max Drawdown']:.1%}")

        st.subheader("Summary")
        st.dataframe(table.style.format({
            "Annual Return": "{:.2%}", "Volatility": "{:.2%}", "Max Drawdown": "{:.2%}",
            "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Beta": "{:.2f}",
        }))
        st.subheader("Charts")
        st.pyplot(plot_growth_and_drawdown(table_input, show=False))
        st.subheader("Correlation")
        st.dataframe(metrics.correlation_matrix(returns[tickers]).style.format("{:.2f}"))
        st.download_button("Download summary (CSV)", table.to_csv(), "portfolio_summary.csv")
    except Exception as e:
        st.error(str(e))
else:
    st.info("Pick tickers and weights in the sidebar, then press Analyze.")
