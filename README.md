# 📈 Quant SharpeScope (QSS)

![tests](https://github.com/zane-rosado/sharpescope/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**Portfolio risk, stress tests, Monte Carlo and optimization for stocks, ETFs and crypto. Built for quants, made to teach.**

Free and open source. Point-and-click web app, no coding needed. Every number comes with a plain-English explanation, so it works as a professional tool *and* a learning tool.

![Growth of $10,000 and drawdown chart](examples/sample_chart.png)

## What it does

| | |
|---|---|
| 🧾 **Plain-English summary** | A few sentences at the top that explain the results to anyone |
| 📈 **Performance** | Total & annual return, volatility, Sharpe, Sortino, max drawdown, beta, **alpha**, **information ratio**, Calmar, monthly-returns heatmap, rolling Sharpe |
| ⚠️ **Value at Risk** | 1-day & 1-month historical VaR, CVaR (expected shortfall), parametric VaR, in % and $; risk contribution by asset |
| 🔮 **Monte Carlo** | Thousands of possible futures (block bootstrap of real returns), monthly contributions, goal probability, optional long-run return override |
| 🌪️ **Stress tests** | Rate shocks (+1%, +2%, -1%), bear markets, stagflation, recession, crypto winter; **build-your-own shock** sliders |
| 🏦 **Rates & bonds** | Each asset's *empirical duration* (sensitivity to the 10-year yield), bond-math calculator, Sharpe vs. risk-free rate |
| 🕰️ **Crisis replays** | Dot-com, 2008, 2018 Q4, COVID, 2022 rate shock replayed on today's portfolio |
| 🎯 **Optimizer** | Max-Sharpe, minimum-volatility and **risk-parity** weights with an efficient-frontier chart and a per-asset cap |
| 🔁 **Rebalancing** | Daily, monthly, quarterly, yearly or never (buy and hold) |
| 💵 **Flexible input** | Enter % weights, $ amounts or share counts |
| ₿ **Crypto** | BTC, ETH, XRP, SOL, BNB, ADA, DOGE, AVAX, LINK, LTC, mixed correctly with stocks |
| 📘 **Learning mode** | One switch opens an explanation next to every number, plus a glossary |
| 📄 **PDF report** | One click: summary, all tables and charts, Monte Carlo and stress tests |
| 💬 **Feedback** | Built-in form to request features |

## Checked against real data

- **Bloomberg PORT:** a 20-stock buy-and-hold portfolio (12/31/25 to 10/30/26) returns **44.90%** here vs. **44.84%** in Bloomberg; IVV **11.75%** in both.
- **Rate sensitivity:** BND's measured duration comes out at **5.6 years** (Vanguard publishes about 6); TLT **14**.
- **Crisis replays:** SPY **-55.2%** in 2007-09, **-33.7%** in the COVID crash, **-24.5%** in 2022, matching the historical record.

## Quick start (Windows)

```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The app opens in your browser. Pick a preset or your own assets, enter weights/dollars/shares and press **Analyze**.

### Command line

```bash
py -m sharpescope.cli VOO VXUS BND --weights 0.6 0.3 0.1 --benchmark VOO --stress
py -m sharpescope.cli --preset "Crypto majors" --mc-years 5 --goal 20000 --pdf crypto.pdf
py -m sharpescope.cli VOO BND --weights 0.7 0.3 --mc-contribution 200 --mc-return 0.07
```

Run `py -m sharpescope.cli --help` for every option.

### As a Python library

```python
from sharpescope import analyze, plain_english
from sharpescope.data import fetch_prices, fetch_factors
from sharpescope.report import build_pdf

prices = fetch_prices(["VOO", "BND", "BTC-USD"], start="2021-01-01")
a = analyze(prices, {"VOO": 0.6, "BND": 0.3, "BTC-USD": 0.1}, risk_free=0.04,
            factors=fetch_factors("2021-01-01"), mc_years=10)
print("\n".join(plain_english(a)))
print(a.scenarios[["Portfolio"]])     # stress tests
open("report.pdf", "wb").write(build_pdf(a))
```

## Methods

- **Sharpe** = mean(excess return) / std × √N, where N = 252 (stocks/mixed) or 365 (crypto only)
- **Alpha** (Jensen) = annualised mean excess return - β × benchmark excess return
- **Historical VaR (95%)** = 5th-percentile daily loss; CVaR = average loss beyond it; multi-day = VaR × √days
- **Monte Carlo** = block bootstrap of 21-day chunks of real daily returns; optional mean shift to a chosen annual return
- **Stress sensitivities** = OLS of each asset's daily return on SPY return and the daily change in the 10-year yield (^TNX); empirical duration = -100 × rate coefficient
- **Crisis replays** = actual window returns; assets without history are estimated as β × S&P 500 return and flagged
- **Max Sharpe / Min Vol / Risk Parity** = SciPy SLSQP, long-only, weights sum to 1, optional cap per asset
- **Bond math** = %ΔP ≈ -D·Δy + ½·C·Δy²

## Project layout

```
sharpescope/
  analysis.py   # analyze(): runs everything; used by the app, CLI and PDF
  metrics.py    # return, risk-adjusted metrics, alpha, info ratio, monthly table
  risk.py       # VaR, CVaR
  simulate.py   # Monte Carlo
  stress.py     # factor sensitivities, scenarios, crisis replays, bond math
  optimize.py   # max-Sharpe, min-volatility, risk parity, efficient frontier
  data.py       # Yahoo Finance prices, SPY/10-year yield, holdings conversion
  universe.py   # ticker list and presets (add a line to add an asset)
  plots.py      # all charts
  report.py     # PDF report
  cli.py        # command line
app.py          # Streamlit web app
tests/          # 49 tests, run automatically on every change
```

## Feedback

Have an idea? Use the **Feedback** section at the bottom of the app or [open an issue](https://github.com/zane-rosado/sharpescope/issues/new/choose).

## Roadmap

- [ ] Hosted version on Streamlit Community Cloud
- [ ] Interactive (hover) charts
- [ ] Upload a brokerage CSV
- [ ] Factor exposure (size, value, momentum)
- [ ] Options and income strategies

## Disclaimer

For education and research only. This is not investment advice. Stress tests and Monte Carlo results are estimates based on past data, not predictions. Price data comes from Yahoo Finance via `yfinance` and may contain errors.

## License

MIT, see [LICENSE](LICENSE). Created by Zane Rosado.
