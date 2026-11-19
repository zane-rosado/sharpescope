# Portfolio Analyzer

![tests](https://github.com/zane-rosado/portfolio-analyzer/actions/workflows/tests.yml/badge.svg)

A free, open-source Python tool that measures the **risk and return of an ETF or stock portfolio**.
Give it tickers and weights; it pulls price history and reports the numbers investors and fund managers actually use to judge a portfolio.

![Growth of $10,000 and drawdown chart](examples/sample_chart.png)
*Example output using the synthetic data in `examples/sample_prices.csv`.*

## What it calculates

| Metric | What it tells you |
|---|---|
| **Annual return (CAGR)** | Compound yearly growth rate |
| **Volatility** | Annualised standard deviation of daily returns |
| **Sharpe ratio** | Return earned per unit of total risk, above the risk-free rate |
| **Sortino ratio** | Like Sharpe, but only penalises downside moves |
| **Max drawdown** | Worst peak-to-trough loss |
| **Beta** | Sensitivity to a benchmark (e.g. VOO for the S&P 500) |
| **Correlation matrix** | How much the holdings move together (diversification) |

The portfolio is modelled as rebalanced daily to fixed, long-only weights.

## Quick start

```bash
git clone https://github.com/zane-rosado/portfolio-analyzer.git
cd portfolio-analyzer
pip install -r requirements.txt
```

**Command line**

```bash
python -m portfolio_analyzer.cli VOO VXUS BND --weights 0.6 0.3 0.1 \
    --start 2021-01-01 --rf 0.04 --benchmark VOO --plot
```

**Web app**

```bash
streamlit run app.py
```

**Offline (no internet needed)**

```bash
python -m portfolio_analyzer.cli --csv examples/sample_prices.csv --weights 0.6 0.3 0.1
```

**As a library**

```python
from portfolio_analyzer import metrics
from portfolio_analyzer.data import fetch_prices

prices = fetch_prices(["VOO", "BND"], start="2021-01-01")
returns = metrics.daily_returns(prices)
port = metrics.portfolio_returns(returns, {"VOO": 0.7, "BND": 0.3})
print(metrics.sharpe_ratio(port, risk_free=0.04))
```

## Formulas

- Daily excess return: `r_t − rf / 252`
- Sharpe: `mean(excess) / std(excess) × √252`
- Sortino: `mean(excess) / downside_deviation × √252`
- Max drawdown: `min(wealth_t / max(wealth_0..t) − 1)`
- Beta: `cov(r, r_benchmark) / var(r_benchmark)`

## Project layout

```
portfolio_analyzer/
  metrics.py   # all calculations (pure functions, fully tested)
  data.py      # Yahoo Finance download + CSV loader
  plots.py     # growth-of-$10k and drawdown charts
  cli.py       # command-line interface
app.py         # Streamlit web app
tests/         # pytest suite, run automatically by GitHub Actions
examples/      # sample data and chart
```

## Roadmap

- [ ] Rebalancing frequency options (monthly, quarterly, none)
- [ ] Rolling Sharpe and rolling volatility charts
- [ ] Value at Risk (VaR) and Conditional VaR
- [ ] Mean-variance optimiser (max-Sharpe and min-volatility weights)
- [ ] Export a one-page PDF report

## Running tests

```bash
pytest -q
```

## Disclaimer

For education and research only. This is not investment advice. Price data comes from Yahoo Finance via `yfinance` and may contain errors.

## License

MIT, see [LICENSE](LICENSE).
