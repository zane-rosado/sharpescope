"""One-click PDF report, built with matplotlib only (no extra installs)."""

from __future__ import annotations

import io
import textwrap
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from . import plots
from .analysis import Analysis, plain_english

PAGE = (8.5, 11)  # US Letter, inches


def _fmt_table(df: pd.DataFrame, pct_cols=(), num_cols=(), money_cols=()) -> pd.DataFrame:
    out = df.copy().astype(object)
    for c in df.columns:
        if c in pct_cols:
            out[c] = df[c].map(lambda v: f"{v:.2%}")
        elif c in money_cols:
            out[c] = df[c].map(lambda v: f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}")
        elif c in num_cols:
            out[c] = df[c].map(lambda v: f"{v:.2f}")
    return out


def _draw_table(ax, df: pd.DataFrame, title: str, font: int = 8):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    t = ax.table(cellText=df.values, rowLabels=list(df.index), colLabels=list(df.columns),
                 loc="upper center", cellLoc="right", colLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(font)
    t.scale(1, 1.35)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold")
        if c == -1 and df.index[r - 1 if r else 0] == "Portfolio" and r:
            cell.set_text_props(fontweight="bold")


def _to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def _fig_page(pdf: PdfPages, figs: list):
    """Stack one or more charts on a letter-size page, scaled to fit."""
    imgs = [_to_img(f) for f in figs]
    max_w, gap, avail = 7.5, 0.3, 9.6
    widths = [min(max_w, i.shape[1] / 140) for i in imgs]  # never enlarge a small chart
    heights = [w * i.shape[0] / i.shape[1] for w, i in zip(widths, imgs)]
    scale = min(1.0, (avail - gap * (len(imgs) - 1)) / sum(heights))
    page = plt.figure(figsize=PAGE)
    top = PAGE[1] - 0.7
    for img, w0, h in zip(imgs, widths, heights):
        w, h = w0 * scale, h * scale
        left = (PAGE[0] - w) / 2
        ax = page.add_axes([left / PAGE[0], (top - h) / PAGE[1], w / PAGE[0], h / PAGE[1]])
        ax.imshow(img)
        ax.axis("off")
        top -= h + gap
    pdf.savefig(page)
    plt.close(page)


def build_pdf(a: Analysis) -> bytes:
    """Return the report as PDF bytes (ready for a download button or a file)."""
    buf = io.BytesIO()
    # parse_math=False stops matplotlib reading "$122 ... $10,000" as a math formula
    with plt.rc_context({"text.parse_math": False}), PdfPages(buf) as pdf:
        # ---- Page 1+: summary, then tables flowing onto new pages as needed ----
        footer = ("For education only. Not investment advice. Past performance does not "
                  "predict future results.")
        page = plt.figure(figsize=PAGE)
        page.text(0.06, 0.955, "Quant SharpeScope Report", fontsize=20, fontweight="bold")
        page.text(0.06, 0.93, f"Generated {date.today():%B %d, %Y}  |  Data: Yahoo Finance (adjusted close)  |  "
                  f"Risk-free rate {a.risk_free:.2%}", fontsize=8, color="#555555")

        y = 0.895
        page.text(0.06, y, "Summary", fontsize=13, fontweight="bold")
        y -= 0.025
        for line in plain_english(a) + a.notes:
            for i, part in enumerate(textwrap.wrap(line, 105)):
                page.text(0.07 if i else 0.06, y, ("• " if i == 0 else "  ") + part, fontsize=8.5)
                y -= 0.018
            y -= 0.004

        tables = [
            ("Return & risk", _fmt_table(a.summary, pct_cols=("Total Return", "Annual Return", "Volatility",
                                                              "Max Drawdown", "Alpha"),
                                         num_cols=("Sharpe", "Sortino", "Beta", "Calmar", "Info Ratio"))),
            (f"Value at Risk ({a.var_level:.0%} confidence, ${a.value:,.0f} portfolio)",
             _fmt_table(a.risk, pct_cols=("1-Day VaR", "1-Day CVaR", "1-Month VaR", "Parametric 1-Day VaR"),
                        money_cols=("1-Day VaR ($)",))),
        ]
        if "Max Sharpe" in a.optimal:
            opt = a.optimal.copy()
            if a.current_weights is not None:
                opt.insert(1, "Weight now (drifted)", a.current_weights)
            tables.append(("Weights: yours vs. optimised (optimised = fixed weights, rebalanced daily)",
                           _fmt_table(opt, pct_cols=tuple(opt.columns))))

        if a.montecarlo is not None:
            st = a.montecarlo.stats()
            mc = pd.DataFrame({"Value": [f"{v:.1%}" if ("Chance" in k or "growth" in k) else f"${v:,.0f}"
                                         for k, v in st.items()]}, index=list(st.keys()))
            tables.append((f"Monte Carlo: {a.montecarlo.years} years ({a.montecarlo.method})", mc))
        if a.exposures is not None:
            ex = a.exposures[["Market beta", "Empirical duration", "R²"]]
            tables.append(("Sensitivities: market beta and rate sensitivity (empirical duration, years)",
                           _fmt_table(ex, num_cols=tuple(ex.columns))))
        if a.scenarios is not None:
            sc = a.scenarios[["Portfolio", "Portfolio ($)"]]
            tables.append(("Stress tests: estimated impact today",
                           _fmt_table(sc, pct_cols=("Portfolio",), money_cols=("Portfolio ($)",))))
        if a.crises is not None and len(a.crises):
            tables.append(("Historical crisis replays (today's weights)",
                           _fmt_table(a.crises, pct_cols=("Portfolio", "S&P 500 (SPY)", "Share of portfolio estimated"),
                                      money_cols=("Portfolio ($)",))))

        top = y - 0.01
        row_h = 0.03 if max(len(t) for _, t in tables) <= 8 else 0.022
        font = 8 if row_h == 0.03 else 7
        for title, tbl in tables:
            h = row_h * (len(tbl) + 2)
            if top - h < 0.06:                      # doesn't fit: start a new page
                page.text(0.06, 0.03, footer, fontsize=7, color="#777777")
                pdf.savefig(page)
                plt.close(page)
                page = plt.figure(figsize=PAGE)
                top = 0.95
            longest = max(len(str(i)) for i in tbl.index)
            left = 0.06 + min(0.38, 0.0072 * longest * font / 8)   # room for the row labels
            _draw_table(page.add_axes([left, top - h, 0.95 - left, h]), tbl, title, font=font)
            top -= h + 0.045
        page.text(0.06, 0.03, footer, fontsize=7, color="#777777")
        pdf.savefig(page)
        plt.close(page)

        # ---- Chart pages ----
        rk = a.risk.loc["Portfolio"]
        _fig_page(pdf, [
            plots.growth_and_drawdown(a.returns, a.value),
            plots.return_histogram(a.returns["Portfolio"], rk["1-Day VaR"], rk["1-Day CVaR"], a.var_level),
        ])
        if a.frontier is not None:
            _fig_page(pdf, [plots.efficient_frontier(a.frontier, a.optimal_stats),
                            plots.weights_chart(a.optimal)])
        page_mc = []
        if a.montecarlo is not None:
            page_mc.append(plots.monte_carlo_fan(a.montecarlo))
        if a.monthly is not None and len(a.monthly):
            page_mc.append(plots.monthly_heatmap(a.monthly))
        if page_mc:
            _fig_page(pdf, page_mc)
        page_stress = []
        if a.scenarios is not None:
            page_stress.append(plots.scenario_bars(a.scenarios, title="Stress tests: estimated portfolio move"))
        if a.crises is not None and len(a.crises):
            page_stress.append(plots.scenario_bars(a.crises, title="Historical crisis replays"))
        if a.rf_sensitivity is not None:
            page_stress.append(plots.sharpe_vs_rf_chart(a.rf_sensitivity, a.risk_free))
        if page_stress:
            _fig_page(pdf, page_stress)
        last = [plots.rolling_sharpe_chart(a.returns["Portfolio"], a.risk_free, a.periods)]
        if len(a.assets) > 1:
            last.append(plots.correlation_heatmap(a.correlation))
        _fig_page(pdf, last)

        info = pdf.infodict()
        info["Title"] = "Quant SharpeScope Report"
        info["Creator"] = "Quant SharpeScope (QSS)"
    return buf.getvalue()
