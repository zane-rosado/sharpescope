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
            out[c] = df[c].map(lambda v: f"${v:,.0f}")
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
        page.text(0.06, 0.955, "Portfolio Analysis Report", fontsize=20, fontweight="bold")
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
            ("Return & risk", _fmt_table(a.summary, pct_cols=("Total Return", "Annual Return", "Volatility", "Max Drawdown"),
                                         num_cols=("Sharpe", "Sortino", "Beta", "Calmar"))),
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
            _draw_table(page.add_axes([0.2, top - h, 0.76, h]), tbl, title, font=font)
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
        last = [plots.rolling_sharpe_chart(a.returns["Portfolio"], a.risk_free, a.periods)]
        if len(a.assets) > 1:
            last.append(plots.correlation_heatmap(a.correlation))
        _fig_page(pdf, last)

        info = pdf.infodict()
        info["Title"] = "Portfolio Analysis Report"
        info["Creator"] = "Portfolio Analyzer"
    return buf.getvalue()
