"""Exploratory data analysis: correlations, category summaries and charts (read-only).

Usage:
    python generate_eda.py data.csv
    python generate_eda.py data.xlsx --sheet 2025 --output results/
    python generate_eda.py big.csv --max-categories 8 --max-scatter-points 3000
    python generate_eda.py export.json --json-path data.orders
    python generate_eda.py orders.csv --date-column shipped_at --period month

Writes a new folder (default ./eda_output/<dataset>_<timestamp>/) containing
report.md, summary.json and charts/*.png. The input file is only ever read; its
SHA-256 is recorded before and checked after the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: no window, works on servers and in CI

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_data import (  # noqa: E402
    DATASET_HELP,
    ENCODING_HELP,
    JSON_PATH_HELP,
    DatasetError,
    classify_columns,
    describe_dataset,
    load_dataset,
    to_jsonable,
)
from report import build_findings, write_report  # noqa: E402
from time_trends import PERIOD_NAMES, time_summary  # noqa: E402

# --------------------------------------------------------------------------
# Defaults and limits
# --------------------------------------------------------------------------

DEFAULT_MAX_CATEGORIES = 10
DEFAULT_MAX_SCATTER_POINTS = 5000
DEFAULT_MAX_CHARTS = 40
TOP_SCATTER_PAIRS = 3
ID_MIN_ROWS = 20  # a 1, 2, 3, ... column this long is treated as a row counter
HEATMAP_MAX_COLUMNS = 20  # more than this and the cells become unreadable
HEATMAP_ANNOTATE_MAX = 12  # print r in each cell up to this many columns
MIN_CORR_PAIRS = 3  # pairs of values needed before a correlation is computed
LOW_N_WARNING = 30
LOG_MIN_SKEW = 1.0  # "highly skewed" threshold, same as inspect_data's skew label
LOG_MIN_RATIO = 20  # max / min: below this a log axis barely changes the picture
SEED = 0  # sampling is reproducible: the same file gives the same charts

# --------------------------------------------------------------------------
# Chart style (decision D7). One color for single-series charts, a blue<->red
# diverging scale with a gray midpoint for correlations, text in ink colors.
# --------------------------------------------------------------------------

COLOR_SERIES = "#2a78d6"  # blue: every single-series chart
COLOR_OTHER = "#b4b2a9"  # gray: the folded "Other" bar
COLOR_SURFACE = "#fcfcfb"
COLOR_INK = "#0b0b0b"
COLOR_INK_2 = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_PROBLEM = "#e34948"  # red, palette slot 8: empty periods in the time chart
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#2a78d6", "#86b6ef", "#f0efec", "#f0a3a2", "#e34948", "#a32b2a"]
)

STYLE = {
    "figure.facecolor": COLOR_SURFACE,
    "axes.facecolor": COLOR_SURFACE,
    "savefig.facecolor": COLOR_SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": COLOR_INK,
    "axes.labelcolor": COLOR_INK_2,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "axes.edgecolor": COLOR_AXIS,
    "axes.linewidth": 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": COLOR_MUTED,
    "ytick.color": COLOR_MUTED,
    "xtick.labelcolor": COLOR_INK_2,
    "ytick.labelcolor": COLOR_INK_2,
    "legend.frameon": False,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def likely_id_columns(df: pd.DataFrame, kinds: dict[str, str]) -> list[str]:
    """Numeric columns that look like row identifiers (order_id 1, 2, 3, ...).

    Their histograms and correlations are meaningless (an ID that grows over
    time "correlates" with anything that trends), so they are left out of the
    charts and correlations. Rule: whole numbers, every value unique, and one of
      - the name says so ("order_id", "customerID", "id"), or it is pandas'
        "Unnamed: 0" column (a saved row index), or
      - the values count up by exactly 1 in row order (1, 2, 3, ...) over at
        least ID_MIN_ROWS rows. Short runs are left alone: a small table of
        years or ranks is real data.
    """
    ids = []
    for name in df.columns:
        key = str(name)
        if kinds.get(key) != "numeric":
            continue
        v = df[name].dropna()
        if len(v) < 2 or v.nunique() != len(v) or not np.all(np.mod(v, 1) == 0):
            continue
        named_id = (
            re.search(r"(^|[_\s-])id$", key, re.IGNORECASE) is not None
            or key.endswith(("ID", "Id"))
            or key.startswith("Unnamed: ")
        )
        row_counter = len(v) >= ID_MIN_ROWS and bool((np.diff(v.to_numpy()) == 1).all())
        if named_id or row_counter:
            ids.append(key)
    return ids


def correlation_columns(df: pd.DataFrame, kinds: dict[str, str], exclude: list[str]) -> list[str]:
    """Numeric columns usable for correlation: not IDs, not constant, enough values."""
    cols = []
    for name in df.columns:
        key = str(name)
        if kinds.get(key) != "numeric" or key in exclude:
            continue
        v = df[name].dropna()
        if len(v) >= MIN_CORR_PAIRS and v.nunique() > 1:
            cols.append(key)
    return cols


def correlations(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Pearson and Spearman correlation matrices plus every pair ranked by strength.

    Pearson measures a straight-line relationship and is sensitive to outliers.
    Spearman correlates the ranks, so it captures any steadily rising or falling
    relationship and is robust to outliers. A big gap between them is a signal:
    the relationship is curved, or a few extreme values drive it.
    """
    if len(columns) < 2:
        return {"columns": columns, "pearson": {}, "spearman": {}, "pairs": []}
    data = df[columns].astype(float)
    pearson = data.corr(method="pearson", min_periods=MIN_CORR_PAIRS)
    spearman = data.corr(method="spearman", min_periods=MIN_CORR_PAIRS)
    present = data.notna().astype(int)
    n_pairs = present.T @ present  # rows where both columns have a value

    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            r, rho = pearson.loc[a, b], spearman.loc[a, b]
            if pd.isna(r) and pd.isna(rho):
                continue
            n = int(n_pairs.loc[a, b])
            strength = np.nanmax([abs(r), abs(rho)])
            pair = {
                "a": a,
                "b": b,
                "pearson": _round(r),
                "spearman": _round(rho),
                "n": n,
                "strength": strength_label(strength),
                "direction": "positive" if np.nansum([r, rho]) > 0 else "negative",
                "_rank": strength,
            }
            notes = []
            if not pd.isna(r) and not pd.isna(rho) and abs(abs(r) - abs(rho)) >= 0.2:
                notes.append("Pearson and Spearman differ by 0.2 or more: the relationship may be "
                             "curved or driven by a few extreme values; check the scatter plot")
            if n < LOW_N_WARNING:
                notes.append(f"only {n} paired values: the estimate is unstable")
            if notes:
                pair["notes"] = notes
            pairs.append(pair)
    pairs.sort(key=lambda p: p["_rank"], reverse=True)
    for p in pairs:
        del p["_rank"]
    return {
        "columns": columns,
        "pearson": _matrix(pearson),
        "spearman": _matrix(spearman),
        "pairs": pairs,
    }


def strength_label(r: float) -> str:
    """Common rule of thumb for |r|. Context matters: 0.3 can be important in social data."""
    r = abs(r)
    if r < 0.1:
        return "negligible"
    if r < 0.3:
        return "weak"
    if r < 0.5:
        return "moderate"
    if r < 0.7:
        return "strong"
    return "very strong"


def categorical_summary(df: pd.DataFrame, kinds: dict[str, str], max_categories: int) -> dict[str, Any]:
    """Top-N categories per categorical/boolean column, the rest folded into "Other".

    Free-text columns are skipped: their values are mostly unique, so a "top
    values" chart says nothing.
    """
    out = {}
    for name in df.columns:
        key = str(name)
        if kinds.get(key) not in ("categorical", "boolean"):
            continue
        v = df[name].dropna()
        counts = v.astype(str).value_counts()
        top = counts.head(max_categories)
        other = int(counts.iloc[max_categories:].sum())
        n = int(len(v))
        out[key] = {
            "kind": kinds[key],
            "count": n,
            "missing": int(df[name].isna().sum()),
            "unique": int(counts.size),
            "categories": [{"value": str(k), "count": int(c), "pct": _pct(c, n)} for k, c in top.items()],
            "other": {"categories": int(max(counts.size - max_categories, 0)), "count": other, "pct": _pct(other, n)},
        }
    return out


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


# Matplotlib 3.10 replaced boxplot(vert=False) with orientation="horizontal".
_HORIZONTAL_BOX = (
    {"orientation": "horizontal"}
    if tuple(int(x) for x in re.findall(r"\d+", matplotlib.__version__)[:2]) >= (3, 10)
    else {"vert": False}
)


def _new_fig(width: float, height: float):
    return plt.subplots(figsize=(width, height))


def _titles(ax, title: str, subtitle: str) -> None:
    """Bold title with a one-line grey subtitle underneath, both left-aligned."""
    ax.set_title(title, pad=26)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=COLOR_INK_2, va="bottom")


def _save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)


def _number_formatter(axis_max: float) -> FuncFormatter:
    """Full numbers below 10,000 (1,500), compact above (25K, 1.2M)."""
    compact = abs(axis_max) >= 10_000

    def fmt(x, _pos):
        if compact:
            for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
                if abs(x) >= div:
                    return f"{x / div:,.3g}{suffix}"
        return f"{x:,.10g}" if abs(x) >= 1 or x == 0 else f"{x:.3g}"

    return FuncFormatter(fmt)


def _bar_height(n_bars: int) -> float:
    """Keep bars thin: with few bars each band is tall, so use less of it."""
    return 0.55 if n_bars >= 5 else 0.35


def _short(label: str, limit: int = 30) -> str:
    return label if len(label) <= limit else label[: limit - 1] + "…"


def chart_numeric(series: pd.Series, name: str, path: Path) -> dict[str, Any]:
    """Histogram with mean/median markers, and a box plot on the same x-axis."""
    v = series.dropna().astype(float)
    fig, (ax_h, ax_b) = plt.subplots(
        2, 1, figsize=(7.5, 4.6), sharex=True, gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08}
    )
    bins = np.histogram_bin_edges(v, bins="auto")
    if len(bins) > 51:  # cap at 50 bins so bars stay readable on large data
        bins = np.histogram_bin_edges(v, bins=50)
    ax_h.hist(v, bins=bins, color=COLOR_SERIES, edgecolor=COLOR_SURFACE, linewidth=0.6)
    ax_h.grid(axis="x", visible=False)
    ax_h.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax_h.yaxis.set_major_formatter(_number_formatter(ax_h.get_ylim()[1]))
    ax_h.set_ylabel("count")

    mean, median = v.mean(), v.median()
    ax_h.axvline(median, color=COLOR_INK, linewidth=1.3, label=f"median {median:,.4g}")
    ax_h.axvline(mean, color=COLOR_INK, linewidth=1.3, linestyle=(0, (4, 3)), label=f"mean {mean:,.4g}")
    ax_h.legend(loc="upper right", fontsize=9, handlelength=2.2, labelcolor=COLOR_INK_2)

    ax_b.boxplot(
        v, **_HORIZONTAL_BOX, widths=0.55, patch_artist=True, whis=1.5,
        boxprops={"facecolor": "#cde2fb", "edgecolor": COLOR_SERIES, "linewidth": 1.2},
        medianprops={"color": COLOR_INK, "linewidth": 1.5},
        whiskerprops={"color": COLOR_SERIES, "linewidth": 1.2},
        capprops={"color": COLOR_SERIES, "linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3.5, "markerfacecolor": COLOR_SERIES,
                    "markeredgecolor": COLOR_SURFACE, "markeredgewidth": 0.5, "alpha": 0.7},
    )
    ax_b.set_yticks([])
    ax_b.grid(axis="y", visible=False)
    ax_b.spines["left"].set_visible(False)
    ax_b.xaxis.set_major_formatter(_number_formatter(v.abs().max()))
    ax_b.set_xlabel(name)
    _titles(ax_h, f"Distribution of {name}", f"n = {len(v):,}  ·  box = middle 50%, dots = beyond 1.5 × IQR")
    _save(fig, path)
    return {"type": "histogram_boxplot", "column": name, "values": int(len(v))}


def needs_log_view(series: pd.Series) -> bool:
    """Is a log-scale histogram worth adding next to the normal one?

    Yes when the column is highly right-skewed, every value is positive (a log
    scale can't show 0 or negatives) and values span at least LOG_MIN_RATIO x.
    On such data the normal histogram piles nearly everything into a few bars.
    """
    v = series.dropna().astype(float)
    if len(v) < 3 or (v <= 0).any():
        return False
    skew = v.skew()
    return bool(pd.notna(skew) and skew > LOG_MIN_SKEW and v.max() / v.min() >= LOG_MIN_RATIO)


def chart_numeric_log(series: pd.Series, name: str, path: Path) -> dict[str, Any]:
    """Histogram on a log x-axis: equal-width bars on a ×10 scale."""
    v = series.dropna().astype(float)
    logs = np.log10(v)
    n_bins = min(len(np.histogram_bin_edges(logs, bins="auto")) - 1, 50)
    bins = np.logspace(logs.min(), logs.max(), n_bins + 1)
    fig, ax = _new_fig(7.5, 4.2)
    ax.hist(v, bins=bins, color=COLOR_SERIES, edgecolor=COLOR_SURFACE, linewidth=0.6)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(_number_formatter(v.max()))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_formatter(_number_formatter(ax.get_ylim()[1]))
    ax.set_ylabel("count")
    ax.set_xlabel(f"{name} (log scale)")
    median, mean = v.median(), v.mean()
    ax.axvline(median, color=COLOR_INK, linewidth=1.3, label=f"median {median:,.4g}")
    ax.axvline(mean, color=COLOR_INK, linewidth=1.3, linestyle=(0, (4, 3)), label=f"mean {mean:,.4g}")
    ax.legend(loc="upper right", fontsize=9, handlelength=2.2, labelcolor=COLOR_INK_2)
    _titles(ax, f"Distribution of {name}, log scale",
            f"n = {len(v):,}  ·  each ×10 step takes the same width, so the long tail spreads out")
    _save(fig, path)
    return {"type": "histogram_log", "column": name, "values": int(len(v))}


def chart_missing(missing: dict[str, Any], path: Path) -> dict[str, Any] | None:
    cols = [c for c in missing["columns"] if c["missing"] > 0]
    if not cols:
        return None
    cols.sort(key=lambda c: c["missing_pct"])
    fig, ax = _new_fig(7.5, max(2.2, 0.38 * len(cols) + 1.2))
    names = [_short(c["name"]) for c in cols]
    pcts = [c["missing_pct"] for c in cols]
    ax.barh(names, pcts, color=COLOR_SERIES, height=_bar_height(len(names)))
    for y, c in enumerate(cols):
        ax.text(c["missing_pct"], y, f"  {c['missing_pct']:.1f}% ({c['missing']:,})", va="center",
                fontsize=9, color=COLOR_INK_2)
    if max(pcts) > 80:
        ax.set_xlim(0, 125)  # room for the label next to a 100% bar...
        ax.set_xticks(range(0, 101, 20))  # ...but no ticks beyond 100%, which can't exist
    else:
        ax.set_xlim(0, min(100, max(pcts) * 1.35))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:g}%"))
    ax.grid(axis="y", visible=False)
    n_rows = missing["total_cells"] // max(len(missing["columns"]), 1)
    _titles(ax, "Missing values by column",
            f"share of {n_rows:,} rows that are empty; columns without gaps are not shown")
    _save(fig, path)
    return {"type": "missing_values", "columns": [c["name"] for c in cols]}


def chart_categories(name: str, summary: dict[str, Any], path: Path) -> dict[str, Any]:
    cats = list(summary["categories"])
    labels = [_short(c["value"]) for c in cats]
    counts = [c["count"] for c in cats]
    colors = [COLOR_SERIES] * len(cats)
    if summary["other"]["count"]:
        labels.append(f"Other ({summary['other']['categories']:,} more)")
        counts.append(summary["other"]["count"])
        colors.append(COLOR_OTHER)
    pcts = [_pct(c, summary["count"]) for c in counts]
    # Largest at the top, "Other" always last.
    labels, counts, colors, pcts = labels[::-1], counts[::-1], colors[::-1], pcts[::-1]
    fig, ax = _new_fig(7.5, max(2.2, 0.38 * len(labels) + 1.2))
    ax.barh(labels, counts, color=colors, height=_bar_height(len(labels)))
    for y, (c, p) in enumerate(zip(counts, pcts)):
        ax.text(c, y, f"  {c:,} ({p:.1f}%)", va="center", fontsize=9, color=COLOR_INK_2)
    ax.set_xlim(0, max(counts) * 1.3)
    ax.xaxis.set_major_formatter(_number_formatter(max(counts)))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.grid(axis="y", visible=False)
    shown = len(summary["categories"])
    _titles(ax, f"Most common values of {name}",
            f"{summary['unique']:,} distinct values, top {shown} shown  ·  {summary['count']:,} non-missing rows")
    _save(fig, path)
    return {"type": "top_categories", "column": name, "categories_shown": shown}


def chart_heatmap(corr: dict[str, Any], path: Path) -> dict[str, Any] | None:
    import seaborn as sns

    columns = corr["columns"]
    if len(columns) < 2:
        return None
    note = None
    if len(columns) > HEATMAP_MAX_COLUMNS:
        # Keep the columns that take part in the strongest pairs.
        keep: list[str] = []
        for p in corr["pairs"]:
            for c in (p["a"], p["b"]):
                if c not in keep and len(keep) < HEATMAP_MAX_COLUMNS:
                    keep.append(c)
        note = f"{len(columns)} numeric columns; the {len(keep)} in the strongest pairs are shown"
        columns = [c for c in columns if c in keep]
    full = pd.DataFrame(corr["pearson"]).loc[columns, columns].astype(float)
    # Show each pair once: the lower triangle without the diagonal (always 1),
    # which also drops the first row and last column (they would be empty).
    matrix = full.iloc[1:, :-1]
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    size = max(4.5, 0.55 * len(columns) + 2.5)
    fig, ax = _new_fig(size + 1.2, size)
    annotate = len(columns) <= HEATMAP_ANNOTATE_MAX
    sns.heatmap(
        matrix, mask=mask, cmap=DIVERGING, vmin=-1, vmax=1, center=0, square=True,
        annot=annotate, fmt=".2f", annot_kws={"fontsize": 8},
        linewidths=2, linecolor=COLOR_SURFACE,
        cbar_kws={"shrink": 0.7, "label": "Pearson r", "ticks": [-1, -0.5, 0, 0.5, 1]}, ax=ax,
    )
    ax.grid(False)
    ax.set_xticklabels([_short(t.get_text(), 20) for t in ax.get_xticklabels()], rotation=45, ha="right")
    ax.set_yticklabels([_short(t.get_text(), 20) for t in ax.get_yticklabels()], rotation=0)
    _titles(ax, "Correlation between numeric columns",
            "blue = move in opposite directions, red = move together, gray = no linear relationship")
    _save(fig, path)
    out = {"type": "correlation_heatmap", "columns": columns}
    if note:
        out["note"] = note
    return out


def chart_scatter(df: pd.DataFrame, pair: dict[str, Any], max_points: int, path: Path) -> dict[str, Any]:
    a, b = pair["a"], pair["b"]
    data = df[[a, b]].dropna().astype(float)
    total = len(data)
    sampled = total > max_points
    if sampled:
        data = data.sample(max_points, random_state=SEED)
    fig, ax = _new_fig(6.5, 5)
    alpha = 0.85 if len(data) < 300 else 0.45 if len(data) < 2000 else 0.25
    size = 22 if len(data) < 300 else 10
    ax.scatter(data[a], data[b], s=size, color=COLOR_SERIES, alpha=alpha, edgecolors=COLOR_SURFACE, linewidths=0.4)
    ax.xaxis.set_major_formatter(_number_formatter(data[a].abs().max()))
    ax.yaxis.set_major_formatter(_number_formatter(data[b].abs().max()))
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    sub = f"Pearson r = {pair['pearson']}, Spearman ρ = {pair['spearman']}  ·  n = {total:,}"
    if sampled:
        sub += f" (random sample of {max_points:,} shown)"
    _titles(ax, f"{b} vs {a}", sub)
    _save(fig, path)
    return {"type": "scatter", "columns": [a, b], "points_shown": int(len(data)), "sampled": sampled}


def chart_time(time: dict[str, Any], path: Path) -> dict[str, Any] | None:
    """Rows per period as a line; below it, the median of one numeric column per period.

    Partial first/last periods are drawn dashed with hollow markers (a
    half-covered week looks like a drop), empty runs flagged as gaps get a red
    band, spikes a ring. Two panels with one shared x-axis, not two y-axes.
    """
    t = time.get("trend")
    if not t:
        return None
    col, period, periods = time["column"], t["period"], t["periods"]
    x = pd.to_datetime([p["start"] for p in periods])
    counts = np.array([p["rows"] for p in periods], dtype=float)
    partial = {i for i, p in enumerate(periods) if p.get("partial")}
    value = t.get("value_column")
    if value:
        fig, (ax, ax_v) = plt.subplots(2, 1, figsize=(8.5, 5.8), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    else:
        fig, ax = _new_fig(8.5, 3.8)
        ax_v = None

    handles = [_time_line(ax, x, counts, partial, f"rows per {period}")]
    for g in t["gaps"]:
        i, j = g["start_index"], g["end_index"]
        left = x[i] - (x[i] - x[i - 1]) / 2
        right = x[j] + (x[j + 1] - x[j]) / 2
        for a in (ax, ax_v) if ax_v is not None else (ax,):
            a.axvspan(left, right, color=COLOR_PROBLEM, alpha=0.12, linewidth=0)
    if t["gaps"]:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=COLOR_PROBLEM, alpha=0.25,
                                     label="no rows: possible missing data"))
    spikes = [s["index"] for s in t["spikes"]]
    if spikes:
        handles.append(ax.scatter(x[spikes], counts[spikes], s=110, facecolors="none", edgecolors=COLOR_INK,
                                  linewidths=1.3, zorder=4, label=f"spike: ≥ 3 × the typical {period}"))
        for i in spikes:
            ax.annotate(f"{int(counts[i]):,}", (x[i], counts[i]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=COLOR_INK_2)
    if partial:
        handles.append(plt.Line2D([], [], color=COLOR_SERIES, linestyle=(0, (4, 3)), marker="o",
                                  markerfacecolor=COLOR_SURFACE, markersize=5, label=f"incomplete {period}"))
    ax.set_ylim(bottom=0, top=max(counts.max() * 1.15, 1))  # a count: keep zero in view
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_formatter(_number_formatter(counts.max()))
    ax.set_ylabel("rows")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, labelcolor=COLOR_INK_2, ncol=2)

    if ax_v is not None:
        med = np.array([np.nan if p.get("value_median") is None else p["value_median"] for p in periods])
        _time_line(ax_v, x, med, partial, f"median {value}")
        finite = med[np.isfinite(med)]
        if finite.size:
            ax_v.yaxis.set_major_formatter(_number_formatter(np.abs(finite).max()))
        ax_v.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_v.set_ylabel(f"median {_short(value, 20)}")
        ax_v.grid(axis="x", visible=False)
    bottom = ax_v if ax_v is not None else ax
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    bottom.xaxis.set_major_locator(locator)
    bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    sub = f"{t['rows_used']:,} rows, {t['first'][:10]} to {t['last'][:10]}"
    if t["rows_excluded_implausible"]:
        sub += f"  ·  {t['rows_excluded_implausible']:,} implausible date(s) left out"
    if value:
        sub += f"  ·  below: median {value} per {period}"
    _titles(ax, f"Rows per {period} by {col}", sub)
    _save(fig, path)
    return {"type": "time_trend", "column": col, "period": period, "value_column": value,
            "points": len(periods)}


def _time_line(ax, x, y: np.ndarray, partial: set[int], label: str):
    """Solid line through full periods, dashed segments to partial first/last points."""
    n = len(y)
    lo = 1 if 0 in partial and n > 2 else 0
    hi = n - 2 if (n - 1) in partial and n > 2 else n - 1
    line, = ax.plot(x[lo:hi + 1], y[lo:hi + 1], color=COLOR_SERIES, linewidth=1.8, marker="o",
                    markersize=3 if n > 40 else 4, label=label, zorder=3)
    if n > 2:
        for k, segment in ((0, slice(0, 2)), (n - 1, slice(n - 2, n))):
            if k in partial:
                ax.plot(x[segment], y[segment], color=COLOR_SERIES, linewidth=1.5, linestyle=(0, (4, 3)), zorder=3)
                ax.plot(x[k], y[k], marker="o", markersize=5, color=COLOR_SERIES,
                        markerfacecolor=COLOR_SURFACE, zorder=4)
    return line


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def default_output_dir(dataset: Path, base: Path = Path("eda_output")) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / f"{_slug(dataset.stem)}_{stamp}"


def prepare_output_dir(out_dir: Path) -> Path:
    """Create the folder. Refuse a non-empty one so nothing existing is overwritten."""
    if out_dir.exists():
        if not out_dir.is_dir():
            raise DatasetError(f"Output path exists and is a file: {out_dir}")
        if any(out_dir.iterdir()):
            raise DatasetError(f"Output folder is not empty: {out_dir}. Choose a new folder.")
    (out_dir / "charts").mkdir(parents=True, exist_ok=True)
    return out_dir


def run_eda(
    df: pd.DataFrame,
    out_dir: Path,
    key_columns: list[str] | None = None,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    max_scatter_points: int = DEFAULT_MAX_SCATTER_POINTS,
    max_charts: int = DEFAULT_MAX_CHARTS,
    date_column: str | None = None,
    period: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Analyse ``df``, write charts, summary.json and report.md into ``out_dir``, return the summary.

    ``now`` is the date that counts as "today" for the future-date check (default: now).
    """
    source_file = _source_file(df)
    sha_before = file_sha256(source_file) if source_file else None

    summary = describe_dataset(df, key_columns=key_columns)
    kinds = classify_columns(df)
    ids = likely_id_columns(df, kinds)
    # Columns given with --key are identifiers by definition, even when they
    # repeat (which is exactly what the key check is there to catch).
    ids += [k for k in key_columns or [] if kinds.get(k) == "numeric" and k not in ids]
    # Before the output folder exists: a wrong --date-column or --period leaves nothing behind.
    time = time_summary(df, kinds, summary["quality"]["type_issues"], date_column=date_column,
                        period=period, exclude_values=ids, now=now)
    out_dir = prepare_output_dir(Path(out_dir))
    charts_dir = out_dir / "charts"
    corr = correlations(df, correlation_columns(df, kinds, ids))
    cats = categorical_summary(df, kinds, max_categories)

    # Planned charts in priority order; the cap cuts from the end.
    plan: list[tuple[str, Any]] = [("missing", None)]
    if time and time["trend"]:
        plan.append(("time", None))
    plan.append(("heatmap", None))
    plan += [("scatter", p) for p in corr["pairs"][:TOP_SCATTER_PAIRS]]
    for c in df.columns:
        if kinds[str(c)] == "numeric" and str(c) not in ids and df[c].notna().any():
            plan.append(("numeric", c))
            if needs_log_view(df[c]):
                plan.append(("numeric_log", c))
    # Dates stored as text look categorical, but "most common dates" says nothing;
    # neither does "most common lists" for lists kept as JSON text.
    no_chart = {i["column"] for i in summary["quality"]["type_issues"]
                if i["issue"] in ("dates_as_text", "nested_values")}
    plan += [("categories", c) for c in cats if c not in no_chart]

    charts: list[dict[str, Any]] = []
    skipped: list[str] = []
    with plt.rc_context(STYLE):
        for kind, item in plan:
            if len(charts) >= max_charts:
                skipped.append(_describe_plan_item(kind, item))
                continue
            prefix = f"{len(charts) + 1:02d}"
            if kind == "missing":
                path = charts_dir / f"{prefix}_missing_values.png"
                info = chart_missing(summary["quality"]["missing"], path)
            elif kind == "heatmap":
                path = charts_dir / f"{prefix}_correlation_heatmap.png"
                info = chart_heatmap(corr, path)
            elif kind == "time":
                path = charts_dir / f"{prefix}_time_{_slug(time['column'])}_by_{time['trend']['period']}.png"
                info = chart_time(time, path)
            elif kind == "scatter":
                path = charts_dir / f"{prefix}_scatter_{_slug(item['a'])}__{_slug(item['b'])}.png"
                info = chart_scatter(df, item, max_scatter_points, path)
            elif kind == "numeric":
                path = charts_dir / f"{prefix}_distribution_{_slug(item)}.png"
                info = chart_numeric(df[item], str(item), path)
            elif kind == "numeric_log":
                path = charts_dir / f"{prefix}_distribution_{_slug(item)}_log.png"
                info = chart_numeric_log(df[item], str(item), path)
            else:
                path = charts_dir / f"{prefix}_categories_{_slug(item)}.png"
                info = chart_categories(item, cats[item], path)
            if info is None:  # nothing to draw (e.g. no missing values)
                continue
            info["file"] = f"charts/{path.name}"
            charts.append(info)

    summary["analysis"] = {
        "excluded_id_columns": ids,
        "correlations": corr,
        "categorical": cats,
        "time": time,
    }
    summary["charts"] = charts
    summary["charts_skipped"] = skipped
    summary["settings"] = {
        "max_categories": max_categories,
        "max_scatter_points": max_scatter_points,
        "max_charts": max_charts,
        "key_columns": key_columns,
        "date_column": date_column,
        "period": period,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary = to_jsonable(summary)
    summary["findings"] = build_findings(summary)
    if sha_before:
        # Proof of the read-only rule: the file's fingerprint before and after.
        summary["source"]["sha256"] = sha_before
        summary["source"]["unchanged_after_run"] = file_sha256(source_file) == sha_before
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, out_dir)
    return summary


def file_sha256(path: Path) -> str:
    """Fingerprint of a file, read in 1 MB chunks so large files don't fill memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_file(df: pd.DataFrame) -> Path | None:
    """The file ``df`` was loaded from (set by load_dataset), if it still exists."""
    path = df.attrs.get("source", {}).get("path")
    return Path(path) if path and Path(path).is_file() else None


def _describe_plan_item(kind: str, item: Any) -> str:
    if kind == "scatter":
        return f"scatter {item['a']} vs {item['b']}"
    return f"{kind} {item}" if item is not None else kind


def _slug(text: Any) -> str:
    s = re.sub(r"[^\w-]+", "_", str(text), flags=re.UNICODE).strip("_").lower()
    return s[:40] or "column"


def _round(x: float) -> float | None:
    return None if pd.isna(x) else round(float(x), 3)


def _pct(part: float, whole: float) -> float:
    return round(100 * part / whole, 2) if whole else 0.0


def _matrix(m: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    return {str(c): {str(r): _round(m.loc[r, c]) for r in m.index} for c in m.columns}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run exploratory data analysis and save charts (read-only).")
    p.add_argument("path", help=DATASET_HELP)
    p.add_argument("--sheet", help="Excel sheet name (default: first sheet)")
    p.add_argument("--json-path", help=JSON_PATH_HELP)
    p.add_argument("--encoding", help=ENCODING_HELP)
    p.add_argument("--delimiter", help="CSV delimiter (default: detected automatically)")
    p.add_argument("--key", help="comma-separated key columns for the duplicate-key check")
    p.add_argument("--output", help="output folder (default: eda_output/<dataset>_<timestamp>/)")
    p.add_argument("--max-categories", type=int, default=DEFAULT_MAX_CATEGORIES,
                   help=f"top-N categories per column; the rest become 'Other' (default {DEFAULT_MAX_CATEGORIES})")
    p.add_argument("--max-scatter-points", type=int, default=DEFAULT_MAX_SCATTER_POINTS,
                   help=f"random sample size for scatter plots (default {DEFAULT_MAX_SCATTER_POINTS})")
    p.add_argument("--max-charts", type=int, default=DEFAULT_MAX_CHARTS,
                   help=f"maximum number of charts (default {DEFAULT_MAX_CHARTS})")
    p.add_argument("--date-column", help="date column for the time trend (default: the one with the most dates)")
    p.add_argument("--period", choices=PERIOD_NAMES,
                   help="period for the time trend (default: chosen so the chart has about 12-120 points)")
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    for opt in ("max_categories", "max_scatter_points"):
        if getattr(args, opt) < 1:
            print(f"Error: --{opt.replace('_', '-')} must be at least 1", file=sys.stderr)
            return 2
    if args.max_charts < 0:
        print("Error: --max-charts must be 0 or more", file=sys.stderr)
        return 2

    delimiter = args.delimiter.encode().decode("unicode_escape") if args.delimiter else None
    dataset = Path(args.path)
    try:
        df = load_dataset(dataset, sheet=args.sheet, encoding=args.encoding, delimiter=delimiter,
                          json_path=args.json_path)
        out_dir = Path(args.output) if args.output else default_output_dir(dataset)
        keys = [k.strip() for k in args.key.split(",")] if args.key else None
        summary = run_eda(df, out_dir, key_columns=keys, max_categories=args.max_categories,
                          max_scatter_points=args.max_scatter_points, max_charts=args.max_charts,
                          date_column=args.date_column, period=args.period)
    except DatasetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"EDA written to {out_dir}")
    print("  report.md")
    print(f"  summary.json  ({summary['overview']['rows']:,} rows x {summary['overview']['columns']} columns)")
    for c in summary["charts"]:
        print(f"  {c['file']}")
    if summary["charts_skipped"]:
        print(f"  {len(summary['charts_skipped'])} chart(s) skipped by --max-charts {args.max_charts}")
    time = summary["analysis"]["time"]
    if time and time["trend"]:
        t = time["trend"]
        print(f"Time trend: {time['column']} by {t['period']}, {t['first'][:10]} to {t['last'][:10]} "
              f"({len(t['periods'])} points)")
    elif time:
        print(f"Time trend: none for {time['column']} ({time['note']})")
    pairs = summary["analysis"]["correlations"]["pairs"][:3]
    if pairs:
        print("Strongest correlations:")
        for p in pairs:
            print(f"  {p['a']} ~ {p['b']}: Pearson {p['pearson']}, Spearman {p['spearman']} ({p['strength']}, n={p['n']})")
    findings = summary["findings"]
    if findings:
        counts = ", ".join(f"{n} {sev}" for sev in ("high", "medium", "low")
                           if (n := sum(f["severity"] == sev for f in findings)))
        print(f"Findings: {counts}")
        for f in findings[:5]:
            print(f"  [{f['severity']}] {f['title'].replace('`', '')}")
    if summary["source"].get("unchanged_after_run") is False:
        print("Error: the input file changed during the run. Check what else is writing to it.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
