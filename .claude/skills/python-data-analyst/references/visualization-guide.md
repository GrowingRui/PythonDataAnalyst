# Visualization guide

Read this when you need a chart that `generate_eda.py` doesn't produce, or
when you're choosing which chart answers a question. The goal of every chart
here is one clear answer to one question, readable without explanation.

**Contents**
1. [Question → chart](#1-question--chart)
2. [Readability rules](#2-readability-rules)
3. [Colors](#3-colors)
4. [Code: charts in the skill's style](#4-code-charts-in-the-skills-style)
5. [Misleading charts](#5-misleading-charts)
6. [Checklist before showing a chart](#6-checklist-before-showing-a-chart)

---

## 1. Question → chart

| Question | Chart | Avoid |
|---|---|---|
| How is one numeric column distributed? | Histogram + box plot (the script does this) | A bar per raw value |
| Is it skewed over orders of magnitude? | Histogram on a log axis (the script adds it) | Cutting off the tail |
| How does a numeric column differ between groups? | Box plots per group, sorted by median | Bars of the mean alone (hide spread and n) |
| Which group is largest? | Horizontal bar chart, sorted | Pie chart with more than 3–4 slices |
| How does something change over time? | Line chart, time on the x-axis | Bars for long series; unsorted dates |
| How do two numeric columns relate? | Scatter plot (sample if > 5,000 points) | A line through unsorted points |
| How do many numeric columns relate? | Correlation heatmap (the script does this) | Scatter matrix with > 6 columns |
| How is each group made up? | 100% stacked horizontal bars | Stacks with more than ~6 segments |
| Two measures with different units over time | Two charts stacked, same x-axis | Dual y-axes |
| Where are the missing values? | Bar of missing % per column (the script does this) | |
| One number that matters | Say it in text: "Median order: €90" | A chart with one bar |

Too many categories (more than ~8 lines or ~15 bars): show the top N and fold
the rest into "Other", or use small multiples (one small chart per group, same
axes).

## 2. Readability rules

- **A title that names the subject, a subtitle that says how to read it.**
  "Revenue by region" + "median and middle 50% of orders; n per group in the label".
- **Bars start at zero.** A bar's length *is* its value; a cut axis exaggerates
  differences. Lines and dots may use a tighter range.
- **Sort categories** by value (largest at the top), unless they have a natural
  order (months, ratings 1–5, age bands).
- **Horizontal bars for labels.** Long category names stay readable without
  rotation.
- **Label directly** where possible (value at the end of the bar, name at the
  end of the line) instead of making the reader match colors to a legend. Label
  selectively: not every point needs a number.
- **Show n.** Every group statistic gets its group size; a group of 4 looks
  just as solid as a group of 4,000 otherwise.
- **Readable numbers.** Thousands separators, units in the axis label, compact
  form for large values (25K, 1.2M), no more decimals than the data supports.
- **Recessive grid and axes.** Light gray, no top/right frame, so the data is
  the darkest thing on the chart.
- **Overplotting.** With thousands of points, use transparency, smaller dots, a
  random sample (say so in the subtitle), or a 2D histogram.
- **Say when the scale is logarithmic,** in the axis label and the subtitle.

## 3. Colors

The skill uses one palette everywhere (decision D7). Colors carry a *job*:

| Job | Colors | Rule |
|---|---|---|
| One series | blue `#2a78d6` | Every single-series chart; no rainbow bars |
| "Other" / context | gray `#b4b2a9` | Folded categories, reference data |
| Several categories (identity) | the 8 slots below, **in this order** | Never cycled: a 9th series goes into "Other" or small multiples |
| Polarity (correlation, change) | blue ↔ gray ↔ red | Gray means zero; same scale on both sides |
| Ordered groups (ratings, tiers) | one hue in steps: `#86b6ef` `#5598e7` `#2a78d6` `#1c5cab` `#104281` | Order is the message; categorical hues would hide it |
| Magnitude (heat) | one hue, light → dark | Never a rainbow |

Categorical order (validated for color-vision deficiencies on the light
background):

| Slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Hue | blue | orange | aqua | yellow | magenta | green | violet | red |
| Hex | `#2a78d6` | `#eb6834` | `#1baf7a` | `#eda100` | `#e87ba4` | `#008300` | `#6250d6` | `#e34948` |

Aqua, yellow and magenta are lighter than 3:1 against the background, so
charts using them **must** have a legend or direct labels, never color alone.

Colors follow the **entity**, not the rank: if "North" is blue in one chart,
it's blue in every chart of the report. Text (labels, values, legends) stays in
ink colors; the colored mark beside it carries the identity.

## 4. Code: charts in the skill's style

Reuse the style from `generate_eda.py` so custom charts match the generated
ones. `scripts_dir` is the skill's `scripts/` folder; `out_dir` is the analysis
output folder.

```python
import sys

import matplotlib

matplotlib.use("Agg")  # no window: save to files
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(scripts_dir))
from generate_eda import COLOR_INK, COLOR_INK_2, COLOR_OTHER, COLOR_SERIES, COLOR_SURFACE, STYLE

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

df = pd.read_csv(path)
df["order_date"] = pd.to_datetime(df["order_date"], format="%d/%m/%Y", errors="coerce")
df["region"] = df["region"].str.strip().str.title()  # only after the user approved this cleaning step


def titles(ax, title: str, subtitle: str) -> None:
    """Bold title, one grey line underneath: the skill's chart heading."""
    ax.set_title(title, pad=26)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=COLOR_INK_2, va="bottom")
```

### Compare a numeric column across groups: box plots

```python
groups = df.dropna(subset=["revenue"]).groupby("region")["revenue"]
order = groups.median().sort_values().index  # smallest at the bottom, largest at the top
data = [groups.get_group(g).to_numpy() for g in order]
labels = [f"{g}  (n={len(groups.get_group(g)):,})" for g in order]

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(order) + 1.6))
    ax.boxplot(
        data, tick_labels=labels, orientation="horizontal", widths=0.5, patch_artist=True,
        boxprops={"facecolor": "#cde2fb", "edgecolor": COLOR_SERIES},
        medianprops={"color": COLOR_INK, "linewidth": 1.5},
        whiskerprops={"color": COLOR_SERIES}, capprops={"color": COLOR_SERIES},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": COLOR_SERIES, "markeredgecolor": COLOR_SURFACE},
    )
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("revenue per order")
    titles(ax, "Revenue per order by region", "line = median, box = middle 50% of orders, dots = beyond 1.5 × IQR")
    fig.savefig(out_dir / "revenue_by_region.png")
    plt.close(fig)
```

On Matplotlib older than 3.10 use `vert=False` instead of
`orientation="horizontal"`, and before 3.9 `labels=` instead of `tick_labels=`.

### Change over time: a line chart

```python
monthly = df.set_index("order_date").resample("MS")["revenue"].agg(["sum", "count"])

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly.index, monthly["sum"], color=COLOR_SERIES, linewidth=2, marker="o", markersize=4)
    ax.set_ylim(bottom=0)  # a total: keep zero in view so changes aren't exaggerated
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("revenue")
    titles(ax, "Revenue per month", f"sum of {int(monthly['count'].sum()):,} orders; the last month may be incomplete")
    fig.autofmt_xdate()
    fig.savefig(out_dir / "revenue_per_month.png")
    plt.close(fig)
```

Check the first and last period: partial months look like sudden drops.

### Which group is largest: a sorted bar chart

```python
totals = df.groupby("region")["revenue"].sum().sort_values()  # ascending: largest ends up at the top

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(totals) + 1.4))
    ax.barh(totals.index, totals.to_numpy(), color=COLOR_SERIES, height=0.55)
    for y, value in enumerate(totals.to_numpy()):
        ax.text(value, y, f"  {value:,.0f}", va="center", fontsize=9, color=COLOR_INK_2)  # direct labels
    ax.set_xlim(0, totals.max() * 1.2)
    ax.grid(axis="y", visible=False)
    titles(ax, "Total revenue by region", f"{len(df):,} orders")
    fig.savefig(out_dir / "revenue_total_by_region.png")
    plt.close(fig)
```

### How each group is made up: 100% stacked bars

Ratings 1–5 are **ordered**, so they get one hue from light to dark (`BLUES`),
not the categorical slots; the placeholder `-` becomes "not rated" in the
context gray, placed last. For unordered groups (product types, channels) use
`CATEGORICAL` in its fixed order instead.

```python
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordered steps, lightest still visible

rating = df["rating"].replace({"-": "not rated"})
shares = pd.crosstab(df["region"], rating, normalize="index").mul(100)
rated = sorted(c for c in shares.columns if c != "not rated")  # natural order 1..5
shares = shares[rated + [c for c in shares.columns if c == "not rated"]]
if len(rated) > len(BLUES):
    raise ValueError("more ordered steps than colors: group them into bands first")
colors = BLUES[: len(rated)] + [COLOR_OTHER] * (shares.shape[1] - len(rated))

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(shares) + 1.8))
    left = pd.Series(0.0, index=shares.index)
    for color, segment in zip(colors, shares.columns):
        ax.barh(shares.index, shares[segment], left=left, color=color, height=0.6,
                edgecolor=COLOR_SURFACE, linewidth=2, label=str(segment))  # 2px gaps between segments
        left += shares[segment]
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda x, _pos: f"{x:g}%")
    ax.grid(axis="y", visible=False)
    ax.legend(title="rating", ncols=shares.shape[1], loc="upper left", bbox_to_anchor=(0, -0.12),
              fontsize=9, title_fontsize=9, alignment="left")
    titles(ax, "Ratings by region", "share of orders per region; each bar adds up to 100%")
    fig.savefig(out_dir / "ratings_by_region.png")
    plt.close(fig)
```

### Two measures over time: small multiples, not two y-axes

```python
from matplotlib.ticker import MaxNLocator

with plt.rc_context(STYLE):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    top.plot(monthly.index, monthly["sum"], color=COLOR_SERIES, linewidth=2)
    top.set_ylabel("revenue")
    bottom.plot(monthly.index, monthly["count"], color=COLOR_SERIES, linewidth=2)
    bottom.set_ylabel("orders")
    bottom.yaxis.set_major_locator(MaxNLocator(integer=True))  # counts: no 12.5 orders
    for ax in (top, bottom):
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
    titles(top, "Revenue and orders per month", "two panels, one shared time axis, each with its own scale")
    fig.autofmt_xdate()
    fig.savefig(out_dir / "revenue_and_orders.png")
    plt.close(fig)
```

Always `plt.close(fig)` after saving; scripts that draw many charts otherwise
keep every figure in memory.

## 5. Misleading charts

Each of these can make a true number tell a false story.

| Pitfall | What goes wrong | Do instead |
|---|---|---|
| Truncated bar axis | 98 vs 100 looks like a 2× difference | Start bars at zero, or use dots |
| Dual y-axes | The crossing point and slopes depend on arbitrary scaling | Two panels, or index both to 100 |
| Mean bars without spread | Hides that groups overlap almost completely | Box plots, or show the IQR |
| No group sizes | A 4-row group looks as reliable as a 4,000-row group | n in the label |
| Cherry-picked time window | A dip or rise appears or disappears | Show the full range; say why if you zoom |
| Incomplete last period | The current month looks like a collapse | Mark or drop the partial period |
| Area or 3D scaling | Doubling a circle's radius quadruples its area | Length (bars), not area or volume |
| Pie charts with many slices | Angles are hard to compare | Sorted bars |
| Rainbow colors | Suggest order or importance that isn't there; fail for color-blind readers | One color, or the categorical order above |
| Trend line on a scatter plot | Reads as "x drives y" | Report the correlation as an association; mention possible third variables |
| Aggregation hides the pattern | Overall trend reverses inside every group (Simpson's paradox) | Check the main split (region, segment) before concluding |
| Log axis not labeled | Readers assume equal steps | "(log scale)" in the axis label and subtitle |
| Unsorted categories | The reader has to search for the largest | Sort by value or natural order |

## 6. Checklist before showing a chart

1. Can you state its message in one sentence? Put that sentence in the report.
2. Does the title name the subject and the subtitle explain how to read it?
3. Do bars start at zero? Is any log scale labeled?
4. Are group sizes visible, and sampling mentioned?
5. Is color doing one job, from the palette, consistent with the other charts?
6. Are labels readable (no overlaps, no rotated text you can avoid)?
7. Open the PNG and look at it. Code that runs can still draw a bad chart.
