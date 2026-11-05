"""Target comparison: how one chosen column (the target) differs across the other columns.

generate_eda.py calls target_summary() when it runs with --target and draws the
charts, so you normally don't run this module yourself. It only computes: no
charts, no files, and no changes to the data. Descriptive only: no
significance tests (a p-value is easily read as proof, and the tests would
need SciPy).

A **numeric target** (revenue) is compared
  - across the groups of each categorical column: count, median, quartiles
    and mean per group, and one strength number per column, and
  - with each numeric column: Pearson and Spearman correlation.
A **categorical target** (churned yes/no, at most MAX_CLASSES classes) gets
  - the rate of each class per group of each categorical column, and
  - the median of each numeric column per class.

Every comparison gets a strength between 0 and 1 on the same scale as a
correlation (0 = no difference, 1 = complete separation):
  - numeric vs numeric: the larger of |Pearson r| and |Spearman rho|
  - numeric vs groups: eta, the correlation ratio, computed on ranks (so
    skew and outliers don't dominate) and corrected for the number of groups
    the way adjusted R^2 is (many small groups always differ a little by chance)
  - groups vs groups: Cramer's V with the bias correction of Bergsma (2013)

A column that predicts the target almost perfectly is flagged as possible
*leakage*: derived from the target (revenue and revenue_eur, an order-size band
computed from revenue) or recorded after the outcome (a cancellation reason).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from inspect_data import DatasetError

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

MAX_CLASSES = 8  # one color per class from the 8-slot palette; more is no longer "a few outcomes"
MIN_GROUP_ROWS = 30  # smaller groups are shown but marked "too small to compare"
LEAK_THRESHOLD = 0.95  # strength (or chance-corrected separation) that counts as "almost perfect"
MIN_ROWS = 3
MISSING_LABEL = "(missing)"
OTHER_LABEL = "Other"
LOG_MIN_SKEW = 1.0  # same rule as the log-scale histogram in generate_eda
LOG_MIN_RATIO = 20
POSITIVE_LABELS = ("true", "1", "1.0", "yes", "y")
TARGET_ISSUES = {  # type problems that make a column unusable as a target until it's converted
    "numbers_as_text": "holds numbers stored as text",
    "dates_as_text": "holds dates stored as text",
    "nested_values": "holds lists or nested objects",
    "mixed_types": "mixes different value types",
}
ID_NAME = re.compile(r"(^|[_\s-])id$", re.IGNORECASE)


# --------------------------------------------------------------------------
# Strength measures
# --------------------------------------------------------------------------


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


def rank_eta(values: pd.Series, groups: pd.Series) -> float | None:
    """Correlation ratio on ranks, corrected for the number of groups (0 to 1).

    eta^2 = between-group sum of squares / total sum of squares of the ranks:
    the share of the variation in the ranks that the groups explain. With k
    groups, chance alone gives about (k - 1) / (n - 1), so it's corrected like
    adjusted R^2, then square-rooted to put it on the same scale as |r|.
    """
    ok = values.notna() & groups.notna()
    v, g = values[ok].astype(float), groups[ok]
    n, k = len(v), g.nunique()
    if n < MIN_ROWS or k < 2 or n <= k:
        return None
    r = v.rank()
    ss_total = float(((r - r.mean()) ** 2).sum())
    if ss_total == 0:
        return None
    stats = r.groupby(g, sort=False).agg(["mean", "size"])
    ss_between = float((stats["size"] * (stats["mean"] - r.mean()) ** 2).sum())
    eta2 = ss_between / ss_total
    adjusted = 1 - (1 - eta2) * (n - 1) / (n - k)
    return float(np.sqrt(min(max(adjusted, 0.0), 1.0)))


def cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """Bias-corrected Cramér's V between two categorical columns (0 to 1)."""
    ok = a.notna() & b.notna()
    table = pd.crosstab(a[ok], b[ok]).to_numpy(dtype=float)
    n = table.sum()
    r, k = table.shape
    if n < MIN_ROWS or r < 2 or k < 2:
        return None
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(min(phi2 / denom, 1.0)))


def separation(values: pd.Series, groups: pd.Series) -> float | None:
    """How cleanly the groups occupy separate ranges of the values, corrected for chance.

    Sort the groups by their median and give each one a block of ranks as wide
    as its size. The share of rows whose rank falls inside their own group's
    block is 1 when the groups don't overlap at all (an order-size band
    computed from revenue) and about sum(p^2) by chance. Corrected like
    Cohen's kappa: (share - chance) / (1 - chance).
    """
    ok = values.notna() & groups.notna()
    v, g = values[ok].astype(float), groups[ok]
    n = len(v)
    if n < MIN_ROWS or g.nunique() < 2:
        return None
    ranks = v.rank()
    order = v.groupby(g, sort=False).median().sort_values(kind="stable").index
    sizes = g.value_counts()
    start, inside = 0.0, 0
    for label in order:
        size = int(sizes[label])
        r = ranks[g == label]
        inside += int(((r > start) & (r <= start + size + 1e-9)).sum())
        start += size
    share = inside / n
    chance = float(((sizes / n) ** 2).sum())
    if chance >= 1:
        return None
    return float(max((share - chance) / (1 - chance), 0.0))


def goodman_kruskal_lambda(groups: pd.Series, target: pd.Series) -> float | None:
    """How much knowing the group improves guessing the class (0 = not at all, 1 = perfectly)."""
    ok = groups.notna() & target.notna()
    table = pd.crosstab(groups[ok], target[ok])
    n = int(table.to_numpy().sum())
    best_overall = int(table.sum(axis=0).max()) if n else 0
    if n < MIN_ROWS or table.shape[1] < 2 or n == best_overall:
        return None
    best_per_group = int(table.max(axis=1).sum())
    return float((best_per_group - best_overall) / (n - best_overall))


# --------------------------------------------------------------------------
# Choosing columns
# --------------------------------------------------------------------------


def target_kind(df: pd.DataFrame, target: str, kinds: dict[str, str], type_issues: list[dict[str, Any]],
                ids: list[str]) -> str:
    """'numeric' or 'categorical'; raises DatasetError with a reason when the column can't be a target."""
    names = [str(c) for c in df.columns]
    if target not in names:
        close = [n for n in names if n.lower() == target.lower()]
        hint = f" Did you mean {close[0]!r}?" if close else f" Columns: {', '.join(names[:30])}."
        raise DatasetError(f"--target column {target!r} not found.{hint}")
    kind = kinds.get(target)
    s = df[target].dropna()
    for issue in type_issues:
        if issue["column"] == target and issue["issue"] in TARGET_ISSUES:
            raise DatasetError(
                f"--target column {target!r} {TARGET_ISSUES[issue['issue']]}. Convert it in a cleaned copy "
                "first (see references/data-cleaning.md), then run the EDA on that copy.")
    if kind == "empty" or s.empty:
        raise DatasetError(f"--target column {target!r} is empty.")
    if target in ids:
        raise DatasetError(f"--target column {target!r} looks like a row ID; an ID can't be a target.")
    if kind == "numeric":
        n_unique = s.nunique()
        if n_unique < 2:
            raise DatasetError(f"--target column {target!r} has the same value in every row.")
        return "categorical" if n_unique == 2 else "numeric"  # 0/1 codes are two classes
    if kind in ("boolean", "categorical"):
        n_unique = s.astype(str).nunique()
        if n_unique < 2:
            raise DatasetError(f"--target column {target!r} has the same value in every row.")
        if n_unique > MAX_CLASSES:
            raise DatasetError(
                f"--target column {target!r} has {n_unique} different values; a categorical target can have "
                f"at most {MAX_CLASSES} (e.g. yes/no, or a few outcomes). Group the values first, or choose "
                "another column.")
        return "categorical"
    raise DatasetError(f"--target column {target!r} holds {kind} values; choose a numeric or categorical column "
                       f"(at most {MAX_CLASSES} distinct values).")


def comparison_columns(df: pd.DataFrame, target: str, kinds: dict[str, str],
                       type_issues: list[dict[str, Any]], ids: list[str]) -> tuple[list[str], list[str], list[dict]]:
    """(group columns, numeric columns, skipped columns with the reason)."""
    issues = {i["column"]: i["issue"] for i in type_issues if i["issue"] in TARGET_ISSUES}
    groups, numeric, skipped = [], [], []
    for name in df.columns:
        col = str(name)
        if col == target:
            continue
        kind = kinds.get(col)
        reason = None
        if col in issues:
            reason = TARGET_ISSUES[issues[col]]
        elif col in ids or (kind in ("categorical", "text") and ID_NAME.search(col)):
            reason = "an ID column"
        elif kind == "empty":
            reason = "empty"
        elif kind == "text":
            reason = "free text (mostly unique values)"
        elif kind == "datetime":
            reason = "dates (see the time trend)"
        elif df[name].nunique() < 2:
            reason = "the same value in every row"
        if reason:
            skipped.append({"column": col, "reason": reason})
        elif kind == "numeric":
            numeric.append(col)
        else:
            groups.append(col)
    return groups, numeric, skipped


def fold_groups(s: pd.Series, max_groups: int) -> pd.Series:
    """Values as text: the max_groups most common kept, the rest "Other", missing values "(missing)"."""
    text = s.astype(object).where(s.isna(), s.astype(str))
    counts = text.dropna().value_counts()
    top = set(counts.index[:max_groups])
    out = text.where(text.isin(top) | text.isna(), OTHER_LABEL)
    return out.fillna(MISSING_LABEL).astype(str)


def focus_class(classes: list[str], counts: pd.Series) -> str | None:
    """For two classes, the one whose rate is shown: the 'yes'-like one, else the rarer one."""
    if len(classes) != 2:
        return None
    for c in classes:
        if c.strip().lower() in POSITIVE_LABELS:
            return c
    return str(counts.sort_values(kind="stable").index[0])


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def target_summary(
    df: pd.DataFrame,
    target: str,
    kinds: dict[str, str],
    type_issues: list[dict[str, Any]],
    ids: list[str],
    max_groups: int = 10,
) -> dict[str, Any]:
    """Everything the report and charts need to compare ``target`` across the other columns."""
    kind = target_kind(df, target, kinds, type_issues, ids)
    group_cols, numeric_cols, skipped = comparison_columns(df, target, kinds, type_issues, ids)
    has_target = df[target].notna()
    data = df[has_target]
    out: dict[str, Any] = {
        "column": target,
        "kind": kind,
        "rows_total": int(len(df)),
        "rows_used": int(has_target.sum()),
        "rows_missing_target": int((~has_target).sum()),
        "min_group_rows": MIN_GROUP_ROWS,
        "max_groups": max_groups,
        "leak_threshold": LEAK_THRESHOLD,
        "skipped": skipped,
    }
    if kind == "numeric":
        y = data[target].astype(float)
        out["overall"] = _numeric_stats(y)
        out["log_scale"] = bool((y > 0).all() and y.skew() > LOG_MIN_SKEW and y.max() / y.min() >= LOG_MIN_RATIO)
        out["by_group"] = [_numeric_by_group(y, fold_groups(data[c], max_groups), c) for c in group_cols]
        out["by_numeric"] = [_numeric_vs_numeric(y, data[c].astype(float), c) for c in numeric_cols]
    else:
        y = data[target].astype(str)
        counts = y.value_counts()
        classes = [str(c) for c in counts.index]
        out["classes"] = [{"value": c, "count": int(counts[c]), "pct": _pct(counts[c], len(y))} for c in classes]
        out["focus_class"] = focus_class(classes, counts)
        out["by_group"] = [_classes_by_group(y, fold_groups(data[c], max_groups), c, classes, out["focus_class"])
                           for c in group_cols]
        out["by_numeric"] = [_numeric_by_class(data[c].astype(float), y, c, classes) for c in numeric_cols]
    for key in ("by_group", "by_numeric"):
        out[key].sort(key=lambda e: -1.0 if e["strength"] is None else e["strength"], reverse=True)
    out["leakage"] = [{"column": e["column"], **e["leakage"]} for key in ("by_group", "by_numeric")
                      for e in out[key] if e.get("leakage")]
    return out


def _numeric_by_group(y: pd.Series, g: pd.Series, col: str) -> dict[str, Any]:
    groups = []
    for label, values in y.groupby(g, sort=False):
        stats = _numeric_stats(values)
        groups.append({"value": str(label), **stats, "small": stats["count"] < MIN_GROUP_ROWS})
    groups.sort(key=lambda x: (x["median"] is None, -(x["median"] or 0)))
    strength = rank_eta(y, g)
    entry = {
        "column": col,
        "measure": "rank_eta",
        "strength": _round(strength),
        "strength_label": None if strength is None else strength_label(strength),
        "groups": groups,
        "other_values": _other_count(g),
    }
    comparable = [x for x in groups if not x["small"] and x["value"] not in (OTHER_LABEL, MISSING_LABEL)]
    if len(comparable) >= 2:
        hi, lo = comparable[0], comparable[-1]
        entry["highest"] = {"value": hi["value"], "median": hi["median"], "count": hi["count"]}
        entry["lowest"] = {"value": lo["value"], "median": lo["median"], "count": lo["count"]}
        if lo["median"] and hi["median"] and lo["median"] > 0:
            entry["median_ratio"] = _round(hi["median"] / lo["median"])
    sep = separation(y, g)
    if sep is not None and sep >= LEAK_THRESHOLD:
        entry["leakage"] = {"measure": "separation", "value": _round(sep),
                            "detail": "its groups cover separate ranges of the target with almost no overlap"}
    return entry


def _numeric_vs_numeric(y: pd.Series, x: pd.Series, col: str) -> dict[str, Any]:
    both = y.notna() & x.notna()
    n = int(both.sum())
    pearson = spearman = None
    if n >= MIN_ROWS and x[both].nunique() > 1:
        pearson = _round(y[both].corr(x[both]))
        spearman = _round(y[both].rank().corr(x[both].rank()))
    present = [abs(v) for v in (pearson, spearman) if v is not None]
    strength = max(present) if present else None
    entry = {
        "column": col,
        "measure": "correlation",
        "pearson": pearson,
        "spearman": spearman,
        "n": n,
        "strength": strength,
        "strength_label": None if strength is None else strength_label(strength),
        "direction": None if strength is None else ("positive" if (spearman or 0) + (pearson or 0) > 0 else "negative"),
    }
    if strength is not None and strength >= LEAK_THRESHOLD:
        entry["leakage"] = {"measure": "correlation", "value": strength,
                            "detail": "it moves almost exactly with the target"}
    return entry


def _classes_by_group(y: pd.Series, g: pd.Series, col: str, classes: list[str], focus: str | None) -> dict[str, Any]:
    table = pd.crosstab(g, y).reindex(columns=classes, fill_value=0)
    groups = []
    for label, row in table.iterrows():
        n = int(row.sum())
        groups.append({"value": str(label), "count": n, "small": n < MIN_GROUP_ROWS,
                       "rates": {c: _pct(row[c], n) for c in classes}})
    if focus is not None:
        groups.sort(key=lambda x: -x["rates"][focus])
    else:
        groups.sort(key=lambda x: -x["count"])
    strength = cramers_v(g, y)
    entry = {
        "column": col,
        "measure": "cramers_v",
        "strength": _round(strength),
        "strength_label": None if strength is None else strength_label(strength),
        "groups": groups,
        "other_values": _other_count(g),
    }
    if focus is not None:
        comparable = [x for x in groups if not x["small"] and x["value"] not in (OTHER_LABEL, MISSING_LABEL)]
        if len(comparable) >= 2:
            hi, lo = comparable[0], comparable[-1]
            entry["highest"] = {"value": hi["value"], "rate": hi["rates"][focus], "count": hi["count"]}
            entry["lowest"] = {"value": lo["value"], "rate": lo["rates"][focus], "count": lo["count"]}
    lam = goodman_kruskal_lambda(g, y)
    if lam is not None and lam >= LEAK_THRESHOLD:
        entry["leakage"] = {"measure": "lambda", "value": _round(lam),
                            "detail": "knowing its value tells you the target's class almost every time"}
    return entry


def _numeric_by_class(x: pd.Series, y: pd.Series, col: str, classes: list[str]) -> dict[str, Any]:
    per_class = []
    for c in classes:
        v = x[y == c].dropna()
        per_class.append({"value": c, "count": int(len(v)),
                          "median": _round(v.median()) if len(v) else None, "small": len(v) < MIN_GROUP_ROWS})
    strength = rank_eta(x, y)
    entry = {
        "column": col,
        "measure": "rank_eta",
        "strength": _round(strength),
        "strength_label": None if strength is None else strength_label(strength),
        "classes": per_class,
        "n": int(x.notna().sum()),
    }
    sep = separation(x, y)
    if sep is not None and sep >= LEAK_THRESHOLD:
        entry["leakage"] = {"measure": "separation", "value": _round(sep),
                            "detail": "its values separate the classes with almost no overlap"}
    return entry


def _numeric_stats(v: pd.Series) -> dict[str, Any]:
    v = v.dropna().astype(float)
    if v.empty:
        return {"count": 0, "median": None, "mean": None, "q1": None, "q3": None}
    return {"count": int(len(v)), "median": _round(v.median()), "mean": _round(v.mean()),
            "q1": _round(v.quantile(0.25)), "q3": _round(v.quantile(0.75))}


def _other_count(g: pd.Series) -> int:
    """How many rows were folded into "Other"."""
    return int((g == OTHER_LABEL).sum())


def _round(x: Any, digits: int = 4) -> float | None:
    if x is None or pd.isna(x):
        return None
    x = float(x)
    return round(x, digits) if abs(x) < 1000 else round(x, 2)


def _pct(part: float, whole: float) -> float:
    return round(100 * float(part) / whole, 2) if whole else 0.0
