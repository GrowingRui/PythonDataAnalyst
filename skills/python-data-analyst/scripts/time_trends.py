"""Time trends for date columns: rows per period, gaps, spikes, partial periods, implausible dates.

generate_eda.py calls time_summary() and draws the chart, so you normally don't
run this module yourself. It only computes: no charts, no files. Dates stored
as text are parsed in memory, with the day/month order that inspect_data
detected; the dataset itself is never changed.

The rules are deliberately simple and each one is explained in the report:
  - the period size (day ... year) is chosen so the chart has roughly
    MIN_POINTS to MAX_POINTS points with enough rows in each
  - an empty period is a *gap* only when it's unlikely to happen by chance
    (Poisson: with a typical 20 rows per week, one empty week has a chance
    of e^-20; with 2 rows per day, empty days are normal)
  - a *spike* is a period with at least SPIKE_RATIO times the typical rows
  - the first and last periods are *partial* when the data covers less than
    PARTIAL_COVERAGE of them, so their counts look like drops
  - future dates and placeholder dates (1900-01-01, 1970-01-01) are
    *implausible*; when they are a small minority they are left out of the
    trend, because one date in 2052 would stretch the axis over decades
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

from inspect_data import DatasetError

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

# name: pandas period alias. Weeks run Monday to Sunday.
PERIODS = {"day": "D", "week": "W-SUN", "month": "M", "quarter": "Q", "year": "Y"}
PERIOD_NAMES = tuple(PERIODS)
MAX_POINTS = 120  # more and the line becomes a blur
MIN_POINTS = 12  # fewer and there's hardly a trend to see
MIN_ROWS_PER_PERIOD = 5  # below this, counts are mostly noise (and empty periods normal)
MIN_TREND_PERIODS = 3
MIN_CHECK_PERIODS = 3  # full, non-empty periods needed before gaps and spikes are judged
PARTIAL_COVERAGE = 0.9  # data covering less than 90% of a period makes it partial
GAP_MIN_EXPECTED = 6.9  # flag an empty run when e^-(typical x length) < 0.001
SPIKE_RATIO = 3.0
SPIKE_MIN_EXTRA = 10  # and at least this many rows above typical (tiny counts jump easily)
FUTURE_MAX_SHARE = 5.0  # % of dates; above this, future dates look intended (due dates, plans)
PLACEHOLDER_DATES = ("1899-12-30", "1899-12-31", "1900-01-01", "1970-01-01")
MAX_EXAMPLES = 5

# Numeric columns whose median per period is most likely to interest someone,
# in order of preference. Matched as a part of the lowercase column name.
VALUE_NAME_HINTS = ("revenue", "sales", "amount", "total", "profit", "income", "spend",
                    "price", "cost", "value", "quantity", "qty")


# --------------------------------------------------------------------------
# Finding and parsing date columns
# --------------------------------------------------------------------------


def date_candidates(df: pd.DataFrame, kinds: dict[str, str],
                    type_issues: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Columns that hold dates: real datetime columns, and text the quality check found to be dates."""
    out: dict[str, dict[str, Any]] = {}
    for name in df.columns:
        if kinds.get(str(name)) == "datetime":
            out[str(name)] = {"source": "datetime", "date_order": None}
    for issue in type_issues:
        if issue["issue"] == "dates_as_text" and issue["column"] not in out:
            out[issue["column"]] = {"source": "text", "date_order": issue.get("date_order")}
    return out


def parse_dates(s: pd.Series, date_order: str | None = None) -> pd.Series:
    """The column as timezone-naive datetimes; values that aren't dates become NaT.

    Text is parsed day-first only when the quality check saw day-first values
    (a part above 12 in the first position). "ambiguous" (no part above 12) is
    read month-first, pandas' default; the report says so. Each distinct value
    is parsed once: fast on large files with repeating dates.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return _naive(s)
    present = s.dropna()
    text = present.astype(str).str.strip()
    uniq = pd.Series(text.unique())
    dayfirst = date_order == "day-first"
    parsed = _to_datetime(uniq, dayfirst=dayfirst, mixed=False)
    retry = parsed.isna() & uniq.str.contains(r"\d", regex=True)
    if retry.any():  # values in a different format than the first one
        parsed = parsed.copy()
        parsed[retry] = _to_datetime(uniq[retry], dayfirst=dayfirst, mixed=True)
    mapping = pd.Series(parsed.to_numpy(), index=uniq.to_numpy())
    out = pd.Series(pd.NaT, index=s.index, dtype=parsed.dtype)
    out[present.index] = text.map(mapping).to_numpy()
    return pd.to_datetime(out)


def _to_datetime(values: pd.Series, dayfirst: bool, mixed: bool) -> pd.Series:
    kwargs: dict[str, Any] = {"errors": "coerce", "dayfirst": dayfirst}
    if mixed:
        kwargs["format"] = "mixed"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = pd.to_datetime(values, **kwargs)
        except (ValueError, TypeError):
            # Mixed time zones ("+01:00" and "+02:00") can't share one dtype:
            # convert everything to UTC first.
            out = pd.to_datetime(values, utc=True, **kwargs)
    return _naive(pd.Series(out, index=values.index))


def _naive(s: pd.Series) -> pd.Series:
    """Drop the time zone but keep the local clock time: 09:00+02:00 stays 09:00."""
    if getattr(s.dt, "tz", None) is not None:
        return s.dt.tz_localize(None)
    return s


def check_dates(dates: pd.Series, now: pd.Timestamp) -> dict[str, Any]:
    """Future dates and placeholder dates in one parsed column."""
    d = dates.dropna()
    n = len(d)
    future = d[d.dt.normalize() > now.normalize()]
    placeholders = d[d.isin(pd.to_datetime(list(PLACEHOLDER_DATES)))]
    future_pct = round(100 * len(future) / n, 2) if n else 0.0
    out: dict[str, Any] = {
        "values": n,
        "min": _iso(d.min()) if n else None,
        "max": _iso(d.max()) if n else None,
        "future": int(len(future)),
        "future_pct": future_pct,
        # A few future dates are probably typos; many of them look intended.
        "future_excluded": bool(len(future)) and future_pct <= FUTURE_MAX_SHARE,
        "future_examples": [_iso(v) for v in sorted(future.unique())[-MAX_EXAMPLES:][::-1]],
        "placeholders": {_iso(k): int(v) for k, v in placeholders.value_counts().sort_index().items()},
        "today": _iso(now.normalize()),
    }
    return out


def plausible_mask(dates: pd.Series, check: dict[str, Any], now: pd.Timestamp) -> pd.Series:
    """True for dates the trend should use: present, not a placeholder, not a stray future date."""
    ok = dates.notna() & ~dates.isin(pd.to_datetime(list(PLACEHOLDER_DATES)))
    if check["future_excluded"]:
        ok &= ~(dates.dt.normalize() > now.normalize())
    return ok


# --------------------------------------------------------------------------
# Periods
# --------------------------------------------------------------------------


def count_periods(first: pd.Timestamp, last: pd.Timestamp, period: str) -> int:
    alias = PERIODS[period]
    return int((last.to_period(alias) - first.to_period(alias)).n) + 1


def choose_period(first: pd.Timestamp, last: pd.Timestamp, n_rows: int) -> str:
    """The finest period with at most MAX_POINTS points and enough rows per point.

    A finer period is skipped when it averages fewer than MIN_ROWS_PER_PERIOD
    rows, as long as the next coarser one still gives MIN_POINTS points.
    """
    for i, name in enumerate(PERIOD_NAMES):
        n = count_periods(first, last, name)
        if n > MAX_POINTS:
            continue
        coarser = PERIOD_NAMES[i + 1] if i + 1 < len(PERIOD_NAMES) else None
        if (n_rows / n < MIN_ROWS_PER_PERIOD and coarser
                and count_periods(first, last, coarser) >= MIN_POINTS):
            continue
        return name
    return "year"


def period_label(p: pd.Period, period: str) -> str:
    start = p.start_time
    if period == "day":
        return start.strftime("%Y-%m-%d")
    if period == "week":
        return f"week of {start:%Y-%m-%d}"
    if period == "month":
        return start.strftime("%Y-%m")
    if period == "quarter":
        return f"{start.year}-Q{p.quarter}"
    return str(start.year)


def _coverage(p: pd.Period, first_day: pd.Timestamp, last_day: pd.Timestamp) -> tuple[int, int]:
    """(days of the period inside [first_day, last_day], days in the period)."""
    start, end = p.start_time.normalize(), p.end_time.normalize()
    days = (end - start).days + 1
    covered = (min(end, last_day) - max(start, first_day)).days + 1
    return max(covered, 0), days


# --------------------------------------------------------------------------
# The summary
# --------------------------------------------------------------------------


def time_summary(
    df: pd.DataFrame,
    kinds: dict[str, str],
    type_issues: list[dict[str, Any]],
    date_column: str | None = None,
    period: str | None = None,
    value_column: str | None = None,
    exclude_values: list[str] | None = None,
    now: pd.Timestamp | None = None,
) -> dict[str, Any] | None:
    """Date checks for every date column and a trend for one of them.

    Returns None when the data has no date columns (and no ``date_column`` was
    asked for). Raises DatasetError when ``date_column`` doesn't exist or
    doesn't hold dates, or ``period`` is unknown.
    """
    if period is not None and period not in PERIODS:
        raise DatasetError(f"Unknown period {period!r}. Choose one of: {', '.join(PERIOD_NAMES)}.")
    candidates = date_candidates(df, kinds, type_issues)
    if date_column is not None:
        names = [str(c) for c in df.columns]
        if date_column not in names:
            raise DatasetError(f"--date-column {date_column!r}: no such column. "
                               f"Columns: {', '.join(names[:30])}{', ...' if len(names) > 30 else ''}")
        if date_column not in candidates:
            found = ", ".join(candidates) or "none"
            raise DatasetError(f"--date-column {date_column!r} doesn't hold dates "
                               f"(fewer than 80% of its values parse as dates). Date columns found: {found}.")
    if not candidates:
        return None

    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    columns = {str(c): c for c in df.columns}
    parsed = {name: parse_dates(df[columns[name]], info["date_order"]) for name, info in candidates.items()}
    checks = {}
    for name, info in candidates.items():
        checks[name] = {**info, **check_dates(parsed[name], now),
                        "unparsed": int(df[columns[name]].notna().sum() - parsed[name].notna().sum())}

    if date_column is not None:
        column, chosen_by = date_column, "--date-column"
    elif len(candidates) == 1:
        column, chosen_by = next(iter(candidates)), "only date column"
    else:
        column = max(candidates, key=lambda c: (checks[c]["values"], -list(candidates).index(c)))
        chosen_by = "most dates"

    dates = parsed[column]
    keep = plausible_mask(dates, checks[column], now)
    result: dict[str, Any] = {"date_columns": checks, "column": column, "chosen_by": chosen_by,
                              "trend": None, "note": None}
    used = dates[keep]
    if used.empty:
        result["note"] = "no usable dates left after removing missing and implausible ones"
        return result
    first, last = used.min(), used.max()
    chosen = period or choose_period(first, last, len(used))
    n_periods = count_periods(first, last, chosen)
    if n_periods < MIN_TREND_PERIODS:
        result["note"] = (f"all usable dates fall within {n_periods} {chosen}"
                          f"{'s' if n_periods != 1 else ''}; a trend needs at least {MIN_TREND_PERIODS}")
        return result

    value = value_column or pick_value_column(df, kinds, exclude_values or [])
    result["trend"] = _trend(df, dates, keep, chosen, period is not None, value, value_column is not None, checks[column])
    return result


def pick_value_column(df: pd.DataFrame, kinds: dict[str, str], exclude: list[str]) -> str | None:
    """The numeric column whose median per period is shown: named like an amount, else the fullest."""
    numeric = [str(c) for c in df.columns
               if kinds.get(str(c)) == "numeric" and str(c) not in exclude
               and df[c].notna().any() and df[c].nunique() > 1]
    for hint in VALUE_NAME_HINTS:
        for name in numeric:
            if hint in name.lower():
                return name
    if not numeric:
        return None
    return max(numeric, key=lambda c: (int(df[c].notna().sum()), -numeric.index(c)))


def _trend(df: pd.DataFrame, dates: pd.Series, keep: pd.Series, period: str, period_given: bool,
           value: str | None, value_given: bool, check: dict[str, Any]) -> dict[str, Any]:
    alias = PERIODS[period]
    used = dates[keep]
    per = used.dt.to_period(alias)
    rng = pd.period_range(per.min(), per.max(), freq=alias)
    counts = per.value_counts().reindex(rng, fill_value=0)
    medians = None
    if value is not None:
        col = next(c for c in df.columns if str(c) == value)
        values = pd.to_numeric(df.loc[keep, col], errors="coerce")
        medians = values.groupby(per).median().reindex(rng)

    first_day, last_day = used.min().normalize(), used.max().normalize()
    partial = []
    if period != "day":
        for which, p in (("first", rng[0]), ("last", rng[-1])):
            if which == "last" and len(rng) == 1:
                break
            covered, days = _coverage(p, first_day, last_day)
            if covered / days < PARTIAL_COVERAGE:
                partial.append({"which": which, "label": period_label(p, period), "days_covered": covered,
                                "days_in_period": days, "rows": int(counts[p])})
    partial_idx = {0 if x["which"] == "first" else len(rng) - 1 for x in partial}

    full = [int(c) for i, c in enumerate(counts) if i not in partial_idx and c > 0]
    checkable = len(full) >= MIN_CHECK_PERIODS
    typical = float(np.median(full)) if full else 0.0

    gaps, chance_empty, weekend_empty = [], 0, 0
    empty = (counts == 0).to_numpy()
    i = 0
    while i < len(rng):
        if not empty[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(rng) and empty[j + 1]:
            j += 1
        length = j - i + 1
        if period == "day" and all(rng[k].start_time.weekday() >= 5 for k in range(i, j + 1)):
            weekend_empty += length  # closed at weekends: no activity, not missing data
        elif checkable and typical * length >= GAP_MIN_EXPECTED:
            gaps.append({"first": period_label(rng[i], period), "last": period_label(rng[j], period),
                         "periods": length, "start_index": i, "end_index": j,
                         "chance": _chance(typical, length)})
        else:
            chance_empty += length
        i = j + 1

    duplicated = df.loc[keep].duplicated(keep="first").to_numpy()
    spikes = []
    if checkable:
        for idx, (p, c) in enumerate(counts.items()):
            if c >= SPIKE_RATIO * typical and c - typical >= SPIKE_MIN_EXTRA:
                in_p = (per == p).to_numpy()
                days = used[in_p].dt.normalize().value_counts()
                spikes.append({"label": period_label(p, period), "index": idx, "rows": int(c),
                               "ratio": round(c / typical, 1), "top_day": _iso(days.index[0]),
                               "top_day_rows": int(days.iloc[0]),
                               "exact_duplicates": int(duplicated[in_p].sum())})

    gap_idx = {k for g in gaps for k in range(g["start_index"], g["end_index"] + 1)}
    spike_idx = {s["index"] for s in spikes}
    rows = []
    for idx, p in enumerate(rng):
        row = {"start": _iso(p.start_time), "label": period_label(p, period), "rows": int(counts.iloc[idx])}
        if medians is not None:
            m = medians.iloc[idx]
            row["value_median"] = None if pd.isna(m) else float(m)
        if idx in partial_idx:
            row["partial"] = True
        if idx in gap_idx:
            row["gap"] = True
        if idx in spike_idx:
            row["spike"] = True
        rows.append(row)

    n_years = (used.max() - used.min()).days / 365.25
    return {
        "period": period,
        "period_chosen_by": "--period" if period_given else "automatic",
        "first": _iso(used.min()),
        "last": _iso(used.max()),
        "span_days": int((last_day - first_day).days) + 1,
        "span_years": round(n_years, 2),
        "rows_used": int(len(used)),
        "rows_missing_date": int(dates.isna().sum()),
        "rows_excluded_implausible": int((dates.notna() & ~keep).sum()),
        "date_source": check["source"],
        "date_order": check["date_order"],
        "value_column": value,
        "value_chosen_by": None if value is None else ("given" if value_given else "automatic"),
        "typical_rows": typical,
        "checks_applied": checkable,
        "periods": rows,
        "partial": partial,
        "gaps": gaps,
        "empty_periods_by_chance": chance_empty,
        "empty_weekend_days": weekend_empty,
        "spikes": spikes,
    }


def _chance(typical: float, length: int) -> str:
    """Rough chance of this many empty periods in a row if rows arrived at random (Poisson)."""
    p = math.exp(-typical * length)
    if p < 1e-6:
        return "less than one in a million"
    return f"about 1 in {round(1 / p):,}"


def _iso(ts: Any) -> str:
    ts = pd.Timestamp(ts)
    if ts == ts.normalize():
        return ts.strftime("%Y-%m-%d")
    return ts.isoformat()
