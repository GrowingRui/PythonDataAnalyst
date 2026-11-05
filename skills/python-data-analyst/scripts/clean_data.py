"""Apply approved cleaning steps to a copy of a dataset, and log every change.

Usage:
    python clean_data.py data.csv --plan plan.json            # dry run: shows the changes, writes nothing
    python clean_data.py data.csv --plan plan.json --apply    # writes the cleaned file and the logs
    python clean_data.py data.csv --plan plan.json --output cleaned/sales.parquet --apply
    python clean_data.py data.csv --plan cleaned/cleaning_log.json --output again/sales.csv --apply

plan.json lists steps from a fixed set and runs them in order:

    {"steps": [
      {"step": "drop_exact_duplicates"},
      {"step": "placeholders_to_missing", "columns": ["rating"], "values": ["-"]},
      {"step": "to_numeric", "columns": ["rating"]}
    ]}

Every step may also carry "fixes" (the finding rule it answers) and "reason"
(why the user approved it); both go into the log. Run with --list-steps to see
every step and its options.

Without --apply nothing is written. With --apply the cleaned file (CSV, or
Parquet for a .parquet output), cleaning_log.md and cleaning_log.json go into a
new or empty folder. The input file is only ever read: its SHA-256 is checked
before and after. cleaning_log.json can be passed back as --plan to rebuild the
same file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_data import (  # noqa: E402
    DATASET_HELP,
    DISGUISED_MISSING,
    ENCODING_HELP,
    ISO_DATE_PATTERN,
    JSON_PATH_HELP,
    DatasetError,
    _day_month_order,
    _inconsistent_labels,
    _string_values,
    file_sha256,
    load_dataset,
    source_description,
    to_jsonable,
)
from report import ADVICE_ALIAS, CLEANING_STEPS  # noqa: E402

OUTPUT_FORMATS = {".csv": "csv", ".parquet": "parquet"}
LOG_MD, LOG_JSON = "cleaning_log.md", "cleaning_log.json"
MAX_EXAMPLES = 5
ROW_NUMBERS = "Row numbers count the data rows of the input from 0 (the header row isn't counted), as in pandas."
IQR_MIN_VALUES = 5  # same rule as inspect_data.iqr_outliers
# Text that becomes a number once these are removed: "$1,234.50", "12 %", "€ 3".
CURRENCY_PCT_SPACES = r"^[\$€£¥]\s*|\s*%$|\s"
DECIMAL_COMMA = r"^[-+]?\d+,\d{1,2}$|\.\d{3},\d"


class PlanError(ValueError):
    """The plan can't be run as written. The message is meant for the user."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_for_cleaning(path: str | Path, sheet: str | None = None, encoding: str | None = None,
                      delimiter: str | None = None, json_path: str | None = None) -> pd.DataFrame:
    """load_dataset(), plus protection for codes that pandas would turn into numbers.

    A CSV column such as postcode "01234" or an 18-digit ID is read as a number
    and written back as 1234 or 1.2345678901234568e+17: a silent change. Such
    columns are kept exactly as text; ``source["kept_as_text"]`` lists them.
    """
    df = load_dataset(path, sheet=sheet, encoding=encoding, delimiter=delimiter, json_path=json_path)
    src = df.attrs["source"]
    if src["format"] != "csv":
        return df
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if not numeric:
        return df
    raw = pd.read_csv(path, sep=src["delimiter"], encoding=src["encoding"], dtype=str)
    kept = []
    for c in numeric:
        text = raw[c].dropna().str.strip()
        digits = text.str.replace(r"\D", "", regex=True).str.len()
        leading_zero = text.str.match(r"^[-+]?0\d").any()
        too_long = pd.api.types.is_float_dtype(df[c]) and bool((digits > 15).any())
        if leading_zero or too_long:
            df[c] = raw[c]
            kept.append(str(c))
    if kept:
        src["kept_as_text"] = kept
    return df


# --------------------------------------------------------------------------
# Plan validation
# --------------------------------------------------------------------------


def _v_columns(v: Any) -> list[str]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or not v or not all(isinstance(c, str) and c for c in v):
        raise ValueError("must be a column name or a non-empty list of column names")
    if len(set(v)) != len(v):
        raise ValueError("lists a column more than once")
    return v


def _v_column(v: Any) -> str:
    if not isinstance(v, str) or not v:
        raise ValueError("must be one column name")
    return v


def _v_choice(*options: str) -> Callable[[Any], str]:
    def check(v: Any) -> str:
        if v not in options:
            raise ValueError(f"must be one of {', '.join(map(repr, options))}")
        return v
    return check


def _v_bool(v: Any) -> bool:
    if not isinstance(v, bool):
        raise ValueError("must be true or false")
    return v


def _v_positive(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
        raise ValueError("must be a number above 0")
    return float(v)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _v_values(v: Any) -> list[Any]:
    if not isinstance(v, list) or not v or not all(isinstance(x, str) or _is_number(x) for x in v):
        raise ValueError('must be a non-empty list of texts and/or numbers, e.g. ["-", "unknown", 999]')
    return v


def _v_labels(v: Any) -> dict[str, str]:
    if not isinstance(v, dict) or not v or not all(isinstance(k, str) and isinstance(x, str) for k, x in v.items()):
        raise ValueError('must map old labels to new ones, e.g. {"north": "North"}')
    return v


def _v_number_map(v: Any) -> dict[str, float]:
    if not isinstance(v, dict) or not all(isinstance(k, str) and _is_number(x) for k, x in v.items()):
        raise ValueError('must map texts to numbers, e.g. {"free": 0}')
    return v


def _v_scalar(v: Any) -> Any:
    if not (isinstance(v, (str, bool)) or _is_number(v)):
        raise ValueError("must be a text, a number or true/false")
    return v


def _v_text(v: Any) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError("must be a non-empty text")
    return v


@dataclass
class Step:
    run: Callable[[pd.DataFrame, dict[str, Any]], tuple[pd.DataFrame, dict[str, Any]]]
    doc: str
    required: dict[str, Callable[[Any], Any]] = field(default_factory=dict)
    optional: dict[str, tuple[Callable[[Any], Any], Any]] = field(default_factory=dict)  # name: (check, default)
    adds_columns: bool = False

    def fixes(self, name: str) -> list[str]:
        """Finding rules this step can fix (from report.CLEANING_STEPS and its aliases)."""
        keys = {k for k, steps in CLEANING_STEPS.items() if name in steps}
        return sorted(keys | {alias for alias, key in ADVICE_ALIAS.items() if key in keys})


COMMON_KEYS = ("step", "fixes", "reason")
STEPS: dict[str, Step] = {}  # filled below, after the step functions


def read_plan(path: str | Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read plan.json, or a cleaning_log.json from an earlier run (for an exact rebuild).

    Returns (plan, earlier_log or None).
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise PlanError(f"Plan file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PlanError(f"{path.name} is not valid JSON: {exc}") from exc
    if isinstance(data, dict) and "plan" in data and "steps" in data and "input" in data:
        return data["plan"], data  # a cleaning log
    if isinstance(data, list):
        data = {"steps": data}
    return data, None


def validate_plan(plan: Any) -> list[dict[str, Any]]:
    """Check the plan's shape before any data is touched; return the steps with defaults filled in."""
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        raise PlanError('A plan is a JSON object with a list of steps: {"steps": [{"step": ...}, ...]}')
    if not plan["steps"]:
        raise PlanError("The plan has no steps.")
    steps = []
    for n, raw in enumerate(plan["steps"], start=1):
        if not isinstance(raw, dict) or "step" not in raw:
            raise PlanError(f'Step {n} must be an object with a "step" name, e.g. {{"step": "drop_exact_duplicates"}}.')
        name = raw["step"]
        if name not in STEPS:
            raise PlanError(f"Step {n}: unknown step {name!r}. Available steps: {', '.join(STEPS)}. "
                            "Only these tested steps can run; do a one-off fix with pandas and say so.")
        spec = STEPS[name]
        where = f"Step {n} ({name})"
        known = set(COMMON_KEYS) | set(spec.required) | set(spec.optional)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise PlanError(f"{where}: unknown option(s) {', '.join(map(repr, unknown))}. "
                            f"Options for this step: {', '.join(sorted(known - {'step'}))}")
        step: dict[str, Any] = {"step": name}
        for key, check in spec.required.items():
            if key not in raw:
                raise PlanError(f"{where}: {key!r} is required.")
            step[key] = _checked(where, key, check, raw[key])
        for key, (check, default) in spec.optional.items():
            step[key] = _checked(where, key, check, raw[key]) if key in raw else default
        fixes = spec.fixes(name)
        if "fixes" in raw:
            if raw["fixes"] not in fixes:
                raise PlanError(f"{where}: 'fixes' is {raw['fixes']!r}, but this step fixes "
                                f"{', '.join(map(repr, fixes))}.")
            step["fixes"] = raw["fixes"]
        else:
            step["fixes"] = None
        step["reason"] = _checked(where, "reason", _v_text, raw["reason"]) if "reason" in raw else None
        _cross_check(where, step)
        steps.append(step)
    return steps


def _checked(where: str, key: str, check: Callable[[Any], Any], value: Any) -> Any:
    try:
        return check(value)
    except ValueError as exc:
        raise PlanError(f"{where}: {key!r} {exc}.") from exc


def _cross_check(where: str, step: dict[str, Any]) -> None:
    name = step["step"]
    if name == "to_datetime" and step["format"] and step["day_first"] is not None:
        raise PlanError(f"{where}: give either 'format' or 'day_first', not both.")
    if name == "fill_missing":
        if step["method"] == "constant" and step["value"] is None:
            raise PlanError(f"{where}: method 'constant' needs a 'value'.")
        if step["method"] != "constant" and step["value"] is not None:
            raise PlanError(f"{where}: 'value' is only used with method 'constant'.")
        if step["by"] and step["method"] == "constant":
            raise PlanError(f"{where}: 'by' (fill per group) works with median or mode, not constant.")
        if step["by"] and step["by"] in step["columns"]:
            raise PlanError(f"{where}: 'by' can't be one of the columns being filled.")


# --------------------------------------------------------------------------
# Steps. Each takes the current table and the checked options, returns a new
# table (never changes its input) and details for the log, including a
# one-sentence "summary".
# --------------------------------------------------------------------------


def _need(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise PlanError(f"column(s) not found: {', '.join(map(repr, missing))}. "
                        f"Columns at this point: {', '.join(map(str, df.columns))}")


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _need_numeric(df: pd.DataFrame, c: str) -> pd.Series:
    s = df[c]
    if not _is_numeric(s):
        raise PlanError(f"{c!r} is not numeric (dtype {s.dtype}); convert it with a to_numeric step first.")
    return s


def _labels(index: pd.Index, limit: int = 10) -> list[Any]:
    return [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in index[:limit]]


def _show(v: Any) -> str:
    """A value as it appears in the log."""
    if v is None or (not isinstance(v, (str, list, dict)) and pd.isna(v)):
        return "missing"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d") if v == v.normalize() else v.isoformat()
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if _is_number(v):
        f = float(v)
        return str(int(f)) if f.is_integer() and abs(f) < 1e15 else f"{f:.6g}"
    return repr(str(v))


def _stats(s: pd.Series) -> dict[str, float | None]:
    v = s.dropna().astype(float)
    return {"count": int(v.size), "mean": _r(v.mean()), "std": _r(v.std()), "min": _r(v.min()), "max": _r(v.max())}


def _r(x: Any) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), 6)


def _fences(s: pd.Series, k: float) -> tuple[float, float] | str:
    v = s.dropna().astype(float)
    if len(v) < IQR_MIN_VALUES:
        return f"only {len(v)} values; at least {IQR_MIN_VALUES} are needed"
    q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
    if q3 == q1:
        return "at least half the values are identical (IQR 0), so the fences would flag every other value"
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


def _insert_after(df: pd.DataFrame, column: str, name: str, values: pd.Series) -> None:
    if name in df.columns:
        raise PlanError(f"can't add column {name!r}: it already exists.")
    df.insert(df.columns.get_loc(column) + 1, name, values)


def step_drop_exact_duplicates(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    dup = df.duplicated(keep=p["keep"])
    n = int(dup.sum())
    return df[~dup].copy(), {
        "summary": f"Removed {n:,} exact duplicate row(s), keeping the {p['keep']} copy of each.",
        "rows_removed": n, "removed_rows": _labels(df.index[dup]),
    }


def step_drop_duplicate_keys(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys, keep = p["columns"], p["keep"]
    _need(df, keys)
    has_key = df[keys].notna().all(axis=1)
    sub = df[has_key]
    if keep == "most_complete":
        gaps = sub.isna().sum(axis=1)
        winners = gaps.groupby([sub[k] for k in keys], sort=False).idxmin()  # ties: the first row
        drop = sub.index.difference(pd.Index(winners.to_numpy()))
    else:
        drop = sub.index[sub.duplicated(subset=keys, keep=keep)]
    repeated = sub[sub.duplicated(subset=keys, keep=False)]
    differ: dict[str, int] = {}
    examples = []
    if not repeated.empty:
        others = [c for c in df.columns if c not in keys]
        if others:
            per_key = repeated.groupby(keys, sort=False)[others].nunique(dropna=False)
            differ = {str(c): int(n) for c, n in (per_key > 1).sum().items() if n}
        for key, grp in list(repeated.groupby(keys, sort=False))[:MAX_EXAMPLES]:
            key = key if isinstance(key, tuple) else (key,)
            examples.append({"key": " / ".join(_show(k) for k in key),
                             "kept": _labels(grp.index.difference(drop)),
                             "dropped": _labels(grp.index.intersection(drop))})
    n = len(drop)
    keep_text = {"first": "the first row", "last": "the last row", "most_complete": "the most complete row"}[keep]
    details: dict[str, Any] = {
        "summary": f"Removed {n:,} row(s) that repeat a key on {', '.join(keys)}, keeping {keep_text} per key.",
        "rows_removed": n, "keys_repeated": int(repeated.drop_duplicates(subset=keys).shape[0]),
        "columns_that_differ": differ, "key_examples": examples, "notes": [],
    }
    if differ:
        cols = ", ".join(f"{c} ({k} key{'s' if k != 1 else ''})" for c, k in differ.items())
        details["notes"].append(f"The repeated rows disagree in {cols}: the dropped versions of those values are lost.")
    if not has_key.all():
        details["notes"].append(f"{int((~has_key).sum()):,} row(s) have no value in {', '.join(keys)}; "
                                "they are never dropped by this step.")
    return df.drop(index=drop), details


def step_placeholders_to_missing(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = p["columns"]
    _need(df, cols)
    values = p["values"]
    texts = {str(v).strip().casefold() for v in (values or sorted(DISGUISED_MISSING)) if isinstance(v, str)}
    numbers = [float(v) for v in values or [] if _is_number(v)]
    texts |= {_show(v) for v in numbers}  # 999 also matches the text "999"
    new = df.copy()
    per_column: dict[str, dict[str, int]] = {}
    notes = []
    for c in cols:
        s = new[c]
        if pd.api.types.is_bool_dtype(s):
            raise PlanError(f"{c!r} holds true/false values; it has no placeholders.")
        if pd.api.types.is_datetime64_any_dtype(s):
            if not values:
                raise PlanError(f"{c!r} holds dates: give the placeholder dates as 'values', e.g. [\"1900-01-01\"].")
            dates = []
            for v in values:
                try:
                    dates.append(pd.Timestamp(str(v)))
                except ValueError as exc:
                    raise PlanError(f"{v!r} is not a date, so it can't be a placeholder in date column {c!r}.") from exc
            mask = s.isin(dates)
        elif _is_numeric(s):
            if not numbers:
                raise PlanError(f"{c!r} is numeric: give the code numbers as 'values', e.g. [999, -1].")
            mask = s.isin(numbers)
        else:
            mask = s.map(lambda x: (isinstance(x, str) and x.strip().casefold() in texts)
                         or (_is_number(x) and float(x) in numbers), na_action="ignore")
            mask = mask.fillna(False).astype(bool)
        found = s[mask].map(_show).value_counts()
        per_column[str(c)] = {str(k): int(v) for k, v in found.items()}
        new[c] = s.mask(mask)
        if values:
            seen = set(per_column[str(c)])
            unused = [v for v in values if _show(v) not in seen and not any(
                isinstance(v, str) and k.strip("'").strip().casefold() == v.strip().casefold() for k in seen)]
            if unused:
                notes.append(f"In {c}, {', '.join(_show(v) for v in unused)} matched nothing.")
    total = sum(sum(v.values()) for v in per_column.values())
    parts = "; ".join(f"{c}: " + ", ".join(f"{k} x{n}" for k, n in v.items()) for c, v in per_column.items() if v)
    return new, {"summary": f"Turned {total:,} placeholder value(s) into missing values" + (f" ({parts})." if parts else "."),
                 "replaced": per_column, "notes": notes}


def step_strip_whitespace(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = p["columns"]
    if cols is None:
        cols = [str(c) for c in df.columns if not _string_values(df[c]).empty]
    _need(df, cols)
    new = df.copy()
    per_column: dict[str, int] = {}
    emptied = 0
    notes = []
    for c in cols:
        strings = _string_values(new[c])
        if strings.empty:
            notes.append(f"{c} has no text values; nothing to trim.")
            continue
        stripped = strings.str.strip()
        changed = strings.index[strings != stripped]
        if len(changed):
            blank = stripped[changed] == ""
            emptied += int(blank.sum())
            s = new[c].copy()
            s.loc[changed] = stripped[changed].mask(blank)
            new[c] = s
            per_column[str(c)] = len(changed)
    if emptied:
        notes.append(f"{emptied:,} value(s) held only spaces and became missing.")
    total = sum(per_column.values())
    parts = ", ".join(f"{c} {n:,}" for c, n in per_column.items())
    return new, {"summary": f"Trimmed spaces from {total:,} value(s)" + (f" ({parts})." if parts else "."),
                 "trimmed": per_column, "became_missing": emptied, "notes": notes}


def step_normalize_labels(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    c, mapping = p["column"], p["mapping"]
    _need(df, [c])
    strings = _string_values(df[c])
    hit = strings.isin(list(mapping))
    counts = strings[hit].value_counts()
    new = df.copy()
    s = new[c].copy()
    s.loc[strings.index[hit]] = strings[hit].map(mapping)
    new[c] = s
    notes = []
    stripped = set(strings.str.strip())
    unused = [k for k in mapping if k not in counts.index]
    for k in unused:
        hint = " (it matches once spaces are trimmed: run strip_whitespace first)" if k.strip() in stripped else ""
        notes.append(f"{k!r} matched no value in {c}{hint}.")
    after = _string_values(s)
    left = _inconsistent_labels(after.str.strip(), MAX_EXAMPLES)
    if left:
        groups = "; ".join(" / ".join(repr(v["value"]) for v in g) for g in left["groups"])
        notes.append(f"{c} still writes some labels in more than one way: {groups}.")
    total = int(counts.sum())
    parts = ", ".join(f"{k!r} -> {mapping[k]!r} x{int(n)}" for k, n in counts.items())
    return new, {"summary": f"Relabelled {total:,} value(s) in {c}" + (f" ({parts})." if parts else "."),
                 "relabelled": {str(k): int(n) for k, n in counts.items()},
                 "distinct_before": int(strings.nunique()), "distinct_after": int(after.nunique()), "notes": notes}


def step_to_numeric(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = p["columns"]
    _need(df, cols)
    mapping = {k.strip().casefold(): float(v) for k, v in (p["map"] or {}).items()}
    new = df.copy()
    per_column: dict[str, dict[str, Any]] = {}
    notes = []
    for c in cols:
        s = new[c]
        if _is_numeric(s):
            notes.append(f"{c} is already numeric; unchanged.")
            continue
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
            raise PlanError(f"{c!r} holds {'dates' if pd.api.types.is_datetime64_any_dtype(s) else 'true/false values'}, "
                            "not numbers stored as text.")
        result = pd.Series(np.nan, index=s.index, dtype="float64")
        non_null = s.dropna()
        strings = _string_values(s).str.strip()
        others = non_null[~non_null.index.isin(strings.index)]  # e.g. real numbers in an Excel text column
        numbers = others[others.map(_is_number).astype(bool)] if not others.empty else others
        if not numbers.empty:
            result.loc[numbers.index] = numbers.astype(float)
        bad = [*others[~others.index.isin(numbers.index)].index]

        mapped = strings.str.casefold().isin(mapping)
        if mapped.any():
            result.loc[strings.index[mapped]] = strings[mapped].str.casefold().map(mapping).astype(float)
        rest = strings[~mapped]
        empty = rest == ""
        rest = rest[~empty]
        cleaned = rest.str.replace(CURRENCY_PCT_SPACES, "", regex=True)
        comma = p["decimal_comma"]
        if comma is None:
            comma = bool(cleaned.str.contains(DECIMAL_COMMA, regex=True).any())
        if comma:
            normalised = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        else:
            normalised = cleaned.str.replace(r"(?<=\d),(?=\d{3}\b)", "", regex=True)
        parsed = pd.to_numeric(normalised, errors="coerce")
        percent = rest.str.endswith("%")
        if p["percent"] == "fraction":
            parsed = parsed.where(~percent, parsed / 100)
        if not rest.empty:
            result.loc[rest.index] = parsed.astype(float)
        failed = rest[parsed.isna()]
        bad += list(failed.index)
        if bad and p["unparseable"] == "error":
            shown = ", ".join(_show(v) for v in s[bad].drop_duplicates()[:MAX_EXAMPLES])
            raise PlanError(f"{len(bad):,} value(s) in {c!r} don't parse as numbers: {shown}. Map them with 'map', "
                            "turn them into missing values first (placeholders_to_missing), or set "
                            "'unparseable' to 'missing'.")
        if result.notna().all() and bool((result % 1 == 0).all()):
            result = result.astype("int64")
        new[c] = result
        failed_counts = s[bad].map(_show).value_counts()
        per_column[str(c)] = {
            "converted": int(result.notna().sum()) - len(numbers) if result.notna().any() else 0,
            "mapped": {k: int(n) for k, n in strings[mapped].str.casefold().value_counts().items()},
            "made_missing": len(bad), "unparseable_values": {str(k): int(n) for k, n in failed_counts.head(10).items()},
            "blank_to_missing": int(empty.sum()), "decimal_comma": comma, "percent_values": int(percent.sum()),
            "dtype_after": str(result.dtype),
        }
        if bad:
            shown = ", ".join(f"{k} x{n}" for k, n in failed_counts.head(MAX_EXAMPLES).items())
            notes.append(f"{len(bad):,} value(s) in {c} don't parse as numbers and became missing: {shown}.")
        if comma:
            notes.append(f"{c} was read with a decimal comma ('120,5' = 120.5).")
        if percent.any():
            how = "divided by 100 (12% = 0.12)" if p["percent"] == "fraction" else "kept as percent numbers (12% = 12)"
            notes.append(f"{int(percent.sum()):,} value(s) in {c} had a % sign: {how}.")
    parts = ", ".join(f"{c} ({v['converted']:,} values"
                      + "".join(f", {n:,} {k!r} mapped to {_show(mapping[k])}" for k, n in v["mapped"].items())
                      + f", {v['made_missing']:,} made missing)" for c, v in per_column.items())
    return new, {"summary": "Converted text to numbers" + (f": {parts}." if parts else ": nothing to convert."),
                 "columns": per_column, "notes": notes}


def step_to_datetime(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = p["columns"]
    _need(df, cols)
    new = df.copy()
    per_column: dict[str, dict[str, Any]] = {}
    notes = []
    for c in cols:
        s = new[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            notes.append(f"{c} already holds dates; unchanged.")
            continue
        if _is_numeric(s) or pd.api.types.is_bool_dtype(s):
            raise PlanError(f"{c!r} holds numbers, not dates stored as text (Excel serial dates aren't supported).")
        strings = _string_values(s).str.strip()
        others = s.dropna()[~s.dropna().index.isin(strings.index)]
        fmt, day_first = p["format"], p["day_first"]
        unique = strings[strings != ""].drop_duplicates()
        if fmt:
            how = f"format {fmt!r}"
        elif day_first is not None:
            how = "day first (31/12/2025)" if day_first else "month first (12/31/2025)"
        else:
            order = _day_month_order(unique)
            if order in ("ambiguous", "inconsistent"):
                raise PlanError(f"the day/month order of {c!r} is {order} (e.g. 03/04/2025 could be 3 April or "
                                "4 March). Say which with 'day_first' (true or false) or give a 'format' such as "
                                "\"%d/%m/%Y\".")
            day_first = order == "day-first"
            how = f"detected {order}" if order else "unambiguous values (e.g. ISO 2025-12-31)"
        try:
            if fmt:
                parsed_unique = pd.to_datetime(unique, format=fmt, errors="coerce")
            else:
                # dayfirst would misread ISO dates (2025-01-02 as 1 February), so they are parsed apart.
                iso = unique.str.match(ISO_DATE_PATTERN)
                parsed_unique = pd.Series(pd.NaT, index=unique.index, dtype="datetime64[ns]")
                if iso.any():
                    parsed_unique[iso] = pd.to_datetime(unique[iso], format="ISO8601", errors="coerce")
                if (~iso).any():
                    parsed_unique[~iso] = pd.to_datetime(unique[~iso], format="mixed", dayfirst=bool(day_first),
                                                         errors="coerce")
        except (ValueError, TypeError) as exc:  # e.g. a mix of time zones
            raise PlanError(f"{c!r} can't be parsed as one kind of date: {exc}") from exc
        lookup = dict(zip(unique, parsed_unique))
        result = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        result.loc[strings.index] = pd.to_datetime(strings.map(lookup), errors="coerce")
        for i, v in others.items():
            if isinstance(v, (datetime, pd.Timestamp, np.datetime64)):
                result.loc[i] = pd.Timestamp(v)
        bad = s.notna() & result.isna()
        blank = strings[strings == ""].index
        bad[blank] = False
        failed_counts = s[bad].map(_show).value_counts()
        if bad.any() and p["unparseable"] == "error":
            shown = ", ".join(f"{k} x{n}" for k, n in failed_counts.head(MAX_EXAMPLES).items())
            raise PlanError(f"{int(bad.sum()):,} value(s) in {c!r} don't parse as dates: {shown}. Turn them into "
                            "missing values first (placeholders_to_missing), or set 'unparseable' to 'missing'.")
        new[c] = result
        ok = result.dropna()
        per_column[str(c)] = {
            "converted": int(ok.size), "read_as": how, "made_missing": int(bad.sum()),
            "unparseable_values": {str(k): int(n) for k, n in failed_counts.head(10).items()},
            "blank_to_missing": len(blank),
            "earliest": _show(ok.min()) if not ok.empty else None, "latest": _show(ok.max()) if not ok.empty else None,
        }
        if bad.any():
            shown = ", ".join(f"{k} x{n}" for k, n in failed_counts.head(MAX_EXAMPLES).items())
            notes.append(f"{int(bad.sum()):,} value(s) in {c} don't parse as dates and became missing: {shown}.")
    parts = ", ".join(f"{c} ({v['converted']:,} dates, read as {v['read_as']}; range {v['earliest']} to {v['latest']})"
                      for c, v in per_column.items())
    return new, {"summary": "Converted text to dates" + (f": {parts}." if parts else ": nothing to convert."),
                 "columns": per_column, "notes": notes}


def step_drop_columns(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = p["columns"]
    _need(df, cols)
    info = {str(c): {"non_missing": int(df[c].notna().sum()), "distinct": int(df[c].nunique())} for c in cols}
    parts = ", ".join(f"{c} ({v['non_missing']:,} values, {v['distinct']:,} distinct)" for c, v in info.items())
    return df.drop(columns=cols), {"summary": f"Dropped {len(cols)} column(s): {parts}.", "dropped": info}


def step_drop_rows_missing(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = p["columns"]
    _need(df, cols)
    gaps = df[cols].isna()
    drop = gaps.any(axis=1) if p["how"] == "any" else gaps.all(axis=1)
    n = int(drop.sum())
    which = ("a missing value in " if p["how"] == "any" or len(cols) == 1 else "every one of ") + ", ".join(cols)
    pct = 100 * n / len(df) if len(df) else 0.0
    return df[~drop].copy(), {"summary": f"Removed {n:,} row(s) ({pct:.1f}%) with {which}.",
                              "rows_removed": n, "removed_rows": _labels(df.index[drop])}


def step_fill_missing(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols, method, by = p["columns"], p["method"], p["by"]
    _need(df, cols + ([by] if by else []))
    new = df.copy()
    per_column: dict[str, dict[str, Any]] = {}
    notes = []
    for c in cols:
        s = new[c]
        missing = s.isna()
        entry: dict[str, Any] = {"filled": int(missing.sum()), "indicator": f"{c}_was_missing"}
        if method == "median":
            _need_numeric(new, c)
            overall = s.median()
            if pd.isna(overall):
                raise PlanError(f"{c!r} has no values to take a median from.")
            fill = s.groupby(new[by]).transform("median") if by else pd.Series(overall, index=s.index)
            filled = s.fillna(fill).fillna(overall)
            entry["value"] = _r(overall)
        elif method == "mode":
            modes = s.mode(dropna=True)
            if modes.empty:
                raise PlanError(f"{c!r} has no values to take a mode from.")
            if len(modes) > 1:
                notes.append(f"{c} has {len(modes)} equally common values; the first in sort order "
                             f"({_show(modes.iloc[0])}) was used.")
            overall = modes.iloc[0]
            if by:
                fill = s.groupby(new[by]).transform(lambda x: x.mode().iloc[0] if x.notna().any() else np.nan)
                filled = s.fillna(fill).fillna(overall)
            else:
                filled = s.fillna(overall)
            entry["value"] = _show(overall)
        else:
            value = p["value"]
            if pd.api.types.is_bool_dtype(s) or (s.dtype == object and _string_values(s).empty and s.notna().any()):
                if not isinstance(value, bool):
                    raise PlanError(f"{c!r} holds true/false values; the fill value must be true or false.")
            elif _is_numeric(s):
                if not _is_number(value):
                    raise PlanError(f"{c!r} is numeric; the fill value must be a number, not {value!r}.")
            elif pd.api.types.is_datetime64_any_dtype(s):
                try:
                    value = pd.Timestamp(str(value))
                except ValueError as exc:
                    raise PlanError(f"{c!r} holds dates; {value!r} is not a date.") from exc
            elif not isinstance(value, str):
                raise PlanError(f"{c!r} holds text; give the fill value as text, e.g. \"{value}\".")
            filled = s.fillna(value)
            entry["value"] = _show(value)
        if by:
            groups = filled[missing].groupby(new.loc[missing, by], dropna=False).first()
            entry["by"] = by
            entry["group_values"] = {_show(k).strip("'"): _show(v) for k, v in list(groups.items())[:20]}
        if _is_numeric(s):
            entry["before"], entry["after"] = _stats(s), _stats(filled)
        new[c] = filled
        _insert_after(new, c, f"{c}_was_missing", missing.astype(bool))
        per_column[str(c)] = entry
    how = {"median": "the median", "mode": "the most common value", "constant": "a fixed value"}[method]
    if by:
        how += f" of each {by} group"
    parts = ", ".join(f"{c} {v['filled']:,}" for c, v in per_column.items())
    spread = [f"{c}: standard deviation {v['before']['std']} -> {v['after']['std']}"
              for c, v in per_column.items() if "before" in v and v["filled"]]
    if spread:
        notes.append("Filling with one value shrinks the spread: " + "; ".join(spread) + ".")
    return new, {"summary": f"Filled missing values with {how} ({parts}); each column got a `<column>_was_missing` "
                            "indicator.", "columns": per_column, "notes": notes}


def step_cap_outliers(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols, k = p["columns"], p["k"]
    _need(df, cols)
    new = df.copy()
    per_column: dict[str, dict[str, Any]] = {}
    notes = []
    for c in cols:
        s = _need_numeric(new, c)
        fences = _fences(s, k)
        if isinstance(fences, str):
            notes.append(f"{c} skipped: {fences}.")
            continue
        lower, upper = fences
        if pd.api.types.is_integer_dtype(s):
            # Keep whole numbers whole: cap to the nearest integers inside the fences.
            lower, upper = math.ceil(lower), math.floor(upper)
        capped = s.clip(lower=lower, upper=upper)
        entry = {"lower_fence": _r(lower), "upper_fence": _r(upper), "k": k,
                 "capped_low": int((s < lower).sum()), "capped_high": int((s > upper).sum()),
                 "before": _stats(s), "after": _stats(capped)}
        new[c] = capped
        per_column[str(c)] = entry
        b, a = entry["before"], entry["after"]
        notes.append(f"{c}: mean {_show(b['mean'])} -> {_show(a['mean'])}, standard deviation "
                     f"{_show(b['std'])} -> {_show(a['std'])}, maximum {_show(b['max'])} -> {_show(a['max'])}. "
                     "Totals and means computed from capped values understate the real ones.")
    parts = ", ".join(f"{c} {v['capped_low'] + v['capped_high']:,} (to {_show(v['lower_fence'])} .. "
                      f"{_show(v['upper_fence'])})" for c, v in per_column.items())
    return new, {"summary": f"Capped values outside Q1 - {k:g}*IQR .. Q3 + {k:g}*IQR" + (f": {parts}." if parts else
                                                                                          ": nothing capped."),
                 "columns": per_column, "notes": notes}


def step_flag_outliers(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols, k = p["columns"], p["k"]
    _need(df, cols)
    new = df.copy()
    per_column: dict[str, dict[str, Any]] = {}
    notes = []
    for c in cols:
        s = _need_numeric(new, c)
        fences = _fences(s, k)
        if isinstance(fences, str):
            notes.append(f"{c} not flagged: {fences}.")
            continue
        lower, upper = fences
        flag = pd.Series(pd.NA, index=s.index, dtype="boolean")
        known = s.notna()
        flag[known] = ((s[known] < lower) | (s[known] > upper)).to_numpy()
        _insert_after(new, c, f"{c}_outlier", flag)
        per_column[str(c)] = {"column_added": f"{c}_outlier", "flagged": int(flag.sum()),
                              "lower_fence": _r(lower), "upper_fence": _r(upper), "k": k}
    parts = ", ".join(f"{c} {v['flagged']:,}" for c, v in per_column.items())
    return new, {"summary": f"Flagged values outside Q1 - {k:g}*IQR .. Q3 + {k:g}*IQR in new `<column>_outlier` "
                            f"columns ({parts or 'none'}); the values themselves are unchanged.",
                 "columns": per_column, "notes": notes}


_UNPARSEABLE = (_v_choice("missing", "error"), "missing")
STEPS.update({
    "drop_exact_duplicates": Step(step_drop_exact_duplicates, "Remove rows identical in every column.",
                                  optional={"keep": (_v_choice("first", "last"), "first")}),
    "drop_duplicate_keys": Step(step_drop_duplicate_keys, "Keep one row per key (rows without a key are kept).",
                                required={"columns": _v_columns},
                                optional={"keep": (_v_choice("first", "last", "most_complete"), "first")}),
    "placeholders_to_missing": Step(step_placeholders_to_missing,
                                    "Turn placeholders (text such as '-', numbers such as 999, dates such as "
                                    "1900-01-01) into missing values. Default values: the standard text placeholders.",
                                    required={"columns": _v_columns}, optional={"values": (_v_values, None)}),
    "strip_whitespace": Step(step_strip_whitespace, "Trim spaces at the start and end of text values "
                             "(default: every text column).", optional={"columns": (_v_columns, None)}),
    "normalize_labels": Step(step_normalize_labels, "Replace labels by an explicit mapping, e.g. north -> North.",
                             required={"column": _v_column, "mapping": _v_labels}),
    "to_numeric": Step(step_to_numeric, "Convert numbers stored as text ($4.99, 12%, 1.234,50).",
                       required={"columns": _v_columns},
                       optional={"map": (_v_number_map, None), "decimal_comma": (_v_bool, None),
                                 "percent": (_v_choice("number", "fraction"), "number"), "unparseable": _UNPARSEABLE}),
    "to_datetime": Step(step_to_datetime, "Convert dates stored as text, with an explicit day/month order.",
                        required={"columns": _v_columns},
                        optional={"format": (_v_text, None), "day_first": (_v_bool, None),
                                  "unparseable": _UNPARSEABLE}),
    "drop_columns": Step(step_drop_columns, "Remove columns.", required={"columns": _v_columns}),
    "drop_rows_missing": Step(step_drop_rows_missing, "Remove rows with missing values in the given columns.",
                              required={"columns": _v_columns}, optional={"how": (_v_choice("any", "all"), "any")}),
    "fill_missing": Step(step_fill_missing, "Fill missing values (median, mode or a constant, optionally per group) "
                         "and add a <column>_was_missing column.",
                         required={"columns": _v_columns, "method": _v_choice("median", "mode", "constant")},
                         optional={"value": (_v_scalar, None), "by": (_v_column, None)}, adds_columns=True),
    "cap_outliers": Step(step_cap_outliers, "Cap values at the IQR fences (winsorise).",
                         required={"columns": _v_columns}, optional={"k": (_v_positive, 1.5)}),
    "flag_outliers": Step(step_flag_outliers, "Add a <column>_outlier true/false column; values are unchanged.",
                          required={"columns": _v_columns}, optional={"k": (_v_positive, 1.5)}, adds_columns=True),
})


# --------------------------------------------------------------------------
# Running a plan
# --------------------------------------------------------------------------


def apply_plan(df: pd.DataFrame, steps: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run validated steps on a copy of ``df``. Returns the cleaned table and one log entry per step."""
    current = df.copy()
    entries = []
    for n, step in enumerate(steps, start=1):
        name = step["step"]
        options = {k: v for k, v in step.items() if k not in COMMON_KEYS}
        try:
            after, details = STEPS[name].run(current, options)
        except PlanError as exc:
            raise PlanError(f"Step {n} ({name}): {exc}") from exc
        entries.append(_describe_step(n, step, current, after, details))
        current = after
    return current, entries


def _describe_step(n: int, step: dict[str, Any], before: pd.DataFrame, after: pd.DataFrame,
                   details: dict[str, Any]) -> dict[str, Any]:
    common = [c for c in after.columns if c in before.columns]
    kept = after.index
    changed: dict[str, int] = {}
    examples = []
    missing_change: dict[str, list[int]] = {}
    dtypes: dict[str, list[str]] = {}
    for c in common:
        a, b = before.loc[kept, c], after[c]
        if str(before[c].dtype) != str(b.dtype):
            dtypes[str(c)] = [str(before[c].dtype), str(b.dtype)]
        m_before, m_after = int(a.isna().sum()), int(b.isna().sum())
        if m_before != m_after:
            missing_change[str(c)] = [m_before, m_after]
        if a.equals(b):
            continue
        same = (a.isna() & b.isna()) | a.astype(object).eq(b.astype(object)).fillna(False).astype(bool)
        diff = ~same
        if diff.any():
            changed[str(c)] = int(diff.sum())
            seen = set()
            for i in a.index[diff.to_numpy()][:500]:
                pair = (_show(a[i]), _show(b[i]))
                if pair in seen:
                    continue
                seen.add(pair)
                if len(examples) < MAX_EXAMPLES * 2 and sum(e["column"] == str(c) for e in examples) < MAX_EXAMPLES:
                    examples.append({"row": _labels(pd.Index([i]))[0], "column": str(c),
                                     "before": pair[0], "after": pair[1]})
    details = dict(details)
    return {
        "n": n,
        "step": step["step"],
        "fixes": step["fixes"],
        "reason": step["reason"],
        "options": {k: v for k, v in step.items() if k not in COMMON_KEYS},
        "summary": details.pop("summary"),
        "notes": details.pop("notes", []),
        "rows_before": int(len(before)), "rows_after": int(len(after)),
        "columns_before": int(before.shape[1]), "columns_after": int(after.shape[1]),
        "missing_before": int(before.isna().sum().sum()), "missing_after": int(after.isna().sum().sum()),
        "columns_added": [str(c) for c in after.columns if c not in before.columns],
        "columns_removed": [str(c) for c in before.columns if c not in after.columns],
        "values_changed": int(sum(changed.values())),
        "values_changed_by_column": changed,
        "missing_by_column": missing_change,
        "dtype_changes": dtypes,
        "examples": examples,
        "details": details,
    }


def default_output(dataset: Path, base: Path = Path("eda_output"), now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    stem = re.sub(r"[^\w-]+", "_", dataset.stem).strip("_") or "data"
    ext = ".parquet" if dataset.suffix.lower() in (".parquet", ".pq") else ".csv"
    return base / f"{stem}_cleaned_{stamp}" / f"{stem}_cleaned{ext}"


def check_output(output: Path, dataset: Path) -> str:
    """Refuse anything that could overwrite data. Returns the output format."""
    fmt = OUTPUT_FORMATS.get(output.suffix.lower())
    if fmt is None:
        raise PlanError(f"The output must end in .csv or .parquet, not {output.suffix or '(nothing)'}: {output}")
    if output.resolve() == dataset.resolve():
        raise PlanError("The output is the input file. Cleaning always writes a new file; choose another --output.")
    folder = output.parent
    if folder.exists():
        if not folder.is_dir():
            raise PlanError(f"The output folder is a file: {folder}")
        if any(folder.iterdir()):
            raise PlanError(f"The output folder is not empty: {folder}. Choose a new or empty folder, so no "
                            "existing file can be overwritten.")
    if fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise PlanError("Writing Parquet needs the 'pyarrow' package (pip install pyarrow), "
                            "or write CSV instead.") from exc
    return fmt


def write_table(df: pd.DataFrame, output: Path, fmt: str) -> None:
    """Write the cleaned table. The same table always gives the same bytes."""
    if fmt == "csv":
        df.to_csv(output, index=False, encoding="utf-8", lineterminator="\n")
        return
    try:
        df.to_parquet(output, engine="pyarrow", index=False)
    except Exception as exc:  # pyarrow refuses e.g. a column of mixed numbers and text
        raise PlanError(f"Could not write Parquet: {exc}. Convert mixed columns first, or write CSV.") from exc


def run_cleaning(dataset: str | Path, plan: dict[str, Any], output: str | Path | None = None, apply: bool = False,
                 sheet: str | None = None, encoding: str | None = None, delimiter: str | None = None,
                 json_path: str | None = None, plan_file: str | Path | None = None,
                 earlier_log: dict[str, Any] | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Validate and run ``plan`` on ``dataset``. Writes files only when ``apply`` is true.

    Returns the log (the content of cleaning_log.json).
    """
    dataset = Path(dataset)
    steps = validate_plan(plan)  # before reading any data
    output = Path(output) if output else default_output(dataset, now=now)
    fmt = check_output(output, dataset)
    sha_before = file_sha256(dataset) if dataset.is_file() else None
    df = load_for_cleaning(dataset, sheet=sheet, encoding=encoding, delimiter=delimiter, json_path=json_path)
    cleaned, entries = apply_plan(df, steps)
    src = df.attrs["source"]
    read_options = {k: v for k, v in {"sheet": sheet, "encoding": encoding, "delimiter": delimiter,
                                      "json_path": json_path}.items() if v is not None}
    log: dict[str, Any] = {
        "tool": "clean_data.py",
        "mode": "applied" if apply else "dry_run",
        "created_at": (now or datetime.now()).isoformat(timespec="seconds"),
        "pandas_version": pd.__version__,
        "input": {
            "path": str(dataset), "file_name": dataset.name, "read_as": source_description(src),
            "read_options": read_options, "sha256": sha_before,
            "rows": int(len(df)), "columns": int(df.shape[1]),
            "kept_as_text": src.get("kept_as_text", []), "iso_date_columns": src.get("iso_date_columns", []),
        },
        "output": {"path": str(output), "format": fmt, "rows": int(len(cleaned)), "columns": int(cleaned.shape[1]),
                   "sha256": None, "written": False},
        "result": {
            "rows_before": int(len(df)), "rows_after": int(len(cleaned)),
            "columns_before": int(df.shape[1]), "columns_after": int(cleaned.shape[1]),
            "missing_before": int(df.isna().sum().sum()), "missing_after": int(cleaned.isna().sum().sum()),
            "columns_added": [str(c) for c in cleaned.columns if c not in df.columns],
            "columns_removed": [str(c) for c in df.columns if c not in cleaned.columns],
        },
        "steps": entries,
        "plan": plan,
        "plan_file": str(plan_file) if plan_file else None,
        "warnings": [],
    }
    if earlier_log is not None:
        earlier_sha = earlier_log.get("input", {}).get("sha256")
        if earlier_sha and earlier_sha != sha_before:
            log["warnings"].append("The plan comes from a cleaning log of a different input file (SHA-256 differs), "
                                   "so the result will not match that earlier output.")
    if apply:
        created = not output.parent.exists()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_table(cleaned, output, fmt)
        except PlanError:
            if output.exists():
                output.unlink()
            if created and not any(output.parent.iterdir()):
                output.parent.rmdir()
            raise
        log["output"].update({"sha256": file_sha256(output), "written": True})
    sha_after = file_sha256(dataset) if sha_before else None
    log["input"]["unchanged_after_run"] = sha_after == sha_before
    log["rebuild"] = _rebuild_command(log, output)
    if apply:
        log_json = output.parent / LOG_JSON
        log_json.write_text(json.dumps(to_jsonable(log), indent=2, ensure_ascii=False), encoding="utf-8")
        (output.parent / LOG_MD).write_text(render_log(log), encoding="utf-8")
    return to_jsonable(log)


def _rebuild_command(log: dict[str, Any], output: Path) -> str:
    def q(x: Any) -> str:
        x = str(x)
        return f'"{x}"' if re.search(r"[\s;&|]", x) or not x else x
    opts = "".join(f" --{k.replace('_', '-')} {q(v)}" for k, v in log["input"]["read_options"].items())
    target = output.parent.with_name(output.parent.name + "_rebuilt") / output.name
    return (f"python clean_data.py {q(log['input']['path'])} --plan {q(output.parent / LOG_JSON)}"
            f"{opts} --output {q(target)} --apply")


# --------------------------------------------------------------------------
# Output: console text and cleaning_log.md
# --------------------------------------------------------------------------


def format_text(log: dict[str, Any]) -> str:
    inp, out, res = log["input"], log["output"], log["result"]
    lines = []
    if log["mode"] == "dry_run":
        lines.append("DRY RUN: nothing was written. Run again with --apply to write the cleaned file.")
    lines.append(f"Input   {inp['file_name']}  {inp['rows']:,} rows x {inp['columns']} columns  ({inp['read_as']})")
    lines.append(f"Output  {out['path']}  ({out['format']}, {'written' if out['written'] else 'would be written'})")
    if inp["kept_as_text"]:
        lines.append(f"Kept as text so codes stay exact (leading zeros, long IDs): {', '.join(inp['kept_as_text'])}")
    lines.append(ROW_NUMBERS)
    for e in log["steps"]:
        lines += ["", f"Step {e['n']}  {e['step']}" + (f"  (fixes {e['fixes']})" if e["fixes"] else "")]
        lines.append(f"  {e['summary'].replace('`', '')}")
        lines.append(f"  rows {e['rows_before']:,} -> {e['rows_after']:,}; columns {e['columns_before']} -> "
                     f"{e['columns_after']}; missing cells {e['missing_before']:,} -> {e['missing_after']:,}; "
                     f"values changed {e['values_changed']:,}")
        for x in e["examples"][:MAX_EXAMPLES]:
            lines.append(f"  e.g. row {x['row']} {x['column']}: {x['before']} -> {x['after']}")
        for row_list in ("removed_rows",):
            if e["details"].get(row_list):
                more = " ..." if e["details"].get("rows_removed", 0) > len(e["details"][row_list]) else ""
                lines.append(f"  removed rows: {', '.join(map(str, e['details'][row_list]))}{more}")
        for k in e["details"].get("key_examples", [])[:3]:
            lines.append(f"  key {k['key']}: kept row {', '.join(map(str, k['kept']))}, "
                         f"dropped {', '.join(map(str, k['dropped'])) or '-'}")
        for note in e["notes"]:
            lines.append(f"  ! {note.replace('`', '')}")
    lines += ["", f"Result: rows {res['rows_before']:,} -> {res['rows_after']:,}, columns {res['columns_before']} -> "
                  f"{res['columns_after']}, missing cells {res['missing_before']:,} -> {res['missing_after']:,}"]
    if res["columns_removed"]:
        lines.append(f"  removed columns: {', '.join(res['columns_removed'])}")
    if res["columns_added"]:
        lines.append(f"  added columns: {', '.join(res['columns_added'])}")
    for w in log["warnings"]:
        lines.append(f"Warning: {w}")
    if out["written"]:
        lines.append(f"Wrote {out['path']} and {LOG_MD} / {LOG_JSON} next to it (sha256 {out['sha256'][:12]}...).")
    return "\n".join(lines)


def render_log(log: dict[str, Any]) -> str:
    inp, out, res = log["input"], log["output"], log["result"]
    lines = [f"# Cleaning log: {inp['file_name']}", "",
             f"Created {log['created_at']} by `clean_data.py` (pandas {log['pandas_version']}). The input file was "
             "only read" + (" and is unchanged (same SHA-256 before and after)." if inp["unchanged_after_run"]
                            else "; **its SHA-256 changed during the run: something else wrote to it.**"), "",
             "| | File | Rows | Columns | Missing cells | SHA-256 |", "|---|---|---|---|---|---|",
             f"| Input | `{inp['path']}` | {res['rows_before']:,} | {res['columns_before']} | {res['missing_before']:,} "
             f"| `{inp['sha256']}` |",
             f"| Output | `{out['path']}` | {res['rows_after']:,} | {res['columns_after']} | {res['missing_after']:,} "
             f"| `{out['sha256']}` |", ""]
    if inp["kept_as_text"]:
        lines += [f"Kept as text so codes stay exact (leading zeros, long IDs): "
                  f"{', '.join(f'`{c}`' for c in inp['kept_as_text'])}.", ""]
    if res["columns_removed"]:
        lines.append(f"- **Columns removed:** {', '.join(f'`{c}`' for c in res['columns_removed'])}")
    if res["columns_added"]:
        lines.append(f"- **Columns added:** {', '.join(f'`{c}`' for c in res['columns_added'])}")
    for w in log["warnings"]:
        lines.append(f"- **Warning:** {w}")
    lines += ["", "## Steps", "", ROW_NUMBERS, "", "| # | Step | Fixes | Rows | Missing cells | Values changed |",
              "|---|---|---|---|---|---|"]
    for e in log["steps"]:
        lines.append(f"| {e['n']} | `{e['step']}` | {e['fixes'] or '-'} | {e['rows_before']:,} → {e['rows_after']:,} "
                     f"| {e['missing_before']:,} → {e['missing_after']:,} | {e['values_changed']:,} |")
    for e in log["steps"]:
        lines += ["", f"### {e['n']}. `{e['step']}`", "", e["summary"]]
        if e["reason"]:
            lines += ["", f"*Reason:* {e['reason']}"]
        extra = []
        if e["values_changed_by_column"]:
            extra.append("Values changed per column: " + ", ".join(
                f"`{c}` {n:,}" for c, n in e["values_changed_by_column"].items()))
        if e["missing_by_column"]:
            extra.append("Missing values per column: " + ", ".join(
                f"`{c}` {a:,} → {b:,}" for c, (a, b) in e["missing_by_column"].items()))
        if e["dtype_changes"]:
            extra.append("Types: " + ", ".join(f"`{c}` {a} → {b}" for c, (a, b) in e["dtype_changes"].items()))
        if e["details"].get("removed_rows"):
            more = " ..." if e["details"].get("rows_removed", 0) > len(e["details"]["removed_rows"]) else ""
            extra.append("Rows removed: " + ", ".join(map(str, e["details"]["removed_rows"])) + more)
        for k in e["details"].get("key_examples", []):
            extra.append(f"Key {k['key']}: kept row {', '.join(map(str, k['kept']))}, dropped "
                         f"{', '.join(map(str, k['dropped'])) or '-'}")
        if extra:
            lines += [""] + [f"- {x}" for x in extra]
        if e["examples"]:
            lines += ["", "| Row | Column | Before | After |", "|---|---|---|---|"]
            lines += [f"| {x['row']} | `{x['column']}` | {x['before']} | {x['after']} |" for x in e["examples"]]
        if e["notes"]:
            lines += [""] + [f"> {n}" for n in e["notes"]]
    lines += ["", "## Rebuild", "", "The plan below is also stored in `cleaning_log.json`. To rebuild the same file "
              "from the same input (`clean_data.py` is in the skill's `scripts/` folder; the output must go to a new "
              "folder):", "", "```bash", log["rebuild"], "```", "",
              "## Plan", "", "```json", json.dumps(log["plan"], indent=2, ensure_ascii=False), "```", ""]
    return "\n".join(lines)


def list_steps() -> str:
    lines = []
    for name, spec in STEPS.items():
        req = ", ".join(spec.required) or "-"
        opt = ", ".join(f"{k}={json.dumps(d)}" for k, (_, d) in spec.optional.items()) or "-"
        lines += [f"{name}: {spec.doc}", f"    required: {req}; optional: {opt}",
                  f"    fixes: {', '.join(spec.fixes(name))}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply approved cleaning steps to a copy of a dataset (dry run by default).")
    p.add_argument("path", nargs="?", help=DATASET_HELP)
    p.add_argument("--plan", help="plan.json with the steps, or a cleaning_log.json to rebuild an earlier result")
    p.add_argument("--output", help="cleaned file, .csv or .parquet, in a new or empty folder "
                                    "(default: eda_output/<dataset>_cleaned_<timestamp>/<dataset>_cleaned.csv)")
    p.add_argument("--apply", action="store_true", help="write the cleaned file and logs (default: dry run only)")
    p.add_argument("--sheet", help="Excel sheet name (default: first sheet)")
    p.add_argument("--json-path", help=JSON_PATH_HELP)
    p.add_argument("--encoding", help=ENCODING_HELP)
    p.add_argument("--delimiter", help="CSV delimiter (default: detected automatically)")
    p.add_argument("--json", action="store_true", help="print the log as JSON instead of text")
    p.add_argument("--list-steps", action="store_true", help="list the available steps and their options")
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_steps:
        print(list_steps())
        return 0
    if not args.path or not args.plan:
        parser.error("a dataset and --plan are required (or use --list-steps)")
    delimiter = args.delimiter.encode().decode("unicode_escape") if args.delimiter else None
    try:
        plan, earlier = read_plan(args.plan)
        log = run_cleaning(args.path, plan, output=args.output, apply=args.apply, sheet=args.sheet,
                           encoding=args.encoding, delimiter=delimiter, json_path=args.json_path,
                           plan_file=args.plan, earlier_log=earlier)
    except (PlanError, DatasetError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(log, indent=2, ensure_ascii=False) if args.json else format_text(log))
    if not log["input"]["unchanged_after_run"]:
        print("Error: the input file changed during the run. Check what else is writing to it.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
