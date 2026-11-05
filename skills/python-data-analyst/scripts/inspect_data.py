"""Load a CSV/TSV/Excel/Parquet/JSON dataset, describe it and check its quality (read-only).

Usage:
    python inspect_data.py data.csv
    python inspect_data.py data.xlsx --sheet "2025" --json
    python inspect_data.py orders.csv --key order_id
    python inspect_data.py export.json --json-path data.orders

The input file is only ever read, never written.
"""

from __future__ import annotations

import argparse
import csv
import decimal
import hashlib
import json
import math
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CSV_EXTENSIONS = {".csv", ".tsv", ".txt"}
EXCEL_EXTENSIONS = {".xlsx", ".xlsm"}
PARQUET_EXTENSIONS = {".parquet", ".pq"}
JSON_EXTENSIONS = {".json"}
JSONL_EXTENSIONS = {".jsonl", ".ndjson"}
SUPPORTED_EXTENSIONS = CSV_EXTENSIONS | EXCEL_EXTENSIONS | PARQUET_EXTENSIONS | JSON_EXTENSIONS | JSONL_EXTENSIONS
# Nested JSON objects become columns joined with this separator:
# {"customer": {"city": ...}} gives the column "customer.city".
JSON_SEP = "."
# How deep to look for lists of records when a JSON file is a single object.
JSON_SEARCH_DEPTH = 4

# Tried in this order. "utf-8-sig" is UTF-8 that also strips a byte-order mark
# (Excel adds one when it saves "CSV UTF-8"). cp1252 is the classic Windows
# encoding used by older Excel CSV exports.
DEFAULT_ENCODINGS = ("utf-8-sig", "cp1252")
CANDIDATE_DELIMITERS = ",;\t|"
SNIFF_BYTES = 64 * 1024

# Column-classification thresholds. A string column is "text" (free text or an
# identifier) rather than "categorical" when values are long or nearly unique.
TEXT_MIN_UNIQUE = 50
TEXT_UNIQUE_RATIO = 0.5
TEXT_MEAN_LENGTH = 50

COLUMN_KINDS = ("numeric", "boolean", "datetime", "categorical", "text", "empty")

# Placeholders that often mean "missing" but survive as text (compared after
# stripping spaces and lower-casing). "" covers whitespace-only cells.
DISGUISED_MISSING = frozenset(
    {"", "-", "--", "---", "?", "??", ".", "n/a", "na", "nan", "null", "none", "nil",
     "unknown", "missing", "not available", "#n/a", "tbd"}
)
# A text column is flagged as "numbers/dates stored as text" when at least this
# share of its non-placeholder values parse as numbers/dates.
TYPE_MATCH_RATIO = 0.8
# IQR outliers: below IQR_MIN_VALUES the check is skipped; below
# IQR_SMALL_SAMPLE the result carries a "small sample" note.
IQR_MIN_VALUES = 5
IQR_SMALL_SAMPLE = 20
# Date detection parses each distinct value; above this many it uses a sample.
DATE_SAMPLE_UNIQUE = 5000
# Numbers often used to mean "unknown" or "not applicable" in exports. Reported
# only when they are also IQR outliers, so a real 99 in a 0-100 score is left alone.
CODE_LIKE_VALUES = (99, 999, 9999, 99999, -1, -9, -99, -999, -9999)
# Text files can't store a date type, so a CSV or JSON column whose every value
# is a strict ISO date ("2025-01-31", "2025-01-31 14:05:00") is the best a text
# file can do: it's read as dates, not reported as "dates stored as text".
ISO_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?)?$"


class DatasetError(ValueError):
    """A dataset could not be loaded. The message is meant for the user."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_dataset(
    path: str | Path,
    sheet: str | None = None,
    encoding: str | None = None,
    delimiter: str | None = None,
    json_path: str | None = None,
) -> pd.DataFrame:
    """Read a CSV, TSV, Excel, Parquet, JSON or JSON Lines file into a DataFrame.

    How the file was read (encoding, delimiter, sheet, ...) is stored in
    ``df.attrs["source"]`` so the report can say so. Lists and objects inside
    cells (JSON, Parquet) are kept as JSON text so they can be counted; the
    columns that hold them are listed in ``source["nested_columns"]``.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"File not found: {path}")
    if path.is_dir():
        raise DatasetError(f"Expected a file but got a folder: {path}")

    ext = path.suffix.lower()
    if ext == ".xls":
        raise DatasetError(
            "Old-style .xls files are not supported. Open the file in Excel and "
            "save it as .xlsx or .csv, then try again."
        )
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DatasetError(f"Unsupported file type '{ext or '(none)'}'. Supported: {supported}")
    if path.stat().st_size == 0:
        raise DatasetError(f"The file is empty (0 bytes): {path}")

    if sheet is not None and ext not in EXCEL_EXTENSIONS:
        raise DatasetError("--sheet only applies to Excel files.")
    if json_path is not None and ext not in JSON_EXTENSIONS:
        raise DatasetError("--json-path only applies to .json files.")

    if ext in EXCEL_EXTENSIONS:
        df, source = _read_excel(path, sheet)
    elif ext in PARQUET_EXTENSIONS:
        df, source = _read_parquet(path)
    elif ext in JSON_EXTENSIONS | JSONL_EXTENSIONS:
        df, source = _read_json(path, encoding, json_path, lines=ext in JSONL_EXTENSIONS)
    else:
        df, source = _read_csv(path, encoding, delimiter)

    if df.shape[1] == 0:
        raise DatasetError(f"No columns found in {path.name}.")
    if df.shape[0] == 0:
        raise DatasetError(f"{path.name} has column headers but no data rows.")

    nested = _freeze_nested(df)
    if nested:
        source["nested_columns"] = nested
    if source["format"] in ("csv", "json"):
        iso = parse_iso_date_columns(df)
        if iso:
            source["iso_date_columns"] = iso
    source.update({"path": str(path), "file_name": path.name, "file_bytes": path.stat().st_size})
    df.attrs["source"] = source
    return df


def _read_excel(path: Path, sheet: str | None) -> tuple[pd.DataFrame, dict]:
    try:
        workbook = pd.ExcelFile(path, engine="openpyxl")
    except ImportError as exc:
        raise DatasetError(
            "Reading Excel files needs the 'openpyxl' package. "
            "Install it with: pip install openpyxl"
        ) from exc
    except Exception as exc:  # corrupt or not really an Excel file
        raise DatasetError(f"Could not open {path.name} as an Excel workbook: {exc}") from exc

    with workbook:
        sheets = list(workbook.sheet_names)
        if sheet is None:
            sheet = sheets[0]
        elif sheet not in sheets:
            available = ", ".join(repr(s) for s in sheets)
            raise DatasetError(f"Sheet {sheet!r} not found in {path.name}. Available sheets: {available}")
        df = pd.read_excel(workbook, sheet_name=sheet)
    return df, {"format": "excel", "sheet": sheet, "available_sheets": sheets}


def _read_csv(path: Path, encoding: str | None, delimiter: str | None) -> tuple[pd.DataFrame, dict]:
    encodings = (encoding,) if encoding else DEFAULT_ENCODINGS
    errors: list[str] = []
    for enc in encodings:
        try:
            sep = delimiter or _sniff_delimiter(path, enc)
            df = pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError as exc:
            errors.append(f"{enc}: {exc.reason} at byte {exc.start}")
            continue
        except LookupError as exc:  # unknown encoding name given by the user
            raise DatasetError(f"Unknown encoding {enc!r}.") from exc
        except pd.errors.EmptyDataError as exc:
            raise DatasetError(f"{path.name} contains no data.") from exc
        except pd.errors.ParserError as exc:
            raise DatasetError(
                f"Could not parse {path.name} with delimiter {sep!r}: {exc}. "
                "Try passing --delimiter explicitly."
            ) from exc
        return df, {"format": "csv", "encoding": enc, "delimiter": sep}

    raise DatasetError(
        f"Could not decode {path.name} with any of: {', '.join(encodings)} "
        f"({'; '.join(errors)}). Pass the right one with --encoding, e.g. --encoding latin-1."
    )


def _read_parquet(path: Path) -> tuple[pd.DataFrame, dict]:
    try:
        import pyarrow  # noqa: F401  (pandas needs it to read Parquet)
    except ImportError as exc:
        raise DatasetError(
            "Reading Parquet files needs the 'pyarrow' package. "
            "Install it with: pip install pyarrow"
        ) from exc
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:  # corrupt or not really a Parquet file
        raise DatasetError(f"Could not read {path.name} as a Parquet file: {exc}") from exc
    # Parquet DECIMAL columns arrive as Python Decimal objects, which pandas
    # treats as text. Analyse them as floats (in memory only).
    decimals = [str(c) for c in df.columns if _all_decimals(df[c])]
    for c in decimals:
        df[c] = df[c].astype(float)
    source: dict[str, Any] = {"format": "parquet", "stored_types": True}
    if decimals:
        source["decimal_columns"] = decimals
    return df, source


def _all_decimals(s: pd.Series) -> bool:
    non_null = s.dropna()
    return (s.dtype == object and not non_null.empty
            and bool(non_null.map(lambda v: isinstance(v, decimal.Decimal)).all()))


def _read_json(path: Path, encoding: str | None, json_path: str | None, lines: bool) -> tuple[pd.DataFrame, dict]:
    enc = encoding or "utf-8-sig"
    try:
        text = path.read_text(encoding=enc)
    except LookupError as exc:
        raise DatasetError(f"Unknown encoding {enc!r}.") from exc
    except UnicodeDecodeError as exc:
        raise DatasetError(
            f"Could not decode {path.name} as {enc} ({exc.reason} at byte {exc.start}). "
            "Pass the right one with --encoding."
        ) from exc

    layout = "json_lines" if lines else "records"
    if lines:
        records = _parse_json_lines(text, path.name)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            if exc.msg != "Extra data" or json_path:
                raise DatasetError(f"{path.name} is not valid JSON: {exc}") from exc
            # Several JSON values one after another: JSON Lines saved as .json.
            records, layout = _parse_json_lines(text, path.name), "json_lines"
        else:
            if json_path:
                data = _follow_json_path(data, json_path, path.name)
            records = _as_records(data, path.name, json_path)

    if not records:
        raise DatasetError(f"{path.name} contains no records.")
    df = pd.json_normalize(records, sep=JSON_SEP)
    top_keys = {k for r in records for k in r}
    flattened = [str(c) for c in df.columns if c not in top_keys]
    source: dict[str, Any] = {"format": "json", "layout": layout, "encoding": enc}
    if json_path:
        source["json_path"] = json_path
    if flattened:
        source["flattened_columns"] = flattened
    return df, source


def _parse_json_lines(text: str, name: str) -> list[dict]:
    records = []
    decoder = json.JSONDecoder()
    for n, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            value, end = decoder.raw_decode(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Line {n} of {name} is not valid JSON: {exc.msg}.") from exc
        if line[end:].strip():
            raise DatasetError(f"Line {n} of {name} holds more than one JSON value; "
                               "JSON Lines needs exactly one record per line.")
        if not isinstance(value, dict):
            raise DatasetError(f"Line {n} of {name} is a {_json_type(value)}, not a record. "
                               'JSON Lines needs one object such as {"id": 1, ...} per line.')
        records.append(value)
    return records


def _follow_json_path(data: Any, json_path: str, name: str) -> Any:
    here: list[str] = []
    for key in json_path.split("."):
        where = ".".join(here) or "the top level"
        if not isinstance(data, dict):
            raise DatasetError(f"--json-path {json_path}: {where} of {name} is a {_json_type(data)}, "
                               f"so it has no key {key!r}.")
        if key not in data:
            keys = ", ".join(repr(k) for k in list(data)[:20]) or "(none)"
            raise DatasetError(f"--json-path {json_path}: key {key!r} not found at {where} of {name}. "
                               f"Keys there: {keys}")
        data = data[key]
        here.append(key)
    return data


def _as_records(data: Any, name: str, json_path: str | None) -> list[dict]:
    """Check that ``data`` is a list of objects (one per row) and return it."""
    where = f"--json-path {json_path}" if json_path else name
    if isinstance(data, list):
        bad = [v for v in data if not isinstance(v, dict)]
        if not bad:
            return data
        if len(bad) == len(data):
            kinds = " and ".join(sorted({_json_type(v) for v in data}))
            raise DatasetError(f"{where} is a list of {kinds} values, not of records. A table needs a "
                               'list of objects such as [{"id": 1, "city": "Rome"}, ...].')
        raise DatasetError(f"{where} mixes records with other values: {len(bad)} of {len(data)} items "
                           f"are not objects (first one: {json.dumps(bad[0])[:80]}).")
    if isinstance(data, dict):
        keys = ", ".join(repr(k) for k in list(data)[:10])
        msg = f"{where} is a single JSON object ({'keys: ' + keys if keys else 'empty'}), not a table of records."
        found = _find_record_lists(data, json_path + "." if json_path else "")
        if found:
            places = ", ".join(f"{p} ({n:,} records)" for p, n in found[:5])
            msg += f" Lists of records found at: {places}. Choose one with --json-path {found[0][0]}"
        raise DatasetError(msg)
    raise DatasetError(f"{where} is a single {_json_type(data)} value, not a table of records.")


def _find_record_lists(data: dict, prefix: str = "", depth: int = 0) -> list[tuple[str, int]]:
    """Paths (a.b.c) to non-empty lists of objects inside ``data``, largest first."""
    found: list[tuple[str, int]] = []
    if depth >= JSON_SEARCH_DEPTH:
        return found
    for key, value in data.items():
        if "." in str(key):  # --json-path couldn't address it
            continue
        here = f"{prefix}{key}"
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            found.append((here, len(value)))
        elif isinstance(value, dict):
            found += _find_record_lists(value, here + ".", depth + 1)
    return sorted(found, key=lambda t: -t[1])


def _json_type(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "list"
    if isinstance(value, bool):
        return "true/false"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "text"
    return "null"


def _freeze_nested(df: pd.DataFrame) -> dict[str, int]:
    """Replace lists, arrays and dicts in cells with their JSON text, in place.

    pandas can't count, compare or deduplicate such values (they aren't
    hashable), so every summary would fail on them. As text they can still be
    counted, and the quality checks report the column as a type problem.
    Returns {column: number of nested values}.
    """
    nested: dict[str, int] = {}
    for c in df.columns:
        if df[c].dtype != object:
            continue
        mask = df[c].map(lambda v: isinstance(v, (list, tuple, dict, np.ndarray))).astype(bool)
        if mask.any():
            df[c] = df[c].where(~mask, df[c][mask].map(_to_json_text))
            nested[str(c)] = int(mask.sum())
    return nested


def _to_json_text(value: Any) -> str:
    def plain(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return [plain(x) for x in v.tolist()]
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        return v
    return json.dumps(plain(value), ensure_ascii=False, default=str)


def parse_iso_date_columns(df: pd.DataFrame) -> list[str]:
    """Turn text columns that hold only strict ISO dates into datetimes, in place.

    A column with even one other value stays text, so the quality check still
    reports it. Returns the names of the converted columns.
    """
    converted = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        strings = _string_values(s)
        if strings.empty or len(strings) != int(s.notna().sum()):
            continue
        if not strings.str.match(ISO_DATE_PATTERN).all():
            continue
        parsed = pd.to_datetime(s, format="ISO8601", errors="coerce")
        if int(parsed.notna().sum()) == len(strings):  # e.g. 2025-02-30 stays text
            df[c] = parsed
            converted.append(str(c))
    return converted


def file_sha256(path: str | Path) -> str:
    """Fingerprint of a file, read in 1 MB chunks so large files don't fill memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sniff_delimiter(path: Path, encoding: str) -> str:
    """Guess the delimiter from the start of the file; fall back on the extension."""
    with open(path, "rb") as fh:
        raw = fh.read(SNIFF_BYTES)
    # Cutting at a fixed byte count may split a multi-byte character, so ignore
    # decode errors here. The full read below still decodes strictly.
    sample = raw.decode(encoding, errors="ignore")
    lines = sample.splitlines()
    if len(raw) == SNIFF_BYTES and len(lines) > 1:
        sample = "\n".join(lines[:-1])  # drop the possibly incomplete last line
    try:
        return csv.Sniffer().sniff(sample, delimiters=CANDIDATE_DELIMITERS).delimiter
    except csv.Error:
        return "\t" if path.suffix.lower() == ".tsv" else ","


# --------------------------------------------------------------------------
# Describing
# --------------------------------------------------------------------------


def classify_columns(df: pd.DataFrame) -> dict[str, str]:
    """Assign each column one of COLUMN_KINDS.

    This looks at the dtype pandas inferred. Values stored with the wrong type
    (numbers or dates kept as text) are reported by the quality checks, not here.
    """
    kinds: dict[str, str] = {}
    for name in df.columns:
        s = df[name]
        non_null = s.dropna()
        if non_null.empty:
            kind = "empty"
        elif pd.api.types.is_bool_dtype(s) or _all_python_bools(non_null):
            kind = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        elif isinstance(s.dtype, pd.CategoricalDtype):
            kind = "categorical"
        else:
            kind = "text" if _looks_like_text(non_null) else "categorical"
        kinds[str(name)] = kind
    return kinds


def _all_python_bools(non_null: pd.Series) -> bool:
    return non_null.dtype == object and non_null.map(lambda v: isinstance(v, (bool, np.bool_))).all()


def _looks_like_text(non_null: pd.Series) -> bool:
    as_str = non_null.astype(str)
    n_unique = as_str.nunique()
    if as_str.str.len().mean() > TEXT_MEAN_LENGTH:
        return True
    return n_unique > TEXT_MIN_UNIQUE and n_unique / len(as_str) > TEXT_UNIQUE_RATIO


def get_overview(df: pd.DataFrame, kinds: dict[str, str] | None = None) -> dict[str, Any]:
    """Shape, memory and a per-column summary of dtype, kind and unique count."""
    kinds = kinds or classify_columns(df)
    memory = int(df.memory_usage(deep=True).sum())
    columns = []
    for name in df.columns:
        s = df[name]
        columns.append(
            {
                "name": str(name),
                "dtype": str(s.dtype),
                "kind": kinds[str(name)],
                "non_null": int(s.notna().sum()),
                "unique": int(s.nunique(dropna=True)),
            }
        )
    kind_counts = {k: sum(1 for v in kinds.values() if v == k) for k in COLUMN_KINDS}
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_bytes": memory,
        "memory_human": _human_bytes(memory),
        "kind_counts": {k: v for k, v in kind_counts.items() if v},
        "column_info": columns,
    }


def descriptive_stats(
    df: pd.DataFrame, kinds: dict[str, str] | None = None, top_n: int = 5
) -> dict[str, dict[str, Any]]:
    """Summary statistics per column, chosen by column kind."""
    kinds = kinds or classify_columns(df)
    stats: dict[str, dict[str, Any]] = {}
    for name in df.columns:
        key = str(name)
        s = df[name]
        kind = kinds[key]
        if kind == "numeric":
            stats[key] = _numeric_stats(s)
        elif kind == "boolean":
            stats[key] = _boolean_stats(s)
        elif kind == "datetime":
            stats[key] = _datetime_stats(s)
        elif kind in ("categorical", "text"):
            stats[key] = _categorical_stats(s, top_n, kind)
        else:
            stats[key] = {"kind": "empty", "count": 0}
    return stats


def _numeric_stats(s: pd.Series) -> dict[str, Any]:
    v = s.dropna().astype(float)
    n = len(v)
    # pandas' skew() is the adjusted Fisher-Pearson coefficient; it needs >= 3 values.
    # A constant column has no spread, so skewness is undefined. pandas 3 returns
    # NaN for it, pandas 2 returns 0.0 ("symmetric"), so decide it ourselves.
    skew = float(v.skew()) if n >= 3 and v.nunique() > 1 else None
    if skew is not None and not math.isfinite(skew):
        skew = None
    return {
        "kind": "numeric",
        "count": n,
        "mean": _f(v.mean()),
        "std": _f(v.std()),  # sample standard deviation (n - 1)
        "min": _f(v.min()),
        "q1": _f(v.quantile(0.25)),
        "median": _f(v.median()),
        "q3": _f(v.quantile(0.75)),
        "max": _f(v.max()),
        "skewness": skew,
        "skew_label": _skew_label(skew),
    }


def _skew_label(skew: float | None) -> str | None:
    """Common rule of thumb: |skew| < 0.5 roughly symmetric, 0.5–1 moderate, > 1 high."""
    if skew is None:
        return None
    side = "right" if skew > 0 else "left"
    if abs(skew) < 0.5:
        return "approximately symmetric"
    if abs(skew) < 1:
        return f"moderately {side}-skewed"
    return f"highly {side}-skewed"


def _boolean_stats(s: pd.Series) -> dict[str, Any]:
    v = s.dropna().astype(bool)
    n = len(v)
    n_true = int(v.sum())
    return {
        "kind": "boolean",
        "count": n,
        "true": n_true,
        "false": n - n_true,
        "true_pct": round(100 * n_true / n, 2) if n else None,
    }


def _datetime_stats(s: pd.Series) -> dict[str, Any]:
    v = s.dropna()
    start, end = v.min(), v.max()
    return {
        "kind": "datetime",
        "count": int(len(v)),
        "unique": int(v.nunique()),
        "min": start.isoformat(),
        "max": end.isoformat(),
        "range_days": round((end - start).total_seconds() / 86400, 2),
    }


def _categorical_stats(s: pd.Series, top_n: int, kind: str) -> dict[str, Any]:
    v = s.dropna()
    n = len(v)
    counts = v.astype(str).value_counts()
    top = [
        {"value": str(val), "count": int(c), "pct": round(100 * c / n, 2)}
        for val, c in counts.head(top_n).items()
    ]
    return {
        "kind": kind,
        "count": n,
        "unique": int(counts.size),
        "top_values": top,
        "other_count": int(n - sum(t["count"] for t in top)),
    }


# --------------------------------------------------------------------------
# Data quality (detects problems, never changes the data)
# --------------------------------------------------------------------------


def missing_values(df: pd.DataFrame, extra_tokens: Iterable[str] = ()) -> dict[str, Any]:
    """Real missing values (NaN/None) and "disguised" ones such as "-" or "unknown".

    pandas already turns "", "NA", "N/A", "null", "NaN", "None" etc. into NaN
    while reading a CSV. Disguised values are the placeholders that survive as
    ordinary text. They are counted separately and NOT included in "missing",
    because whether "unknown" means missing is a decision for the analyst.
    """
    tokens = DISGUISED_MISSING | {t.strip().casefold() for t in extra_tokens}
    n_rows = len(df)
    columns = []
    for name in df.columns:
        s = df[name]
        missing = int(s.isna().sum())
        strings = _string_values(s)
        disguised_mask = strings.str.strip().str.casefold().isin(tokens)
        disguised = strings[disguised_mask]
        columns.append(
            {
                "name": str(name),
                "missing": missing,
                "missing_pct": _pct(missing, n_rows),
                "disguised": int(disguised.size),
                "disguised_pct": _pct(disguised.size, n_rows),
                "disguised_values": {repr(k): int(v) for k, v in disguised.value_counts().items()},
            }
        )
    total_cells = int(df.size)
    total_missing = int(df.isna().sum().sum())
    # One all-empty column makes every row "incomplete", which hides how
    # complete the rest of the data is. So also count rows ignoring such columns.
    used = df.loc[:, df.notna().any()]
    gaps_in_used = int(used.isna().any(axis=1).sum()) if used.shape[1] else 0
    return {
        "total_cells": total_cells,
        "total_missing": total_missing,
        "total_missing_pct": _pct(total_missing, total_cells),
        "total_disguised": sum(c["disguised"] for c in columns),
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "rows_with_missing_pct": _pct(int(df.isna().any(axis=1).sum()), n_rows),
        "complete_rows": int(df.notna().all(axis=1).sum()),
        "rows_with_missing_ignoring_empty_columns": gaps_in_used,
        "rows_with_missing_ignoring_empty_columns_pct": _pct(gaps_in_used, n_rows),
        "complete_rows_ignoring_empty_columns": n_rows - gaps_in_used,
        "columns": columns,
    }


def find_duplicates(df: pd.DataFrame, subset: list[str] | None = None, max_examples: int = 5) -> dict[str, Any]:
    """Count duplicate rows, either fully identical or identical on key columns.

    ``duplicate_rows`` counts the *extra* copies: three identical rows give 2,
    because removing duplicates would keep one of them.
    """
    if subset:
        unknown = [c for c in subset if c not in df.columns]
        if unknown:
            raise DatasetError(
                f"Key column(s) not found: {', '.join(unknown)}. "
                f"Available: {', '.join(map(str, df.columns))}"
            )
    extra = df.duplicated(subset=subset, keep="first")
    involved = df.duplicated(subset=subset, keep=False)
    groups = df[involved]
    examples = []
    if extra.any():
        keys = subset or list(df.columns)
        for _, grp in list(groups.groupby(keys, dropna=False, sort=False))[:max_examples]:
            examples.append({"rows": [_index_label(i) for i in grp.index], "count": int(len(grp))})
    return {
        "subset": subset,
        "mode": "key columns" if subset else "exact (all columns)",
        "duplicate_rows": int(extra.sum()),
        "duplicate_pct": _pct(int(extra.sum()), len(df)),
        "rows_involved": int(involved.sum()),
        "groups": int(groups.drop_duplicates(subset=subset).shape[0]),
        "examples": examples,
    }


def detect_type_issues(
    df: pd.DataFrame, kinds: dict[str, str] | None = None, max_examples: int = 5
) -> list[dict[str, Any]]:
    """Values stored with the wrong type, and inconsistent text labels.

    Issue types:
      numbers_as_text      most values are numbers, but the column is text
      dates_as_text        most values are dates, but the column is text
      mixed_types          one column holds several Python types (common in Excel)
      inconsistent_labels  the same label written differently ("North" / "north ")
      untrimmed_whitespace values with spaces at the start or end
      nested_values        lists or objects inside cells (JSON, Parquet), kept as JSON text
    """
    kinds = kinds or classify_columns(df)
    nested = df.attrs.get("source", {}).get("nested_columns", {})
    issues: list[dict[str, Any]] = []
    for name in df.columns:
        key = str(name)
        s = df[name]
        if kinds[key] not in ("categorical", "text"):
            continue
        if key in nested:
            # The text checks (numbers, dates, labels) make no sense on JSON text.
            n, total = nested[key], int(s.notna().sum())
            examples = [v for v in s.dropna().unique() if isinstance(v, str) and v[:1] in "[{"]
            issues.append(
                {"column": key, "issue": "nested_values", "count": n,
                 "detail": f"{n} of {total} values are lists or objects, kept as JSON text; they can be "
                           "counted but not summarised as numbers or categories",
                 "examples": [repr(v if len(v) <= 60 else v[:57] + "...") for v in examples[:max_examples]]}
            )
            continue

        mixed = _mixed_types(s)
        if mixed:
            issues.append(
                {"column": key, "issue": "mixed_types", "count": int(s.notna().sum()),
                 "detail": "Python types found: " + ", ".join(f"{t} {n}" for t, n in mixed.items()),
                 "types": mixed}
            )

        strings = _string_values(s)
        stripped = strings.str.strip()
        candidates = stripped[~stripped.str.casefold().isin(DISGUISED_MISSING)]

        numeric = _numeric_like(candidates, max_examples)
        if numeric:
            issues.append({"column": key, **numeric})
        else:
            dates = _date_like(candidates, max_examples)
            if dates:
                issues.append({"column": key, **dates})

        labels = _inconsistent_labels(candidates, max_examples)
        if labels:
            issues.append({"column": key, **labels})

        untrimmed = strings[(strings != stripped) & (stripped != "")]
        if not untrimmed.empty:
            issues.append(
                {"column": key, "issue": "untrimmed_whitespace", "count": int(untrimmed.size),
                 "detail": f"{untrimmed.size} value(s) have leading or trailing spaces",
                 "examples": [repr(v) for v in untrimmed.unique()[:max_examples]]}
            )
    return issues


def iqr_outliers(df: pd.DataFrame, kinds: dict[str, str] | None = None, k: float = 1.5) -> dict[str, dict[str, Any]]:
    """Flag values outside [Q1 - k*IQR, Q3 + k*IQR] in each numeric column.

    Values outside the fences are *unusual*, not necessarily *wrong*. In skewed
    data (incomes, order values) many legitimate values fall outside them.
    """
    kinds = kinds or classify_columns(df)
    result: dict[str, dict[str, Any]] = {}
    for name in df.columns:
        key = str(name)
        if kinds[key] != "numeric":
            continue
        v = df[name].dropna().astype(float)
        n = len(v)
        if n < IQR_MIN_VALUES:
            result[key] = {"status": "skipped", "count": n,
                           "note": f"only {n} values; at least {IQR_MIN_VALUES} are needed for quartiles to mean anything"}
            continue
        q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            result[key] = {"status": "zero_iqr", "count": n, "q1": q1, "q3": q3, "iqr": 0.0,
                           "values_not_equal_to_q1": int((v != q1).sum()),
                           "note": "at least half the values are identical, so IQR is 0 and every other "
                                   "value would be flagged; the IQR method doesn't apply here"}
            continue
        lower, upper = q1 - k * iqr, q3 + k * iqr
        low, high = v[v < lower], v[v > upper]
        entry = {
            "status": "ok", "count": n, "q1": q1, "q3": q3, "iqr": iqr, "k": k,
            "lower_fence": lower, "upper_fence": upper,
            "n_low": int(low.size), "n_high": int(high.size),
            "n_outliers": int(low.size + high.size), "outlier_pct": _pct(low.size + high.size, n),
            "lowest_outliers": sorted(low.tolist())[:5], "highest_outliers": sorted(high.tolist())[-5:][::-1],
        }
        codes = {}
        for code in CODE_LIKE_VALUES:
            # Negative codes only count in columns that are otherwise non-negative.
            if (code > 0 and code > upper) or (code < 0 and code < lower and q1 >= 0):
                hits = int((v == code).sum())
                if hits:
                    codes[str(code)] = hits
        if codes:
            entry["code_like_values"] = codes
        if n < IQR_SMALL_SAMPLE:
            entry["note"] = f"small sample ({n} values): quartiles are unstable, treat flags as tentative"
        result[key] = entry
    return result


def data_quality(df: pd.DataFrame, kinds: dict[str, str] | None = None,
                 key_columns: list[str] | None = None) -> dict[str, Any]:
    kinds = kinds or classify_columns(df)
    quality = {
        "missing": missing_values(df),
        "duplicates": find_duplicates(df),
        "type_issues": detect_type_issues(df, kinds),
        "outliers": iqr_outliers(df, kinds),
    }
    if key_columns:
        quality["key_duplicates"] = find_duplicates(df, subset=key_columns)
    return quality


# --- quality helpers -------------------------------------------------------


def _string_values(s: pd.Series) -> pd.Series:
    """Non-null values that are Python strings (other types in object columns are skipped)."""
    non_null = s.dropna()
    if pd.api.types.is_numeric_dtype(non_null) or pd.api.types.is_datetime64_any_dtype(non_null):
        return pd.Series([], dtype=object)
    if non_null.dtype != object:  # pandas string dtype: every value is a str
        return non_null.astype(str)
    mask = non_null.map(lambda v: isinstance(v, str)).astype(bool)
    return non_null[mask].astype(str)


def _mixed_types(s: pd.Series) -> dict[str, int] | None:
    if s.dtype != object:
        return None
    types = s.dropna().map(lambda v: type(v).__name__).value_counts()
    return {str(t): int(n) for t, n in types.items()} if len(types) > 1 else None


def _numeric_like(values: pd.Series, max_examples: int) -> dict[str, Any] | None:
    """Do most values parse as numbers once currency signs, % and separators are removed?"""
    if values.empty:
        return None
    cleaned = values.str.replace(r"^[\$€£¥]\s*|\s*%$|\s", "", regex=True)
    decimal_comma = bool(cleaned.str.contains(r"^[-+]?\d+,\d{1,2}$|\.\d{3},\d", regex=True).any())
    if decimal_comma:
        normalised = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        normalised = cleaned.str.replace(r"(?<=\d),(?=\d{3}\b)", "", regex=True)
    parsed = pd.to_numeric(normalised, errors="coerce")
    ok = int(parsed.notna().sum())
    if ok == 0 or ok / len(values) < TYPE_MATCH_RATIO:
        return None
    failed = values[parsed.isna()]
    notes = []
    if decimal_comma:
        notes.append("decimal comma (e.g. '120,5' = 120.5)")
    if (cleaned != values.str.replace(r"\s", "", regex=True)).any():
        notes.append("currency symbols or % signs")
    if values.str.match(r"^0\d").any():
        notes.append("leading zeros: may be a code (ID, postcode) that should stay text")
    return {
        "issue": "numbers_as_text",
        "count": ok,
        "detail": f"{ok} of {len(values)} text values ({_pct(ok, len(values))}%) parse as numbers"
                  + (f"; {', '.join(notes)}" if notes else ""),
        "non_numeric_examples": [repr(v) for v in failed.unique()[:max_examples]],
        "decimal_comma": decimal_comma,
    }


def _date_like(values: pd.Series, max_examples: int) -> dict[str, Any] | None:
    """Do most values parse as dates? Values without a digit are never treated as dates.

    Flexible date parsing is slow, so each distinct value is parsed once, and
    columns with more than DATE_SAMPLE_UNIQUE distinct values are judged on a
    fixed random sample of them (the result says so).
    """
    if values.empty:
        return None
    counts = values.value_counts()
    sampled = len(counts) > DATE_SAMPLE_UNIQUE
    if sampled:
        counts = counts.sample(DATE_SAMPLE_UNIQUE, random_state=0)
    uniques = counts.index.to_series(index=counts.index)
    parsed = pd.Series(pd.NaT, index=counts.index, dtype="datetime64[ns]")
    has_digit = uniques.str.contains(r"\d", regex=True)
    if has_digit.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed[has_digit] = pd.to_datetime(uniques[has_digit], errors="coerce", format="mixed")
    checked = int(counts.sum())
    share = int(counts[parsed.notna()].sum()) / checked
    if share == 0 or share < TYPE_MATCH_RATIO:
        return None
    ok = round(share * len(values))
    order = _day_month_order(uniques)
    if sampled:
        detail = (f"about {ok} of {len(values)} text values ({_pct(share, 1)}%) parse as dates "
                  f"(estimated from a sample of {DATE_SAMPLE_UNIQUE} distinct values)")
    else:
        detail = f"{ok} of {len(values)} text values ({_pct(share, 1)}%) parse as dates"
    if order == "ambiguous":
        detail += "; day/month order is ambiguous (e.g. 03/04 could be 3 April or 4 March)"
    elif order == "inconsistent":
        detail += "; some values are clearly day-first and others month-first"
    elif order:
        detail += f"; values look {order}"
    return {
        "issue": "dates_as_text",
        "count": ok,
        "detail": detail,
        "non_date_examples": [repr(v) for v in uniques[parsed.isna()].head(max_examples)],
        "date_order": order,
        "sampled": sampled,
    }


def _day_month_order(values: pd.Series) -> str | None:
    """For dates like 03/04/2025: which part is the day? Decided by any part > 12."""
    parts = values.str.extract(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}").dropna().astype(int)
    if parts.empty:
        return None
    day_first, month_first = (parts[0] > 12).any(), (parts[1] > 12).any()
    if day_first and month_first:
        return "inconsistent"
    if day_first:
        return "day-first"
    if month_first:
        return "month-first"
    return "ambiguous"


def _inconsistent_labels(values: pd.Series, max_examples: int) -> dict[str, Any] | None:
    """Group labels that differ only in case or inner spacing ("North", "north", "North  East").

    ``values`` are already stripped; untrimmed values are reported separately.
    """
    if values.empty:
        return None
    counts = values.value_counts()
    table = pd.DataFrame({
        "value": counts.index.astype(str),
        "count": counts.to_numpy(),
        "norm": counts.index.str.casefold().str.replace(r"\s+", " ", regex=True),
    })
    table = table[table.groupby("norm")["value"].transform("size") > 1]
    if table.empty:
        return None
    groups = [
        [{"value": v, "count": int(c)} for v, c in zip(g["value"], g["count"])]
        for _, g in table.groupby("norm", sort=False)
    ]
    return {
        "issue": "inconsistent_labels",
        "count": int(table["count"].sum()),
        "detail": f"{len(groups)} label(s) are written in more than one way (case or spacing)",
        "groups": groups[:max_examples],
    }


def _index_label(i: Any) -> Any:
    return int(i) if isinstance(i, (int, np.integer)) else str(i)


def _pct(part: float, whole: float) -> float:
    return round(100 * part / whole, 2) if whole else 0.0


def describe_dataset(df: pd.DataFrame, top_n: int = 5, key_columns: list[str] | None = None) -> dict[str, Any]:
    """Everything this script reports, as one JSON-ready dict."""
    kinds = classify_columns(df)
    stats = descriptive_stats(df, kinds, top_n)
    return to_jsonable(
        {
            "source": df.attrs.get("source", {}),
            "overview": get_overview(df, kinds),
            "statistics": stats,
            "quality": data_quality(df, kinds, key_columns),
        }
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _f(x: Any) -> float | None:
    """Float, or None for NaN/inf (JSON has no NaN)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars and NaN into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return _f(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:,.4g}" if abs(x) < 1e4 else f"{x:,.0f}"
    return str(x)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    """Plain-text table with aligned columns (no extra dependency)."""
    cells = [[_fmt(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in cells]
    return "\n".join([line, sep, *body])


def source_description(src: dict[str, Any]) -> str:
    """How the file was read, in a few words (plain text)."""
    fmt = src.get("format")
    if fmt == "excel":
        return f"Excel, sheet {src.get('sheet')!r} of {len(src.get('available_sheets', []))}"
    if fmt == "csv":
        return f"text, encoding {src.get('encoding')}, delimiter {src.get('delimiter')!r}"
    if fmt == "parquet":
        return "Parquet, column types stored in the file"
    if fmt == "json":
        how = "JSON Lines" if src.get("layout") == "json_lines" else "JSON list of records"
        if src.get("json_path"):
            how += f" at {src['json_path']}"
        return how
    return "in-memory data"


def format_text(result: dict[str, Any]) -> str:
    """Human-readable version of describe_dataset()."""
    src, ov, st = result["source"], result["overview"], result["statistics"]
    out: list[str] = []

    out.append(f"DATASET  {src.get('file_name', '?')}  ({source_description(src)})")
    kinds = ", ".join(f"{v} {k}" for k, v in ov["kind_counts"].items())
    out.append(f"{ov['rows']:,} rows x {ov['columns']} columns  |  memory {ov['memory_human']}  |  {kinds}")
    if src.get("flattened_columns"):
        out.append("Flattened from nested objects: " + ", ".join(src["flattened_columns"]))
    if src.get("nested_columns"):
        out.append("Lists/objects kept as JSON text: " + ", ".join(src["nested_columns"]))
    if src.get("decimal_columns"):
        out.append("Parquet decimals analysed as floats: " + ", ".join(src["decimal_columns"]))
    if src.get("iso_date_columns"):
        out.append("ISO dates read as dates: " + ", ".join(src["iso_date_columns"]))
    out.append("")

    out.append("COLUMNS")
    out.append(
        _table(
            ["column", "kind", "dtype", "non-null", "unique"],
            [[c["name"], c["kind"], c["dtype"], c["non_null"], c["unique"]] for c in ov["column_info"]],
        )
    )

    numeric = {k: v for k, v in st.items() if v["kind"] == "numeric"}
    if numeric:
        out += ["", "NUMERIC COLUMNS"]
        out.append(
            _table(
                ["column", "count", "mean", "std", "min", "q1", "median", "q3", "max", "skew"],
                [
                    [k, v["count"], v["mean"], v["std"], v["min"], v["q1"], v["median"], v["q3"], v["max"], v["skewness"]]
                    for k, v in numeric.items()
                ],
            )
        )
        for k, v in numeric.items():
            if v["skew_label"] and v["skew_label"] != "approximately symmetric":
                out.append(f"  * {k}: {v['skew_label']} (mean {_fmt(v['mean'])} vs median {_fmt(v['median'])})")

    for kind, title in (("boolean", "BOOLEAN COLUMNS"), ("datetime", "DATETIME COLUMNS")):
        cols = {k: v for k, v in st.items() if v["kind"] == kind}
        if not cols:
            continue
        out += ["", title]
        if kind == "boolean":
            rows = [[k, v["count"], v["true"], v["false"], v["true_pct"]] for k, v in cols.items()]
            out.append(_table(["column", "count", "true", "false", "true %"], rows))
        else:
            rows = [[k, v["count"], v["min"], v["max"], v["range_days"]] for k, v in cols.items()]
            out.append(_table(["column", "count", "earliest", "latest", "range (days)"], rows))

    cats = {k: v for k, v in st.items() if v["kind"] in ("categorical", "text")}
    if cats:
        out += ["", "CATEGORICAL AND TEXT COLUMNS"]
        for k, v in cats.items():
            tops = ", ".join(f"{t['value']!r} {t['count']} ({t['pct']}%)" for t in v["top_values"])
            out.append(f"  {k} [{v['kind']}]: {v['unique']} unique of {v['count']}. Top: {tops}")

    empty = [k for k, v in st.items() if v["kind"] == "empty"]
    if empty:
        out += ["", f"EMPTY COLUMNS (all values missing): {', '.join(empty)}"]

    if "quality" in result:
        out += ["", *_format_quality(result["quality"])]
    return "\n".join(out)


def _format_quality(q: dict[str, Any]) -> list[str]:
    out = ["DATA QUALITY"]

    m = q["missing"]
    out.append(
        f"Missing: {m['total_missing']:,} of {m['total_cells']:,} cells ({m['total_missing_pct']}%); "
        f"{m['rows_with_missing']:,} rows ({m['rows_with_missing_pct']}%) have at least one gap"
    )
    if m["complete_rows_ignoring_empty_columns"] != m["complete_rows"]:
        out.append(f"  ignoring all-empty columns: {m['complete_rows_ignoring_empty_columns']:,} complete rows")
    rows = [[c["name"], c["missing"], c["missing_pct"], c["disguised"],
             ", ".join(f"{k} x{v}" for k, v in c["disguised_values"].items()) or "-"]
            for c in m["columns"] if c["missing"] or c["disguised"]]
    if rows:
        out.append(_table(["column", "missing", "missing %", "disguised", "disguised values"], rows))

    for label, d in (("Duplicates", q["duplicates"]), ("Key duplicates", q.get("key_duplicates"))):
        if d is None:
            continue
        what = "all columns" if not d["subset"] else f"on {', '.join(d['subset'])}"
        line = f"{label} ({what}): {d['duplicate_rows']:,} extra rows ({d['duplicate_pct']}%)"
        if d["examples"]:
            line += "; e.g. rows " + "; ".join(", ".join(map(str, e["rows"])) for e in d["examples"][:3])
        out.append(line)

    out.append("Type issues:" + ("" if q["type_issues"] else " none found"))
    for i in q["type_issues"]:
        if i.get("groups"):
            extra = "; ".join(" / ".join(f"{v['value']!r} x{v['count']}" for v in g) for g in i["groups"])
        else:
            extra = ", ".join(i.get("non_numeric_examples") or i.get("non_date_examples") or i.get("examples") or [])
        label = {"numbers_as_text": "not numbers", "dates_as_text": "not dates"}.get(i["issue"], "examples")
        out.append(f"  * {i['column']}: {i['issue']}. {i['detail']}" + (f". {label.capitalize()}: {extra}" if extra else ""))

    out.append("IQR outliers (values outside Q1 - 1.5*IQR .. Q3 + 1.5*IQR):")
    for col, o in q["outliers"].items():
        if o["status"] != "ok":
            out.append(f"  * {col}: {o['status']}. {o['note']}")
            continue
        line = (f"  * {col}: {o['n_outliers']} ({o['outlier_pct']}%), {o['n_low']} low / {o['n_high']} high; "
                f"fences {_fmt(o['lower_fence'])} .. {_fmt(o['upper_fence'])}")
        if o["highest_outliers"]:
            line += f"; highest {', '.join(_fmt(v) for v in o['highest_outliers'])}"
        if o.get("code_like_values"):
            codes = ", ".join(f"{v} x{c}" for v, c in o["code_like_values"].items())
            line += f"; possible code for unknown: {codes}"
        if o.get("note"):
            line += f" ({o['note']})"
        out.append(line)
    if not q["outliers"]:
        out.append("  (no numeric columns)")
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


DATASET_HELP = "dataset file (.csv, .tsv, .txt, .xlsx, .xlsm, .parquet, .json, .jsonl)"
JSON_PATH_HELP = "dot-separated key that holds the records in a .json file, e.g. data.orders"
ENCODING_HELP = "text encoding for CSV/JSON files (default: CSV tries utf-8, then cp1252; JSON utf-8)"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Load a dataset file and describe it (read-only).")
    p.add_argument("path", help=DATASET_HELP)
    p.add_argument("--sheet", help="Excel sheet name (default: first sheet)")
    p.add_argument("--json-path", help=JSON_PATH_HELP)
    p.add_argument("--encoding", help=ENCODING_HELP)
    p.add_argument("--delimiter", help="CSV delimiter (default: detected automatically)")
    p.add_argument("--top", type=int, default=5, help="top values shown per categorical column (default 5)")
    p.add_argument("--key", help="comma-separated key columns for an extra duplicate check, e.g. order_id")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    return p


def main(argv: list[str] | None = None) -> int:
    # Column names and values may contain any character; don't crash on a
    # Windows console that can't display them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    delimiter = args.delimiter.encode().decode("unicode_escape") if args.delimiter else None  # allow "\t"
    try:
        df = load_dataset(args.path, sheet=args.sheet, encoding=args.encoding, delimiter=delimiter,
                          json_path=args.json_path)
        keys = [k.strip() for k in args.key.split(",")] if args.key else None
        result = describe_dataset(df, top_n=args.top, key_columns=keys)
    except DatasetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_text(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
