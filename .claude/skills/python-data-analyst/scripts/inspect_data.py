"""Load a CSV/TSV/Excel dataset and describe it (read-only).

Usage:
    python inspect_data.py data.csv
    python inspect_data.py data.xlsx --sheet "2025" --json

The input file is only ever read, never written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CSV_EXTENSIONS = {".csv", ".tsv", ".txt"}
EXCEL_EXTENSIONS = {".xlsx", ".xlsm"}
SUPPORTED_EXTENSIONS = CSV_EXTENSIONS | EXCEL_EXTENSIONS

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
) -> pd.DataFrame:
    """Read a CSV, TSV or Excel file into a DataFrame.

    How the file was read (encoding, delimiter, sheet) is stored in
    ``df.attrs["source"]`` so the report can say so.
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

    if ext in EXCEL_EXTENSIONS:
        df, source = _read_excel(path, sheet)
    else:
        if sheet is not None:
            raise DatasetError("--sheet only applies to Excel files.")
        df, source = _read_csv(path, encoding, delimiter)

    if df.shape[1] == 0:
        raise DatasetError(f"No columns found in {path.name}.")
    if df.shape[0] == 0:
        raise DatasetError(f"{path.name} has column headers but no data rows.")

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
    skew = float(v.skew()) if n >= 3 else None
    if skew is not None and not math.isfinite(skew):
        skew = None  # constant column: skewness is undefined
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


def describe_dataset(df: pd.DataFrame, top_n: int = 5) -> dict[str, Any]:
    """Everything this script reports, as one JSON-ready dict."""
    kinds = classify_columns(df)
    stats = descriptive_stats(df, kinds, top_n)
    return to_jsonable(
        {
            "source": df.attrs.get("source", {}),
            "overview": get_overview(df, kinds),
            "statistics": stats,
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
        return f"{x:,.4g}" if abs(x) < 1e6 else f"{x:,.0f}"
    return str(x)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    """Plain-text table with aligned columns (no extra dependency)."""
    cells = [[_fmt(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in cells]
    return "\n".join([line, sep, *body])


def format_text(result: dict[str, Any]) -> str:
    """Human-readable version of describe_dataset()."""
    src, ov, st = result["source"], result["overview"], result["statistics"]
    out: list[str] = []

    how = (
        f"sheet {src.get('sheet')!r} of {len(src.get('available_sheets', []))}"
        if src.get("format") == "excel"
        else f"encoding {src.get('encoding')}, delimiter {src.get('delimiter')!r}"
    )
    out.append(f"DATASET  {src.get('file_name', '?')}  ({how})")
    kinds = ", ".join(f"{v} {k}" for k, v in ov["kind_counts"].items())
    out.append(f"{ov['rows']:,} rows x {ov['columns']} columns  |  memory {ov['memory_human']}  |  {kinds}")
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
    return "\n".join(out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Load a CSV/TSV/Excel file and describe it (read-only).")
    p.add_argument("path", help="dataset file (.csv, .tsv, .txt, .xlsx, .xlsm)")
    p.add_argument("--sheet", help="Excel sheet name (default: first sheet)")
    p.add_argument("--encoding", help="text encoding for CSV files (default: try utf-8, then cp1252)")
    p.add_argument("--delimiter", help="CSV delimiter (default: detected automatically)")
    p.add_argument("--top", type=int, default=5, help="top values shown per categorical column (default 5)")
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
        df = load_dataset(args.path, sheet=args.sheet, encoding=args.encoding, delimiter=delimiter)
    except DatasetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    result = describe_dataset(df, top_n=args.top)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_text(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
