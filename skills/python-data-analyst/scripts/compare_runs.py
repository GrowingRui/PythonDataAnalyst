"""Compare two EDA runs, e.g. before and after cleaning: which findings went away, which are new.

Usage:
    python compare_runs.py eda_output/sales_20261105-1200 eda_output/sales_cleaned_20261105-1230
    python compare_runs.py before/summary.json after/summary.json --json

Each argument is an EDA output folder (or its summary.json). Findings are
matched by rule and columns; a matched finding whose title changed (e.g. a
different count) is listed as changed. Nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SEVERITIES = ("high", "medium", "low")


def load_summary(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"No EDA summary found at {path}. Pass an EDA output folder or its summary.json.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def _key(f: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return f["rule"], tuple(f["columns"])


def _brief(f: dict[str, Any]) -> dict[str, Any]:
    return {"rule": f["rule"], "severity": f["severity"], "columns": f["columns"], "title": f["title"]}


def _facts(s: dict[str, Any]) -> dict[str, Any]:
    q = s["quality"]
    facts = {
        "file": s["source"].get("file_name"),
        "rows": s["overview"]["rows"],
        "columns": s["overview"]["columns"],
        "missing_cells": q["missing"]["total_missing"],
        "placeholder_values": q["missing"]["total_disguised"],
        "exact_duplicate_rows": q["duplicates"]["duplicate_rows"],
        "type_issues": len(q["type_issues"]),
        "findings": {sev: sum(f["severity"] == sev for f in s["findings"]) for sev in SEVERITIES},
    }
    if q.get("key_duplicates"):
        facts["repeated_key_rows"] = q["key_duplicates"]["duplicate_rows"]
    return facts


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Match findings by (rule, columns). Repeats of the same key are paired in order."""
    old: dict[tuple, list[dict]] = {}
    for f in before["findings"]:
        old.setdefault(_key(f), []).append(f)
    new, changed, same = [], [], []
    for f in after["findings"]:
        matches = old.get(_key(f))
        if not matches:
            new.append(_brief(f))
            continue
        b = matches.pop(0)
        if b["title"] != f["title"] or b["severity"] != f["severity"]:
            changed.append({**_brief(f), "before_title": b["title"], "before_severity": b["severity"]})
        else:
            same.append(_brief(f))
    leftover = [f for rest in old.values() for f in rest]
    # One finding of a rule on both sides, with different columns (e.g. which
    # columns have gaps): the same finding, changed.
    for rule in {f["rule"] for f in leftover}:
        olds = [f for f in leftover if f["rule"] == rule]
        news = [f for f in new if f["rule"] == rule]
        if len(olds) == 1 and len(news) == 1:
            leftover.remove(olds[0])
            new.remove(news[0])
            changed.append({**news[0], "before_title": olds[0]["title"], "before_severity": olds[0]["severity"],
                            "before_columns": olds[0]["columns"]})
    gone = [_brief(f) for f in leftover]
    return {"before": _facts(before), "after": _facts(after),
            "gone": gone, "new": new, "changed": changed, "unchanged": same}


def format_text(c: dict[str, Any]) -> str:
    def plain(t: str) -> str:
        return t.replace("`", "")

    def counts(f: dict[str, Any]) -> str:
        return ", ".join(f"{f['findings'][s]} {s}" for s in SEVERITIES)

    b, a = c["before"], c["after"]
    lines = [f"Before: {b['file']}  {b['rows']:,} rows x {b['columns']} columns, findings {counts(b)}",
             f"After:  {a['file']}  {a['rows']:,} rows x {a['columns']} columns, findings {counts(a)}", ""]
    for label, key in (("missing cells", "missing_cells"), ("placeholder values", "placeholder_values"),
                       ("exact duplicate rows", "exact_duplicate_rows"), ("repeated key rows", "repeated_key_rows"),
                       ("type issues", "type_issues")):
        if key in b or key in a:
            lines.append(f"  {label}: {b.get(key, '-')} -> {a.get(key, '-')}")
    sections = (("Gone", c["gone"]), ("New", c["new"]), ("Changed", c["changed"]))
    for title, items in sections:
        lines += ["", f"{title} ({len(items)}):"]
        for f in items:
            lines.append(f"  [{f['severity']}] {f['rule']}: {plain(f['title'])}")
            if title == "Changed":
                lines.append(f"      before [{f['before_severity']}]: {plain(f['before_title'])}")
        if not items:
            lines.append("  none")
    lines += ["", f"Unchanged: {len(c['unchanged'])} finding(s)"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Compare the findings of two EDA runs (read-only).")
    p.add_argument("before", help="EDA output folder or summary.json of the first run")
    p.add_argument("after", help="EDA output folder or summary.json of the second run")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = p.parse_args(argv)
    try:
        result = compare(load_summary(args.before), load_summary(args.after))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else format_text(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
