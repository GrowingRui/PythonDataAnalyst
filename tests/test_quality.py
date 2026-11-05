"""Tests for the data-quality checks in inspect_data.py.

Each check must find the problems planted in the test data, and nothing more.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from inspect_data import (
    DatasetError,
    data_quality,
    detect_type_issues,
    find_duplicates,
    iqr_outliers,
    missing_values,
)


def _issues(df, column=None):
    found = detect_type_issues(df)
    return {(i["column"], i["issue"]): i for i in found if column is None or i["column"] == column}


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A dataset with no quality problems at all."""
    # Evenly spaced amounts, not random ones: about 0.7% of normally distributed
    # values fall outside the IQR fences, so random data isn't reliably "clean".
    return pd.DataFrame(
        {
            "order_id": range(1, 41),
            "amount": np.linspace(80, 120, 40).round(2),
            "region": ["North", "South", "East", "West"] * 10,
            "month": ["March", "April", "May", "June"] * 10,  # words, not dates
            "sku": [f"A{i}" for i in range(40)],  # codes, not numbers
        }
    )


# --------------------------------------------------------------------------
# Clean data: nothing should be flagged
# --------------------------------------------------------------------------


def test_clean_data_has_no_findings(clean_df):
    q = data_quality(clean_df)
    assert q["missing"]["total_missing"] == 0
    assert q["missing"]["total_disguised"] == 0
    assert q["duplicates"]["duplicate_rows"] == 0
    assert q["type_issues"] == []
    assert all(o["n_outliers"] == 0 for o in q["outliers"].values())


# --------------------------------------------------------------------------
# missing_values
# --------------------------------------------------------------------------


def test_missing_counts():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": ["x", "y", None, "z"], "c": [1, 2, 3, 4]})
    m = missing_values(df)
    assert m["total_cells"] == 12
    assert m["total_missing"] == 3
    assert m["total_missing_pct"] == 25.0
    assert m["rows_with_missing"] == 3
    assert m["complete_rows"] == 1
    cols = {c["name"]: c for c in m["columns"]}
    assert cols["a"]["missing"] == 2 and cols["a"]["missing_pct"] == 50.0
    assert cols["c"]["missing"] == 0


def test_disguised_missing_found_but_not_counted_as_missing():
    df = pd.DataFrame({"city": ["Paris", "-", "?", "Unknown", "   ", "N/A", "Rome", "Lyon"]})
    col = missing_values(df)["columns"][0]
    assert col["missing"] == 0  # real NaN only
    assert col["disguised"] == 5
    assert col["disguised_values"] == {"'-'": 1, "'?'": 1, "'Unknown'": 1, "'   '": 1, "'N/A'": 1}


def test_disguised_extra_tokens():
    df = pd.DataFrame({"x": ["ok", "999", "ok"]})
    assert missing_values(df)["columns"][0]["disguised"] == 0
    assert missing_values(df, extra_tokens=["999"])["columns"][0]["disguised"] == 1


def test_numeric_columns_have_no_disguised_values():
    df = pd.DataFrame({"n": [0, -1, 99]})
    assert missing_values(df)["columns"][0]["disguised"] == 0


# --------------------------------------------------------------------------
# find_duplicates
# --------------------------------------------------------------------------


def test_exact_duplicates_count_extra_copies():
    df = pd.DataFrame({"id": [1, 2, 2, 2, 3], "v": ["a", "b", "b", "b", "c"]})
    d = find_duplicates(df)
    assert d["duplicate_rows"] == 2  # three identical rows -> two extra copies
    assert d["rows_involved"] == 3
    assert d["groups"] == 1
    assert d["examples"] == [{"rows": [1, 2, 3], "count": 3}]
    assert d["duplicate_pct"] == 40.0


def test_key_duplicates_differ_from_exact():
    # Order 7 appears twice with different amounts: not an exact duplicate,
    # but a duplicate key, which usually means a data-entry problem.
    df = pd.DataFrame({"order_id": [6, 7, 7, 8], "amount": [10, 20, 25, 30]})
    assert find_duplicates(df)["duplicate_rows"] == 0
    d = find_duplicates(df, subset=["order_id"])
    assert d["duplicate_rows"] == 1
    assert d["examples"] == [{"rows": [1, 2], "count": 2}]


def test_duplicates_with_missing_values_still_match():
    df = pd.DataFrame({"a": [1, 1], "b": [np.nan, np.nan]})
    assert find_duplicates(df)["duplicate_rows"] == 1


def test_unknown_key_column():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(DatasetError, match="Key column.*not found: nope"):
        find_duplicates(df, subset=["nope"])


# --------------------------------------------------------------------------
# detect_type_issues
# --------------------------------------------------------------------------


def test_numbers_as_text_with_placeholder():
    df = pd.DataFrame({"revenue": ["120", "80", "-", "95.5", "110"]})
    i = _issues(df)[("revenue", "numbers_as_text")]
    assert i["count"] == 4  # "-" is a placeholder, not a failed number
    assert not i["decimal_comma"]


def test_numbers_with_currency_thousands_and_percent():
    df = pd.DataFrame({"price": ["$1,200", "$950", "$3,400.50", "$80"], "rate": ["15%", "20%", "7.5%", "0%"]})
    found = _issues(df)
    assert found[("price", "numbers_as_text")]["count"] == 4
    assert "currency" in found[("price", "numbers_as_text")]["detail"]
    assert found[("rate", "numbers_as_text")]["count"] == 4


def test_decimal_comma_detected():
    df = pd.DataFrame({"weight": ["1,5", "2,25", "1.234,5", "3"]})
    i = _issues(df)[("weight", "numbers_as_text")]
    assert i["decimal_comma"] is True
    assert i["count"] == 4


def test_leading_zeros_warn_about_codes():
    df = pd.DataFrame({"zip": ["01234", "02134", "10001"]})
    assert "leading zeros" in _issues(df)[("zip", "numbers_as_text")]["detail"]


def test_mostly_text_is_not_flagged_as_numbers():
    df = pd.DataFrame({"code": ["12", "AB", "CD", "EF", "34"]})  # 40% numeric
    assert ("code", "numbers_as_text") not in _issues(df)


def test_dates_as_text_iso():
    df = pd.DataFrame({"d": ["2025-01-03", "2025-02-14", "2025-03-01", "not a date", "2025-04-30"]})
    i = _issues(df)[("d", "dates_as_text")]
    assert i["count"] == 4
    assert i["non_date_examples"] == ["'not a date'"]
    assert i["date_order"] is None  # ISO dates are unambiguous


@pytest.mark.parametrize(
    "values, order",
    [
        (["03/04/2025", "05/06/2025", "01/02/2025"], "ambiguous"),
        (["25/01/2025", "03/04/2025", "13/12/2025"], "day-first"),
        (["01/25/2025", "03/04/2025", "12/13/2025"], "month-first"),
        (["25/01/2025", "01/25/2025", "03/04/2025"], "inconsistent"),
    ],
)
def test_day_month_order(values, order):
    assert _issues(pd.DataFrame({"d": values}))[("d", "dates_as_text")]["date_order"] == order


def test_month_names_are_not_dates(clean_df):
    assert _issues(clean_df, "month") == {}


def test_mixed_python_types():
    df = pd.DataFrame({"qty": pd.Series([1, 2, "three", 4], dtype=object)})
    i = _issues(df)[("qty", "mixed_types")]
    assert i["types"] == {"int": 3, "str": 1}


def test_inconsistent_labels():
    df = pd.DataFrame({"region": ["North", "north", "NORTH", "South", "South", "South  East", "south east"]})
    i = _issues(df)[("region", "inconsistent_labels")]
    assert i["detail"].startswith("2 label(s)")
    assert i["count"] == 5  # 3 North variants + 2 South East variants
    groups = sorted(sorted(v["value"] for v in g) for g in i["groups"])
    assert groups == [["NORTH", "North", "north"], ["South  East", "south east"]]


def test_untrimmed_whitespace():
    df = pd.DataFrame({"region": ["North ", "North", " South", "South"]})
    found = _issues(df)
    assert found[("region", "untrimmed_whitespace")]["count"] == 2
    assert ("region", "inconsistent_labels") not in found  # same label after trimming


def test_whitespace_only_is_placeholder_not_untrimmed():
    df = pd.DataFrame({"x": ["a", "   ", "b"]})
    assert ("x", "untrimmed_whitespace") not in _issues(df)


def test_numeric_and_datetime_columns_are_not_checked(mixed_df):
    assert detect_type_issues(mixed_df) == []


# --------------------------------------------------------------------------
# iqr_outliers
# --------------------------------------------------------------------------


def test_iqr_known_answer():
    df = pd.DataFrame({"x": list(range(1, 11)) + [1000]})
    o = iqr_outliers(df)["x"]
    # Q1 = 3.5, Q3 = 8.5 (linear interpolation), IQR = 5 -> fences -4 and 16
    assert (o["q1"], o["q3"], o["iqr"]) == (3.5, 8.5, 5.0)
    assert (o["lower_fence"], o["upper_fence"]) == (-4.0, 16.0)
    assert o["n_outliers"] == 1
    assert o["n_high"] == 1 and o["n_low"] == 0
    assert o["highest_outliers"] == [1000.0]


def test_iqr_low_outliers_and_custom_k():
    df = pd.DataFrame({"x": [-500] + list(range(1, 21))})
    assert iqr_outliers(df)["x"]["n_low"] == 1
    assert iqr_outliers(df, k=1000)["x"]["n_outliers"] == 0


def test_iqr_zero_iqr_is_not_applied():
    df = pd.DataFrame({"x": [5] * 10 + [6, 7]})
    o = iqr_outliers(df)["x"]
    assert o["status"] == "zero_iqr"
    assert o["values_not_equal_to_q1"] == 2
    assert "n_outliers" not in o


def test_iqr_skips_tiny_samples():
    o = iqr_outliers(pd.DataFrame({"x": [1, 2, 100, np.nan]}))["x"]
    assert o["status"] == "skipped"
    assert o["count"] == 3


def test_iqr_small_sample_note():
    assert "small sample" in iqr_outliers(pd.DataFrame({"x": range(10)}))["x"]["note"]
    assert "note" not in iqr_outliers(pd.DataFrame({"x": range(30)}))["x"]


def test_iqr_only_numeric_columns(mixed_df):
    assert set(iqr_outliers(mixed_df)) == {"id", "price"}


# --------------------------------------------------------------------------
# Whole pipeline and CLI
# --------------------------------------------------------------------------

MESSY_CSV = (
    "order_id;region;revenue;order_date;comment\n"
    "1;North;120,5;03/01/2025;fine\n"
    "2;south;80;04/01/2025;-\n"
    "3;North ;9500;05/01/2025;big order\n"
    "4;East;95;06/01/2025;unknown\n"
    "5;North;110;07/01/2025;ok\n"
    "5;North;110;07/01/2025;ok\n"
    "6;north;-;25/01/2025;\n"
)


def test_cli_finds_planted_problems_and_leaves_file_untouched(scripts_dir, tmp_path):
    path = tmp_path / "messy.csv"
    path.write_text(MESSY_CSV, encoding="utf-8")
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    r = subprocess.run(
        [sys.executable, str(scripts_dir / "inspect_data.py"), str(path), "--json", "--key", "order_id"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    q = json.loads(r.stdout)["quality"]

    assert q["duplicates"]["duplicate_rows"] == 1
    assert q["key_duplicates"]["duplicate_rows"] == 1
    found = {(i["column"], i["issue"]) for i in q["type_issues"]}
    assert found == {
        ("region", "inconsistent_labels"),
        ("region", "untrimmed_whitespace"),
        ("revenue", "numbers_as_text"),
        ("order_date", "dates_as_text"),
    }
    comment = next(c for c in q["missing"]["columns"] if c["name"] == "comment")
    assert (comment["missing"], comment["disguised"]) == (1, 2)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_cli_bad_key_column(scripts_dir, csv_comma):
    r = subprocess.run(
        [sys.executable, str(scripts_dir / "inspect_data.py"), str(csv_comma), "--key", "nope"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 2
    assert "Key column" in r.stderr


def test_date_check_samples_many_distinct_values(monkeypatch):
    import inspect_data

    monkeypatch.setattr(inspect_data, "DATE_SAMPLE_UNIQUE", 50)
    dates = pd.date_range("2020-01-01", periods=300).strftime("%Y-%m-%d").tolist()
    i = _issues(pd.DataFrame({"d": dates}))[("d", "dates_as_text")]
    assert i["sampled"] is True
    assert i["count"] == 300  # estimated share (100%) applied to all values
    assert "estimated from a sample of 50" in i["detail"]
