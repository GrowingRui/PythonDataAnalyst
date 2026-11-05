"""Tests for inspect_data.py: loading, column classification, overview and statistics."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from inspect_data import (
    DatasetError,
    classify_columns,
    describe_dataset,
    descriptive_stats,
    get_overview,
    load_dataset,
)

# --------------------------------------------------------------------------
# load_dataset
# --------------------------------------------------------------------------


def test_load_comma_csv(csv_comma):
    df = load_dataset(csv_comma)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["a", "b", "c"]
    assert df.attrs["source"]["delimiter"] == ","


def test_sniffs_semicolon_delimiter(csv_semicolon):
    df = load_dataset(csv_semicolon)
    assert list(df.columns) == ["a", "b", "c"]
    assert df.attrs["source"]["delimiter"] == ";"


def test_tsv(tsv_file):
    df = load_dataset(tsv_file)
    assert df.shape == (2, 2)
    assert df.attrs["source"]["delimiter"] == "\t"


def test_explicit_delimiter_overrides_sniffing(csv_comma):
    df = load_dataset(csv_comma, delimiter=";")
    assert df.shape[1] == 1  # whole line becomes one column, as asked


def test_utf8_bom_is_stripped(csv_utf8_bom):
    df = load_dataset(csv_utf8_bom)
    assert list(df.columns) == ["name", "city"]  # not "﻿name"
    assert df.loc[0, "city"] == "Zürich"


def test_falls_back_to_cp1252(csv_cp1252):
    df = load_dataset(csv_cp1252)
    assert df.attrs["source"]["encoding"] == "cp1252"
    assert df["item"].tolist() == ["café", "thé"]
    assert df["price"].tolist() == ["€3", "€2"]


def test_wrong_explicit_encoding_gives_clear_error(csv_cp1252):
    with pytest.raises(DatasetError, match="--encoding"):
        load_dataset(csv_cp1252, encoding="utf-8")


def test_unknown_encoding(csv_comma):
    with pytest.raises(DatasetError, match="Unknown encoding"):
        load_dataset(csv_comma, encoding="not-a-codec")


def test_excel_first_sheet_by_default(excel_two_sheets):
    df = load_dataset(excel_two_sheets)
    assert list(df.columns) == ["x"]
    assert df.attrs["source"]["sheet"] == "first"
    assert df.attrs["source"]["available_sheets"] == ["first", "2025"]


def test_excel_named_sheet(excel_two_sheets):
    df = load_dataset(excel_two_sheets, sheet="2025")
    assert list(df.columns) == ["y", "z"]


def test_excel_wrong_sheet_lists_available(excel_two_sheets):
    with pytest.raises(DatasetError, match=r"Available sheets: 'first', '2025'"):
        load_dataset(excel_two_sheets, sheet="2024")


def test_sheet_on_csv_is_an_error(csv_comma):
    with pytest.raises(DatasetError, match="only applies to Excel"):
        load_dataset(csv_comma, sheet="x")


def test_file_not_found(tmp_path):
    with pytest.raises(DatasetError, match="File not found"):
        load_dataset(tmp_path / "missing.csv")


def test_folder_instead_of_file(tmp_path):
    with pytest.raises(DatasetError, match="folder"):
        load_dataset(tmp_path)


def test_empty_file(empty_file):
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(empty_file)


def test_header_only_file(header_only):
    with pytest.raises(DatasetError, match="no data rows"):
        load_dataset(header_only)


@pytest.mark.parametrize("name", ["data.feather", "data.xml", "data"])
def test_unsupported_format(tmp_path, name):
    path = tmp_path / name
    path.write_text("{}")
    with pytest.raises(DatasetError, match="Unsupported file type"):
        load_dataset(path)


def test_old_xls_explains_how_to_convert(tmp_path):
    path = tmp_path / "old.xls"
    path.write_bytes(b"\xd0\xcf\x11\xe0")
    with pytest.raises(DatasetError, match="save it as .xlsx"):
        load_dataset(path)


def test_loading_does_not_modify_file(csv_cp1252):
    before = csv_cp1252.read_bytes()
    load_dataset(csv_cp1252)
    assert csv_cp1252.read_bytes() == before


# --------------------------------------------------------------------------
# classify_columns
# --------------------------------------------------------------------------


def test_classify_one_of_each(mixed_df):
    assert classify_columns(mixed_df) == {
        "id": "numeric",
        "price": "numeric",
        "region": "categorical",
        "active": "boolean",
        "joined": "datetime",
        "notes": "empty",
    }


def test_classify_text_by_uniqueness():
    df = pd.DataFrame({"email": [f"user{i}@example.com" for i in range(100)]})
    assert classify_columns(df) == {"email": "text"}


def test_classify_text_by_length():
    df = pd.DataFrame({"comment": ["This product arrived late and the box was badly damaged."] * 3})
    assert classify_columns(df) == {"comment": "text"}


def test_classify_pandas_categorical_dtype():
    df = pd.DataFrame({"size": pd.Categorical(["S", "M", "L"])})
    assert classify_columns(df) == {"size": "categorical"}


def test_classify_numeric_with_missing():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    assert classify_columns(df) == {"x": "numeric"}


# --------------------------------------------------------------------------
# get_overview
# --------------------------------------------------------------------------


def test_overview(mixed_df):
    ov = get_overview(mixed_df)
    assert ov["rows"] == 5
    assert ov["columns"] == 6
    assert ov["memory_bytes"] > 0
    assert ov["kind_counts"] == {
        "numeric": 2,
        "boolean": 1,
        "datetime": 1,
        "categorical": 1,
        "empty": 1,
    }
    region = next(c for c in ov["column_info"] if c["name"] == "region")
    assert region == {"name": "region", "dtype": str(mixed_df["region"].dtype), "kind": "categorical",
                      "non_null": 5, "unique": 3}
    active = next(c for c in ov["column_info"] if c["name"] == "active")
    assert active["non_null"] == 4


# --------------------------------------------------------------------------
# descriptive_stats
# --------------------------------------------------------------------------


def test_numeric_stats_known_values(mixed_df):
    s = descriptive_stats(mixed_df)["price"]
    assert s["count"] == 5
    assert s["mean"] == 30
    assert s["median"] == 30
    assert s["min"] == 10 and s["max"] == 50
    assert s["q1"] == 20 and s["q3"] == 40
    assert s["std"] == pytest.approx(15.8113883)  # sample std (n - 1)
    assert s["skewness"] == pytest.approx(0.0)
    assert s["skew_label"] == "approximately symmetric"


def test_right_skew_detected():
    df = pd.DataFrame({"revenue": [1, 1, 2, 2, 3, 3, 4, 50]})
    s = descriptive_stats(df)["revenue"]
    assert s["skewness"] > 1
    assert s["skew_label"] == "highly right-skewed"
    assert s["mean"] > s["median"]  # the long tail pulls the mean up


def test_skew_undefined_for_small_or_constant():
    df = pd.DataFrame({"two": [1.0, 2.0, np.nan], "const": [5.0, 5.0, 5.0]})
    st = descriptive_stats(df)
    assert st["two"]["skewness"] is None
    assert st["const"]["skewness"] is None
    assert st["const"]["std"] == 0


def test_categorical_top_values(mixed_df):
    s = descriptive_stats(mixed_df, top_n=2)["region"]
    assert s["kind"] == "categorical"
    assert s["unique"] == 3
    assert s["top_values"][0] == {"value": "North", "count": 3, "pct": 60.0}
    assert len(s["top_values"]) == 2
    assert s["other_count"] == 1


def test_boolean_stats(mixed_df):
    s = descriptive_stats(mixed_df)["active"]
    assert s == {"kind": "boolean", "count": 4, "true": 3, "false": 1, "true_pct": 75.0}


def test_datetime_stats(mixed_df):
    s = descriptive_stats(mixed_df)["joined"]
    assert s["min"].startswith("2024-01-01")
    assert s["max"].startswith("2024-03-01")
    assert s["range_days"] == 60
    assert s["unique"] == 4


def test_describe_dataset_is_json_serialisable(mixed_df):
    result = describe_dataset(mixed_df)
    text = json.dumps(result)  # raises if numpy types or NaN slipped through
    assert "NaN" not in text
    assert set(result) == {"source", "overview", "statistics", "quality"}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _run(scripts_dir, *args):
    return subprocess.run(
        [sys.executable, str(scripts_dir / "inspect_data.py"), *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_cli_text_output(scripts_dir, csv_cp1252):
    r = _run(scripts_dir, csv_cp1252)
    assert r.returncode == 0, r.stderr
    assert "legacy.csv" in r.stdout
    assert "2 rows x 2 columns" in r.stdout
    assert "café" in r.stdout


def test_cli_json_output(scripts_dir, excel_two_sheets):
    r = _run(scripts_dir, excel_two_sheets, "--sheet", "2025", "--json")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["overview"]["rows"] == 2
    assert data["statistics"]["z"]["mean"] == 2.0


def test_cli_error_exit_code(scripts_dir, tmp_path):
    r = _run(scripts_dir, tmp_path / "nope.csv")
    assert r.returncode == 2
    assert "File not found" in r.stderr
    assert "Traceback" not in r.stderr
