"""Parquet, JSON and JSON Lines input (Phase 11)."""

from __future__ import annotations

import decimal
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from generate_eda import run_eda
from inspect_data import DatasetError, describe_dataset, format_text, load_dataset
from report import render_report

SAMPLE_CSV = Path(__file__).resolve().parents[1] / "sample_data" / "sample_sales.csv"


@pytest.fixture
def pyarrow():
    return pytest.importorskip("pyarrow")


def _write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _issues(df: pd.DataFrame) -> set[tuple[str, str]]:
    return {(i["column"], i["issue"]) for i in describe_dataset(df)["quality"]["type_issues"]}


# --------------------------------------------------------------------------
# Parquet
# --------------------------------------------------------------------------


def test_parquet_keeps_stored_types(tmp_path, pyarrow):
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "price": [9.5, 12.0, 3.25],
        "day": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
        "ok": [True, False, True],
        "city": ["Rome", "Oslo", "Rome"],
    })
    path = tmp_path / "typed.parquet"
    df.to_parquet(path)
    back = load_dataset(path)
    assert back.attrs["source"]["format"] == "parquet"
    assert back.attrs["source"]["stored_types"] is True
    assert pd.api.types.is_datetime64_any_dtype(back["day"])
    assert back["ok"].dtype == bool
    assert "Parquet" in format_text(describe_dataset(back))


def test_parquet_text_columns_are_still_checked(tmp_path, pyarrow):
    """Parquet only prevents type problems in typed columns; text stays text."""
    df = pd.DataFrame({"amount": ["$1,200", "$85", "$3,400", "$12", "$990"],
                       "region": ["North", "north", "South", "South", "North "]})
    path = tmp_path / "text.parquet"
    df.to_parquet(path)
    issues = _issues(load_dataset(path))
    assert ("amount", "numbers_as_text") in issues
    assert ("region", "inconsistent_labels") in issues
    assert ("region", "untrimmed_whitespace") in issues


def test_typed_parquet_reports_why_type_problems_are_absent(tmp_path, pyarrow):
    """The sample data with properly typed columns: the text-type problems vanish,
    and the report says that's the format at work, not a clean bill of health."""
    df = load_dataset(SAMPLE_CSV)
    df["order_date"] = pd.to_datetime(df["order_date"], format="%d/%m/%Y", errors="coerce")
    for col in ("shipping_cost", "rating"):
        df[col] = pd.to_numeric(df[col].str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")
    path = tmp_path / "typed_sales.parquet"
    df.to_parquet(path)

    loaded = load_dataset(path)
    issues = _issues(loaded)
    assert not {i for i in issues if i[1] in ("numbers_as_text", "dates_as_text")}
    assert ("region", "inconsistent_labels") in issues  # text problems are still found

    summary = run_eda(loaded, tmp_path / "out", max_charts=0)
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Parquet, column types stored in the file" in report
    assert "The file stores each column's type" in report
    assert "not proof that the values are right" in report
    rules = {f["rule"] for f in summary["findings"]}
    assert {"possible_missing_code", "duplicate_rows"} <= rules  # value problems remain


def test_csv_report_has_no_stored_types_note(tmp_path):
    df = load_dataset(SAMPLE_CSV)
    run_eda(df, tmp_path / "out", max_charts=0)
    assert "stores each column's type" not in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_parquet_without_pyarrow_explains_install(tmp_path, monkeypatch):
    path = tmp_path / "x.parquet"
    path.write_bytes(b"PAR1 not really")
    monkeypatch.setitem(sys.modules, "pyarrow", None)  # makes "import pyarrow" fail
    with pytest.raises(DatasetError, match="pip install pyarrow"):
        load_dataset(path)


def test_corrupt_parquet(tmp_path, pyarrow):
    path = tmp_path / "broken.parquet"
    path.write_bytes(b"this is not parquet")
    with pytest.raises(DatasetError, match="Could not read broken.parquet as a Parquet file"):
        load_dataset(path)


def test_parquet_decimals_become_floats(tmp_path, pyarrow):
    table = pyarrow.table({"amount": pyarrow.array([decimal.Decimal("1.10"), decimal.Decimal("2.25"), None],
                                                   type=pyarrow.decimal128(10, 2))})
    path = tmp_path / "dec.parquet"
    import pyarrow.parquet as pq
    pq.write_table(table, path)
    df = load_dataset(path)
    assert df["amount"].dtype == float
    assert df["amount"].tolist()[:2] == [1.1, 2.25]
    assert df.attrs["source"]["decimal_columns"] == ["amount"]


def test_parquet_list_column_is_reported(tmp_path, pyarrow):
    df = pd.DataFrame({"id": [1, 2, 3], "tags": [["a", "b"], ["a"], []]})
    path = tmp_path / "lists.parquet"
    df.to_parquet(path)
    back = load_dataset(path)
    assert back["tags"].tolist() == ['["a", "b"]', '["a"]', "[]"]
    assert back.attrs["source"]["nested_columns"] == {"tags": 3}
    assert ("tags", "nested_values") in _issues(back)


# --------------------------------------------------------------------------
# JSON and JSON Lines
# --------------------------------------------------------------------------

ORDERS = [
    {"id": 1, "amount": 30.0, "customer": {"city": "Rome", "tier": "gold"}, "tags": ["gift", "express"]},
    {"id": 2, "amount": 12.5, "customer": {"city": "Oslo"}, "tags": ["gift"]},
    {"id": 3, "amount": 8.0, "customer": {"city": "Rome", "tier": "basic"}, "tags": []},
    {"id": 4, "amount": None, "customer": {"city": "Lima", "tier": "gold"}},
]


def test_json_records_are_flattened(tmp_path):
    df = load_dataset(_write_json(tmp_path / "orders.json", ORDERS))
    src = df.attrs["source"]
    assert src["format"] == "json" and src["layout"] == "records"
    assert list(df.columns) == ["id", "amount", "tags", "customer.city", "customer.tier"]
    assert src["flattened_columns"] == ["customer.city", "customer.tier"]
    assert df["customer.tier"].isna().sum() == 1  # absent key = missing value


def test_json_lists_stay_and_are_a_type_problem(tmp_path):
    df = load_dataset(_write_json(tmp_path / "orders.json", ORDERS))
    assert df["tags"].tolist()[:3] == ['["gift", "express"]', '["gift"]', "[]"]
    assert pd.isna(df["tags"].iloc[3])
    assert df.attrs["source"]["nested_columns"] == {"tags": 3}
    issues = describe_dataset(df)["quality"]["type_issues"]
    tags = [i for i in issues if i["column"] == "tags"]
    assert [i["issue"] for i in tags] == ["nested_values"]  # and no label/number checks on JSON text
    assert tags[0]["count"] == 3


def test_json_report_mentions_flattening_and_lists(tmp_path):
    df = load_dataset(_write_json(tmp_path / "orders.json", ORDERS))
    summary = run_eda(df, tmp_path / "out")
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "JSON list of records" in report
    assert "**Flattened from nested objects:** `customer.city`, `customer.tier`" in report
    assert "**Nested JSON objects were flattened**" in report
    finding = next(f for f in summary["findings"] if f["rule"] == "nested_values")
    assert finding["columns"] == ["tags"]
    assert "Lists or objects inside cells" in report  # cleaning advice
    assert not any("tags" in c["file"] for c in summary["charts"])  # no "most common lists" chart
    assert summary["source"]["unchanged_after_run"] is True


def test_jsonl(tmp_path):
    path = tmp_path / "orders.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ORDERS) + "\n\n", encoding="utf-8")
    df = load_dataset(path)
    assert df.attrs["source"]["layout"] == "json_lines"
    assert len(df) == 4 and "customer.city" in df.columns


def test_json_lines_saved_as_json(tmp_path):
    path = tmp_path / "orders.json"
    path.write_text("\n".join(json.dumps(r) for r in ORDERS), encoding="utf-8")
    df = load_dataset(path)
    assert df.attrs["source"]["layout"] == "json_lines" and len(df) == 4


def test_jsonl_bad_line_names_the_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\n{"a": 2\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="Line 2 of bad.jsonl is not valid JSON"):
        load_dataset(path)


def test_jsonl_line_that_is_not_a_record(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="Line 2 of bad.jsonl is a list, not a record"):
        load_dataset(path)


def test_json_path(tmp_path):
    path = _write_json(tmp_path / "export.json", {"meta": {"v": 2}, "data": {"orders": ORDERS}})
    df = load_dataset(path, json_path="data.orders")
    assert len(df) == 4
    assert df.attrs["source"]["json_path"] == "data.orders"


def test_settings_object_is_not_a_table(tmp_path):
    path = _write_json(tmp_path / "settings.json", {"theme": "dark", "retries": 3})
    with pytest.raises(DatasetError, match="single JSON object .*'theme', 'retries'.*not a table"):
        load_dataset(path)


def test_object_error_suggests_json_path(tmp_path):
    path = _write_json(tmp_path / "export.json",
                       {"meta": {"v": 2}, "data": {"orders": ORDERS, "notes": [{"x": 1}]}})
    with pytest.raises(DatasetError, match=r"data\.orders \(4 records\), data\.notes \(1 records\)"
                                           r".*--json-path data\.orders"):
        load_dataset(path)


def test_json_path_missing_key_lists_keys(tmp_path):
    path = _write_json(tmp_path / "export.json", {"data": {"orders": ORDERS}})
    with pytest.raises(DatasetError, match="key 'order' not found at data of export.json. Keys there: 'orders'"):
        load_dataset(path, json_path="data.order")


def test_json_path_through_a_list(tmp_path):
    path = _write_json(tmp_path / "export.json", {"data": ORDERS})
    with pytest.raises(DatasetError, match="data of export.json is a list, so it has no key 'orders'"):
        load_dataset(path, json_path="data.orders")


def test_json_path_only_for_json(csv_comma):
    with pytest.raises(DatasetError, match="--json-path only applies to .json files"):
        load_dataset(csv_comma, json_path="data")


@pytest.mark.parametrize("data, message", [
    ([1, 2, 3], "list of number values, not of records"),
    ([{"a": 1}, 5], "mixes records with other values: 1 of 2 items"),
    ([], "contains no records"),
    ("hello", "single text value"),
])
def test_json_that_is_not_table_shaped(tmp_path, data, message):
    with pytest.raises(DatasetError, match=message):
        load_dataset(_write_json(tmp_path / "x.json", data))


def test_invalid_json(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('[{"a": 1},', encoding="utf-8")
    with pytest.raises(DatasetError, match="x.json is not valid JSON"):
        load_dataset(path)


def test_json_encoding(tmp_path):
    path = tmp_path / "latin.json"
    path.write_bytes(json.dumps([{"city": "Zürich"}], ensure_ascii=False).encode("cp1252"))
    with pytest.raises(DatasetError, match="Pass the right one with --encoding"):
        load_dataset(path)
    assert load_dataset(path, encoding="cp1252")["city"].iloc[0] == "Zürich"


# --------------------------------------------------------------------------
# Same data, same findings
# --------------------------------------------------------------------------


def _findings(df: pd.DataFrame, out: Path) -> list[tuple[str, str, str]]:
    summary = run_eda(df, out, max_charts=0)
    return [(f["rule"], f["severity"], f["title"]) for f in summary["findings"]]


def test_sample_data_gives_the_same_findings_in_every_format(tmp_path, pyarrow):
    """Done-when of Phase 11: the sample saved as Parquet, JSON and JSONL gives the CSV's findings."""
    df = load_dataset(SAMPLE_CSV)
    df.to_parquet(tmp_path / "sales.parquet")
    df.to_json(tmp_path / "sales.json", orient="records")
    df.to_json(tmp_path / "sales.jsonl", orient="records", lines=True)
    expected = _findings(df, tmp_path / "out_csv")
    assert len(expected) > 10
    for name in ("sales.parquet", "sales.json", "sales.jsonl"):
        assert _findings(load_dataset(tmp_path / name), tmp_path / f"out_{name}") == expected, name


def test_render_report_without_file_source():
    """An in-memory summary (no source) still renders."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": list("xyzxy")})
    summary = describe_dataset(df)
    summary["findings"] = []
    assert "in-memory data" in render_report(summary)


def test_cli_json_path(scripts_dir, tmp_path):
    path = _write_json(tmp_path / "export.json", {"data": {"orders": ORDERS}})
    run = subprocess.run([sys.executable, str(scripts_dir / "inspect_data.py"), str(path),
                          "--json-path", "data.orders"], capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    assert "JSON list of records at data.orders" in run.stdout
    assert "Flattened from nested objects: customer.city, customer.tier" in run.stdout
    assert "tags: nested_values" in run.stdout

    run = subprocess.run([sys.executable, str(scripts_dir / "inspect_data.py"), str(path)],
                         capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 2 and "--json-path data.orders" in run.stderr
