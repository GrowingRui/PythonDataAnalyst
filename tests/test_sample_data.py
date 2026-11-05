"""End to end: the sample dataset is reproducible, and the EDA finds every planted problem."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from generate_eda import run_eda
from inspect_data import load_dataset

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_sample_data", SAMPLE_DIR / "make_sample_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_generator()


@pytest.fixture(scope="module")
def sample_run(generator, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sample")
    path = generator.write_csv(generator.make_sales(), tmp / "sample_sales.csv")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = run_eda(load_dataset(path), tmp / "out", key_columns=["order_id"])
    return path, before, summary, tmp / "out"


def _find(summary, rule, column=None):
    return [f for f in summary["findings"] if f["rule"] == rule and (column is None or column in f["columns"])]


def test_generator_is_reproducible(generator, tmp_path):
    a = generator.write_csv(generator.make_sales(), tmp_path / "a.csv").read_bytes()
    b = generator.write_csv(generator.make_sales(), tmp_path / "b.csv").read_bytes()
    assert a == b
    c = generator.write_csv(generator.make_sales(seed=1), tmp_path / "c.csv").read_bytes()
    assert a != c


def test_committed_sample_matches_the_generator(generator, tmp_path):
    fresh = generator.write_csv(generator.make_sales(), tmp_path / "fresh.csv").read_bytes()
    committed = (SAMPLE_DIR / "sample_sales.csv").read_bytes()
    assert committed == fresh, "run `python sample_data/make_sample_data.py` and commit the result"


def test_shape(sample_run):
    _, _, summary, _ = sample_run
    assert summary["overview"]["rows"] == 1215  # 1,200 orders + 12 exact duplicates + 3 edited repeats
    assert summary["overview"]["columns"] == 16


@pytest.mark.parametrize(
    ("rule", "column"),
    [
        ("dates_as_text", "order_date"),
        ("inconsistent_labels", "region"),
        ("untrimmed_whitespace", "region"),
        ("missing_values", "discount_pct"),
        ("missing_values", "delivery_days"),
        ("possible_missing_code", "delivery_days"),
        ("numbers_as_text", "shipping_cost"),
        ("numbers_as_text", "rating"),
        ("disguised_missing", "rating"),
        ("constant_column", "currency"),
        ("empty_columns", "legacy_code"),
        ("outliers", "quantity"),
        ("skewed", "revenue"),
        ("skewed", "unit_price"),
        ("strong_correlation", "revenue"),
        ("duplicate_rows", None),
        ("duplicate_keys", "order_id"),
        ("incomplete_rows", None),
        ("date_gaps", "order_date"),
        ("date_spike", "order_date"),
        ("implausible_dates", "order_date"),
        ("partial_period", "order_date"),
    ],
)
def test_every_planted_problem_is_found(sample_run, rule, column):
    _, _, summary, _ = sample_run
    assert _find(summary, rule, column), f"{rule} on {column} not found"


def test_planted_counts(sample_run):
    _, _, summary, _ = sample_run
    q = summary["quality"]
    assert q["duplicates"]["duplicate_rows"] == 12
    assert q["key_duplicates"]["duplicate_rows"] == 15  # the 12 exact copies + 3 edited orders
    assert q["outliers"]["delivery_days"]["code_like_values"] == {"999": 3}
    missing = {c["name"]: c for c in q["missing"]["columns"]}
    assert missing["rating"]["disguised_values"] == {"'-'": 92}  # 90 planted + 2 among the duplicated rows
    assert missing["legacy_code"]["missing_pct"] == 100.0
    assert summary["analysis"]["excluded_id_columns"] == ["order_id"]


def test_planted_time_problems(sample_run):
    _, _, summary, _ = sample_run
    time = summary["analysis"]["time"]
    dates = time["date_columns"]["order_date"]
    assert (dates["source"], dates["date_order"]) == ("text", "day-first")
    assert dates["future"] == 1 and dates["future_examples"][0].startswith("2052-")
    assert dates["placeholders"] == {"1900-01-01": 1}
    t = time["trend"]
    assert (t["period"], t["first"], t["last"]) == ("week", "2025-01-01", "2025-12-31")
    assert t["rows_excluded_implausible"] == 2
    assert [(g["first"], g["periods"]) for g in t["gaps"]] == [("week of 2025-08-04", 4)]
    # Only the late load is a spike, not the busy November and December weeks.
    assert [(sp["label"], sp["top_day"]) for sp in t["spikes"]] == [("week of 2025-09-15", "2025-09-15")]
    assert [p["which"] for p in t["partial"]] == ["first", "last"]
    assert t["value_column"] == "revenue"
    # The outage's orders (and their duplicated copies) all carry the load date.
    spike = t["spikes"][0]
    assert (spike["rows"], spike["top_day_rows"], t["typical_rows"]) == (104, 93, 23)


def test_no_false_alarms_on_clean_columns(sample_run):
    _, _, summary, _ = sample_run
    flagged = {c for f in summary["findings"] for c in f["columns"]}
    for clean in ("channel", "category", "returning_customer", "customer_id"):
        assert clean not in flagged, clean


def test_report_and_input_untouched(sample_run):
    path, before, summary, out = sample_run
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert summary["source"]["unchanged_after_run"] is True
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "categories_order_date" not in report  # no "most common dates" chart
    assert "Only 64.9% of rows are complete, not counting the empty column `legacy_code`" in report
