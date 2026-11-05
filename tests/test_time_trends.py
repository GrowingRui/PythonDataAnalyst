"""Tests for time_trends.py and the time trend in generate_eda.py / report.py."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from generate_eda import run_eda
from inspect_data import DatasetError, classify_columns, detect_type_issues
from report import REPORT_SECTIONS, build_findings
from time_trends import choose_period, parse_dates, pick_value_column, time_summary

NOW = pd.Timestamp("2026-01-15")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _summary(df, **kwargs):
    kinds = classify_columns(df)
    return time_summary(df, kinds, detect_type_issues(df, kinds), now=NOW, **kwargs)


def _rows_per_week(counts, start="2025-01-06"):
    """One row per order, ``counts[i]`` orders in week i, spread over Monday to Friday."""
    dates = []
    for week, n in enumerate(counts):
        monday = pd.Timestamp(start) + pd.Timedelta(weeks=week)
        dates += [monday + pd.Timedelta(days=i % 5) for i in range(n)]
    return pd.DataFrame({"order_date": dates, "revenue": np.arange(len(dates), dtype=float) % 50 + 1})


# --------------------------------------------------------------------------
# Choosing the period
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "rows", "expected"),
    [
        ("2025-01-01", "2025-01-30", 3000, "day"),  # 30 days
        ("2025-01-01", "2025-12-31", 1200, "week"),  # 365 days: too many for days
        ("2025-01-01", "2025-12-31", 100, "month"),  # 53 weeks of ~2 rows: months have enough
        ("2021-01-01", "2025-12-31", 5000, "month"),  # 60 months
        ("2006-01-01", "2025-12-31", 5000, "quarter"),  # 240 months, 80 quarters
        ("1966-01-01", "2025-12-31", 5000, "year"),
        ("2025-01-01", "2025-01-10", 5, "day"),  # sparse, but weeks would give only 2 points
    ],
)
def test_choose_period(start, end, rows, expected):
    assert choose_period(pd.Timestamp(start), pd.Timestamp(end), rows) == expected


def test_period_override_and_unknown_period():
    df = _rows_per_week([20] * 30)
    assert _summary(df, period="month")["trend"]["period"] == "month"
    assert _summary(df, period="month")["trend"]["period_chosen_by"] == "--period"
    with pytest.raises(DatasetError, match="Unknown period"):
        _summary(df, period="fortnight")


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_day_first_text_is_parsed_day_first():
    out = parse_dates(pd.Series(["13/02/2025", "01/03/2025", None]), "day-first")
    assert list(out[:2]) == [pd.Timestamp("2025-02-13"), pd.Timestamp("2025-03-01")]
    assert pd.isna(out[2])


def test_ambiguous_text_is_read_month_first_and_labelled():
    df = pd.DataFrame({"d": [f"0{m}/0{d}/2025" for m in range(1, 10) for d in range(1, 4)]})
    s = _summary(df)
    assert s["date_columns"]["d"]["date_order"] == "ambiguous"
    # Month-first, "09/03/2025" is 3 September; day-first it would be 9 March.
    assert s["trend"]["last"] == "2025-09-03"


def test_mixed_formats_and_junk():
    s = parse_dates(pd.Series(["2025-01-05", "5 March 2025", "no idea", "2025-02-01"]), None)
    assert s[0] == pd.Timestamp("2025-01-05") and s[1] == pd.Timestamp("2025-03-05")
    assert pd.isna(s[2]) and s[3] == pd.Timestamp("2025-02-01")


def test_timezone_aware_dates_keep_their_clock_time():
    s = pd.Series(pd.date_range("2025-01-01 09:00", periods=3, freq="D", tz="Europe/Berlin"))
    out = parse_dates(s)
    assert out.dt.tz is None and out[0] == pd.Timestamp("2025-01-01 09:00")


def test_excel_style_datetime_column_needs_no_parsing():
    df = pd.DataFrame({"when": pd.date_range("2025-01-01", periods=200, freq="D")})
    s = _summary(df)
    assert s["date_columns"]["when"]["source"] == "datetime"
    assert s["trend"]["period"] == "week"


# --------------------------------------------------------------------------
# Gaps, spikes, partial periods
# --------------------------------------------------------------------------


def test_gap_is_found_when_unlikely_to_be_chance():
    counts = [20] * 10 + [0, 0] + [20] * 10
    t = _summary(_rows_per_week(counts))["trend"]
    assert t["period"] == "week"
    assert len(t["gaps"]) == 1
    g = t["gaps"][0]
    assert (g["first"], g["last"], g["periods"]) == ("week of 2025-03-17", "week of 2025-03-24", 2)
    assert "million" in g["chance"]


def test_empty_periods_in_sparse_data_are_not_gaps():
    counts = [1, 0, 2, 1, 0, 1, 3, 0, 1, 2, 1, 0, 1, 1, 2, 0, 1, 1, 2, 1] * 3
    t = _summary(_rows_per_week(counts), period="week")["trend"]
    assert t["gaps"] == []
    assert t["empty_periods_by_chance"] == counts.count(0)


def test_empty_weekends_in_daily_data_are_not_gaps():
    days = pd.bdate_range("2025-01-06", periods=40)  # Monday to Friday only, 8 weeks
    df = pd.DataFrame({"d": np.repeat(days, 12)})
    t = _summary(df)["trend"]
    assert t["period"] == "day"
    assert t["gaps"] == [] and t["empty_weekend_days"] == 14


def test_spike_is_found_with_its_busiest_day():
    counts = [20] * 15 + [95] + [20] * 10
    df = _rows_per_week(counts)
    df.loc[df["order_date"].dt.to_period("W-SUN") == pd.Period("2025-04-21", "W-SUN"), "order_date"] = \
        pd.Timestamp("2025-04-23")  # the whole spike week loaded on one day
    t = _summary(df)["trend"]
    assert len(t["spikes"]) == 1
    sp = t["spikes"][0]
    assert sp["label"] == "week of 2025-04-21" and sp["rows"] == 95
    assert sp["top_day"] == "2025-04-23" and sp["top_day_rows"] == 95
    assert sp["ratio"] == pytest.approx(4.8)


def test_spike_counts_exact_duplicates():
    df = _rows_per_week([20] * 12)
    week = df[df["order_date"].dt.to_period("W-SUN") == pd.Period("2025-02-10", "W-SUN")]
    df = pd.concat([df, week, week, week], ignore_index=True)  # the same week imported 3 more times
    sp = _summary(df)["trend"]["spikes"][0]
    assert sp["rows"] == 80 and sp["exact_duplicates"] == 60


def test_seasonal_peak_is_not_a_spike():
    counts = [20] * 40 + [36] * 8  # a busy November/December: 1.8x
    assert _summary(_rows_per_week(counts))["trend"]["spikes"] == []


def test_partial_first_and_last_periods():
    df = _rows_per_week([20] * 20, start="2025-01-01")  # starts on a Wednesday
    df = df[df["order_date"] <= "2025-05-14"]  # and ends on one
    t = _summary(df)["trend"]
    assert [p["which"] for p in t["partial"]] == ["first", "last"]
    first, last = t["partial"]
    assert (first["label"], first["days_covered"], first["days_in_period"]) == ("week of 2024-12-30", 5, 7)
    assert (last["label"], last["days_covered"]) == ("week of 2025-05-12", 3)
    full = t["periods"][1:-1]
    assert not any(p.get("partial") for p in full)


def test_whole_months_are_not_partial():
    df = pd.DataFrame({"d": pd.date_range("2024-01-01", "2025-12-31", freq="D")})
    t = _summary(df, period="month")["trend"]
    assert t["partial"] == []


def test_checks_need_enough_full_periods():
    t = _summary(_rows_per_week([20, 0, 0, 20], start="2025-01-08"), period="week")["trend"]
    assert not t["checks_applied"] and t["gaps"] == []


# --------------------------------------------------------------------------
# Implausible dates
# --------------------------------------------------------------------------


def test_future_and_placeholder_dates_are_flagged_and_left_out():
    df = _rows_per_week([20] * 30)
    df.loc[0, "order_date"] = pd.Timestamp("2052-03-01")
    df.loc[1, "order_date"] = pd.Timestamp("1900-01-01")
    df.loc[2, "order_date"] = pd.Timestamp("1970-01-01")
    s = _summary(df)
    c = s["date_columns"]["order_date"]
    assert c["future"] == 1 and c["future_excluded"] and c["future_examples"] == ["2052-03-01"]
    assert c["placeholders"] == {"1900-01-01": 1, "1970-01-01": 1}
    t = s["trend"]
    assert t["rows_excluded_implausible"] == 3
    assert t["first"].startswith("2025-01") and t["last"].startswith("2025-08")  # range not stretched
    assert t["period"] == "week"


def test_mostly_future_dates_look_intended():
    df = pd.DataFrame({"due": pd.date_range("2025-12-01", periods=200, freq="D")})  # NOW is 2026-01-15
    s = _summary(df)
    c = s["date_columns"]["due"]
    assert c["future"] > 100 and not c["future_excluded"]
    assert s["trend"]["rows_excluded_implausible"] == 0
    summary = {"overview": {"rows": 200, "columns": 1, "column_info": []}, "statistics": {},
               "quality": {"missing": {"columns": [], "rows_with_missing": 0, "rows_with_missing_pct": 0},
                           "duplicates": {"duplicate_rows": 0}, "type_issues": [], "outliers": {}},
               "analysis": {"time": s}}
    assert not [f for f in build_findings(summary) if f["rule"] == "implausible_dates"]


# --------------------------------------------------------------------------
# Choosing columns
# --------------------------------------------------------------------------


def test_no_dates_gives_no_time_summary():
    assert _summary(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})) is None


def test_most_complete_date_column_is_used_and_others_checked():
    n = 100
    df = pd.DataFrame({
        "shipped": pd.Series(pd.date_range("2025-01-01", periods=n, freq="D")).where(np.arange(n) % 2 == 0),
        "ordered": pd.date_range("2025-01-01", periods=n, freq="D"),
    })
    df.loc[3, "shipped"] = pd.Timestamp("1900-01-01")
    s = _summary(df)
    assert s["column"] == "ordered" and s["chosen_by"] == "most dates"
    assert s["date_columns"]["shipped"]["placeholders"] == {"1900-01-01": 1}
    assert _summary(df, date_column="shipped")["column"] == "shipped"


def test_bad_date_column_is_explained():
    df = _rows_per_week([20] * 5)
    with pytest.raises(DatasetError, match="no such column"):
        _summary(df, date_column="nope")
    with pytest.raises(DatasetError, match="doesn't hold dates.*order_date"):
        _summary(df, date_column="revenue")


def test_value_column_prefers_amount_names_and_skips_ids():
    df = pd.DataFrame({"order_id": range(30), "age": range(30), "net_revenue": range(30), "qty": [1, 2] * 15})
    kinds = classify_columns(df)
    assert pick_value_column(df, kinds, ["order_id"]) == "net_revenue"
    assert pick_value_column(df.drop(columns=["net_revenue", "qty"]), kinds, ["order_id"]) == "age"
    assert pick_value_column(df[["order_id"]], kinds, ["order_id"]) is None


def test_all_dates_on_one_day():
    df = pd.DataFrame({"d": ["2025-03-01"] * 40, "x": range(40)})
    s = _summary(df)
    assert s["trend"] is None and "at least 3" in s["note"]


# --------------------------------------------------------------------------
# In the EDA run and the report
# --------------------------------------------------------------------------


@pytest.fixture
def dated_run(tmp_path):
    counts = [20] * 12 + [0, 0, 0] + [20] * 6 + [90] + [20] * 8
    df = _rows_per_week(counts, start="2025-01-01")
    df["order_date"] = df["order_date"].dt.strftime("%d/%m/%Y")  # dates as text, day-first
    df.loc[5, "order_date"] = "01/01/1900"
    summary = run_eda(df, tmp_path / "out", now=NOW)
    return summary, tmp_path / "out"


def test_eda_run_has_time_findings_chart_and_section(dated_run):
    summary, out = dated_run
    rules = {f["rule"] for f in summary["findings"]}
    assert {"date_gaps", "date_spike", "partial_period", "implausible_dates"} <= rules
    chart = next(c for c in summary["charts"] if c["type"] == "time_trend")
    assert chart["file"] == "charts/01_time_order_date_by_week.png"  # no missing values: it comes first
    assert chart["value_column"] == "revenue"
    assert (out / chart["file"]).read_bytes()[:8] == PNG_MAGIC
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "### Trend over time" in report
    assert "### Rows per week by `order_date`" in report
    assert "**Dates parsed from text** in `order_date` (day first, as detected)" in report
    assert "isn't a forecast" in report
    assert [line[3:] for line in report.splitlines() if line.startswith("## ")] == list(REPORT_SECTIONS)
    saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert saved["analysis"]["time"]["trend"]["gaps"][0]["periods"] == 3


def test_eda_without_dates_has_no_time_section(tmp_path):
    df = pd.DataFrame({"a": range(50), "b": ["x", "y"] * 25})
    summary = run_eda(df, tmp_path / "out")
    assert summary["analysis"]["time"] is None
    assert not any(c["type"] == "time_trend" for c in summary["charts"])
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Trend over time" not in report and "Time trend" not in report


def test_eda_with_one_day_explains_why_there_is_no_trend(tmp_path):
    df = pd.DataFrame({"d": pd.to_datetime(["2025-03-01"] * 10), "x": range(10)})
    summary = run_eda(df, tmp_path / "out", now=NOW)
    assert not any(c["type"] == "time_trend" for c in summary["charts"])
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "No trend for `d`" in report


def test_bad_date_column_leaves_no_output_folder(tmp_path):
    df = _rows_per_week([20] * 5)
    with pytest.raises(DatasetError):
        run_eda(df, tmp_path / "out", date_column="revenue")
    assert not (tmp_path / "out").exists()


def test_cli_date_column_and_period(tmp_path, scripts_dir):
    df = _rows_per_week([20] * 30)
    df["shipped"] = df["order_date"] + pd.Timedelta(days=2)
    path = tmp_path / "orders.csv"
    df.to_csv(path, index=False)
    out = tmp_path / "out"
    run = subprocess.run(
        [sys.executable, str(scripts_dir / "generate_eda.py"), str(path), "--output", str(out),
         "--date-column", "shipped", "--period", "month"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert run.returncode == 0, run.stderr
    assert "Time trend: shipped by month" in run.stdout
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["settings"]["date_column"] == "shipped" and summary["settings"]["period"] == "month"

    bad = subprocess.run(
        [sys.executable, str(scripts_dir / "generate_eda.py"), str(path), "--output", str(tmp_path / "o2"),
         "--period", "fortnight"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert bad.returncode == 2 and "invalid choice" in bad.stderr
