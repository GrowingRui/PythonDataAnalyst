"""Tests for report.py (findings, report.md) and the report step of generate_eda.py."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from generate_eda import run_eda
from inspect_data import load_dataset
from report import (
    INTERPRETATION_MARKER,
    REPORT_SECTIONS,
    build_findings,
    code,
    md_table,
    render_report,
)


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """No gaps, no duplicates, uniform (no IQR outliers), unrelated columns, balanced groups."""
    rng = np.random.default_rng(10)
    n = 300
    return pd.DataFrame(
        {
            "x": rng.uniform(0, 100, n),
            "y": rng.uniform(0, 1, n),
            "group": np.tile(["a", "b", "c"], n // 3),
        }
    )


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """One planted problem per rule, each with a known size."""
    rng = np.random.default_rng(11)
    n = 100
    x = rng.uniform(1, 100, n)
    df = pd.DataFrame(
        {
            "x": x,
            "double_x": 2 * x,  # derived column: near-perfect correlation
            "income": rng.lognormal(3, 1.3, n),  # highly right-skewed
            "half_empty": np.where(np.arange(n) % 2 == 0, np.nan, 1.0) * rng.uniform(1, 2, n),  # 50% missing
            "some_gaps": np.where(np.arange(n) < 15, np.nan, rng.uniform(0, 1, n)),  # 15% missing
            "few_gaps": np.where(np.arange(n) < 3, np.nan, rng.uniform(0, 1, n)),  # 3% missing
            "nothing": np.nan,  # completely empty
            "country": "DE",  # constant
            "answer": rng.choice(["yes", "no", "unknown"], n),  # "unknown" is a placeholder
            "amount_text": [f"{v:.2f}" for v in rng.uniform(0, 50, n)],  # numbers as text
        }
    )
    return pd.concat([df, df.iloc[:5]], ignore_index=True)  # 5 exact duplicates


def _rules(findings):
    return {f["rule"] for f in findings}


def _by_rule(findings, rule):
    return [f for f in findings if f["rule"] == rule]


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


def test_clean_data_has_no_findings(clean_df, tmp_path):
    summary = run_eda(clean_df, tmp_path / "out")
    assert summary["findings"] == []
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "No rule flagged anything" in report
    assert "No cleaning is suggested" in report


def test_planted_problems_are_found(messy_df, tmp_path):
    findings = run_eda(messy_df, tmp_path / "out")["findings"]
    assert {
        "empty_columns", "missing_values", "missing_minor", "disguised_missing", "duplicate_rows",
        "numbers_as_text", "constant_column", "skewed", "near_duplicate_columns",
    } <= _rules(findings)

    assert _by_rule(findings, "empty_columns")[0]["columns"] == ["nothing"]
    missing = {f["columns"][0]: f["severity"] for f in _by_rule(findings, "missing_values")}
    assert missing == {"half_empty": "high", "some_gaps": "medium"}
    assert _by_rule(findings, "missing_minor")[0]["columns"] == ["few_gaps"]
    assert _by_rule(findings, "constant_column")[0]["columns"] == ["country"]
    assert _by_rule(findings, "disguised_missing")[0]["columns"] == ["answer"]
    assert _by_rule(findings, "skewed")[0]["columns"] == ["income"]
    assert _by_rule(findings, "near_duplicate_columns")[0]["columns"] == ["x", "double_x"]
    dup = _by_rule(findings, "duplicate_rows")[0]
    assert dup["title"].startswith("5 exact duplicate rows")
    assert dup["severity"] == "medium"  # 5 of 105 rows = 4.8%, just under the 5% "high" threshold


def test_findings_are_ranked_by_severity(messy_df, tmp_path):
    findings = run_eda(messy_df, tmp_path / "out")["findings"]
    order = {"high": 0, "medium": 1, "low": 2}
    ranks = [order[f["severity"]] for f in findings]
    assert ranks == sorted(ranks)
    assert findings[0]["severity"] == "high"


def test_skewed_column_outliers_are_low_severity(tmp_path):
    # Lognormal data always has IQR "outliers"; on skewed data they're expected.
    rng = np.random.default_rng(12)
    findings = run_eda(pd.DataFrame({"price": rng.lognormal(3, 1, 500)}), tmp_path / "out")["findings"]
    outliers = _by_rule(findings, "outliers")[0]
    assert outliers["severity"] == "low"
    assert "expected" in outliers["detail"]


def test_symmetric_column_with_many_outliers_is_medium(tmp_path):
    # Outliers placed symmetrically around 12: skewness is exactly 0, 10% are flagged.
    values = [10.0, 11, 12, 13, 14] * 18 + [-76.0] * 5 + [100.0] * 5
    findings = run_eda(pd.DataFrame({"v": values}), tmp_path / "out")["findings"]
    assert "skewed" not in _rules(findings)
    out = _by_rule(findings, "outliers")[0]
    assert out["severity"] == "medium"
    assert out["title"] == "`v`: 10 values (10.0%) outside the IQR fences"


def test_correlation_strength_thresholds(tmp_path):
    rng = np.random.default_rng(13)
    n = 400
    x = rng.uniform(0, 1, n)
    df = pd.DataFrame(
        {
            "x": x,
            "strong": x + rng.normal(0, 0.12, n),  # r around 0.9: medium
            "weak": x + rng.normal(0, 1.5, n),  # r around 0.2: no finding
        }
    )
    findings = run_eda(df, tmp_path / "out")["findings"]
    pairs = {tuple(f["columns"]): f for f in findings if f["rule"] == "strong_correlation"}
    assert ("x", "strong") in pairs and pairs[("x", "strong")]["severity"] == "medium"
    assert ("x", "weak") not in pairs
    assert "not evidence that one causes" in pairs[("x", "strong")]["detail"]


def test_correlation_title_says_when_only_ranks_agree(tmp_path):
    # y = e^x: perfectly monotonic (Spearman 1) but far from a straight line.
    x = np.linspace(0.5, 20, 60)
    findings = run_eda(pd.DataFrame({"x": x, "y": np.exp(x)}), tmp_path / "out")["findings"]
    title = _by_rule(findings, "strong_correlation")[0]["title"]
    assert "by rank, much less linearly" in title


def test_code_values_get_their_own_finding(tmp_path):
    df = pd.DataFrame({"days": [1.0, 2, 3, 4, 5, 6, 7] * 10 + [999] * 3})
    findings = run_eda(df, tmp_path / "out")["findings"]
    code_finding = _by_rule(findings, "possible_missing_code")[0]
    assert code_finding["severity"] == "medium" and "999" in code_finding["title"]
    outliers = _by_rule(findings, "outliers")[0]
    assert "probably legitimate" not in outliers["detail"]  # codes aren't "expected" skew


def test_incomplete_rows_ignore_the_empty_column(tmp_path):
    n = 40
    df = pd.DataFrame(
        {
            "a": [np.nan if i % 4 == 0 else float(i) for i in range(n)],  # 25% missing
            "b": [np.nan if i % 4 == 1 else float(i) for i in range(n)],  # another 25%
            "empty": [np.nan] * n,
        }
    )
    findings = run_eda(df, tmp_path / "out")["findings"]
    title = _by_rule(findings, "incomplete_rows")[0]["title"]
    assert title == "Only 50.0% of rows are complete, not counting the empty column `empty`"


def test_no_category_chart_for_dates_stored_as_text(tmp_path):
    dates = pd.date_range("2025-01-01", periods=30).strftime("%d/%m/%Y")
    df = pd.DataFrame({"when": list(dates) * 2, "team": ["a", "b", "c"] * 20})
    summary = run_eda(df, tmp_path / "out")
    charted = {c.get("column") for c in summary["charts"] if c["type"] == "top_categories"}
    assert charted == {"team"}


def test_key_columns_count_as_ids_even_when_repeated(tmp_path):
    rng = np.random.default_rng(14)
    df = pd.DataFrame({"order_id": list(range(1, 51)) + [1, 2], "v": rng.uniform(0, 1, 52),
                       "w": rng.uniform(0, 1, 52)})
    summary = run_eda(df, tmp_path / "out", key_columns=["order_id"])
    assert summary["analysis"]["excluded_id_columns"] == ["order_id"]
    assert "order_id" not in summary["analysis"]["correlations"]["columns"]
    assert "duplicate_keys" in _rules(summary["findings"])
    assert _by_rule(summary["findings"], "duplicate_keys")[0]["severity"] == "high"


def test_build_findings_is_pure(messy_df, tmp_path):
    summary = run_eda(messy_df, tmp_path / "out")
    before = json.dumps(summary, sort_keys=True)
    assert build_findings(summary) == summary["findings"]
    assert json.dumps(summary, sort_keys=True) == before  # the summary isn't modified


# --------------------------------------------------------------------------
# report.md
# --------------------------------------------------------------------------


def test_report_has_every_section_in_order(messy_df, tmp_path):
    run_eda(messy_df, tmp_path / "out")
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    positions = [report.index(f"## {s}") for s in REPORT_SECTIONS]
    assert positions == sorted(positions)
    # The analyst's interpretation goes after the findings and before the charts.
    assert positions[3] < report.index(INTERPRETATION_MARKER) < positions[4]


def test_chart_links_point_to_existing_files(messy_df, tmp_path):
    summary = run_eda(messy_df, tmp_path / "out")
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report)
    assert len(links) == len(summary["charts"]) > 0
    for link in links:
        assert (tmp_path / "out" / link).is_file()


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"city": ["Paris", "Rome", "Paris"]}),  # no numeric columns
        pd.DataFrame({"v": [1.0, 2, 2, 3, 9]}),  # one numeric column, tiny
        pd.DataFrame({"a": [np.nan] * 4, "b": [np.nan] * 4}),  # everything empty
        pd.DataFrame({"weird | name `x`": [1.0, 2, 3, 4, 5, 6], "Région": list("ababab")}),
    ],
    ids=["no-numeric", "single-column", "all-empty", "odd-names"],
)
def test_report_edge_cases_render_every_section(df, tmp_path):
    run_eda(df, tmp_path / "out")
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    for s in REPORT_SECTIONS:
        assert f"## {s}" in report


def test_small_dataset_limitation(tmp_path):
    run_eda(pd.DataFrame({"v": [1.0, 2, 3, 4, 5]}), tmp_path / "out")
    assert "Small dataset" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_excel_other_sheets_are_mentioned(excel_two_sheets, tmp_path):
    df = load_dataset(excel_two_sheets)
    run_eda(df, tmp_path / "out")
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Only sheet `first` was analysed" in report and "`2025`" in report


def test_render_report_without_stored_findings(messy_df, tmp_path):
    summary = run_eda(messy_df, tmp_path / "out")
    del summary["findings"]
    text = render_report(summary)
    assert "[high]" in text  # findings were rebuilt


def test_md_table_escapes_and_aligns():
    table = md_table(["name", "n", "share"], [["a|b", 1200, "5.0%"], ["c\nd", None, "1.0%"]])
    lines = table.splitlines()
    assert lines[1] == "|---|---:|---|"  # only the all-number column is right-aligned
    assert "a\\|b" in lines[2] and "1,200" in lines[2]
    assert "c d" in lines[3] and "–" in lines[3]


def test_code_survives_backticks():
    assert code("plain") == "`plain`"
    assert code("has `tick`") == "`` has `tick` ``"


# --------------------------------------------------------------------------
# Read-only guarantee and CLI
# --------------------------------------------------------------------------


def test_input_hash_recorded_and_unchanged(messy_df, tmp_path):
    data = tmp_path / "messy.csv"
    messy_df.to_csv(data, index=False)
    before = hashlib.sha256(data.read_bytes()).hexdigest()
    summary = run_eda(load_dataset(data), tmp_path / "out")
    assert summary["source"]["sha256"] == before
    assert summary["source"]["unchanged_after_run"] is True
    assert hashlib.sha256(data.read_bytes()).hexdigest() == before
    assert before in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_in_memory_data_has_no_hash(clean_df, tmp_path):
    summary = run_eda(clean_df, tmp_path / "out")
    assert "sha256" not in summary["source"]


def test_cli_produces_report_folder(scripts_dir, messy_df, tmp_path):
    data = tmp_path / "messy.csv"
    messy_df.to_csv(data, index=False)
    before = hashlib.sha256(data.read_bytes()).hexdigest()
    r = subprocess.run(
        [sys.executable, str(scripts_dir / "generate_eda.py"), str(data), "--output", str(tmp_path / "out")],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    assert "report.md" in r.stdout and "Findings:" in r.stdout
    out = tmp_path / "out"
    assert (out / "report.md").is_file() and (out / "summary.json").is_file()

    def no_nan(token):
        raise AssertionError(f"summary.json contains {token}, which isn't valid JSON")

    saved = json.loads((out / "summary.json").read_text(encoding="utf-8"), parse_constant=no_nan)
    assert saved["findings"] and saved["source"]["sha256"] == before
    assert hashlib.sha256(data.read_bytes()).hexdigest() == before
