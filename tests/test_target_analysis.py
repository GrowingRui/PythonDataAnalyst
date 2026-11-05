"""--target: group comparisons, strength measures, leakage, charts and report section 4b."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from generate_eda import likely_id_columns, run_eda
from inspect_data import DatasetError, classify_columns, detect_type_issues
from report import REPORT_SECTIONS, TARGET_SECTION, build_findings
from target_analysis import (
    MIN_GROUP_ROWS,
    cramers_v,
    fold_groups,
    goodman_kruskal_lambda,
    rank_eta,
    separation,
    target_summary,
)


def _summary(df: pd.DataFrame, target: str, max_groups: int = 10) -> dict:
    kinds = classify_columns(df)
    return target_summary(df, target, kinds, detect_type_issues(df, kinds), likely_id_columns(df, kinds),
                          max_groups=max_groups)


@pytest.fixture
def sales() -> pd.DataFrame:
    """Revenue that really differs by category, not by channel; one leaky column of each kind."""
    rng = np.random.default_rng(13)
    n = 600
    category = rng.choice(["Electronics", "Home", "Books"], n)
    base = pd.Series(category).map({"Electronics": 200.0, "Home": 60.0, "Books": 15.0}).to_numpy()
    revenue = np.round(base * rng.lognormal(0, 0.5, n), 2)
    revenue[:12] = np.nan  # rows without a target
    return pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "category": category,
        "channel": rng.choice(["Online", "Store"], n),
        "region": np.where(np.arange(n) < 20, "Islands", rng.choice(["North", "South"], n)),  # 20 rows: too small
        "quantity": rng.integers(1, 5, n),
        "revenue": revenue,
        "revenue_eur": np.round(revenue * 0.92, 2),  # the same amount in another currency
        "size_band": np.where(revenue < 40, "S", np.where(revenue < 150, "M", "L")),  # computed from revenue
    })


@pytest.fixture
def churn() -> pd.DataFrame:
    """A yes/no target that depends on the plan; a reason column filled in only after churning."""
    rng = np.random.default_rng(7)
    n = 800
    plan = rng.choice(["basic", "plus", "pro"], n)
    p = pd.Series(plan).map({"basic": 0.35, "plus": 0.15, "pro": 0.05}).to_numpy()
    churned = np.where(rng.random(n) < p, "yes", "no")
    tenure = np.round(rng.gamma(3, 6, n) * np.where(churned == "yes", 0.5, 1.0), 1)
    return pd.DataFrame({
        "plan": plan,
        "country": rng.choice(["DE", "FR", "IT"], n),
        "tenure_months": tenure,
        "cancel_reason": np.where(churned == "yes", rng.choice(["price", "service"], n), "none"),
        "churned": churned,
    })


# --------------------------------------------------------------------------
# Strength measures
# --------------------------------------------------------------------------


def test_rank_eta_is_zero_without_and_one_with_complete_separation():
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.normal(size=3000))
    groups = pd.Series(rng.choice(list("abc"), 3000))
    assert rank_eta(noise, groups) < 0.06
    values = pd.Series(np.r_[np.full(50, 1.0), np.full(50, 2.0), np.full(50, 3.0)])
    labels = pd.Series(["a"] * 50 + ["b"] * 50 + ["c"] * 50)
    assert rank_eta(values, labels) == pytest.approx(1.0)


def test_rank_eta_corrects_for_many_small_groups():
    """300 random groups of 3 rows 'explain' a lot of variance by chance; the correction removes it."""
    rng = np.random.default_rng(2)
    values = pd.Series(rng.normal(size=900))
    groups = pd.Series(np.repeat(np.arange(300), 3).astype(str))
    stats = values.rank().groupby(groups).agg(["mean", "size"])
    r = values.rank()
    raw = (stats["size"] * (stats["mean"] - r.mean()) ** 2).sum() / ((r - r.mean()) ** 2).sum()
    assert raw > 0.3
    assert rank_eta(values, groups) < 0.2


def test_rank_eta_ignores_a_few_extreme_values():
    """On ranks, one huge order doesn't make two otherwise identical groups 'differ'."""
    rng = np.random.default_rng(3)
    v = rng.lognormal(3, 0.5, 400)
    g = np.repeat(["a", "b"], 200)
    v[0] = 1e7
    assert rank_eta(pd.Series(v), pd.Series(g)) < 0.15


def test_cramers_v():
    a = pd.Series(list("xy") * 200)
    assert cramers_v(a, a) == pytest.approx(1.0, abs=0.01)
    rng = np.random.default_rng(4)
    assert cramers_v(pd.Series(rng.choice(list("abc"), 2000)), pd.Series(rng.choice(list("xy"), 2000))) < 0.05
    assert cramers_v(pd.Series(["a"] * 10), pd.Series(list("xy") * 5)) is None  # one group: nothing to compare


def test_separation_and_lambda():
    values = pd.Series(np.arange(300, dtype=float))
    bands = pd.Series(np.where(values < 100, "low", np.where(values < 200, "mid", "high")))
    assert separation(values, bands) == pytest.approx(1.0)
    rng = np.random.default_rng(5)
    assert separation(values, pd.Series(rng.choice(["a", "b"], 300))) < 0.15
    target = pd.Series(["yes"] * 30 + ["no"] * 270)
    reason = pd.Series(["price"] * 30 + ["none"] * 270)
    assert goodman_kruskal_lambda(reason, target) == pytest.approx(1.0)
    assert goodman_kruskal_lambda(pd.Series(rng.choice(["a", "b"], 300)), target) == pytest.approx(0.0)


def test_fold_groups_keeps_top_values_and_marks_missing():
    s = pd.Series(["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["d"] + [None, None])
    folded = fold_groups(s, 2)
    assert folded.value_counts().to_dict() == {"a": 5, "b": 4, "Other": 4, "(missing)": 2}


# --------------------------------------------------------------------------
# Choosing the target
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("revnue", "not found"),
        ("REVENUE", "Did you mean 'revenue'"),
        ("order_id", "row ID"),
        ("notes", "holds text values"),
        ("price_text", "numbers stored as text"),
        ("city", "at most 8"),
        ("constant", "same value in every row"),
    ],
)
def test_unusable_targets_are_explained(target, message):
    n = 100
    df = pd.DataFrame({
        "order_id": np.arange(n),
        "revenue": np.linspace(1, 100, n),
        "notes": [f"customer note number {i} with some words" for i in range(n)],
        "price_text": [f"${i}.50" for i in range(n)],
        "city": [f"city{i % 12}" for i in range(n)],
        "constant": ["x"] * n,
    })
    with pytest.raises(DatasetError, match=message.replace("(", r"\(")):
        _summary(df, target)


def test_numeric_with_two_values_is_a_categorical_target():
    df = pd.DataFrame({"churn": [0, 1] * 50, "plan": ["a", "b", "c", "d"] * 25})
    t = _summary(df, "churn")
    assert t["kind"] == "categorical" and t["focus_class"] == "1"


def test_focus_class_is_the_yes_like_one_or_else_the_rarer_one():
    df = pd.DataFrame({"outcome": ["won"] * 30 + ["lost"] * 70, "g": ["a", "b"] * 50})
    assert _summary(df, "outcome")["focus_class"] == "won"
    df = pd.DataFrame({"active": [True] * 80 + [False] * 20, "g": ["a", "b"] * 50})
    assert _summary(df, "active")["focus_class"] == "True"


# --------------------------------------------------------------------------
# Numeric target
# --------------------------------------------------------------------------


def test_numeric_target_ranks_the_real_difference_first(sales):
    t = _summary(sales, "revenue")
    assert t["kind"] == "numeric"
    assert (t["rows_used"], t["rows_missing_target"]) == (588, 12)
    clean = [e for e in t["by_group"] if not e.get("leakage")]
    assert [e["column"] for e in clean][0] == "category"
    assert all(e["strength"] < 0.1 for e in clean[1:])  # region and channel are noise here
    cat = clean[0]
    assert cat["strength_label"] in ("strong", "very strong")
    assert [g["value"] for g in cat["groups"]] == ["Electronics", "Home", "Books"]  # by median, highest first
    assert (cat["highest"]["value"], cat["lowest"]["value"]) == ("Electronics", "Books")
    assert t["log_scale"] is True  # lognormal amounts spanning more than 20x


def test_small_groups_are_marked_and_not_named_highest(sales):
    region = next(e for e in _summary(sales, "revenue")["by_group"] if e["column"] == "region")
    islands = next(g for g in region["groups"] if g["value"] == "Islands")
    assert islands["small"] and islands["count"] < MIN_GROUP_ROWS
    assert "Islands" not in (region["highest"]["value"], region["lowest"]["value"])


def test_numeric_columns_are_correlated_with_the_target(sales):
    t = _summary(sales, "revenue")
    qty = next(e for e in t["by_numeric"] if e["column"] == "quantity")
    assert qty["n"] == 588 and abs(qty["spearman"]) < 0.15
    assert "order_id" not in [e["column"] for e in t["by_numeric"]]  # IDs are skipped
    assert {"column": "order_id", "reason": "an ID column"} in t["skipped"]


def test_leakage_is_flagged_for_a_derived_amount_and_a_band(sales):
    t = _summary(sales, "revenue")
    leaks = {e["column"]: e["measure"] for e in t["leakage"]}
    assert leaks == {"revenue_eur": "correlation", "size_band": "separation"}


def test_other_and_missing_groups(sales):
    df = sales.copy()
    df.loc[:49, "category"] = None
    t = _summary(df, "revenue", max_groups=2)
    cat = next(e for e in t["by_group"] if e["column"] == "category")
    values = [g["value"] for g in cat["groups"]]
    assert "Other" in values and "(missing)" in values and cat["other_values"] > 0


# --------------------------------------------------------------------------
# Categorical target
# --------------------------------------------------------------------------


def test_categorical_target_rates_and_leakage(churn):
    t = _summary(churn, "churned")
    assert t["kind"] == "categorical" and t["focus_class"] == "yes"
    assert [c["value"] for c in t["classes"]] == ["no", "yes"]
    plan = next(e for e in t["by_group"] if e["column"] == "plan")
    assert [g["value"] for g in plan["groups"]] == ["basic", "plus", "pro"]  # highest churn rate first
    assert plan["highest"]["rate"] > plan["lowest"]["rate"] + 15
    country = next(e for e in t["by_group"] if e["column"] == "country")
    assert country["strength"] < plan["strength"]
    tenure = next(e for e in t["by_numeric"] if e["column"] == "tenure_months")
    medians = {c["value"]: c["median"] for c in tenure["classes"]}
    assert medians["yes"] < medians["no"]
    assert [e["column"] for e in t["leakage"]] == ["cancel_reason"]
    assert t["leakage"][0]["measure"] == "lambda"


def test_multiclass_target_has_no_focus_class():
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"tier": rng.choice(["gold", "silver", "bronze"], 300), "g": rng.choice(["a", "b"], 300)})
    t = _summary(df, "tier")
    assert t["focus_class"] is None
    assert set(t["by_group"][0]["groups"][0]["rates"]) == {"gold", "silver", "bronze"}


# --------------------------------------------------------------------------
# Findings, report and charts
# --------------------------------------------------------------------------


@pytest.fixture
def target_run(sales, tmp_path):
    out = tmp_path / "out"
    return run_eda(sales, out, target="revenue"), out


def test_findings_flag_leakage_and_group_differences(target_run):
    summary, _ = target_run
    rules = [(f["rule"], tuple(f["columns"])) for f in summary["findings"]]
    assert ("possible_leakage", ("revenue_eur", "revenue")) in rules
    assert ("possible_leakage", ("size_band", "revenue")) in rules
    assert ("target_difference", ("category", "revenue")) in rules
    assert ("target_difference", ("channel", "revenue")) not in rules
    # The leaky pair is reported once, as leakage, not again as near-duplicate columns.
    assert not [f for f in summary["findings"] if f["rule"] == "near_duplicate_columns"
                and set(f["columns"]) == {"revenue", "revenue_eur"}]
    assert summary["findings"][0]["severity"] == "high"


def test_report_has_section_4b_and_target_charts(target_run):
    summary, out = target_run
    report = (out / "report.md").read_text(encoding="utf-8")
    headings = [line[3:] for line in report.splitlines() if line.startswith("## ")]
    assert headings == [*REPORT_SECTIONS[:4], TARGET_SECTION, *REPORT_SECTIONS[4:]]
    assert "### `revenue` by `category`" in report
    assert "12 row(s) without a value are left out" in report
    assert "too small to compare" in report
    assert "Simpson's paradox" in report
    assert "no significance tests" in report
    charts = [c for c in summary["charts"] if c["type"] == "target_box"]
    assert len(charts) == 3
    assert all((out / c["file"]).is_file() for c in charts)
    assert summary["charts"][1]["type"] == "target_box"  # right after the missing-values chart


def test_without_target_there_is_no_section_4b(sales, tmp_path):
    summary = run_eda(sales, tmp_path / "out")
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert summary["analysis"]["target"] is None
    assert TARGET_SECTION not in report
    assert not [f for f in summary["findings"] if f["rule"] in ("possible_leakage", "target_difference")]
    assert "--target" in report  # suggested as a next step


def test_categorical_target_run_draws_rate_charts(churn, tmp_path):
    summary = run_eda(churn, tmp_path / "out", target="churned")
    rates = [c for c in summary["charts"] if c["type"] == "target_rates"]
    assert rates and rates[0]["column"] in ("cancel_reason", "plan")
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Numeric columns by class (medians)" in report
    assert "| yes % |" in report


def test_numeric_target_is_the_value_in_the_time_chart(tmp_path):
    rng = np.random.default_rng(11)
    n = 400
    df = pd.DataFrame({
        "day": pd.date_range("2025-01-01", periods=n, freq="D"),
        "amount": rng.lognormal(3, 0.4, n),
        "visits": rng.integers(1, 50, n),
        "shop": rng.choice(["a", "b"], n),
    })
    summary = run_eda(df, tmp_path / "out", target="visits")
    assert summary["analysis"]["time"]["trend"]["value_column"] == "visits"


def test_bad_target_leaves_no_output_folder(sales, tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        run_eda(sales, tmp_path / "out", target="profit")
    assert not (tmp_path / "out").exists()


def test_build_findings_without_target_key_still_works(target_run):
    summary, _ = target_run
    old = json.loads(json.dumps(summary))
    del old["analysis"]["target"]
    rules = {f["rule"] for f in build_findings(old)}
    assert "possible_leakage" not in rules and "near_duplicate_columns" in rules


def test_cli_target(sales, tmp_path, scripts_dir):
    path = tmp_path / "sales.csv"
    sales.to_csv(path, index=False)
    out = tmp_path / "out"
    run = subprocess.run(
        [sys.executable, str(scripts_dir / "generate_eda.py"), str(path), "--output", str(out), "--target", "revenue"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert run.returncode == 0, run.stderr
    assert "Target: revenue (numeric, 588 rows used)" in run.stdout
    assert "possible leakage" in run.stdout
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["settings"]["target"] == "revenue"

    bad = subprocess.run(
        [sys.executable, str(scripts_dir / "generate_eda.py"), str(path), "--output", str(tmp_path / "o2"),
         "--target", "category_text"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert bad.returncode == 2 and "not found" in bad.stderr
    assert not (tmp_path / "o2").exists()
