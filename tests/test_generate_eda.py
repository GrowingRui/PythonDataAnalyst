"""Tests for generate_eda.py: correlations, category summaries, charts and limits."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from generate_eda import (
    categorical_summary,
    correlation_columns,
    correlations,
    likely_id_columns,
    needs_log_view,
    run_eda,
    strength_label,
)
from inspect_data import DatasetError, classify_columns

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def shop_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0, 100, n)
    return pd.DataFrame(
        {
            "order_id": range(1, n + 1),
            "x": x,
            "y_linear": 2 * x + rng.normal(0, 5, n),  # strong positive
            "y_negative": -x + rng.normal(0, 5, n),  # strong negative
            "noise": rng.normal(0, 1, n),  # unrelated
            "region": rng.choice(["North", "South", "East"], n),
            "paid": rng.choice([True, False], n),
        }
    )


def _corr(df):
    kinds = classify_columns(df)
    return correlations(df, correlation_columns(df, kinds, likely_id_columns(df, kinds)))


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def test_perfect_linear_correlation():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
    pair = _corr(df)["pairs"][0]
    assert pair["pearson"] == 1.0 and pair["spearman"] == 1.0
    assert pair["direction"] == "positive"
    assert pair["strength"] == "very strong"


def test_spearman_catches_curved_relationship():
    # y = e^x always rises (Spearman 1) but is far from a straight line (Pearson < 1).
    x = np.linspace(0.5, 20, 40)
    pair = _corr(pd.DataFrame({"x": x, "y": np.exp(x)}))["pairs"][0]
    assert pair["spearman"] == 1.0
    assert pair["pearson"] < 0.8
    assert any("curved" in n for n in pair["notes"])


def test_single_outlier_inflates_pearson_not_spearman():
    rng = np.random.default_rng(1)
    a = list(rng.normal(0, 1, 50)) + [100]
    b = list(rng.normal(0, 1, 50)) + [100]
    pair = _corr(pd.DataFrame({"a": a, "b": b}))["pairs"][0]
    assert pair["pearson"] > 0.9  # driven entirely by one point
    assert abs(pair["spearman"]) < 0.4


def test_pairs_ranked_by_strength_and_ids_excluded(shop_df):
    corr = _corr(shop_df)
    assert "order_id" not in corr["columns"]
    top = corr["pairs"][0]
    assert {top["a"], top["b"]} == {"x", "y_linear"}
    strengths = [max(abs(p["pearson"]), abs(p["spearman"])) for p in corr["pairs"]]
    assert strengths == sorted(strengths, reverse=True)
    neg = next(p for p in corr["pairs"] if {p["a"], p["b"]} == {"x", "y_negative"})
    assert neg["direction"] == "negative" and neg["pearson"] < -0.9


def test_constant_and_sparse_columns_are_left_out():
    df = pd.DataFrame({"a": [1.0, 2, 3, 4], "const": [7.0] * 4, "sparse": [1.0, np.nan, np.nan, np.nan]})
    kinds = classify_columns(df)
    assert correlation_columns(df, kinds, []) == ["a"]
    assert _corr(df)["pairs"] == []


def test_pair_count_uses_rows_where_both_present():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6.0], "b": [2, 4, np.nan, 8, 10, 12.0]})
    pair = _corr(df)["pairs"][0]
    assert pair["n"] == 5
    assert any("unstable" in n for n in pair["notes"])  # n < 30


@pytest.mark.parametrize(
    "r, label", [(0.05, "negligible"), (-0.2, "weak"), (0.45, "moderate"), (-0.6, "strong"), (0.9, "very strong")]
)
def test_strength_labels(r, label):
    assert strength_label(r) == label


def test_likely_id_columns():
    df = pd.DataFrame(
        {
            "order_id": [10, 11, 12, 13],  # named id
            "customerID": [904, 17, 233, 51],  # named id, not consecutive
            "price": [3, 9, 4, 12],  # unique but neither
            "identity_score": [1, 5, 9, 13],  # "id" inside the name doesn't count
            "dup_id": [1, 1, 2, 3],  # not unique
            "year": [2021, 2022, 2023, 2024],  # short consecutive run: real data
        }
    )
    assert likely_id_columns(df, classify_columns(df)) == ["order_id", "customerID"]


def test_long_row_counter_is_an_id():
    df = pd.DataFrame({"n": range(1, 31), "shuffled": np.random.default_rng(0).permutation(30)})
    assert likely_id_columns(df, classify_columns(df)) == ["n"]  # counts up in row order


def test_categorical_summary_folds_other():
    df = pd.DataFrame({"c": list("aaaabbbccd") + [None]})
    s = categorical_summary(df, classify_columns(df), max_categories=2)["c"]
    assert [c["value"] for c in s["categories"]] == ["a", "b"]
    assert s["other"] == {"categories": 2, "count": 3, "pct": 30.0}
    assert s["count"] == 10 and s["missing"] == 1 and s["unique"] == 4
    assert sum(c["count"] for c in s["categories"]) + s["other"]["count"] == s["count"]


def test_categorical_summary_skips_text_and_numeric():
    df = pd.DataFrame({"email": [f"u{i}@x.com" for i in range(100)], "n": range(100), "flag": [True, False] * 50})
    assert set(categorical_summary(df, classify_columns(df), 10)) == {"flag"}


# --------------------------------------------------------------------------
# run_eda and charts
# --------------------------------------------------------------------------


def _pngs(out):
    return sorted(p.name for p in (out / "charts").glob("*.png"))


def test_run_eda_writes_charts_and_summary(shop_df, tmp_path):
    shop_df.loc[::10, "noise"] = np.nan
    out = tmp_path / "out"
    summary = run_eda(shop_df, out)
    files = _pngs(out)
    assert files[0] == "01_missing_values.png"
    assert "02_correlation_heatmap.png" in files
    assert sum("_scatter_" in f for f in files) == 3
    assert sum("_distribution_" in f for f in files) == 4  # x, y_linear, y_negative, noise (not order_id)
    assert sum("_categories_" in f for f in files) == 2  # region, paid
    for f in files:
        assert (out / "charts" / f).read_bytes()[:8] == PNG_MAGIC
    assert [c["file"] for c in summary["charts"]] == [f"charts/{f}" for f in files]

    saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert saved["analysis"]["excluded_id_columns"] == ["order_id"]
    assert {"overview", "statistics", "quality", "analysis", "charts", "settings"} <= set(saved)


def test_no_missing_values_means_no_missing_chart(shop_df, tmp_path):
    run_eda(shop_df, tmp_path / "out")
    assert "missing_values" not in " ".join(_pngs(tmp_path / "out"))


def test_max_charts_cap(shop_df, tmp_path):
    summary = run_eda(shop_df, tmp_path / "out", max_charts=3)
    assert len(_pngs(tmp_path / "out")) == 3
    assert len(summary["charts"]) == 3
    assert summary["charts_skipped"]  # the rest are listed, not silently dropped


def test_log_view_only_for_skewed_positive_wide_data():
    rng = np.random.default_rng(4)
    skewed = pd.Series(rng.lognormal(3, 1.2, 500))  # spans several orders of magnitude
    assert needs_log_view(skewed)
    assert not needs_log_view(pd.Series(rng.normal(100, 10, 500)))  # symmetric
    assert not needs_log_view(pd.concat([skewed, pd.Series([0.0])]))  # a zero can't go on a log axis
    assert not needs_log_view(pd.concat([skewed, pd.Series([-5.0])]))  # nor a negative
    narrow = pd.Series([10.0] * 50 + [11, 12, 15, 30, 60])  # skewed but only spans 6x
    assert not needs_log_view(narrow)


def test_log_chart_follows_its_linear_chart(tmp_path):
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"amount": rng.lognormal(3, 1.2, 500), "score": rng.normal(50, 5, 500)})
    summary = run_eda(df, tmp_path / "out")
    names = [c["file"] for c in summary["charts"] if c["type"] in ("histogram_boxplot", "histogram_log")]
    assert names == [
        "charts/03_distribution_amount.png",  # after heatmap and one scatter
        "charts/04_distribution_amount_log.png",
        "charts/05_distribution_score.png",
    ]


def test_scatter_sampling(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 3000)
    df = pd.DataFrame({"x": x, "y": x + rng.normal(0, 0.5, 3000)})
    summary = run_eda(df, tmp_path / "out", max_scatter_points=500)
    scatter = next(c for c in summary["charts"] if c["type"] == "scatter")
    assert scatter["sampled"] is True and scatter["points_shown"] == 500


def test_no_numeric_columns(tmp_path):
    df = pd.DataFrame({"city": ["Paris", "Rome", "Paris"], "team": ["a", "b", "b"]})
    summary = run_eda(df, tmp_path / "out")
    assert summary["analysis"]["correlations"]["pairs"] == []
    assert {c["type"] for c in summary["charts"]} == {"top_categories"}


def test_single_numeric_column(tmp_path):
    summary = run_eda(pd.DataFrame({"v": [1.0, 2, 2, 3, 9]}), tmp_path / "out")
    assert [c["type"] for c in summary["charts"]] == ["histogram_boxplot"]


def test_constant_column_and_all_missing_column(tmp_path):
    df = pd.DataFrame({"const": [5.0] * 6, "empty": [np.nan] * 6, "v": [1.0, 2, 3, 4, 5, 6]})
    summary = run_eda(df, tmp_path / "out")
    types = [c["type"] for c in summary["charts"]]
    assert types.count("histogram_boxplot") == 2  # const and v; nothing to draw for "empty"
    assert "missing_values" in types


def test_unusual_column_names(tmp_path):
    df = pd.DataFrame({"Revenue (€) / month": [1.0, 2, 3, 4], "Région": ["a", "b", "a", "b"]})
    run_eda(df, tmp_path / "out")
    assert len(_pngs(tmp_path / "out")) == 2


def test_refuses_non_empty_output_folder(shop_df, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    with pytest.raises(DatasetError, match="not empty"):
        run_eda(shop_df, out)
    assert (out / "keep.txt").read_text() == "mine"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_end_to_end_leaves_input_untouched(scripts_dir, shop_df, tmp_path):
    data = tmp_path / "shop.csv"
    shop_df.to_csv(data, index=False)
    before = hashlib.sha256(data.read_bytes()).hexdigest()
    r = subprocess.run(
        [sys.executable, str(scripts_dir / "generate_eda.py"), str(data), "--max-categories", "2"],
        capture_output=True, text=True, encoding="utf-8", cwd=tmp_path,
    )
    assert r.returncode == 0, r.stderr
    assert "Strongest correlations" in r.stdout
    runs = list((tmp_path / "eda_output").iterdir())
    assert len(runs) == 1 and runs[0].name.startswith("shop_")
    assert (runs[0] / "summary.json").exists()
    assert hashlib.sha256(data.read_bytes()).hexdigest() == before


def test_cli_rejects_bad_limits(scripts_dir, csv_comma):
    r = subprocess.run(
        [sys.executable, str(scripts_dir / "generate_eda.py"), str(csv_comma), "--max-categories", "0"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 2
    assert "--max-categories" in r.stderr
