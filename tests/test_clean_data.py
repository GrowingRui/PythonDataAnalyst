"""clean_data.py: every step against known answers, the safeguards, and the full cycle on the sample data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import clean_data as cd
from compare_runs import compare
from generate_eda import run_eda
from inspect_data import load_dataset
from report import CLEANING_STEPS

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data" / "sample_sales.csv"
SAMPLE_PLAN = ROOT / "sample_data" / "cleaning_plan.json"
GUIDE = ROOT / "skills" / "python-data-analyst" / "references" / "data-cleaning.md"


def run(df: pd.DataFrame, *steps: dict):
    return cd.apply_plan(df, cd.validate_plan({"steps": list(steps)}))


def one(df: pd.DataFrame, step: dict):
    out, log = run(df, step)
    return out, log[0]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Plan validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("plan, message", [
    ({"steps": []}, "no steps"),
    ([], "no steps"),
    ({"steps": [{"step": "custom_pandas", "code": "df.x = 1"}]}, "unknown step 'custom_pandas'"),
    ({"steps": [{"step": "drop_columns"}]}, "'columns' is required"),
    ({"steps": [{"step": "drop_columns", "columns": ["a"], "colums": ["b"]}]}, "unknown option(s) 'colums'"),
    ({"steps": [{"step": "drop_columns", "columns": []}]}, "non-empty list"),
    ({"steps": [{"step": "drop_columns", "columns": ["a"], "fixes": "outliers"}]}, "this step fixes"),
    ({"steps": [{"step": "to_datetime", "columns": ["d"], "format": "%d/%m/%Y", "day_first": True}]}, "not both"),
    ({"steps": [{"step": "fill_missing", "columns": ["x"], "method": "constant"}]}, "needs a 'value'"),
    ({"steps": [{"step": "fill_missing", "columns": ["x"], "method": "median", "value": 0}]}, "only used with"),
    ({"steps": [{"step": "cap_outliers", "columns": ["x"], "k": 0}]}, "above 0"),
    ({"steps": [{"step": "drop_exact_duplicates", "keep": "middle"}]}, "must be one of"),
    ({"steps": ["drop_columns"]}, 'must be an object with a "step"'),
])
def test_invalid_plans_are_refused_before_any_data_is_read(plan, message):
    with pytest.raises(cd.PlanError, match=message.replace("(", r"\(").replace(")", r"\)")):
        cd.validate_plan(plan if isinstance(plan, dict) else {"steps": plan})


def test_defaults_and_fixes_are_filled_in():
    [step] = cd.validate_plan({"steps": [{"step": "cap_outliers", "columns": "x", "fixes": "outliers",
                                          "reason": "agreed with sales"}]})
    assert step == {"step": "cap_outliers", "columns": ["x"], "k": 1.5, "fixes": "outliers",
                    "reason": "agreed with sales"}


def test_every_step_is_linked_to_the_findings_it_fixes():
    linked = {s for steps in CLEANING_STEPS.values() for s in steps}
    assert linked == set(cd.STEPS)
    assert "missing_minor" in cd.STEPS["fill_missing"].fixes("fill_missing")  # aliases count too


def test_every_step_is_documented_in_the_cleaning_guide():
    text = GUIDE.read_text(encoding="utf-8")
    assert [s for s in cd.STEPS if f"`{s}`" not in text] == []


def test_read_plan_accepts_a_list_and_a_cleaning_log(tmp_path):
    (tmp_path / "list.json").write_text('[{"step": "drop_exact_duplicates"}]', encoding="utf-8")
    plan, earlier = cd.read_plan(tmp_path / "list.json")
    assert plan == {"steps": [{"step": "drop_exact_duplicates"}]} and earlier is None
    log = {"input": {"sha256": "x"}, "steps": [], "plan": {"steps": [{"step": "drop_exact_duplicates"}]}}
    (tmp_path / "log.json").write_text(json.dumps(log), encoding="utf-8")
    plan, earlier = cd.read_plan(tmp_path / "log.json")
    assert plan == log["plan"] and earlier == log


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


def test_drop_exact_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2, 1], "b": ["x", "x", "y", "x"]})
    out, e = one(df, {"step": "drop_exact_duplicates"})
    assert list(out.index) == [0, 2] and e["details"]["removed_rows"] == [1, 3]
    out, _ = one(df, {"step": "drop_exact_duplicates", "keep": "last"})
    assert list(out.index) == [2, 3]
    assert e["rows_before"] == 4 and e["rows_after"] == 2 and e["values_changed"] == 0


def test_drop_duplicate_keys_first_last_and_most_complete():
    df = pd.DataFrame({"id": [1, 1, 2, None, None], "qty": [5, np.nan, 3, 7, 8], "note": [None, "b", "c", "d", "e"]})
    keep = {mode: list(one(df, {"step": "drop_duplicate_keys", "columns": ["id"], "keep": mode})[0].index)
            for mode in ("first", "last", "most_complete")}
    assert keep == {"first": [0, 2, 3, 4], "last": [1, 2, 3, 4], "most_complete": [0, 2, 3, 4]}
    df.loc[0, "qty"] = np.nan  # now row 1 has fewer gaps (0 vs 2)
    out, e = one(df, {"step": "drop_duplicate_keys", "columns": ["id"], "keep": "most_complete"})
    assert list(out.index) == [1, 2, 3, 4]
    assert e["details"]["columns_that_differ"] == {"note": 1}
    assert any("2 row(s) have no value in id" in n for n in e["notes"])  # rows without a key are never dropped


def test_placeholders_in_text_numbers_and_dates():
    df = pd.DataFrame({"t": ["a", " - ", "Unknown", "n/a!", None],
                       "n": [1.0, 999.0, 3.0, 999.0, -1.0],
                       "d": pd.to_datetime(["2025-01-01", "1900-01-01", "2025-02-01", None, "1900-01-01"])})
    out, e = one(df, {"step": "placeholders_to_missing", "columns": ["t"]})  # the standard text placeholders
    assert out["t"].isna().tolist() == [False, True, True, False, True]
    out, e = run(df, {"step": "placeholders_to_missing", "columns": ["n"], "values": [999, -9]},
                 {"step": "placeholders_to_missing", "columns": ["d"], "values": ["1900-01-01"]})
    assert out["n"].isna().sum() == 2 and out["d"].isna().sum() == 3
    assert e[0]["details"]["replaced"] == {"n": {"999": 2}}
    assert e[0]["notes"] == ["In n, -9 matched nothing."]


@pytest.mark.parametrize("df, step, message", [
    (pd.DataFrame({"n": [1, 2]}), {"columns": ["n"]}, "give the code numbers"),
    (pd.DataFrame({"b": [True, False]}), {"columns": ["b"], "values": ["-"]}, "true/false"),
    (pd.DataFrame({"d": pd.to_datetime(["2025-01-01"])}), {"columns": ["d"], "values": ["soon"]}, "not a date"),
])
def test_placeholder_errors(df, step, message):
    with pytest.raises(cd.PlanError, match=message):
        one(df, {"step": "placeholders_to_missing", **step})


def test_text_placeholder_also_matches_a_number_written_as_text():
    out, _ = one(pd.DataFrame({"t": ["999", "5", "x"]}), {"step": "placeholders_to_missing", "columns": ["t"],
                                                           "values": [999]})
    assert out["t"].isna().tolist() == [True, False, False]


def test_strip_whitespace():
    df = pd.DataFrame({"a": [" x", "y ", "z", "   "], "b": ["ok ", "ok", None, "ok"], "n": [1, 2, 3, 4]})
    out, e = one(df, {"step": "strip_whitespace"})  # default: every text column
    assert out["a"].tolist()[:3] == ["x", "y", "z"] and pd.isna(out["a"][3])
    assert out["b"].tolist()[:2] == ["ok", "ok"]
    assert e["details"]["trimmed"] == {"a": 3, "b": 1} and e["details"]["became_missing"] == 1
    assert e["values_changed"] == 4 and e["missing_by_column"] == {"a": [0, 1]}


def test_normalize_labels():
    df = pd.DataFrame({"r": ["North", "north", "South ", "NORTH", "East"]})
    out, e = one(df, {"step": "normalize_labels", "column": "r", "mapping": {"north": "North", "South": "South"}})
    assert out["r"].tolist() == ["North", "North", "South ", "NORTH", "East"]
    assert e["details"]["relabelled"] == {"north": 1}
    assert any("'South' matched no value" in n and "strip_whitespace first" in n for n in e["notes"])
    assert any("'North' / 'NORTH'" in n for n in e["notes"])  # still inconsistent


def test_to_numeric_handles_currency_percent_separators_and_a_mapping():
    df = pd.DataFrame({"price": ["$1,234.50", "$4.99", "free", " 12 ", "-", None],
                       "pct": ["12%", "5 %", "100%", "0%", "50%", "1%"],
                       "eu": ["1.234,50", "3,5", "10", "2,25", None, "7"]})
    out, e = run(df, {"step": "to_numeric", "columns": ["price"], "map": {"FREE": 0}},
                 {"step": "to_numeric", "columns": ["pct"], "percent": "fraction"},
                 {"step": "to_numeric", "columns": ["eu"]})
    assert out["price"].tolist()[:4] == [1234.5, 4.99, 0.0, 12.0] and out["price"][4:].isna().all()
    assert e[0]["details"]["columns"]["price"]["made_missing"] == 1  # "-": listed, not silently lost
    assert e[0]["details"]["columns"]["price"]["unparseable_values"] == {"'-'": 1}
    assert out["pct"].tolist() == [0.12, 0.05, 1.0, 0.0, 0.5, 0.01]
    assert out["eu"].tolist()[:4] == [1234.5, 3.5, 10.0, 2.25]
    assert e[2]["details"]["columns"]["eu"]["decimal_comma"] is True


def test_to_numeric_keeps_whole_numbers_whole_and_can_refuse_bad_values():
    out, e = one(pd.DataFrame({"n": ["1", "2", "3"]}), {"step": "to_numeric", "columns": ["n"]})
    assert str(out["n"].dtype) == "int64" and e["dtype_changes"]["n"][1] == "int64"
    with pytest.raises(cd.PlanError, match=r"Step 1 \(to_numeric\): 1 value\(s\) in 'n' don't parse"):
        one(pd.DataFrame({"n": ["1", "x"]}), {"step": "to_numeric", "columns": ["n"], "unparseable": "error"})


def test_to_numeric_converts_excel_style_mixed_columns():
    out, _ = one(pd.DataFrame({"n": pd.Series([1, "2,5", 3.5, "x"], dtype=object)}),
                 {"step": "to_numeric", "columns": ["n"]})
    assert out["n"].tolist()[:3] == [1.0, 2.5, 3.5] and pd.isna(out["n"][3])


def test_to_datetime_day_first_format_and_iso():
    df = pd.DataFrame({"d": ["31/12/2025", "02/01/2025", "2025-01-03", "", "soon"]})
    out, e = one(df, {"step": "to_datetime", "columns": ["d"], "day_first": True})
    assert out["d"].tolist()[:3] == [pd.Timestamp("2025-12-31"), pd.Timestamp("2025-01-02"),
                                     pd.Timestamp("2025-01-03")]  # ISO isn't misread as 1 March
    info = e["details"]["columns"]["d"]
    assert info["made_missing"] == 1 and info["blank_to_missing"] == 1 and info["unparseable_values"] == {"'soon'": 1}
    out, _ = one(pd.DataFrame({"d": ["01/02/2025"]}), {"step": "to_datetime", "columns": ["d"], "format": "%m/%d/%Y"})
    assert out["d"][0] == pd.Timestamp("2025-01-02")
    out, e = one(pd.DataFrame({"d": ["13/02/2025", "01/03/2025"]}), {"step": "to_datetime", "columns": ["d"]})
    assert out["d"].tolist() == [pd.Timestamp("2025-02-13"), pd.Timestamp("2025-03-01")]
    assert e["details"]["columns"]["d"]["read_as"] == "detected day-first"


def test_to_datetime_refuses_to_guess_the_day_month_order():
    with pytest.raises(cd.PlanError, match="ambiguous.*day_first"):
        one(pd.DataFrame({"d": ["03/04/2025", "05/06/2025"]}), {"step": "to_datetime", "columns": ["d"]})


def test_to_datetime_converts_time_zones_to_utc():
    # Exports and logs often end timestamps in Z or an offset; mixed offsets must compare.
    df = pd.DataFrame({"t": ["2025-10-01T10:00:00Z", "2025-10-01T10:00:00+02:00", "2025-10-01 10:00 -0500",
                             "01/10/2025 10:00 UTC"]})
    out, e = one(df, {"step": "to_datetime", "columns": ["t"], "day_first": True})
    assert out["t"].tolist() == [pd.Timestamp("2025-10-01 10:00"), pd.Timestamp("2025-10-01 08:00"),
                                 pd.Timestamp("2025-10-01 15:00"), pd.Timestamp("2025-10-01 10:00")]
    assert str(out["t"].dtype) == "datetime64[ns]"
    info = e["details"]["columns"]["t"]
    assert info["time_zones_converted_to_utc"] and "UTC" in info["read_as"]
    assert any("UTC" in n for n in e["notes"])


def test_to_datetime_refuses_a_mix_of_zoned_and_plain_times():
    df = pd.DataFrame({"t": ["2025-10-01T10:00:00Z", "2025-10-02 11:00:00"]})
    with pytest.raises(cd.PlanError, match="time zone.*without one"):
        one(df, {"step": "to_datetime", "columns": ["t"]})


def test_drop_columns_and_rows():
    df = pd.DataFrame({"a": [1, None, None, 4], "b": [None, 2, None, 4], "c": ["x"] * 4})
    out, e = one(df, {"step": "drop_columns", "columns": ["c"]})
    assert list(out.columns) == ["a", "b"] and e["columns_removed"] == ["c"]
    out, e = one(df, {"step": "drop_rows_missing", "columns": ["a", "b"]})
    assert list(out.index) == [3] and e["details"]["rows_removed"] == 3
    out, _ = one(df, {"step": "drop_rows_missing", "columns": ["a", "b"], "how": "all"})
    assert list(out.index) == [0, 1, 3]


def test_fill_missing_median_adds_an_indicator_and_shrinks_the_spread():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 10.0, np.nan], "y": ["a"] * 5})
    out, e = one(df, {"step": "fill_missing", "columns": ["x"], "method": "median"})
    assert out["x"].tolist() == [1.0, 2.0, 2.0, 10.0, 2.0]
    assert list(out.columns) == ["x", "x_was_missing", "y"]
    assert out["x_was_missing"].tolist() == [False, False, True, False, True]
    info = e["details"]["columns"]["x"]
    assert info["value"] == 2.0 and info["after"]["std"] < info["before"]["std"]
    assert any("shrinks the spread" in n for n in e["notes"])


def test_fill_missing_per_group_mode_and_constant():
    df = pd.DataFrame({"g": ["a", "a", "a", "b", "b", None],
                       "x": [1.0, 3.0, np.nan, 10.0, np.nan, np.nan],
                       "c": ["u", "u", None, "v", None, "w"]})
    out, e = one(df, {"step": "fill_missing", "columns": ["x"], "method": "median", "by": "g"})
    assert out["x"].tolist() == [1.0, 3.0, 2.0, 10.0, 10.0, 3.0]  # no group: the overall median
    assert e["details"]["columns"]["x"]["group_values"] == {"a": "2", "b": "10", "missing": "3"}
    out, _ = one(df, {"step": "fill_missing", "columns": ["c"], "method": "mode"})
    assert out["c"].tolist() == ["u", "u", "u", "v", "u", "w"]
    out, _ = one(df, {"step": "fill_missing", "columns": ["c"], "method": "constant", "value": "none"})
    assert out["c"][2] == "none"
    with pytest.raises(cd.PlanError, match="must be a number"):
        one(df, {"step": "fill_missing", "columns": ["x"], "method": "constant", "value": "zero"})
    with pytest.raises(cd.PlanError, match="not numeric.*to_numeric"):
        one(df, {"step": "fill_missing", "columns": ["c"], "method": "median"})


def test_fill_missing_refuses_to_overwrite_an_existing_indicator():
    df = pd.DataFrame({"x": [1.0, np.nan], "x_was_missing": [False, True]})
    with pytest.raises(cd.PlanError, match="already exists"):
        one(df, {"step": "fill_missing", "columns": ["x"], "method": "median"})


def test_cap_outliers_moves_the_mean_and_the_standard_deviation():
    x = [10.0, 11, 12, 13, 14, 15, 16, 17, 18, 100]
    out, e = one(pd.DataFrame({"x": x}), {"step": "cap_outliers", "columns": ["x"]})
    q1, q3 = pd.Series(x).quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)  # 12.25, 16.75 -> 23.5
    assert out["x"].max() == upper and out["x"].tolist()[:9] == x[:9]
    info = e["details"]["columns"]["x"]
    assert info["capped_high"] == 1 and info["capped_low"] == 0
    assert info["before"]["mean"] == 22.6 and info["after"]["mean"] == pytest.approx(14.95)  # (126 + 23.5) / 10
    assert info["after"]["std"] < info["before"]["std"] / 3


def test_cap_outliers_keeps_integers_whole_and_skips_zero_iqr():
    out, e = one(pd.DataFrame({"q": [1, 2, 2, 3, 3, 3, 4, 300]}), {"step": "cap_outliers", "columns": ["q"]})
    assert str(out["q"].dtype) == "int64" and out["q"].max() == 5  # fence 3.25 + 1.5 * 1.25 = 5.125, floored
    out, e = one(pd.DataFrame({"q": [5, 5, 5, 5, 5, 9]}), {"step": "cap_outliers", "columns": ["q"]})
    assert out["q"].tolist() == [5, 5, 5, 5, 5, 9] and "IQR 0" in e["notes"][0]


def test_flag_outliers_adds_a_column_and_changes_nothing():
    df = pd.DataFrame({"x": [10.0, 11, 12, 13, np.nan, 14, 100]})
    out, e = one(df, {"step": "flag_outliers", "columns": ["x"]})
    assert out["x"].equals(df["x"]) and e["values_changed"] == 0
    assert out["x_outlier"].tolist()[:4] == [False] * 4 and pd.isna(out["x_outlier"][4])
    assert out["x_outlier"][6] is np.True_ or bool(out["x_outlier"][6]) is True
    assert e["columns_added"] == ["x_outlier"]


def test_a_missing_column_names_the_step_and_the_columns_at_that_point():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(cd.PlanError, match=r"Step 2 \(cap_outliers\): column\(s\) not found: 'a'.*Columns at this "
                                           r"point: b"):
        run(df, {"step": "drop_columns", "columns": ["a"]}, {"step": "cap_outliers", "columns": ["a"]})


def test_steps_never_change_the_table_they_are_given():
    df = pd.DataFrame({"t": [" a", "-"], "n": [1.0, np.nan]})
    copy = df.copy()
    run(df, {"step": "strip_whitespace"}, {"step": "placeholders_to_missing", "columns": ["t"]},
        {"step": "fill_missing", "columns": ["n"], "method": "median"})
    pd.testing.assert_frame_equal(df, copy)


# --------------------------------------------------------------------------
# Files, safeguards and reproducibility
# --------------------------------------------------------------------------


PLAN = {"steps": [{"step": "strip_whitespace"}, {"step": "to_numeric", "columns": ["price"]}]}


@pytest.fixture
def small_csv(tmp_path) -> Path:
    path = tmp_path / "in" / "shop.csv"
    path.parent.mkdir()
    path.write_text("zip,price,city\n01234,$5,Rome \n09999,$7.50,Oslo\n10115,free,Rome\n", encoding="utf-8")
    return path


def test_dry_run_writes_nothing(small_csv, tmp_path):
    out = tmp_path / "out" / "clean.csv"
    log = cd.run_cleaning(small_csv, PLAN, output=out)
    assert log["mode"] == "dry_run" and not out.parent.exists()
    assert sorted(p.name for p in tmp_path.rglob("*")) == ["in", "shop.csv"]
    assert log["output"]["written"] is False and log["input"]["unchanged_after_run"] is True


def test_apply_writes_the_file_and_both_logs(small_csv, tmp_path):
    before = sha(small_csv)
    out = tmp_path / "out" / "clean.csv"
    log = cd.run_cleaning(small_csv, PLAN, output=out, apply=True)
    assert sorted(p.name for p in out.parent.iterdir()) == ["clean.csv", "cleaning_log.json", "cleaning_log.md"]
    assert sha(small_csv) == before == log["input"]["sha256"]
    assert log["output"]["sha256"] == sha(out)
    text = out.read_text(encoding="utf-8")
    assert text == "zip,price,city\n01234,5.0,Rome\n09999,7.5,Oslo\n10115,,Rome\n"  # zip codes kept exactly
    assert log["input"]["kept_as_text"] == ["zip"]
    md = (out.parent / "cleaning_log.md").read_text(encoding="utf-8")
    assert "## Steps" in md and "`strip_whitespace`" in md and before in md and "## Plan" in md
    saved = json.loads((out.parent / "cleaning_log.json").read_text(encoding="utf-8"))
    assert saved["plan"] == PLAN and saved["steps"][1]["details"]["columns"]["price"]["made_missing"] == 1


@pytest.mark.parametrize("make_output, message", [
    (lambda inp, tmp: inp, "The output is the input file"),
    (lambda inp, tmp: inp.parent / "clean.csv", "not empty"),
    (lambda inp, tmp: tmp / "x" / "clean.xlsx", "must end in .csv or .parquet"),
])
def test_refuses_to_overwrite(small_csv, tmp_path, make_output, message):
    with pytest.raises(cd.PlanError, match=message):
        cd.run_cleaning(small_csv, PLAN, output=make_output(small_csv, tmp_path), apply=True)
    assert sorted(p.name for p in small_csv.parent.iterdir()) == ["shop.csv"]


def test_a_failing_plan_leaves_no_folder_behind(small_csv, tmp_path):
    out = tmp_path / "out" / "clean.csv"
    with pytest.raises(cd.PlanError, match="not found"):
        cd.run_cleaning(small_csv, {"steps": [{"step": "drop_columns", "columns": ["nope"]}]}, output=out, apply=True)
    assert not out.parent.exists()


@pytest.mark.parametrize("ext", [".csv", ".parquet"])
def test_same_plan_twice_gives_byte_identical_output(small_csv, tmp_path, ext):
    if ext == ".parquet":
        pytest.importorskip("pyarrow")
    a = tmp_path / "a" / f"clean{ext}"
    b = tmp_path / "b" / f"clean{ext}"
    cd.run_cleaning(small_csv, PLAN, output=a, apply=True)
    plan, earlier = cd.read_plan(a.parent / "cleaning_log.json")  # rebuild from the log
    log = cd.run_cleaning(small_csv, plan, output=b, apply=True, earlier_log=earlier)
    assert a.read_bytes() == b.read_bytes() and log["warnings"] == []


def test_parquet_output_keeps_the_types(small_csv, tmp_path):
    pytest.importorskip("pyarrow")
    out = tmp_path / "o" / "clean.parquet"
    cd.run_cleaning(small_csv, {"steps": [{"step": "to_numeric", "columns": ["price"], "map": {"free": 0}}]},
                    output=out, apply=True)
    back = pd.read_parquet(out)
    assert back["price"].tolist() == [5.0, 7.5, 0.0] and back["zip"].tolist() == ["01234", "09999", "10115"]


def test_rebuilding_from_a_log_of_another_file_warns(small_csv, tmp_path):
    first = tmp_path / "a" / "clean.csv"
    cd.run_cleaning(small_csv, PLAN, output=first, apply=True)
    other = tmp_path / "other.csv"
    other.write_text("zip,price,city\n1,$1,X\n", encoding="utf-8")
    plan, earlier = cd.read_plan(first.parent / "cleaning_log.json")
    log = cd.run_cleaning(other, plan, output=tmp_path / "b" / "c.csv", earlier_log=earlier)
    assert "different input file" in log["warnings"][0]


def test_default_output_is_a_new_folder_and_keeps_parquet():
    now = pd.Timestamp("2026-11-05 12:00:00").to_pydatetime()
    assert cd.default_output(Path("data/sales.csv"), now=now) == \
        Path("eda_output/sales_cleaned_20261105-120000/sales_cleaned.csv")
    assert cd.default_output(Path("x.parquet"), now=now).suffix == ".parquet"


def test_cli_dry_run_apply_and_errors(small_csv, tmp_path, capsys):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(PLAN), encoding="utf-8")
    out = tmp_path / "out" / "c.csv"
    assert cd.main([str(small_csv), "--plan", str(plan), "--output", str(out)]) == 0
    assert "DRY RUN" in capsys.readouterr().out and not out.exists()
    assert cd.main([str(small_csv), "--plan", str(plan), "--output", str(out), "--apply"]) == 0
    assert out.exists()
    assert cd.main([str(small_csv), "--plan", str(plan), "--output", str(out), "--apply"]) == 2
    assert "not empty" in capsys.readouterr().err
    assert cd.main(["--list-steps"]) == 0
    assert "fill_missing" in capsys.readouterr().out


# --------------------------------------------------------------------------
# ISO dates in text files (inspect_data), needed to re-check a cleaned CSV
# --------------------------------------------------------------------------


def test_iso_dates_in_csv_are_read_as_dates(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a,b,c\n2025-01-31,2025-01-31 14:05:00,2025-01-31\n2025-02-01,,not a date\n", encoding="utf-8")
    df = load_dataset(path)
    assert pd.api.types.is_datetime64_any_dtype(df["a"]) and pd.api.types.is_datetime64_any_dtype(df["b"])
    assert df["c"].dtype != "datetime64[ns]"  # one other value: stays text and gets reported
    assert df.attrs["source"]["iso_date_columns"] == ["a", "b"]


# --------------------------------------------------------------------------
# The full cycle on the sample data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_cycle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cycle")
    before_sha = sha(SAMPLE)
    plan, _ = cd.read_plan(SAMPLE_PLAN)
    out = tmp / "clean" / "sample_sales_cleaned.csv"
    log = cd.run_cleaning(SAMPLE, plan, output=out, apply=True)
    eda_before = run_eda(load_dataset(SAMPLE), tmp / "eda_before", key_columns=["order_id"], max_charts=0)
    eda_after = run_eda(load_dataset(out), tmp / "eda_after", key_columns=["order_id"], max_charts=0)
    return before_sha, log, out, eda_before, eda_after


def test_sample_plan_leaves_the_input_unchanged(sample_cycle):
    before_sha, log, *_ = sample_cycle
    assert sha(SAMPLE) == before_sha == log["input"]["sha256"] and log["input"]["unchanged_after_run"]


def test_sample_plan_fixes_the_planted_problems(sample_cycle):
    _, log, _, before, after = sample_cycle
    assert (log["result"]["rows_before"], log["result"]["rows_after"]) == (1215, 1200)
    fixed = {"duplicate_rows", "duplicate_keys", "empty_columns", "constant_column", "dates_as_text",
             "numbers_as_text", "inconsistent_labels", "untrimmed_whitespace", "disguised_missing",
             "possible_missing_code"}
    rules_before = {f["rule"] for f in before["findings"]}
    rules_after = {f["rule"] for f in after["findings"]}
    assert fixed <= rules_before and not fixed & rules_after
    [dates] = [f for f in after["findings"] if f["rule"] == "implausible_dates"]
    assert "1 implausible date" in dates["title"]  # the 2052 typo is left for the data owner
    for kept in ("date_gaps", "date_spike", "skewed", "missing_values"):  # not errors to edit away
        assert kept in rules_after


def test_sample_plan_changes_nothing_else(sample_cycle):
    _, log, out, *_ = sample_cycle
    original = load_dataset(SAMPLE).drop_duplicates()
    original = original[~original.duplicated(subset=["order_id"])]
    cleaned = pd.read_csv(out)
    assert list(cleaned["order_id"]) == list(original["order_id"])
    untouched = ["order_id", "customer_id", "channel", "category", "quantity", "unit_price", "discount_pct",
                 "revenue", "order_size", "returning_customer"]
    pd.testing.assert_frame_equal(cleaned[untouched], original[untouched].reset_index(drop=True))
    changed = {c for e in log["steps"] for c in e["values_changed_by_column"]}
    assert changed == {"region", "rating", "delivery_days", "order_date", "shipping_cost"}
    assert log["result"]["columns_removed"] == ["currency", "legacy_code"] and log["result"]["columns_added"] == []


def test_compare_runs_on_the_sample(sample_cycle):
    *_, before, after = sample_cycle
    c = compare(before, after)
    gone = {f["rule"] for f in c["gone"]}
    assert {"duplicate_rows", "duplicate_keys", "numbers_as_text", "dates_as_text", "possible_missing_code"} <= gone
    assert [f["columns"] for f in c["new"]] == [["rating"]]  # "-" became missing: the cost of step 5
    assert c["before"]["exact_duplicate_rows"] == 12 and c["after"]["exact_duplicate_rows"] == 0


def test_compare_pairs_a_finding_whose_columns_changed():
    def summary(findings):
        return {"source": {"file_name": "f"}, "overview": {"rows": 1, "columns": 1},
                "quality": {"missing": {"total_missing": 0, "total_disguised": 0},
                            "duplicates": {"duplicate_rows": 0}, "type_issues": []},
                "findings": findings}
    a = summary([{"rule": "incomplete_rows", "severity": "medium", "columns": ["x", "y"], "title": "60%"},
                 {"rule": "skewed", "severity": "low", "columns": ["x"], "title": "s"}])
    b = summary([{"rule": "incomplete_rows", "severity": "medium", "columns": ["y"], "title": "70%"},
                 {"rule": "skewed", "severity": "low", "columns": ["x"], "title": "s"}])
    c = compare(a, b)
    assert c["gone"] == [] and c["new"] == [] and len(c["unchanged"]) == 1
    assert c["changed"][0]["before_columns"] == ["x", "y"]
