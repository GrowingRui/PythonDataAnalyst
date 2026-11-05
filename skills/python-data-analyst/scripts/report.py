"""Rule-based findings and the Markdown report, built from an EDA summary.

generate_eda.py calls build_findings() and write_report() at the end of every
run, so you normally don't run this module yourself. Everything here works on
the plain summary dict (the same data as summary.json): no pandas, no file
reading, nothing that could touch the dataset.

The findings are deliberately mechanical: thresholds, counts and percentages.
They tell you *where to look*. Deciding what matters for the user's goal is
the analyst's job (decision D4), which is why the report leaves a marked spot
for an "Analyst interpretation" section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Thresholds (rules of thumb, not laws: each one is explained in the report)
# --------------------------------------------------------------------------

MISSING_HIGH_PCT = 40.0  # a column this empty is barely usable
MISSING_MEDIUM_PCT = 10.0  # below this, gaps are listed together as one minor finding
INCOMPLETE_ROWS_PCT = 20.0  # share of rows with any gap before dropna() becomes costly
DUPLICATE_HIGH_PCT = 5.0
DOMINANT_PCT = 95.0  # one category this common makes the column nearly constant
OUTLIER_MEDIUM_PCT = 5.0
CORR_FINDING = 0.5  # |r| or |rho| from "strong" upwards becomes a finding
CORR_MEDIUM = 0.7
CORR_REDUNDANT = 0.98  # almost perfect: one column may be derived from the other
CORR_FINDINGS_MAX = 10  # wide data could otherwise produce hundreds of pairs
SMALL_DATA_ROWS = 30
TARGET_FINDING = 0.3  # strength (eta, Cramér's V) from "moderate" upwards becomes a finding
TARGET_MEDIUM = 0.5
TARGET_DETAIL_COLUMNS = 3  # group columns with a full per-group table in section 4b

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

REPORT_SECTIONS = (
    "1. Dataset overview",
    "2. Data quality assessment",
    "3. Descriptive statistics",
    "4. Key findings",
    "5. Visualization summaries",
    "6. Suggested next steps",
    "7. Limitations",
)
TARGET_SECTION = "4b. Target comparison"  # only when --target is used, between sections 4 and 5

# The agent replaces this marker with an "Analyst interpretation" subsection.
INTERPRETATION_MARKER = (
    "<!-- analyst-interpretation: the analyst adds an \"Analyst interpretation\" "
    "subsection here after reviewing the charts -->"
)

# What each kind of problem can be done about, and what that costs. Shown in
# "Suggested next steps". Nothing here is ever applied automatically (D5).
CLEANING_ADVICE: dict[str, tuple[str, str, str]] = {
    # rule: (problem label, options, cost)
    "empty_columns": (
        "Empty columns",
        "Drop them.",
        "Nothing is lost, provided they are empty because they're unused and not because an export failed.",
    ),
    "missing_values": (
        "Missing values",
        "Leave them missing and report counts per analysis; drop incomplete rows (`dropna()`); or fill them "
        "(median/mode, or a model), ideally with an extra \"was missing\" indicator column.",
        "Dropping rows shrinks the sample and biases results when values aren't missing at random. "
        "Filling with one value shrinks the spread and weakens correlations.",
    ),
    "disguised_missing": (
        "Placeholder values",
        "Turn the placeholders into real missing values, e.g. `pd.read_csv(..., na_values=['-', 'unknown'])`.",
        "The missing counts go up and the column may become numeric. First check that the placeholder never "
        "means something real (\"unknown\" can be a valid survey answer).",
    ),
    "possible_missing_code": (
        "Code values such as 999",
        "Confirm with the data owner what the code means, then turn it into a missing value "
        "(`df[col].mask(df[col] == 999)`).",
        "If the code is a real value after all, you remove genuine extremes. Left in, it inflates the "
        "mean, the spread and the skew.",
    ),
    "duplicate_rows": (
        "Exact duplicate rows",
        "Remove them with `drop_duplicates()`, keeping the first copy.",
        "Right if each row should be a unique record; wrong if identical rows are genuine repeat events, "
        "which you would then undercount.",
    ),
    "duplicate_keys": (
        "Repeated keys",
        "Investigate the source, then decide which row wins per key (latest, most complete) or aggregate.",
        "Keeping one version discards the others; aggregating changes what a row means.",
    ),
    "numbers_as_text": (
        "Numbers stored as text",
        "Strip currency signs, % and thousands separators, then `pd.to_numeric(..., errors='coerce')`.",
        "Values that still don't parse become missing. List them first so nothing disappears silently.",
    ),
    "dates_as_text": (
        "Dates stored as text",
        "Parse with `pd.to_datetime(..., format='...')` using an explicit format.",
        "Without a format, 03/04 can be read as 3 April or March 4; unparseable values become missing.",
    ),
    "mixed_types": (
        "Mixed value types",
        "Find out why the types are mixed (often a numeric column with a few text notes), then convert to one type.",
        "The odd values become missing or change meaning.",
    ),
    "inconsistent_labels": (
        "Inconsistent labels",
        "Standardise them: trim spaces and pick one case or spelling (e.g. `.str.strip().str.title()`).",
        "Merges groups. Check that no two variants are genuinely different codes.",
    ),
    "untrimmed_whitespace": (
        "Leading or trailing spaces",
        "Trim with `.str.strip()`.",
        "Safe in almost every case; only matters when spacing is meaningful (fixed-width codes).",
    ),
    "nested_values": (
        "Lists or objects inside cells",
        "Count the items per row (a new length column); turn the list into one row per item with "
        "`df.explode(col)` (after `json.loads`) for a separate table; or make one yes/no column per item.",
        "Exploding repeats every other column once per item, so sums and counts over the exploded table "
        "inflate. Counting items keeps the rows but drops what the items were.",
    ),
    "constant_column": (
        "Constant columns",
        "Leave them out of the analysis.",
        "Nothing is lost within this file, but check that the column isn't constant because of a filter or export setting.",
    ),
    "dominant_category": (
        "Nearly constant columns",
        "Keep them, but treat comparisons between groups with care.",
        "The small groups have few rows, so any result for them is uncertain.",
    ),
    "skewed": (
        "Skewed columns",
        "Describe them with the median and IQR; use a log scale in charts or a log transform in models.",
        "A log transform changes the meaning (ratios instead of differences) and can't handle zeros or negatives.",
    ),
    "outliers": (
        "Outliers",
        "Check the most extreme values against the source first. Then keep them, cap them (winsorise), "
        "or analyse with and without them.",
        "Removing real values biases means and totals and hides the cases that may matter most. "
        "Capping keeps the rows but changes the values.",
    ),
    "implausible_dates": (
        "Implausible dates",
        "Check them against the source. Placeholders (`1900-01-01`, `1970-01-01`) become missing values "
        "(`df[col].mask(df[col].isin([...]))`); obvious typos (2052 for 2025) are corrected only with the "
        "data owner's confirmation.",
        "Left in, one wrong date stretches every time chart and date range. Guessing the \"right\" year "
        "invents data; a missing date is more honest than a guessed one.",
    ),
    "date_gaps": (
        "Periods without rows",
        "Ask the data owner whether the period is missing from the export (outage, failed load) or really "
        "had no activity. Compare periods only outside the gap, or label the gap in every chart.",
        "Filling the gap with zeros or interpolated values invents data. Ignoring it understates totals "
        "and averages over the whole range.",
    ),
    "date_spike": (
        "Spikes in rows per period",
        "Check the spike period for duplicate or late-loaded rows: rows sharing a key, or many rows stamped "
        "with one date. If it's a loading artefact, deduplicate or use a better date column if there is one.",
        "Removing a real peak (a sale, a campaign) hides the most important event in the data.",
    ),
    "possible_leakage": (
        "Possible target leakage",
        "Find out how the column is made and when it's filled in. If it's computed from the target or recorded "
        "after the outcome, leave it out of any model or comparison meant to explain the target.",
        "Leaving out a genuinely strong predictor loses information; keeping a leaky one makes a model look far "
        "better than it will be on new data, where the column isn't known yet.",
    ),
    "partial_period": (
        "Incomplete first or last period",
        "Leave partial periods out when comparing periods, or compare rows per day instead of per period.",
        "Dropping their rows loses real data from totals; comparing them as if complete makes them look "
        "like drops.",
    ),
}
# Several findings share the same advice.
ADVICE_ALIAS = {"missing_minor": "missing_values", "incomplete_rows": "missing_values"}


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


def build_findings(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Rule-based findings, most important first.

    Each finding is a dict with ``rule``, ``severity`` (high / medium / low),
    ``columns``, ``title`` and ``detail`` (both Markdown). Severity means "how
    much this can distort an analysis if ignored", not "how interesting it is".
    """
    ov = summary["overview"]
    st = summary["statistics"]
    q = summary["quality"]
    analysis = summary.get("analysis", {})
    ids = set(analysis.get("excluded_id_columns", []))
    rows = ov["rows"]
    out: list[dict[str, Any]] = []

    def add(rule: str, severity: str, columns: list[str], title: str, detail: str, metric: float = 0.0) -> None:
        out.append({"rule": rule, "severity": severity, "columns": columns,
                    "title": title, "detail": detail, "_metric": float(metric)})

    # --- missing values -------------------------------------------------------
    miss_cols = q["missing"]["columns"]
    empty = [c["name"] for c in miss_cols if rows and c["missing"] == rows]
    if empty:
        add("empty_columns", "high", empty,
            f"{_plural(len(empty), 'column is', 'columns are')} completely empty: {_codes(empty)}",
            f"Every value is missing, so {'it carries' if len(empty) == 1 else 'they carry'} no information.", 100)

    minor = []
    for c in miss_cols:
        if c["name"] in empty or not c["missing"]:
            continue
        pct = c["missing_pct"]
        if pct >= MISSING_MEDIUM_PCT:
            add("missing_values", "high" if pct >= MISSING_HIGH_PCT else "medium", [c["name"]],
                f"{code(c['name'])} is missing {_pct(pct)} of its values ({c['missing']:,} of {rows:,} rows)",
                "Any analysis that drops incomplete rows loses these rows. If the gaps aren't random "
                "(e.g. only one region or period is affected), results computed on the rest are biased. "
                "Check whether the gaps cluster in particular groups.", pct)
        else:
            minor.append(c)
    if minor:
        listed = ", ".join(f"{code(c['name'])} ({_pct(c['missing_pct'])})" for c in minor)
        add("missing_minor", "low", [c["name"] for c in minor],
            f"{_plural(len(minor), 'column has', 'columns have')} a few missing values "
            f"(under {MISSING_MEDIUM_PCT:g}% each)",
            f"{listed}.", max(c["missing_pct"] for c in minor))

    m = q["missing"]
    gappy = [c for c in miss_cols if c["missing"] and c["name"] not in empty]
    # Count without the all-empty columns: they'd make every row incomplete.
    gap_rows = m.get("rows_with_missing_ignoring_empty_columns", m["rows_with_missing"])
    gap_pct = m.get("rows_with_missing_ignoring_empty_columns_pct", m["rows_with_missing_pct"])
    if len(gappy) >= 2 and gap_pct >= INCOMPLETE_ROWS_PCT:
        ignoring = f", not counting the empty column{'s' if len(empty) > 1 else ''} {_codes(empty)}" if empty else ""
        add("incomplete_rows", "medium", [c["name"] for c in gappy],
            f"Only {_pct(100 - gap_pct)} of rows are complete{ignoring}",
            f"{gap_rows:,} rows ({_pct(gap_pct)}) have a gap in at least one of {_codes([c['name'] for c in gappy])}. "
            f"Dropping those rows (`dropna(subset=[...])`) would leave {rows - gap_rows:,} of {rows:,} rows.",
            gap_pct)

    for c in miss_cols:
        if c["disguised"]:
            values = ", ".join(f"{code(_unrepr(k))} ×{v}" for k, v in c["disguised_values"].items())
            add("disguised_missing", "medium", [c["name"]],
                f"{code(c['name'])} has {_plural(c['disguised'], 'placeholder value', 'placeholder values')} "
                f"that probably mean \"missing\"",
                f"{values}. They are stored as text, so pandas counts them as present. Counting them, the "
                f"column is missing {_pct(c['missing_pct'] + c['disguised_pct'])} instead of "
                f"{_pct(c['missing_pct'])}.", c["disguised_pct"])

    # --- duplicates -------------------------------------------------------------
    d = q["duplicates"]
    if d["duplicate_rows"]:
        add("duplicate_rows", "high" if d["duplicate_pct"] >= DUPLICATE_HIGH_PCT else "medium", [],
            f"{_plural(d['duplicate_rows'], 'exact duplicate row', 'exact duplicate rows')} "
            f"({_pct(d['duplicate_pct'])})",
            "Every column matches an earlier row. If each row should be a unique record, duplicates inflate "
            "counts and totals; they can also be genuine (two identical purchases)."
            + _dup_examples(d), d["duplicate_pct"])
    kd = q.get("key_duplicates")
    if kd and kd["duplicate_rows"]:
        keys = kd["subset"] or []
        add("duplicate_keys", "high", list(keys),
            f"{_plural(kd['duplicate_rows'], 'row repeats', 'rows repeat')} a key on {_codes(keys)} "
            f"({_pct(kd['duplicate_pct'])})",
            "These rows share a key that should be unique; where they differ in other columns it isn't "
            "clear which version is correct." + _dup_examples(kd), 100 + kd["duplicate_pct"])

    # --- type problems ------------------------------------------------------------
    for issue in q["type_issues"]:
        add(issue["issue"], "low" if issue["issue"] == "untrimmed_whitespace" else "medium", [issue["column"]],
            _type_issue_title(issue), _type_issue_detail(issue), issue.get("count", 0))

    # --- constant and nearly constant columns -------------------------------------
    for c in ov["column_info"]:
        name, s = c["name"], st.get(c["name"], {})
        if c["kind"] == "empty" or not c["non_null"]:
            continue
        if c["unique"] == 1:
            add("constant_column", "medium", [name],
                f"{code(name)} has the same value ({code(_single_value(s))}) in every non-missing row",
                "A column that never changes can't explain any differences in the data.", 100)
            continue
        top = _dominant(s)
        if top and top[1] >= DOMINANT_PCT:
            add("dominant_category", "low", [name],
                f"{code(name)} is nearly constant: {code(top[0])} makes up {_pct(top[1])} of values",
                "The other values have very few rows, so comparisons between groups rest on small numbers.",
                top[1])

    # --- distributions ------------------------------------------------------------
    skewed = set()
    for name, s in st.items():
        if s.get("kind") != "numeric" or name in ids or not (s.get("skew_label") or "").startswith("highly"):
            continue
        skewed.add(name)
        detail = _skew_detail(s)
        codes = q["outliers"].get(name, {}).get("code_like_values")
        if codes:
            detail += (f" Part of the skew may come from the code value{'s' if len(codes) > 1 else ''} "
                       f"{', '.join(codes)}; check again once {'they are' if len(codes) > 1 else 'it is'} resolved.")
        add("skewed", "medium", [name],
            f"{code(name)} is {s['skew_label']} (skewness {_num(s['skewness'], 2)})", detail,
            abs(s["skewness"]))

    for name, o in q["outliers"].items():
        if name in ids or o.get("status") != "ok" or not o["n_outliers"]:
            continue
        codes = o.get("code_like_values") or {}
        if codes:
            listed = ", ".join(f"{code(v)} ×{c}" for v, c in codes.items())
            add("possible_missing_code", "medium", [name],
                f"{code(name)} contains {', '.join(codes)}, a number often used to mean \"unknown\"",
                f"{listed}, far outside the IQR fences ({_num(o['lower_fence'])} to {_num(o['upper_fence'])}). "
                "Exports often use such codes for missing or not-applicable values; the skill can't tell them "
                "apart from real numbers. If it is a code, it distorts the mean, spread and skew.",
                sum(codes.values()))
        expected = name in skewed and not codes
        severity = "medium" if o["outlier_pct"] >= OUTLIER_MEDIUM_PCT and not expected else "low"
        parts = [f"Fences {_num(o['lower_fence'])} to {_num(o['upper_fence'])} "
                 f"({o['n_low']:,} below, {o['n_high']:,} above)."]
        if o["highest_outliers"]:
            parts.append(f"Highest: {', '.join(_num(v) for v in o['highest_outliers'])}.")
        if o["lowest_outliers"]:
            parts.append(f"Lowest: {', '.join(_num(v) for v in o['lowest_outliers'])}.")
        parts.append("Unusual doesn't mean wrong.")
        if expected:
            parts.append("The column is highly skewed, so many flags are expected and are probably legitimate.")
        if o.get("note"):
            parts.append(_sentence(o["note"]))
        add("outliers", severity, [name],
            f"{code(name)}: {_plural(o['n_outliers'], 'value', 'values')} ({_pct(o['outlier_pct'])}) "
            f"outside the IQR fences", " ".join(parts), o["outlier_pct"])

    # --- relationships ------------------------------------------------------------
    target = analysis.get("target") or {}
    leaked = {(target["column"], e["column"]) for e in target.get("leakage", [])} if target else set()
    pairs = [p for p in analysis.get("correlations", {}).get("pairs", []) if _pair_strength(p) >= CORR_FINDING
             and (p["a"], p["b"]) not in leaked and (p["b"], p["a"]) not in leaked]  # reported as leakage
    for p in pairs[:CORR_FINDINGS_MAX]:
        strength = _pair_strength(p)
        redundant = p["pearson"] is not None and abs(p["pearson"]) >= CORR_REDUNDANT
        moves = "rise together" if p["direction"] == "positive" else "move in opposite directions"
        detail = f"The two columns tend to {moves}."
        if redundant:
            detail += (" The relationship is almost perfect, so one column may be derived from the other "
                       "(a total and its parts, the same quantity in two units). Use only one of them in a model.")
        for note in p.get("notes", []):
            detail += f" {_sentence(note)}"
        detail += " This is an association, not evidence that one causes the other."
        r, rho = abs(p["pearson"] or 0), abs(p["spearman"] or 0)
        how = ""
        if rho - r >= 0.2:
            how = " by rank, much less linearly"  # Spearman sees it, Pearson doesn't
        elif r - rho >= 0.2:
            how = " linearly, much less by rank"  # Pearson driven by a few extreme points
        add("near_duplicate_columns" if redundant else "strong_correlation",
            "medium" if strength >= CORR_MEDIUM else "low", [p["a"], p["b"]],
            f"{code(p['a'])} and {code(p['b'])} are {p['strength']}ly {p['direction']}ly correlated{how} "
            f"(Pearson r = {_num(p['pearson'], 2)}, Spearman ρ = {_num(p['spearman'], 2)}, n = {p['n']:,})",
            detail, strength)
    if len(pairs) > CORR_FINDINGS_MAX:
        add("more_correlations", "low", [],
            f"{len(pairs) - CORR_FINDINGS_MAX:,} more strongly correlated pairs are not listed here",
            "All pairs, ranked by strength, are in `summary.json` under `analysis.correlations.pairs`.")

    _time_findings(analysis.get("time"), add)
    _target_findings(analysis.get("target"), add)

    if ids:
        add("id_columns", "low", sorted(ids),
            f"{_plural(len(ids), 'column was treated as a row ID', 'columns were treated as row IDs')}: "
            f"{_codes(sorted(ids))}",
            "Key columns given with `--key`, and unique whole numbers named like an ID or counting 1, 2, 3, "
            "..., are left out of the charts and correlations. If one is a real measurement, analyse it "
            "separately.")

    out.sort(key=lambda f: (SEVERITY_ORDER[f["severity"]], -f["_metric"]))
    for f in out:
        del f["_metric"]
    return out


def _time_findings(time: dict[str, Any] | None, add) -> None:
    """Implausible dates (every date column); gaps, spikes and partial periods (the trend column)."""
    if not time:
        return
    for name, c in time["date_columns"].items():
        parts, n = [], 0
        if c["future_excluded"]:
            n += c["future"]
            parts.append(f"{_plural(c['future'], 'date lies', 'dates lie')} in the future (after "
                         f"{c['today']}): {', '.join(code(d) for d in c['future_examples'])}")
        if c["placeholders"]:
            count = sum(c["placeholders"].values())
            n += count
            listed = ", ".join(f"{code(d)} ×{k}" for d, k in c["placeholders"].items())
            parts.append(f"{_plural(count, 'date is a common placeholder', 'dates are common placeholders')} "
                         f"(Excel's day zero, the Unix epoch): {listed}")
        if not n:
            continue
        detail = _sentence("; ".join(parts))
        if name == time["column"] and time.get("trend"):
            detail += " They were left out of the time trend, so they don't stretch its date range."
        if c["future_excluded"]:
            detail += (" A future date is fine if the column holds planned dates (delivery, due date); in a "
                       "column of past events it's usually a typo.")
        if c["placeholders"]:
            detail += " Placeholders usually mean the date was unknown when the row was entered."
        add("implausible_dates", "medium", [name],
            f"{code(name)} has {_plural(n, 'implausible date', 'implausible dates')}", detail, n)

    t = time.get("trend")
    if not t:
        return
    col, period = time["column"], t["period"]
    typical = _num(t["typical_rows"])
    for g in t["gaps"]:
        span = g["first"] if g["periods"] == 1 else f"{g['first']} to {g['last']}"
        add("date_gaps", "medium", [col],
            f"{code(col)} has no rows for {_plural(g['periods'], period, period + 's')} in a row ({span})",
            f"A typical {period} has {typical} rows, so an empty stretch this long is very unlikely to be "
            f"chance ({g['chance']} if rows arrived at random). Possibly missing data (an outage, a failed "
            "export); check with the data owner before comparing these periods with others.",
            50 + g["periods"])
    for sp in t["spikes"]:
        detail = f"A typical {period} has {typical} rows."
        if period != "day":
            detail += f" {sp['top_day_rows']:,} of the {sp['rows']:,} rows fall on {code(sp['top_day'])}."
        if sp["exact_duplicates"]:
            detail += (" " + _plural(sp["exact_duplicates"], f"row in this {period} is an exact duplicate",
                                     f"rows in this {period} are exact duplicates") + " of an earlier row.")
        detail += (" A spike can be a real event (a sale, a campaign) or a data artefact: a duplicate import, "
                   "or late rows stamped with the date they were loaded.")
        if t["gaps"]:
            detail += " Compare it with the gap: rows that arrived late would explain both."
        add("date_spike", "medium", [col],
            f"{code(col)}: the {sp['label']} has {sp['rows']:,} rows, {_num(sp['ratio'], 2)}× a typical {period}",
            detail, sp["ratio"])
    if t["partial"]:
        which = " and ".join(p["which"] for p in t["partial"])
        many = len(t["partial"]) > 1
        covers = "; ".join(f"the {p['which']} ({p['label']}) covers {p['days_covered']} of "
                           f"{p['days_in_period']} days and has {p['rows']:,} rows" for p in t["partial"])
        add("partial_period", "low", [col],
            f"The {which} {period}{'s' if many else ''} of {code(col)} {'are' if many else 'is'} incomplete",
            f"The data runs from {t['first'][:10]} to {t['last'][:10]}: {covers}. Their counts look like "
            f"drops but cover only part of the {period}; the chart draws them dashed. Leave them out when "
            "comparing periods.", 0)


def _target_findings(target: dict[str, Any] | None, add) -> None:
    """Possible leakage, and the group differences in the target worth a look."""
    if not target:
        return
    name = target["column"]
    for e in target["by_group"] + target["by_numeric"]:
        leak = e.get("leakage")
        if not leak:
            continue
        how = {
            "correlation": f"|r| or |ρ| = {_strength(leak['value'])}",
            "separation": f"chance-corrected separation {_strength(leak['value'])}",
            "lambda": f"Goodman–Kruskal λ = {_strength(leak['value'])}",
        }[leak["measure"]]
        add("possible_leakage", "high", [e["column"], name],
            f"{code(e['column'])} predicts {code(name)} almost perfectly: possible target leakage ({how})",
            f"{_sentence(leak['detail'])} Such a column is usually computed from the target (a band, the same "
            "amount in another currency) or filled in after the outcome is known (a cancellation reason). "
            "It explains nothing about why the target varies, and a model using it fails on new data. Check "
            "how the column is made before using it.", 200 + leak["value"])

    for e in target["by_group"] + (target["by_numeric"] if target["kind"] == "categorical" else []):
        s = e["strength"]
        if s is None or s < TARGET_FINDING or e.get("leakage"):
            continue
        measure = "Cramér's V" if e["measure"] == "cramers_v" else "η"
        title = (f"{code(name)} differs {e['strength_label']}ly by {code(e['column'])} "
                 f"({measure} = {_strength(s)})")
        detail = _target_difference_detail(target, e)
        detail += (" A difference between groups isn't a cause: another column (the mix of products, "
                   "customers or time) can produce it. No significance test was run.")
        add("target_difference", "medium" if s >= TARGET_MEDIUM else "low", [e["column"], name], title, detail, s)


def _target_difference_detail(target: dict[str, Any], e: dict[str, Any]) -> str:
    name = target["column"]
    hi, lo = e.get("highest"), e.get("lowest")
    if target["kind"] == "numeric" and hi and lo:
        text = (f"Median {code(name)} is highest for {code(hi['value'])} ({_num(hi['median'])}, n = {hi['count']:,}) "
                f"and lowest for {code(lo['value'])} ({_num(lo['median'])}, n = {lo['count']:,})")
        if e.get("median_ratio"):
            text += f", {_num(e['median_ratio'], 2)}×"
        return text + "."
    if target["kind"] == "categorical" and hi and lo:
        focus = target["focus_class"]
        return (f"The share of {code(focus)} is highest for {code(hi['value'])} ({_pct(hi['rate'])}, "
                f"n = {hi['count']:,}) and lowest for {code(lo['value'])} ({_pct(lo['rate'])}, n = {lo['count']:,}).")
    if e.get("classes"):
        meds = ", ".join(f"{code(c['value'])} {_num(c['median'])}" for c in e["classes"])
        return f"Median {code(e['column'])} per class of {code(name)}: {meds}."
    return "See section 4b for the values per group."


def _type_issue_title(i: dict[str, Any]) -> str:
    col = code(i["column"])
    return {
        "numbers_as_text": f"{col} holds numbers stored as text",
        "dates_as_text": f"{col} holds dates stored as text",
        "mixed_types": f"{col} mixes different value types",
        "inconsistent_labels": f"{col} writes the same label in different ways",
        "untrimmed_whitespace": f"{col} has values with leading or trailing spaces",
        "nested_values": f"{col} holds lists or nested objects",
    }.get(i["issue"], f"{col}: {i['issue'].replace('_', ' ')}")


def _type_issue_detail(i: dict[str, Any]) -> str:
    parts = [_sentence(i.get("detail", ""))]
    issue = i["issue"]
    if issue == "numbers_as_text":
        parts.append("Until converted, the column can't be summed, averaged or correlated.")
        if i.get("non_numeric_examples"):
            parts.append(f"Values that don't parse: {_examples(i['non_numeric_examples'])}.")
    elif issue == "dates_as_text":
        parts.append("Until parsed, it can't be sorted or grouped by date reliably.")
        if i.get("non_date_examples"):
            parts.append(f"Values that don't parse: {_examples(i['non_date_examples'])}.")
    elif issue == "inconsistent_labels" and i.get("groups"):
        groups = "; ".join(" / ".join(f"{code(v['value'])} ×{v['count']}" for v in g) for g in i["groups"][:3])
        parts.append(f"Counts per category are split between the variants: {groups}.")
    elif i.get("examples"):
        parts.append(f"Examples: {_examples(i['examples'])}.")
    return " ".join(p for p in parts if p)


def _skew_detail(s: dict[str, Any]) -> str:
    mean, median = s["mean"], s["median"]
    right = s["skewness"] > 0
    pull = "a few large values pull the average up" if right else "a few small values pull the average down"
    if median and mean is not None and (mean > 0) == (median > 0):
        ratio = mean / median
        comparison = (f"The mean ({_num(mean)}) is {_num(ratio, 2)}× the median ({_num(median)})"
                      if ratio >= 1 else f"The mean ({_num(mean)}) is {_pct(100 * ratio)} of the median ({_num(median)})")
    else:
        comparison = f"The mean is {_num(mean)} and the median {_num(median)}"
    return (f"{comparison}: {pull}. Use the median to describe a typical value; "
            "for charts and models, a log scale often helps.")


def _single_value(s: dict[str, Any]) -> str:
    kind = s.get("kind")
    if kind == "numeric":
        return _num(s.get("min"))
    if kind == "boolean":
        return "True" if s.get("true") else "False"
    if kind == "datetime":
        return str(s.get("min"))
    tops = s.get("top_values") or []
    return tops[0]["value"] if tops else "?"


def _dominant(s: dict[str, Any]) -> tuple[str, float] | None:
    """(value, percent) of the most common value for categorical/boolean columns."""
    if s.get("kind") == "boolean" and s.get("true_pct") is not None:
        t = s["true_pct"]
        return ("True", t) if t >= 50 else ("False", round(100 - t, 2))
    if s.get("kind") == "categorical" and s.get("top_values"):
        top = s["top_values"][0]
        return top["value"], top["pct"]
    return None


def _pair_strength(p: dict[str, Any]) -> float:
    return max(abs(v) for v in (p.get("pearson"), p.get("spearman"), 0.0) if v is not None)


def _dup_examples(d: dict[str, Any]) -> str:
    if not d.get("examples"):
        return ""
    groups = "; ".join(", ".join(str(r) for r in e["rows"]) for e in d["examples"][:3])
    return f" Example rows (0-based position): {groups}."


# --------------------------------------------------------------------------
# Next steps and limitations
# --------------------------------------------------------------------------


def cleaning_options(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per kind of problem found, with the columns it affects."""
    merged: dict[str, dict[str, Any]] = {}
    for f in findings:
        key = ADVICE_ALIAS.get(f["rule"], f["rule"])
        if key not in CLEANING_ADVICE:
            continue
        label, options, cost = CLEANING_ADVICE[key]
        entry = merged.setdefault(key, {"problem": label, "columns": [], "options": options, "cost": cost})
        for c in f["columns"]:
            if c not in entry["columns"]:
                entry["columns"].append(c)
    return list(merged.values())


def analysis_suggestions(summary: dict[str, Any], findings: list[dict[str, Any]]) -> list[str]:
    st = summary["statistics"]
    ids = set(summary.get("analysis", {}).get("excluded_id_columns", []))
    rules = {f["rule"] for f in findings}
    numeric = [k for k, v in st.items() if v.get("kind") == "numeric" and k not in ids and v.get("count")]
    dates = [k for k, v in st.items() if v.get("kind") == "datetime"]
    dates += [i["column"] for i in summary["quality"]["type_issues"] if i["issue"] == "dates_as_text"]
    # Dates, numbers and lists stored as text look categorical but aren't groups.
    not_groups = set(dates) | {i["column"] for i in summary["quality"]["type_issues"]
                               if i["issue"] in ("numbers_as_text", "nested_values")}
    categorical = [k for k, v in st.items() if v.get("kind") in ("categorical", "boolean") and v.get("count")
                   and k not in not_groups]

    out = ["Agree on the question this data should answer. It decides which of the findings above matter most."]
    trend = (summary.get("analysis", {}).get("time") or {}).get("trend")
    if trend:
        if trend["span_years"] >= 2:
            out.append(f"To see seasonality, compare each {trend['period']} with the same {trend['period']} a "
                       "year earlier, not with the one just before it.")
        else:
            out.append("With less than two years of data, a seasonal pattern can't be told apart from a "
                       "trend: one busy November is not yet a pattern.")
    elif dates and numeric:
        out.append(f"Plot how {_codes(numeric[:3])} change over time using {code(dates[0])}"
                   + (" (after parsing the dates)." if dates[0] not in st or st[dates[0]].get("kind") != "datetime"
                      else "."))
    target = summary.get("analysis", {}).get("target")
    if target:
        out += _target_suggestions(target)
    elif categorical and numeric:
        out.append(f"Compare {code(numeric[0])} across the groups of {code(categorical[0])}: medians and "
                   "box plots per group (`df.groupby(...)[...].describe()`). Run again with "
                   f"`--target {numeric[0]}` to do this for every column.")
    if rules & {"strong_correlation", "near_duplicate_columns"}:
        out.append("For the strongest pairs, look at the scatter plots and ask whether a third variable "
                   "(time, size, region) could drive both.")
    if "skewed" in rules:
        skewed = [f["columns"][0] for f in findings if f["rule"] == "skewed"]
        out.append(f"Use medians or a log scale when comparing groups on {_codes(skewed[:4])}.")
    gaps = [f["columns"][0] for f in findings if f["rule"] == "missing_values"]
    if gaps:
        out.append(f"Check whether the gaps in {_codes(gaps[:3])} depend on other columns (compare the "
                   "missing rate across groups). That shows whether dropping rows would bias the results.")
    return out


def _target_suggestions(target: dict[str, Any]) -> list[str]:
    out = []
    name = target["column"]
    usable = [e["column"] for e in target["by_group"] if e["strength"] is not None and not e.get("leakage")]
    if len(usable) >= 2:
        a, b = usable[0], usable[1]
        how = (f"`df.groupby(['{b}', '{a}'])['{name}'].median().unstack()`" if target["kind"] == "numeric"
               else f"`pd.crosstab([df['{b}'], df['{a}']], df['{name}'], normalize='index')`")
        out.append(f"Check whether the differences in {code(name)} by {code(a)} hold within each group of "
                   f"{code(b)} ({how}). A difference that shrinks or reverses inside the groups comes from the "
                   f"mix of {code(b)}, not from {code(a)} itself (Simpson's paradox).")
    if target["leakage"]:
        out.append(f"Leave {_codes([e['column'] for e in target['leakage']])} out when explaining or predicting "
                   f"{code(name)} until you know how {'it is' if len(target['leakage']) == 1 else 'they are'} made.")
    out.append("Before acting on a difference, check that it also shows up in other data (another period, "
               "another export): with many columns compared, some differences appear by chance.")
    return out


def limitations(summary: dict[str, Any]) -> list[str]:
    ov, q = summary["overview"], summary["quality"]
    analysis = summary.get("analysis", {})
    src = summary.get("source", {})
    out = []
    if ov["rows"] < SMALL_DATA_ROWS:
        out.append(f"**Small dataset.** With {ov['rows']:,} rows, every statistic here is uncertain and a "
                   "single row can change the picture.")
    out += [
        "**IQR outliers** are a rule of thumb for roughly symmetric data. On skewed data they flag many "
        "legitimate values; on small samples the quartiles themselves are unstable.",
        "**Correlation.** Pearson measures straight-line relationships and is sensitive to outliers; "
        "Spearman measures any steadily rising or falling relationship. Neither detects U-shaped "
        "relationships, and neither shows cause and effect.",
        "**Only numeric columns are correlated.** Relationships involving categorical columns are not "
        "measured here.",
        "**Column types are detected by rules** (for example, text with mostly unique values counts as "
        "free text). Check the overview table if a column was classified unexpectedly.",
        "**Disguised missing values** are matched against a fixed list of placeholders (`-`, `?`, "
        "`unknown`, `n/a`, ...). Numeric codes such as `999` are only reported when they are also outliers; "
        "a code inside the normal range (`0` for \"no answer\") can't be told apart from a real value.",
    ]
    if analysis.get("excluded_id_columns"):
        out.append(f"**ID columns excluded** from charts and correlations: "
                   f"{_codes(analysis['excluded_id_columns'])}.")
    na = [k for k, o in q["outliers"].items() if o.get("status") in ("zero_iqr", "skipped")]
    if na:
        out.append(f"**IQR not applied** to {_codes(na)}: too few values, or more than half the values identical.")
    sampled = [c for c in summary.get("charts", []) if c.get("sampled")]
    if sampled:
        out.append(f"**Scatter plots are sampled.** {len(sampled)} scatter plot(s) show a random sample of "
                   f"{sampled[0]['points_shown']:,} points; the correlations use every row.")
    if any(i.get("sampled") for i in q["type_issues"]):
        out.append("**Date detection was sampled** on columns with many distinct values; counts of "
                   "date-like text are estimates.")
    if summary.get("charts_skipped"):
        out.append(f"**{len(summary['charts_skipped'])} chart(s) skipped** by the chart limit "
                   "(`--max-charts`); they are listed in `summary.json` under `charts_skipped`.")
    for c in summary.get("charts", []):
        if c.get("note"):
            out.append(f"**Heatmap trimmed:** {c['note']}.")
    sheets = src.get("available_sheets") or []
    if src.get("format") == "excel" and len(sheets) > 1:
        others = [s for s in sheets if s != src.get("sheet")]
        out.append(f"**Only sheet {code(src.get('sheet'))} was analysed.** The file also contains "
                   f"{_codes(others)}.")
    if src.get("flattened_columns"):
        out.append(f"**Nested JSON objects were flattened** into {_codes(src['flattened_columns'])}. A key "
                   "that is absent from some records counts as a missing value there.")
    if src.get("nested_columns"):
        out.append(f"**Lists and objects were kept as JSON text** in {_codes(list(src['nested_columns']))}: "
                   "they are counted, but their items aren't analysed or charted.")
    if analysis.get("time"):
        out += _time_limitations(analysis["time"])
    if analysis.get("target"):
        out += _target_limitations(analysis["target"])
    if src.get("decimal_columns"):
        out.append(f"**Decimal columns were analysed as floats** ({_codes(src['decimal_columns'])}); "
                   "sums of many values can differ from exact decimal arithmetic in the last digits.")
    return out


def _target_limitations(target: dict[str, Any]) -> list[str]:
    name = target["column"]
    out = [
        f"**Target comparison is descriptive.** It shows how {code(name)} differs between groups; it doesn't "
        "show why. A difference can come from another column (Simpson's paradox), and no significance tests "
        "were run, so small differences, especially in small groups, may be noise.",
        f"**Groups with fewer than {target['min_group_rows']} rows** are shown but marked too small to compare. "
        f"Only the {target['max_groups']} most common values of each column are kept; the rest are combined "
        "as `Other` (set with `--max-categories`).",
        "**Leakage is only caught when it's almost perfect** (strength ≥ "
        f"{target['leak_threshold']:g}). A column filled in after the outcome but noisy isn't flagged: know "
        "when each column is recorded.",
    ]
    if target["rows_missing_target"]:
        pct = 100 * target["rows_missing_target"] / max(target["rows_total"], 1)
        out.append(f"**{target['rows_missing_target']:,} row(s) ({_pct(pct)}) have no {code(name)}** and are "
                   "left out of the target comparison.")
    skipped = [s for s in target["skipped"] if s["reason"] not in ("empty", "the same value in every row")]
    if skipped:
        listed = "; ".join(f"{code(s['column'])} ({s['reason']})" for s in skipped)
        out.append(f"**Not compared with the target:** {listed}.")
    return out


def _time_limitations(time: dict[str, Any]) -> list[str]:
    out = []
    t, col = time.get("trend"), time["column"]
    info = time["date_columns"][col]
    others = [c for c in time["date_columns"] if c != col]
    if t:
        how = "chosen automatically" if t["period_chosen_by"] == "automatic" else "set with `--period`"
        line = (f"**Time trend** uses {code(col)} only, by {t['period']} ({how}). It describes the past; "
                "it isn't a forecast.")
        if others:
            line += (f" The other date columns ({_codes(others)}) only get the implausible-date check; "
                     "choose one with `--date-column`.")
        out.append(line)
        out.append("**Gaps and spikes** are judged by a rule of thumb that assumes rows arrive at a steady "
                   "random rate. Real activity is burstier, so treat them as places to look, not proof.")
        if not t["checks_applied"]:
            out.append("**Gap and spike checks skipped:** fewer than 3 complete periods with rows.")
    if info["source"] == "text":
        how = {"day-first": "day first, as detected",
               "ambiguous": "month first, because no value shows the order (03/04 could be either): check it",
               "inconsistent": "value by value, because some look day-first and others month-first: check them",
               }.get(info["date_order"], "in their own year-first format")
        out.append(f"**Dates parsed from text** in {code(col)} ({how}), in memory only; the file is unchanged.")
    if info.get("unparsed"):
        out.append(f"**{info['unparsed']:,} value(s) in {code(col)} aren't dates** and are left out of the "
                   "time analysis.")
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_report(summary: dict[str, Any]) -> str:
    """The full report.md text. Uses summary["findings"] if present, else builds them."""
    findings = summary.get("findings")
    if findings is None:
        findings = build_findings(summary)
    parts = [
        _header(summary),
        _section_overview(summary),
        _section_quality(summary),
        _section_statistics(summary),
        _section_findings(findings),
        _section_target(summary),
        _section_charts(summary),
        _section_next_steps(summary, findings),
        _section_limitations(summary),
    ]
    return "\n\n".join(p.strip("\n") for p in parts if p) + "\n"


def write_report(summary: dict[str, Any], out_dir: Path) -> Path:
    path = Path(out_dir) / "report.md"
    path.write_text(render_report(summary), encoding="utf-8")
    return path


def _header(s: dict[str, Any]) -> str:
    src = s.get("source", {})
    name = src.get("file_name") or "dataset"
    when = s.get("settings", {}).get("generated_at", "")
    lines = [f"# EDA report: {code(name)}", ""]
    lines.append(f"Generated {when.replace('T', ' ')} by `generate_eda.py`. Every number below was computed "
                 "from the data; the findings are rule-based and still need interpretation." if when else
                 "Every number below was computed from the data; the findings are rule-based and still need "
                 "interpretation.")
    if src.get("sha256"):
        state = ("unchanged after the run" if src.get("unchanged_after_run")
                 else "**CHANGED during the run, investigate**")
        lines += ["", f"The input file was only read. SHA-256 `{src['sha256']}`, {state}."]
    return "\n".join(lines)


def _section_overview(s: dict[str, Any]) -> str:
    ov, src = s["overview"], s.get("source", {})
    missing = {c["name"]: c["missing_pct"] for c in s["quality"]["missing"]["columns"]}
    if src.get("format") == "excel":
        how = f"Excel, sheet {code(src.get('sheet'))}"
    elif src.get("format") == "csv":
        how = f"text, encoding `{src.get('encoding')}`, delimiter {code(_delim(src.get('delimiter')))}"
    elif src.get("format") == "parquet":
        how = "Parquet, column types stored in the file"
    elif src.get("format") == "json":
        how = "JSON Lines" if src.get("layout") == "json_lines" else "JSON list of records"
        if src.get("json_path"):
            how += f" at {code(src['json_path'])}"
    else:
        how = "in-memory data"
    lines = [f"## {REPORT_SECTIONS[0]}", ""]
    file_line = f"- **File:** {code(src.get('file_name', '?'))} ({how})"
    if src.get("file_bytes") is not None:
        file_line += f", {_bytes(src['file_bytes'])} on disk"
    lines.append(file_line)
    lines.append(f"- **Size:** {ov['rows']:,} rows × {ov['columns']:,} columns, {ov['memory_human']} in memory")
    lines.append("- **Column kinds:** " + ", ".join(f"{v} {k}" for k, v in ov["kind_counts"].items()))
    if src.get("flattened_columns"):
        lines.append(f"- **Flattened from nested objects:** {_codes(src['flattened_columns'])}")
    if src.get("nested_columns"):
        lines.append(f"- **Lists or objects kept as JSON text:** {_codes(list(src['nested_columns']))}")
    lines += ["", md_table(
        ["Column", "Kind", "dtype", "Non-missing", "Missing", "Unique"],
        [[code(c["name"]), c["kind"], f"`{c['dtype']}`", c["non_null"], _pct(missing.get(c["name"], 0.0)), c["unique"]]
         for c in ov["column_info"]],
    )]
    return "\n".join(lines)


def _section_quality(s: dict[str, Any]) -> str:
    q, rows = s["quality"], s["overview"]["rows"]
    m = q["missing"]
    lines = [f"## {REPORT_SECTIONS[1]}", "", "### Missing values", ""]
    lines.append(f"{m['total_missing']:,} of {m['total_cells']:,} cells are empty ({_pct(m['total_missing_pct'])}). "
                 f"{m['rows_with_missing']:,} rows ({_pct(m['rows_with_missing_pct'])}) have at least one gap; "
                 f"{m['complete_rows']:,} are complete.")
    complete_used = m.get("complete_rows_ignoring_empty_columns")
    if complete_used is not None and complete_used != m["complete_rows"]:
        lines.append(f"Ignoring the completely empty column(s), {complete_used:,} rows are complete.")
    if m.get("total_disguised"):
        lines.append(f"Another {m['total_disguised']:,} values are placeholders such as `-` or `unknown` that "
                     "probably mean missing.")
    gap_rows = [[code(c["name"]), c["missing"], _pct(c["missing_pct"]), c["disguised"],
                 ", ".join(f"{code(_unrepr(k))} ×{v}" for k, v in c["disguised_values"].items()) or "–"]
                for c in sorted(m["columns"], key=lambda c: -(c["missing_pct"] + c["disguised_pct"]))
                if c["missing"] or c["disguised"]]
    if gap_rows:
        lines += ["", md_table(["Column", "Missing", "Missing %", "Placeholders", "Placeholder values"], gap_rows)]

    lines += ["", "### Duplicates", ""]
    for label, d in (("Exact duplicate rows (all columns equal)", q["duplicates"]),
                     ("Repeated keys", q.get("key_duplicates"))):
        if d is None:
            continue
        if label == "Repeated keys":
            label = f"Repeated keys on {_codes(d['subset'] or [])}"
        lines.append(f"- **{label}:** {d['duplicate_rows']:,} extra rows ({_pct(d['duplicate_pct'])}), "
                     f"{d['groups']:,} distinct repeated value(s)." if d["duplicate_rows"] else f"- **{label}:** none.")
    if q.get("key_duplicates") is None:
        lines.append("- **Repeated keys:** not checked. Run with `--key <column>` if rows have an ID that "
                     "should be unique.")

    lines += ["", "### Type problems", ""]
    if q["type_issues"]:
        lines.append(md_table(
            ["Column", "Problem", "Affected values", "Detail"],
            [[code(i["column"]), i["issue"].replace("_", " "), i.get("count", "–"), _sentence(i.get("detail", ""))]
             for i in q["type_issues"]],
        ))
    else:
        lines.append("None found: no numbers or dates stored as text, mixed types, inconsistent labels or "
                     "stray spaces.")
    if s.get("source", {}).get("stored_types"):
        kinds = [c["kind"] for c in s["overview"]["column_info"]]
        typed = sum(k in ("numeric", "datetime", "boolean") for k in kinds)
        texts = sum(k in ("categorical", "text") for k in kinds)
        lines += ["", f"The file stores each column's type. {_plural(typed, 'column', 'columns')} arrived as "
                      "numbers, dates or true/false, so they can't hold numbers or dates as text. Fewer type "
                      "problems here is a property of the format, not proof that the values are right; "
                      f"{_plural(texts, 'text column was', 'text columns were')} checked as usual."]

    lines += ["", "### Outliers (IQR method)", ""]
    if q["outliers"]:
        lines.append("A value is flagged when it lies below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR "
                     "(IQR = Q3 − Q1). Flagged means unusual, not wrong.")
        out_rows = []
        for name, o in q["outliers"].items():
            if o.get("status") == "ok":
                out_rows.append([code(name), o["n_outliers"], _pct(o["outlier_pct"]), o["n_low"], o["n_high"],
                                 f"{_num(o['lower_fence'])} to {_num(o['upper_fence'])}", _sentence(o.get("note", "")) or "–"])
            else:
                out_rows.append([code(name), "–", "–", "–", "–", "–", _sentence(o.get("note", o.get("status", "")))])
        lines += ["", md_table(["Column", "Outliers", "Share", "Low", "High", "Fences", "Note"], out_rows)]
    else:
        lines.append("No numeric columns, so no outlier check.")
    if rows == 0:
        lines += ["", "The dataset has no rows."]
    return "\n".join(lines)


def _section_statistics(s: dict[str, Any]) -> str:
    st = s["statistics"]
    lines = [f"## {REPORT_SECTIONS[2]}"]
    by_kind: dict[str, dict[str, Any]] = {}
    for name, v in st.items():
        by_kind.setdefault(v.get("kind", "?"), {})[name] = v

    if "numeric" in by_kind:
        lines += ["", "### Numeric columns", "", md_table(
            ["Column", "Count", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max", "Skewness", "Shape"],
            [[code(k), v["count"], v["mean"], v["std"], v["min"], v["q1"], v["median"], v["q3"], v["max"],
              None if v["skewness"] is None else round(v["skewness"], 2), v.get("skew_label") or "–"]
             for k, v in by_kind["numeric"].items()],
        )]
        lines += ["", "Std is the sample standard deviation. When mean and median differ a lot, the median is "
                      "the better \"typical\" value."]
    if "boolean" in by_kind:
        lines += ["", "### True/false columns", "", md_table(
            ["Column", "Count", "True", "False", "True %"],
            [[code(k), v["count"], v["true"], v["false"], _pct(v["true_pct"]) if v["true_pct"] is not None else "–"]
             for k, v in by_kind["boolean"].items()],
        )]
    if "datetime" in by_kind:
        lines += ["", "### Date columns", "", md_table(
            ["Column", "Count", "Earliest", "Latest", "Span (days)", "Unique"],
            [[code(k), v["count"], v["min"], v["max"], v["range_days"], v["unique"]] for k, v in by_kind["datetime"].items()],
        )]
    cat = {**by_kind.get("categorical", {}), **by_kind.get("text", {})}
    if cat:
        lines += ["", "### Categorical and text columns", "", md_table(
            ["Column", "Kind", "Count", "Unique", "Most common values"],
            [[code(k), v["kind"], v["count"], v["unique"],
              ", ".join(f"{code(t['value'])} {_pct(t['pct'])}" for t in v["top_values"]) or "–"]
             for k, v in cat.items()],
        )]
    if "empty" in by_kind:
        lines += ["", f"**Empty columns** (every value missing): {_codes(list(by_kind['empty']))}."]
    lines += _time_statistics(s.get("analysis", {}).get("time"))
    if len(lines) == 1:
        lines += ["", "No columns to describe."]
    return "\n".join(lines)


def _time_statistics(time: dict[str, Any] | None) -> list[str]:
    if not time:
        return []
    col, t = time["column"], time.get("trend")
    chosen = {"--date-column": "chosen with `--date-column`", "only date column": "the only date column",
              "most dates": "the date column with the most dates"}[time["chosen_by"]]
    lines = ["", "### Trend over time", ""]
    if not t:
        lines.append(f"No trend for {code(col)} ({chosen}): {time['note']}.")
        return lines
    source = ", parsed from text" if t["date_source"] == "text" else ""
    lines.append(f"{code(col)} ({chosen}{source}), grouped by {t['period']}.")
    counts = [p["rows"] for p in t["periods"]]
    period = t["period"]
    rows = [
        ["Date range", f"{t['first'][:10]} to {t['last'][:10]} ({t['span_days']:,} days)"],
        ["Rows used", f"{t['rows_used']:,}"],
        [f"Rows per {period}", f"typical (median) {_num(t['typical_rows'])}, lowest {min(counts):,}, "
                               f"highest {max(counts):,}, over {len(counts):,} {period}s"],
    ]
    if t["rows_missing_date"]:
        rows.append(["Rows without a date", f"{t['rows_missing_date']:,}"])
    if t["rows_excluded_implausible"]:
        rows.append(["Implausible dates left out", f"{t['rows_excluded_implausible']:,}"])
    if t["partial"]:
        rows.append([f"Incomplete {period}s", ", ".join(p["label"] for p in t["partial"])])
    if t["gaps"]:
        rows.append([f"Empty {period}s (gaps)", "; ".join(
            g["first"] if g["periods"] == 1 else f"{g['first']} to {g['last']}" for g in t["gaps"])])
    if t["empty_periods_by_chance"]:
        rows.append([f"Other empty {period}s", f"{t['empty_periods_by_chance']:,}: too few rows per "
                                               f"{period} to call them gaps"])
    if t["empty_weekend_days"]:
        rows.append(["Empty weekend days", f"{t['empty_weekend_days']:,}: probably no activity, not missing data"])
    if t["spikes"]:
        rows.append(["Spikes", "; ".join(f"{sp['label']} ({sp['rows']:,} rows)" for sp in t["spikes"])])
    if t["value_column"]:
        rows.append(["Also shown per period", f"median of {code(t['value_column'])}"])
    lines += ["", md_table(["Measure", "Value"], rows)]
    return lines


def _section_findings(findings: list[dict[str, Any]]) -> str:
    lines = [f"## {REPORT_SECTIONS[3]}", ""]
    if not findings:
        lines.append("No rule flagged anything: no notable gaps, duplicates, type problems, skew, outliers or "
                     "strong correlations.")
    else:
        counts = {sev: sum(f["severity"] == sev for f in findings) for sev in SEVERITY_ORDER}
        lines.append("Ranked by how much each one could distort an analysis if ignored: "
                     + ", ".join(f"{n} {sev}" for sev, n in counts.items() if n) + ". "
                     "Rule-based: they show where to look, not what it means for your question.")
        lines.append("")
        for i, f in enumerate(findings, 1):
            lines.append(f"{i}. **[{f['severity']}] {f['title']}.** {f['detail']}")
    lines += ["", INTERPRETATION_MARKER]
    return "\n".join(lines)


def _section_target(s: dict[str, Any]) -> str:
    target = s.get("analysis", {}).get("target")
    if not target:
        return ""
    name = target["column"]
    lines = [f"## {TARGET_SECTION}", ""]
    missing = (f"; {target['rows_missing_target']:,} row(s) without a value are left out"
               if target["rows_missing_target"] else "")
    if target["kind"] == "numeric":
        o = target["overall"]
        lines.append(f"Target {code(name)} (numeric): {target['rows_used']:,} rows used{missing}. Overall median "
                     f"{_num(o['median'])} (middle 50%: {_num(o['q1'])} to {_num(o['q3'])}), mean {_num(o['mean'])}.")
        lines += ["", "Medians, not means: a few large values can move a group's mean a long way. **Strength** "
                      "is on the same 0–1 scale for every column: for groups, η (eta) on ranks, corrected for "
                      "the number of groups; for numeric columns, the larger of |Pearson r| and |Spearman ρ|. "
                      "0 = no difference, 1 = the column fixes the target completely."]
    else:
        focus = target["focus_class"]
        classes = ", ".join(f"{code(c['value'])} {c['count']:,} ({_pct(c['pct'])})" for c in target["classes"])
        lines.append(f"Target {code(name)} (categorical): {target['rows_used']:,} rows used{missing}. "
                     f"Classes: {classes}.")
        lines += ["", ("Rates are the share of each group's rows in a class"
                       + (f"; with two classes the share of {code(focus)} is shown" if focus else "")
                       + ". **Strength** (0–1): Cramér's V (bias-corrected) for groups, η on ranks for numeric "
                         "columns. 0 = no difference, 1 = the column fixes the class completely.")]
    if target["leakage"]:
        lines += ["", f"**Possible leakage:** {_codes([e['column'] for e in target['leakage']])} "
                      f"{'predicts' if len(target['leakage']) == 1 else 'predict'} {code(name)} almost perfectly. "
                      "See the findings above."]

    ranked = sorted(target["by_group"] + target["by_numeric"], key=lambda e: -(e["strength"] or -1))
    if ranked:
        rows = []
        for e in ranked:
            kind = "groups" if e in target["by_group"] else "numeric"
            rows.append([code(e["column"]), kind, _strength(e["strength"]),
                         e["strength_label"] or "–", _target_summary_cell(target, e)])
        lines += ["", "### Ranking", "", md_table(["Column", "Compared as", "Strength", "Label", "Summary"], rows)]

    for e in [e for e in target["by_group"] if e["strength"] is not None][:TARGET_DETAIL_COLUMNS]:
        lines += ["", f"### {code(name)} by {code(e['column'])}", ""]
        lines.append(_group_table(target, e))
        if e["other_values"]:
            lines += ["", f"{e['other_values']:,} rows with rarer values are combined as `Other`."]
    if target["kind"] == "categorical" and target["by_numeric"]:
        classes = [c["value"] for c in target["classes"]]
        rows = [[code(e["column"]), *[_num(c["median"]) for c in e["classes"]],
                 _strength(e["strength"])] for e in target["by_numeric"]]
        lines += ["", "### Numeric columns by class (medians)", "",
                  md_table(["Column", *[f"{_cell(c)}" for c in classes], "Strength (η)"], rows)]
    small = target["min_group_rows"]
    lines += ["", f"Groups marked *too small* have fewer than {small} rows: their medians and rates can change "
                  "a lot with a few rows, so don't rank them."]
    return "\n".join(lines)


def _target_summary_cell(target: dict[str, Any], e: dict[str, Any]) -> str:
    if e.get("leakage"):
        return "possible leakage"
    if e["measure"] == "correlation":
        if e["strength"] is None:
            return "–"
        return f"r = {_num(e['pearson'], 2)}, ρ = {_num(e['spearman'], 2)}, n = {e['n']:,}"
    hi, lo = e.get("highest"), e.get("lowest")
    if hi and lo and target["kind"] == "numeric":
        return f"highest {code(hi['value'])} {_num(hi['median'])}, lowest {code(lo['value'])} {_num(lo['median'])}"
    if hi and lo:
        return f"highest {code(hi['value'])} {_pct(hi['rate'])}, lowest {code(lo['value'])} {_pct(lo['rate'])}"
    if e.get("classes"):
        return ", ".join(f"{code(c['value'])} {_num(c['median'])}" for c in e["classes"])
    return "–"


def _group_table(target: dict[str, Any], e: dict[str, Any]) -> str:
    note = lambda g: "too small to compare" if g["small"] else ""  # noqa: E731
    if target["kind"] == "numeric":
        return md_table(["Group", "Rows", "Median", "Middle 50%", "Mean", "Note"],
                        [[code(g["value"]), g["count"], g["median"], f"{_num(g['q1'])} to {_num(g['q3'])}",
                          g["mean"], note(g)] for g in e["groups"]])
    classes = [c["value"] for c in target["classes"]]
    focus = target["focus_class"]
    shown = [focus] if focus else classes
    return md_table(["Group", "Rows", *[f"{_cell(c)} %" for c in shown], "Note"],
                    [[code(g["value"]), g["count"], *[_pct(g["rates"][c]) for c in shown], note(g)]
                     for g in e["groups"]])


def _section_charts(s: dict[str, Any]) -> str:
    charts = s.get("charts", [])
    lines = [f"## {REPORT_SECTIONS[4]}"]
    if not charts:
        lines += ["", "No charts were produced."]
    for c in charts:
        title, text = _chart_text(c, s)
        lines += ["", f"### {title}", "", text, "", f"![{_alt(title)}]({_link(c['file'])})"]
    if s.get("charts_skipped"):
        lines += ["", f"Not drawn because of the chart limit: {', '.join(s['charts_skipped'])}."]
    return "\n".join(lines)


def _chart_text(c: dict[str, Any], s: dict[str, Any]) -> tuple[str, str]:
    st, q = s["statistics"], s["quality"]
    analysis = s.get("analysis", {})
    kind = c["type"]
    if kind == "histogram_boxplot":
        col = c["column"]
        v, o = st.get(col, {}), q["outliers"].get(col, {})
        text = (f"{c['values']:,} values. Median {_num(v.get('median'))}, mean {_num(v.get('mean'))}, "
                f"range {_num(v.get('min'))} to {_num(v.get('max'))}")
        text += f"; {v['skew_label']}." if v.get("skew_label") else "."
        if o.get("status") == "ok":
            text += (f" The box plot marks {o['n_outliers']:,} value(s) ({_pct(o['outlier_pct'])}) beyond "
                     "1.5 × IQR." if o["n_outliers"] else " No values beyond 1.5 × IQR.")
        return f"Distribution of {code(col)}", text
    if kind == "histogram_log":
        return (f"Distribution of {code(c['column'])}, log scale",
                "The same values on a log axis. Each ×10 step gets the same width, so the long right tail "
                "spreads out and the shape of the bulk of the data becomes visible.")
    if kind == "missing_values":
        cols = [x for x in q["missing"]["columns"] if x["name"] in c["columns"]]
        worst = max(cols, key=lambda x: x["missing_pct"]) if cols else None
        text = f"{_plural(len(c['columns']), 'column has', 'columns have')} gaps"
        text += f"; the most is {code(worst['name'])} with {_pct(worst['missing_pct'])} missing." if worst else "."
        return "Missing values by column", text
    if kind == "correlation_heatmap":
        pairs = analysis.get("correlations", {}).get("pairs", [])
        text = f"Pearson r for {len(c['columns'])} numeric columns, each pair shown once."
        if pairs:
            p = pairs[0]
            text += (f" Strongest pair: {code(p['a'])} and {code(p['b'])} "
                     f"(r = {_num(p['pearson'], 2)}, ρ = {_num(p['spearman'], 2)}, {p['strength']}).")
        if c.get("note"):
            text += f" {_sentence(c['note'])}"
        return "Correlation heatmap", text
    if kind == "scatter":
        a, b = c["columns"]
        p = next((p for p in analysis.get("correlations", {}).get("pairs", []) if (p["a"], p["b"]) == (a, b)), None)
        text = ""
        if p:
            text = (f"Pearson r = {_num(p['pearson'], 2)}, Spearman ρ = {_num(p['spearman'], 2)} "
                    f"({p['strength']}, {p['direction']}), n = {p['n']:,}.")
            for note in p.get("notes", []):
                text += f" {_sentence(note)}"
        if c.get("sampled"):
            text += f" A random sample of {c['points_shown']:,} points is shown."
        return f"{code(b)} vs {code(a)}", text.strip() or "Scatter plot."
    if kind == "top_categories":
        col = c["column"]
        cs = analysis.get("categorical", {}).get(col, {})
        cats = cs.get("categories", [])
        text = f"{_plural(cs.get('unique', 0), 'distinct value', 'distinct values')}."
        if cats:
            text += f" Most common: {code(cats[0]['value'])} ({_pct(cats[0]['pct'])})."
            other = cs.get("other", {})
            if other.get("count"):
                text += (f" The top {len(cats)} cover {_pct(100 - other['pct'])} of non-missing rows; "
                         f"{_plural(other['categories'], 'rarer value is', 'rarer values are')} grouped as Other.")
            elif len(cats) > 1:
                text += " Every value is shown."
        return f"Most common values of {code(col)}", text
    if kind == "time_trend":
        t = (analysis.get("time") or {}).get("trend") or {}
        period = c["period"]
        text = (f"Rows per {period} from {t.get('first', '')[:10]} to {t.get('last', '')[:10]}; a typical "
                f"{period} has {_num(t.get('typical_rows'))} rows.")
        marks = []
        if t.get("partial"):
            marks.append(f"dashed ends are incomplete {period}s")
        if t.get("gaps"):
            marks.append(f"red bands are empty {period}s")
        if t.get("spikes"):
            marks.append("rings mark spikes")
        if marks:
            text += " " + _sentence("; ".join(marks))
        if c.get("value_column"):
            text += f" The lower panel shows the median {code(c['value_column'])} per {period}."
        return f"Rows per {period} by {code(c['column'])}", text
    if kind in ("target_box", "target_rates"):
        target = analysis.get("target") or {}
        e = next((x for x in target.get("by_group", []) if x["column"] == c["column"]), {})
        name, col = c["target"], c["column"]
        text = f"{_plural(c['groups'], 'group', 'groups')}; strength {_strength(e.get('strength'))} ({e.get('strength_label')})."
        if e.get("leakage"):
            text += " The groups barely overlap: possible leakage (see the findings)."
        else:
            detail = _target_difference_detail(target, e) if e.get("highest") else ""
            text += f" {detail}" if detail else ""
        if kind == "target_box":
            text += " Sorted by median; gray boxes are groups under " \
                    f"{target.get('min_group_rows', 30)} rows, Other or missing."
            if c.get("log_scale"):
                text += " Log axis, because the target is highly skewed."
            return f"{code(name)} by {code(col)}", text
        title = (f"{code(name)} = {code(target['focus_class'])}: rate by {code(col)}" if target.get("focus_class")
                 else f"{code(name)} by {code(col)}")
        return title, text
    return kind.replace("_", " ").capitalize(), ""


def _section_next_steps(s: dict[str, Any], findings: list[dict[str, Any]]) -> str:
    lines = [f"## {REPORT_SECTIONS[5]}", "", "### Cleaning options", ""]
    options = cleaning_options(findings)
    if options:
        lines.append("Nothing has been changed. Each option has a cost; choose deliberately and write the "
                     "result to a new file, never over the original.")
        lines.append("")
        for o in options:
            cols = f" ({_codes(o['columns'][:6])}{', ...' if len(o['columns']) > 6 else ''})" if o["columns"] else ""
            lines.append(f"- **{o['problem']}**{cols}: {o['options']} *Cost:* {o['cost']}")
    else:
        lines.append("No cleaning is suggested by the checks.")
    lines += ["", "### Further analysis", ""]
    lines += [f"- {x}" for x in analysis_suggestions(s, findings)]
    return "\n".join(lines)


def _section_limitations(s: dict[str, Any]) -> str:
    return "\n".join([f"## {REPORT_SECTIONS[6]}", "", *[f"- {x}" for x in limitations(s)]])


# --------------------------------------------------------------------------
# Markdown helpers (decision D6: no tabulate)
# --------------------------------------------------------------------------


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """GitHub-flavoured Markdown table. Numeric columns are right-aligned."""
    numeric = [
        all(isinstance(r[i], (int, float)) and not isinstance(r[i], bool) for r in rows if r[i] is not None)
        and any(r[i] is not None for r in rows)
        for i in range(len(headers))
    ]
    out = ["| " + " | ".join(_cell(h) for h in headers) + " |",
           "|" + "|".join("---:" if n else "---" for n in numeric) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_cell(_num(c) if isinstance(c, (int, float)) and not isinstance(c, bool)
                                           else c) for c in r) + " |")
    return "\n".join(out)


def code(text: Any) -> str:
    """Inline code that survives backticks, pipes and line breaks inside the text."""
    t = str(text).replace("\r", " ").replace("\n", " ")
    if "`" not in t:
        return f"`{t}`" if t else "` `"
    return f"`` {t} ``"


def _cell(x: Any) -> str:
    if x is None:
        return "–"
    return str(x).replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def _num(x: Any, digits: int = 4) -> str:
    """1,234 / 12.35 / 0.0123; '–' for missing values."""
    if x is None:
        return "–"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return f"{x:,}"
    x = float(x)
    if x.is_integer() and abs(x) < 1e15:
        return f"{int(x):,}"
    if abs(x) >= 10 ** digits:
        return f"{x:,.0f}"
    return f"{x:,.{digits}g}"


def _pct(p: float | None) -> str:
    if p is None:
        return "–"
    if 0 < p < 0.1:
        return "<0.1%"
    if 99.9 < p < 100:
        return ">99.9%"
    return f"{p:.1f}%"


def _plural(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def _codes(names: list[Any]) -> str:
    return ", ".join(code(n) for n in names)


def _examples(values: list[str]) -> str:
    return ", ".join(code(_unrepr(v)) for v in values[:5])


def _unrepr(v: str) -> str:
    """inspect_data stores example values as repr() strings ("'abc'"); show them without quotes."""
    s = str(v)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def _sentence(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _delim(d: Any) -> str:
    return {"\t": "tab", ",": ",", ";": ";", "|": "|"}.get(d, str(d))


def _bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _alt(title: str) -> str:
    return title.replace("`", "").replace("[", "(").replace("]", ")")


def _link(path: str) -> str:
    """Relative link; spaces and parentheses would break Markdown image syntax."""
    return path.replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def _strength(x: float | None) -> str:
    """A 0-1 strength with two decimals, so 0.60 and 1.00 read as measurements."""
    return "–" if x is None else f"{x:.2f}"
