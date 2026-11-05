# Data cleaning: options and what they cost

Read this before recommending how to handle a data-quality problem. Every
cleaning step changes the results of the analysis that follows, so the user
decides. Your job is to lay out the options, what each one costs, and how many
rows or values it touches.

**Contents**
1. [Ground rules](#1-ground-rules)
2. [Missing values](#2-missing-values)
3. [Placeholder values](#3-placeholder-values)
4. [Duplicates](#4-duplicates)
5. [Type problems](#5-type-problems)
6. [Outliers](#6-outliers)
7. [Skewed, constant and nearly constant columns](#7-skewed-constant-and-nearly-constant-columns)
8. [Dates over time](#8-dates-over-time)
9. [Proposing changes to the user](#9-proposing-changes-to-the-user)
10. [Applying approved changes](#10-applying-approved-changes)

Each section names the `rule` values it covers in `summary.json` → `findings`,
so you can go from a finding straight to its options, and the
`clean_data.py` step that applies an approved option (see
[section 10](#10-applying-approved-changes)). The pandas examples show what
each step does and are useful for one-off checks; they use the same sales
data as `pandas-guide.md` (`path`, `out_dir`).

```python
from pathlib import Path

import numpy as np
import pandas as pd

df = pd.read_csv(path, dtype={"customer_id": "str"})
clean = df.copy()  # every example changes `clean`, never `df` and never the file
```

---

## 1. Ground rules

1. **Ask first.** No step runs without the user's agreement (SKILL.md, rule 2).
2. **Count before and after.** For every step, report rows and missing values
   before and after, and how many values changed.
3. **Write a new file.** Approved steps produce e.g. `sales_cleaned.csv` next
   to the analysis output, never over the original.
4. **Keep a log.** A list of what was done, in order, with the counts. It
   belongs in the report and makes the cleaning reproducible.
5. **Smallest change that solves the problem.** Prefer flagging over deleting,
   and fixing a value over dropping a row.
6. **Order matters.** Fix types and placeholders first (they create or reveal
   missing values), then duplicates, then missing values, then outliers.

## 2. Missing values

Finding rules: `missing_values`, `missing_minor`, `incomplete_rows`, `empty_columns`.
Steps: `drop_rows_missing` (in the given columns), `fill_missing` (median,
mode or constant, optionally `by` a group column; always adds
`<column>_was_missing`), `drop_columns`.

**Why the reason for missingness matters.** Statisticians use three cases:

| Type | Meaning | Example | Dropping rows... |
|---|---|---|---|
| MCAR, missing completely at random | Gaps unrelated to anything | A sensor drops random readings | loses data, but doesn't bias results |
| MAR, missing at random | Gaps depend on *other, recorded* columns | Discount missing more often in one region | biases results, but you can adjust with the other columns |
| MNAR, missing not at random | Gaps depend on the *missing value itself* | High earners skip the income question | biases results; no method on this data alone fully fixes it |

You can't prove MCAR from the data, but you can find evidence against it:
compare the missing rate across groups and periods.

```python
missing_by_region = df["discount"].isna().groupby(df["region"]).mean().mul(100).round(1)
# Very different rates per group: not MCAR. Dropping rows would under-represent some groups.
```

**Options**

| Option | Fits when | Cost |
|---|---|---|
| Leave as missing | Always a valid choice | Every analysis must say how many rows it used |
| Drop rows (`dropna(subset=[...])`) | Few rows affected (a few %) and plausibly MCAR | Smaller sample; biased if MAR/MNAR. Dropping on *all* columns can delete far more rows than any single column suggests |
| Drop the column | Mostly empty (e.g. > 60%) or irrelevant to the goal | Lose whatever the column did say |
| Fill with median / mode | Few gaps, a quick baseline | Shrinks the spread, weakens correlations, creates a spike at one value |
| Fill with the group median | Values differ by group (MAR on that group) | Same as above, but less biased |
| Forward-fill (`ffill`) | Time series where a value holds until it changes (prices, status) | Wrong if the gap is a real change; sort by time first |
| Model-based imputation | Gaps matter and other columns predict them | Complex; needs explaining; can look more certain than it is |
| Add a "was missing" indicator | Combined with any fill | One more column, but the information isn't lost |

```python
# Always keep the information that a value was missing before filling it.
clean["discount_was_missing"] = clean["discount"].isna()

# Fill with the median of the region, falling back to the overall median.
group_median = clean.groupby("region")["discount"].transform("median")
clean["discount"] = clean["discount"].fillna(group_median).fillna(clean["discount"].median())

# How many rows would dropna on every column remove? Often surprising.
rows_lost = len(df) - len(df.dropna())
```

`0` is not the same as missing. Filling a missing discount with `0` claims
"no discount was given", which is a statement about the world, not a
neutral default. Only use it when the user confirms that's what a gap means.

## 3. Placeholder values

Finding rule: `disguised_missing`. Step: `placeholders_to_missing` (text
placeholders; without `values` it uses the standard list the checks look for).

Values like `-`, `?`, `n/a`, `unknown` are stored as text, so pandas counts
them as present. A numeric column containing one becomes a text column.

| Option | Cost |
|---|---|
| Convert to real missing values | Missing counts go up. Correct only if the placeholder never means something real |
| Keep as its own category | Right when "unknown" is a genuine answer (a survey option), not a gap |

```python
placeholders = ["-", "?", "n/a", "unknown"]
is_placeholder = clean["rating"].str.strip().str.casefold().isin(placeholders)
clean["rating"] = clean["rating"].mask(is_placeholder)  # placeholder -> missing
changed = int(is_placeholder.sum())

# Or at load time: pd.read_csv(path, na_values=placeholders)
```

The skill detects only a fixed list of text placeholders.

**Numeric codes** (finding rule `possible_missing_code`). Exports often write
`999`, `-1` or `9999` for "unknown". They look like numbers, so they inflate
the mean, the spread and the skew. The skill reports them when they are also
IQR outliers (a delivery time of 999 days). Codes inside the normal range
(`0` for "no answer" in a 0–10 score) can't be detected: ask about them.
Step: `placeholders_to_missing` with the confirmed codes, e.g. `"values": [999]`.

```python
codes = clean["quantity"].isin([999, -1]).sum()  # how many rows use the code
clean["quantity"] = clean["quantity"].mask(clean["quantity"].isin([999, -1]))  # code -> missing, once confirmed
```

## 4. Duplicates

Finding rules: `duplicate_rows`, `duplicate_keys`. Steps:
`drop_exact_duplicates` (keep `first` or `last`), `drop_duplicate_keys`
(keep `first`, `last` or `most_complete`; rows without a key are never
dropped, and the log lists the columns in which the repeats disagree).

**Exact duplicates** (every column equal) are usually export or join errors,
but not always: two identical purchases on the same day can be real. Ask
whether each row is meant to be a unique record.

**Repeated keys** (same ID, other columns may differ) mean the data disagrees
with itself. Look at them before deciding anything:

```python
repeated = df[df.duplicated(subset=["order_id"], keep=False)].sort_values("order_id")
```

| Option | Fits when | Cost |
|---|---|---|
| Drop exact duplicates, keep the first | Rows should be unique | Undercounts if the repeats were genuine events |
| Keep the latest version per key | Rows are updates (there's a timestamp or version) | Discards earlier versions; requires a reliable order |
| Keep the most complete row per key | Repeats differ only in gaps | Arbitrary when rows conflict in real values |
| Aggregate per key (sum, mean) | Repeats are partial records of one thing | Changes what a row means |
| Investigate the source | Conflicting values | Takes time, but it's the only way to know which is right |

```python
before = len(clean)
clean = clean.drop_duplicates()  # exact duplicates
removed_exact = before - len(clean)

# Repeated keys: keep the row with the fewest gaps (ties: the first one).
clean = (
    clean.assign(_gaps=clean.isna().sum(axis=1))
    .sort_values(["order_id", "_gaps"], kind="stable")
    .drop_duplicates(subset=["order_id"], keep="first")
    .drop(columns="_gaps")
    .sort_index()
)
```

## 5. Type problems

### Numbers stored as text

Finding rule: `numbers_as_text` (and `mixed_types` when the odd values are
numbers). Step: `to_numeric`: currency signs, `%`, thousands separators and
decimal commas are handled, `map` turns words into numbers (`{"free": 0}`),
and the values that still don't parse are counted and listed before they
become missing (or stop the run, with `"unparseable": "error"`).

Caused by currency signs, `%`, thousands separators, decimal commas, or a few
text values mixed in. Until converted, the column can't be summed or
correlated.

```python
raw = clean["rating"]
converted = pd.to_numeric(raw.str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")

# Values that had content but didn't parse would silently become missing. Show them first.
not_parsed = raw[raw.notna() & converted.isna()].value_counts()
clean["rating"] = converted

# Currency and thousands separators ("$1,234.50"):
prices = pd.Series(["$1,234.50", "$99", "12%"])
as_numbers = pd.to_numeric(prices.str.replace(r"[$,%\s]", "", regex=True), errors="coerce")

# Decimal commas ("1.234,50"): remove the thousands dots, then swap the comma.
euro = pd.Series(["1.234,50", "3,5"])
as_numbers_eu = pd.to_numeric(euro.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
```

Cost: unparseable values become missing; a `%` column becomes `12` rather
than `0.12` (decide which and say so).

### Dates stored as text

Finding rule: `dates_as_text`. Step: `to_datetime` with `day_first` or an
explicit `format`; without either it uses the order the values settle and
refuses to guess when they don't.

```python
parsed = pd.to_datetime(clean["order_date"], format="%d/%m/%Y", errors="coerce")
failed = clean.loc[parsed.isna() & clean["order_date"].notna(), "order_date"].unique()
clean["order_date"] = parsed
```

Cost: unparseable values become missing. The real risk is the day/month
order: with `03/04/2025` both readings are valid and the wrong one silently
shifts dates. `inspect_data.py` reports `date_order` when the values settle
it (a day above 12 appears); otherwise ask the user.

Time zones: when every value carries one (`2025-10-01T10:00:00Z`,
`+02:00`, `UTC`), the step converts them to UTC and drops the zone, so
rows from different zones compare correctly and the CSV keeps them. Tell
the user that local clock times are gone (10:00+02:00 becomes 08:00). If
local time matters (e.g. "orders by hour of day"), keep the original
column as well. A column that mixes zoned and plain times is refused,
because nobody knows which zone the plain ones are in.

### Mixed types

Finding rule: `mixed_types`.

```python
type_counts = df["rating"].map(type).value_counts()  # e.g. str 280, float 24 (the NaNs)
```

Usually a numeric column with a few notes ("approx. 5", "see email"). Decide
per value: convert, set missing, or move the note to its own column. Cost:
the odd values change meaning or disappear.

### Lists and nested objects

Finding rule: `nested_values`.

JSON and Parquet cells can hold a list (`["red", "blue"]`) or an object. The
scripts keep these as JSON text so they can be counted, but a list isn't a
category or a number. Pick the shape the question needs:

```python
import json

orders = pd.DataFrame({"order_id": [1, 2, 3], "revenue": [30.0, 12.5, 8.0],
                       "tags": ['["gift", "express"]', '["gift"]', None]})
tags = orders["tags"].map(json.loads, na_action="ignore")

# Keep one row per order: count the items.
orders["n_tags"] = tags.str.len()

# One yes/no column per item, still one row per order.
has_gift = tags.map(lambda t: "gift" in t, na_action="ignore")

# One row per item, as a separate table. Never sum `revenue` here: it repeats.
order_tags = orders[["order_id"]].assign(tag=tags).explode("tag")
```

Cost: exploding repeats every other column once per item, so totals over the
exploded table count an order several times; keep it as its own table keyed
by the ID. Counting items keeps one row per record but drops what the items
were. Lists that are missing and lists that are empty (`[]`) mean different
things: check which ones the file has before treating them alike.

### Inconsistent labels and stray spaces

Finding rules: `inconsistent_labels`, `untrimmed_whitespace`. Steps:
`strip_whitespace`, then `normalize_labels` with an explicit mapping (the
log warns about mapping keys that match nothing and labels still written
in more than one way).

`"North"`, `"north"` and `"North "` are three groups to pandas, so counts
and group statistics are split.

```python
before_groups = clean["region"].nunique()
clean["region"] = clean["region"].str.strip().str.title()
after_groups = clean["region"].nunique()

# Different spellings need an explicit mapping. Show it to the user before applying.
clean["region"] = clean["region"].replace({"N.": "North", "Nord": "North"})

# Trim every text column at once.
for col in clean.select_dtypes(include="str").columns:
    clean[col] = clean[col].str.strip()
```

Cost: merges groups. Check that no variants are genuinely different codes
(`"US"` vs `"us"` rarely, but product codes sometimes). Trimming is safe
almost always, except in fixed-width codes where spaces are part of the value.

## 6. Outliers

Finding rule: `outliers`. Steps: `flag_outliers` (adds `<column>_outlier`,
changes nothing) and `cap_outliers` (to the IQR fences; the log shows the
mean, standard deviation and maximum before and after).

An IQR outlier is *unusual*, not *wrong*. Three very different situations
look the same in the box plot:

| Situation | Example | Right response |
|---|---|---|
| Data error | Age 230, price typed in cents | Fix it if the true value is known, otherwise set missing |
| Different population | A wholesale order among retail orders | Analyse it separately or filter by a rule the user agrees on |
| Real, rare value | The one huge customer | **Keep it.** It's often the most important row |

Look first:

```python
extremes = df.nlargest(5, "revenue")  # the whole rows, not just the values
q1, q3 = df["revenue"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
is_high = df["revenue"] > upper
```

| Option | Cost |
|---|---|
| Keep and report robust statistics (median, IQR) | None to the data; the mean still moves with extremes |
| Analyse with and without them, report both | More output, but honest about how much they matter |
| Cap (winsorise) at e.g. the 1st/99th percentile | Keeps the rows but changes real values; totals become wrong |
| Log-transform | Changes the scale's meaning; only for positive values |
| Remove | Biases means and totals downwards; hides the cases that may matter most. Needs a documented reason per row or a rule agreed in advance |

```python
# With vs without: how much do the extremes move the answer?
comparison = pd.Series({
    "mean, all rows": df["revenue"].mean(),
    "mean, without high outliers": df.loc[~is_high, "revenue"].mean(),
    "median, all rows": df["revenue"].median(),
})

# Capping at the 1st and 99th percentiles (only after the user agrees).
low, high = clean["revenue"].quantile([0.01, 0.99])
capped = int(((clean["revenue"] < low) | (clean["revenue"] > high)).sum())
clean["revenue"] = clean["revenue"].clip(lower=low, upper=high)
```

On skewed data (incomes, order values, durations) the IQR method flags many
legitimate values. That's a limitation of the method, not a data problem.

## 7. Skewed, constant and nearly constant columns

Finding rules: `skewed`, `constant_column`, `dominant_category`,
`near_duplicate_columns`, `possible_leakage`.

**Skewed columns** don't need cleaning. They need the right summary: median
and IQR instead of mean and standard deviation. For charts, a log scale; for
models, sometimes a log transform:

```python
log_revenue = np.log1p(df["revenue"])  # log(1 + x): works with zeros, not with negatives
```

Cost of a log transform: differences become ratios (a coefficient means "10%
more", not "10 more"), and results must be transformed back to explain them.

**Constant columns** (one value) carry no information within this file
(step: `drop_columns`). Drop them from the analysis, but ask why they're constant: a filter applied during
export (`country = DE` only) limits what the whole analysis can claim.

**Nearly constant columns** (one value ≥ 95%) are kept. Warn that statistics
for the rare groups rest on very few rows.

**Near-duplicate columns** (finding rule `near_duplicate_columns`, |r| ≥ 0.98)
are usually one column derived from another (a total and its parts, the same
measure in two units). Not a data error; keep both in the file, but use only
one in a model.

**Possible target leakage** (finding rule `possible_leakage`, only with
`--target`): a column that predicts the target almost perfectly. Usually it's
computed from the target (an order-size band from revenue, revenue in another
currency) or filled in after the outcome (a cancellation reason for a churn
target). Nothing in the file is wrong, so there is nothing to clean; the
column just can't *explain* the target. Ask how and when it's recorded:

| Finding | Options | Cost |
|---|---|---|
| `possible_leakage` | Leave the column out when explaining or predicting the target; keep it in the file | If it's a genuine, known-in-advance predictor, you lose real information. Keeping a leaky one makes every comparison and model look far better than it will be on new data |

## 8. Dates over time

Finding rules: `implausible_dates`, `date_gaps`, `date_spike`, `partial_period`.

`generate_eda.py` groups the main date column into periods (rows per week,
month, ...) and flags what looks wrong over time. None of these is fixed by
editing values without the data owner: a date is evidence of *when* something
happened, and a guessed date is worse than a missing one.

```python
dates = pd.to_datetime(clean["order_date"], format="%d/%m/%Y", errors="coerce")

# Implausible dates: count and show them before deciding anything.
today = pd.Timestamp.today().normalize()
placeholder = dates.isin(pd.to_datetime(["1899-12-30", "1900-01-01", "1970-01-01"]))
future = dates > today
print(f"{int(future.sum())} future and {int(placeholder.sum())} placeholder dates")
dates = dates.mask(placeholder)  # placeholders mean "unknown" (only after the user agrees)

# Gaps and spikes: rows per week, with the empty weeks included.
weekly = dates.dt.to_period("W-SUN").value_counts().sort_index()
weekly = weekly.reindex(pd.period_range(weekly.index.min(), weekly.index.max(), freq="W-SUN"), fill_value=0)
empty_weeks = weekly[weekly == 0]
# Inside the busiest week: one date holding most rows points to a load date, not real activity.
busiest = weekly.idxmax()
print(dates[dates.dt.to_period("W-SUN") == busiest].dt.date.value_counts().head(3))

# Partial periods: rows per covered day stay comparable when a month is cut short.
first_day, last_day = dates.min(), dates.max()
monthly = dates.dt.to_period("M").value_counts().sort_index()
covered = pd.Series({p: (min(p.end_time, last_day) - max(p.start_time, first_day)).days + 1
                     for p in monthly.index})
per_day = monthly / covered
```

| Finding | Options | Cost |
|---|---|---|
| `implausible_dates`: future dates | Check the source; correct an obvious typo (2052 for 2025) only when the owner confirms it, otherwise make it missing | A future date can be right: due dates and planned deliveries are in the future by design |
| `implausible_dates`: placeholders (`1900-01-01`, `1970-01-01`) | Make them missing (`mask`, above; step `placeholders_to_missing`, on the text before `to_datetime` or on the dates after it) | More missing dates, but no fake history: a 1900 order date stretches every range and age |
| `date_gaps` | Ask whether the period is missing from the export or really had no activity. Compare periods outside the gap only, or mark it in every chart | **Never** fill a gap with zeros or interpolated values: that invents data. Ignoring it understates totals and averages |
| `date_spike` | Look at the rows in the spike period: many rows on one date, repeated keys or exact duplicates point to a double or late load. Deduplicate, or use a different date column (`--date-column`) | Removing a real peak (a sale, a launch) deletes the most important event in the data |
| `partial_period` | Leave the first or last period out of comparisons, or compare rows per covered day (`per_day`, above) | Dropping the rows themselves loses real data from totals; they're only a problem when compared as if complete |

A gap followed by a spike, as in the sample data, often has one cause: rows
that arrived late and were stamped with the day they were loaded. Then the
right fix is a better date (the original order date from the source), not
deleting the spike.

## 9. Proposing changes to the user

Present each proposed step in this shape, and wait for an answer:

```markdown
**Proposed step 2 of 4: turn "-" in `rating` into missing values**
- What it does: 51 values (16.8% of rows) change from "-" to missing; the
  column can then be converted to numbers.
- Why: "-" is used for "no rating"; as text it blocks any average.
- Cost: `rating` will be missing in 16.8% of rows. Averages will cover the
  other 253 rows only.
- Alternative: keep "-" as its own category "not rated".
- Your call: apply / skip / use the alternative?
```

Group small, safe steps (trimming spaces) into one proposal; give every step
that removes rows or changes values its own.

## 10. Applying approved changes

Approved steps are applied by `scripts/clean_data.py`, not by ad-hoc pandas:
each step is tested, the input file is only read, and every change is
logged. Write the approved options into a plan, in the order of the ground
rules (types and placeholders, then duplicates, then missing values, then
outliers):

```json
{
  "steps": [
    {"step": "placeholders_to_missing", "columns": ["rating"], "values": ["-"],
     "fixes": "disguised_missing", "reason": "\"-\" means no rating (confirmed)"},
    {"step": "to_numeric", "columns": ["rating"], "fixes": "numbers_as_text"},
    {"step": "strip_whitespace", "columns": ["region"]},
    {"step": "normalize_labels", "column": "region", "mapping": {"north": "North"}},
    {"step": "drop_exact_duplicates", "fixes": "duplicate_rows"}
  ]
}
```

`fixes` (the finding rule) and `reason` (what the user agreed to) are
optional but go into the log. `--list-steps` prints every step with its
options. Then:

```bash
python <skill-dir>/scripts/clean_data.py sales.csv --plan plan.json            # dry run: writes nothing
python <skill-dir>/scripts/clean_data.py sales.csv --plan plan.json --apply    # after the user says yes again
```

The dry run shows, per step, the rows and missing cells before and after,
how many values change, examples (`row 19 rating: '-' -> missing`) and
warnings (values that won't parse, mapping keys that match nothing, the
mean and spread that capping changes). Show it to the user before
`--apply`. The script stops with a clear error, and writes nothing, when a
step names a column that doesn't exist at that point, the output is the
input file, or the output folder isn't empty.

`--apply` writes the cleaned file (CSV, or Parquet with `--output x.parquet`,
the default for Parquet input) with `cleaning_log.md` and
`cleaning_log.json` next to it: every step with its counts, the SHA-256 of
input and output, and the plan. Passing `cleaning_log.json` back as `--plan`
rebuilds the same file byte for byte. CSV columns of codes such as
`01234` are kept as text, so no leading zero is lost.

Then rerun the EDA on the cleaned file and compare the two runs:

```bash
python <skill-dir>/scripts/generate_eda.py <cleaned file> --output eda_output/sales_cleaned
python <skill-dir>/scripts/compare_runs.py eda_output/sales_before eda_output/sales_cleaned
```

Report the findings that went away, the new ones (usually the cost of a
step: placeholders turned into missing values raise a missing-value count)
and the ones that changed.

**A fix outside the fixed steps** (merging two columns, correcting one
known value) is done with pandas on a copy, written to a new file, and
reported to the user as not covered by `cleaning_log.md`:

```python
fixed = df.copy()
wrong = fixed["order_id"] == 100127
fixed.loc[wrong, "quantity"] = 3  # confirmed by the data owner for this one order
manual_log = [{"change": "order 100127: quantity set to 3", "rows": int(wrong.sum())}]
target = out_dir / "sales_fixed.csv"
assert target.resolve() != Path(path).resolve(), "never overwrite the input file"
fixed.to_csv(target, index=False)
```
