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
8. [Proposing changes to the user](#8-proposing-changes-to-the-user)
9. [Applying approved changes](#9-applying-approved-changes)

Each section names the `rule` values it covers in `summary.json` → `findings`,
so you can go from a finding straight to its options. The examples use the
same sales data as `pandas-guide.md` (`path`, `out_dir`).

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

Finding rule: `disguised_missing`.

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

```python
codes = clean["quantity"].isin([999, -1]).sum()  # how many rows use the code
clean["quantity"] = clean["quantity"].mask(clean["quantity"].isin([999, -1]))  # code -> missing, once confirmed
```

## 4. Duplicates

Finding rules: `duplicate_rows`, `duplicate_keys`.

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

Finding rule: `numbers_as_text`.

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

Finding rule: `dates_as_text`.

```python
parsed = pd.to_datetime(clean["order_date"], format="%d/%m/%Y", errors="coerce")
failed = clean.loc[parsed.isna() & clean["order_date"].notna(), "order_date"].unique()
clean["order_date"] = parsed
```

Cost: unparseable values become missing. The real risk is the day/month
order: with `03/04/2025` both readings are valid and the wrong one silently
shifts dates. `inspect_data.py` reports `date_order` when the values settle
it (a day above 12 appears); otherwise ask the user.

### Mixed types

Finding rule: `mixed_types`.

```python
type_counts = df["rating"].map(type).value_counts()  # e.g. str 280, float 24 (the NaNs)
```

Usually a numeric column with a few notes ("approx. 5", "see email"). Decide
per value: convert, set missing, or move the note to its own column. Cost:
the odd values change meaning or disappear.

### Inconsistent labels and stray spaces

Finding rules: `inconsistent_labels`, `untrimmed_whitespace`.

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

Finding rule: `outliers`.

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

Finding rules: `skewed`, `constant_column`, `dominant_category`.

**Skewed columns** don't need cleaning. They need the right summary: median
and IQR instead of mean and standard deviation. For charts, a log scale; for
models, sometimes a log transform:

```python
log_revenue = np.log1p(df["revenue"])  # log(1 + x): works with zeros, not with negatives
```

Cost of a log transform: differences become ratios (a coefficient means "10%
more", not "10 more"), and results must be transformed back to explain them.

**Constant columns** (one value) carry no information within this file. Drop
them from the analysis, but ask why they're constant: a filter applied during
export (`country = DE` only) limits what the whole analysis can claim.

**Nearly constant columns** (one value ≥ 95%) are kept. Warn that statistics
for the rare groups rest on very few rows.

**Near-duplicate columns** (finding rule `near_duplicate_columns`, |r| ≥ 0.98)
are usually one column derived from another (a total and its parts, the same
measure in two units). Not a data error; keep both in the file, but use only
one in a model.

## 8. Proposing changes to the user

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

## 9. Applying approved changes

Apply only the approved steps, in the order of the ground rules, and keep a
log with counts:

```python
log = []


def step(description: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
    log.append({
        "step": description,
        "rows_before": len(before),
        "rows_after": len(after),
        "missing_before": int(before.isna().sum().sum()),
        "missing_after": int(after.isna().sum().sum()),
    })


result = df.copy()

new = result.assign(region=result["region"].str.strip().str.title())
step("Trim and title-case region", result, new)
result = new

new = result.drop_duplicates()
step("Drop exact duplicate rows", result, new)
result = new

target = out_dir / "sales_cleaned.csv"
assert target.resolve() != Path(path).resolve(), "never overwrite the input file"
result.to_csv(target, index=False)
pd.DataFrame(log).to_csv(out_dir / "cleaning_log.csv", index=False)
```

Then rerun `generate_eda.py` on the cleaned file and compare the key numbers
with the original run. Name any finding that changed.
