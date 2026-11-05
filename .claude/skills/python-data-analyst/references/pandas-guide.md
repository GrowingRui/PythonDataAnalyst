# pandas guide

Read this when a file won't load cleanly, the data is large, or you need pandas
code beyond what `inspect_data.py` and `generate_eda.py` produce. Written for
pandas 3.x.

**Contents**
1. [Reading files](#1-reading-files)
2. [First inspection](#2-first-inspection)
3. [Selecting and filtering](#3-selecting-and-filtering)
4. [Dates and time](#4-dates-and-time)
5. [Grouping, pivoting, cross-tabs](#5-grouping-pivoting-cross-tabs)
6. [Large data and memory](#6-large-data-and-memory)
7. [Pitfalls that silently change results](#7-pitfalls-that-silently-change-results)
8. [Saving results](#8-saving-results)

The examples use a sales file with the columns `order_id`, `order_date` (text,
day/month/year), `region`, `customer_id` (codes with leading zeros),
`quantity`, `unit_price`, `revenue`, `discount` (has gaps) and `rating`
(uses `-` for "no rating"). `path` is the CSV file, `xlsx_path` an Excel
workbook, `out_dir` a folder for results.

---

## 1. Reading files

```python
import numpy as np
import pandas as pd

df = pd.read_csv(path)
```

`read_csv` guesses a lot. Override the guess whenever it's wrong:

| Problem | Symptom | Fix |
|---|---|---|
| Wrong delimiter | One single column, or `ParserError: Expected N fields` | `sep=";"`, or `sep=None, engine="python"` to sniff |
| Wrong encoding | `UnicodeDecodeError`, or `Ã©` instead of `é` | `encoding="cp1252"` (Windows Excel) or `"latin-1"`; `"utf-8-sig"` strips a BOM |
| European numbers | `1.234,50` read as text | `decimal=",", thousands="."` |
| Codes lose leading zeros | `00123` becomes `123` | `dtype={"customer_id": "str"}` |
| Placeholders for missing | `-`, `n/a`, `unknown` counted as values | `na_values=["-", "unknown"]` |
| Real values read as missing | Country code `NA` (Namibia) becomes NaN | `keep_default_na=False, na_values=[""]` |
| Header not on line 1 | Column names like `Unnamed: 1` | `skiprows=3` or `header=2` |
| Dates stay text | dtype `str` for a date column | parse after loading, see [section 4](#4-dates-and-time) |

```python
# Keep codes as text and treat "-" as missing.
df = pd.read_csv(path, dtype={"customer_id": "str"}, na_values=["-"])

# Let pandas sniff the delimiter (slower, but robust for unknown files).
sniffed = pd.read_csv(path, sep=None, engine="python")

# A European export: ';' between fields, ',' as decimal mark, '.' for thousands.
from io import StringIO

euro = pd.read_csv(StringIO("item;price\nTea;1.234,50\n"), sep=";", decimal=",", thousands=".")
assert euro.loc[0, "price"] == 1234.5
```

**Excel.** `sheet_name=None` reads every sheet into a dict, which is the
quickest way to see what a workbook contains. `.xls` (the old format) needs
the `xlrd` package; the skill asks the user to re-save as `.xlsx` instead.

```python
with pd.ExcelFile(xlsx_path) as book:  # "with" closes the file again
    sheets = book.sheet_names  # names only, without reading the data
first = pd.read_excel(xlsx_path)  # the first sheet
all_sheets = pd.read_excel(xlsx_path, sheet_name=None)  # {"name": DataFrame, ...}
one = pd.read_excel(xlsx_path, sheet_name=sheets[-1], skiprows=0)
```

Excel pitfalls: merged header cells become `Unnamed: n` columns; totals rows
at the bottom get analysed as data; dates may arrive as numbers (days since
1899-12-30): `pd.to_datetime(s, unit="D", origin="1899-12-30")`.

## 2. First inspection

```python
df.shape  # (rows, columns)
df.dtypes  # one dtype per column: int64, float64, str, bool, datetime64[ns], ...
df.head()
df.describe()  # numeric columns: count, mean, std, min, quartiles, max
df.describe(include="all")  # adds unique / top / freq for text columns
df.isna().sum()  # missing values per column
df.isna().mean().mul(100).round(1)  # ... as percentages
df.nunique()  # distinct values per column
df["region"].value_counts(dropna=False)  # dropna=False also counts NaN
df.duplicated().sum()  # exact duplicate rows
df.duplicated(subset=["order_id"]).sum()  # repeated keys
```

`df.info()` prints dtypes, non-null counts and memory in one go.

Reading dtypes: `str` is text (pandas 3 default), `object` means *mixed*
Python objects (often a sign of a problem), `float64` for a column of whole
numbers usually means it has missing values (see [pitfalls](#7-pitfalls-that-silently-change-results)).

## 3. Selecting and filtering

```python
df.loc[df["region"] == "North", ["order_id", "revenue"]]  # rows by condition, columns by name
df.iloc[:5, :3]  # by position

# Combine conditions with & | ~ and put each one in parentheses.
big_north = df[(df["revenue"] > 100) & (df["region"].isin(["North", "South"]))]
df.query("revenue > 100 and quantity >= 2")  # the same idea, as a string
df[df["unit_price"].between(10, 50)]  # inclusive on both ends

# Text matching: na=False treats missing values as "no match".
df[df["region"].str.contains("orth", case=False, na=False)]
```

`and` / `or` don't work on columns (`ValueError: The truth value of a Series
is ambiguous`); use `&` / `|`. Without parentheses, `&` binds before `>`, so
`df.a > 1 & df.b < 2` is evaluated wrongly.

## 4. Dates and time

Always pass an explicit `format` when you know it. Without one, `03/04/2025`
is ambiguous (3 April or March 4) and pandas guesses from the first value.

```python
dated = df.assign(order_date=pd.to_datetime(df["order_date"], format="%d/%m/%Y", errors="coerce"))

# errors="coerce" turns unparseable values into NaT. Always look at what failed:
failed = df.loc[dated["order_date"].isna() & df["order_date"].notna(), "order_date"]

dated["month"] = dated["order_date"].dt.to_period("M")
dated["weekday"] = dated["order_date"].dt.day_name()

# Totals per calendar month ("MS" = month start labels).
monthly = dated.set_index("order_date").resample("MS")["revenue"].sum()
```

Common formats: `%Y-%m-%d` (2025-03-04), `%d/%m/%Y` (04/03/2025, Europe),
`%m/%d/%Y` (03/04/2025, US), `%d.%m.%Y` (Germany), `%Y-%m-%d %H:%M:%S`. If
values mix formats, use `format="mixed", dayfirst=True` and check the result.

Time series operations (`diff`, `cumsum`, `shift`, `rolling`, `ffill`)
follow **row order**. Sort by date first: `dated.sort_values("order_date")`.

## 5. Grouping, pivoting, cross-tabs

```python
by_region = (
    dated.groupby("region", dropna=False)  # dropna=False keeps a group for missing regions
    .agg(
        orders=("order_id", "count"),
        revenue_total=("revenue", "sum"),
        revenue_median=("revenue", "median"),
        avg_quantity=("quantity", "mean"),
    )
    .sort_values("revenue_total", ascending=False)
)

# transform returns one value per row: here each order's share of its region's revenue.
share = dated["revenue"] / dated.groupby("region")["revenue"].transform("sum")

# Rows x columns table of a numeric value.
pivot = dated.pivot_table(index="region", columns="month", values="revenue", aggfunc="sum", fill_value=0)

# Counts (or shares) of two categorical columns together.
pd.crosstab(dated["region"], dated["rating"], normalize="index")  # each row sums to 1
```

Notes:
- `count` counts non-missing values; `size` counts rows. They differ when the
  column has gaps.
- `groupby` drops rows whose key is missing unless `dropna=False`.
- On `category` columns pass `observed=True`, otherwise every unused category
  appears as an empty group.
- Report the group size next to every group statistic. A median of 3 rows
  isn't comparable to a median of 3,000.

## 6. Large data and memory

```python
df.memory_usage(deep=True).sum() / 1e6  # MB; deep=True measures text properly

# Read only what you need.
sample = pd.read_csv(path, usecols=["order_date", "region", "revenue"], nrows=1_000)

# Repetitive text as category: stores each label once.
compact = df.assign(region=df["region"].astype("category"))

# Smaller numeric types.
compact["quantity"] = pd.to_numeric(compact["quantity"], downcast="integer")

# Files too big for memory: aggregate chunk by chunk.
totals = None
for chunk in pd.read_csv(path, chunksize=100_000, usecols=["region", "revenue"]):
    part = chunk.groupby("region")["revenue"].sum()
    totals = part if totals is None else totals.add(part, fill_value=0)
```

Rules of thumb: pandas needs roughly 2–10× the file size in RAM. Medians and
distinct counts can't be combined across chunks the way sums can; for those,
work on a random sample and say so.

## 7. Pitfalls that silently change results

None of these raise an error. Each one can still change a number in a report.

**Missing values are skipped without warning.** `mean()`, `sum()` and
`corr()` ignore NaN, so a mean over 40% missing data looks just as confident
as one over complete data. An all-missing column sums to `0`, not missing:

```python
assert pd.Series([np.nan]).sum() == 0
assert np.isnan(pd.Series([np.nan]).sum(min_count=1))  # min_count=1: "no values" stays missing
```

**Whole numbers with gaps become floats.** `[1, 2, None]` is `float64`, so IDs
print as `1.0`. Use the nullable type: `astype("Int64")` (capital I).

```python
assert pd.Series([1, 2, None]).dtype == "float64"
assert pd.Series([1, 2, None], dtype="Int64").dtype == "Int64"
```

**`value_counts()` and `groupby()` drop NaN** unless you pass `dropna=False`.
Percentages are then shares of the non-missing rows only.

**Chained assignment never works (pandas 3).** `df[df.x > 0]["y"] = 1`
changes a temporary copy and leaves `df` unchanged (pandas warns with
`ChainedAssignmentError`). Use one `.loc` call:

```python
fixed = df.copy()
fixed.loc[fixed["discount"].isna(), "discount"] = 0.0
```

**Merges can multiply rows.** If the right table repeats a key, every
matching left row is duplicated and sums go up. Ask pandas to check:

```python
regions = pd.DataFrame({"region": ["North", "South", "East"], "manager": ["Ana", "Ben", "Cai"]})
merged = df.merge(regions, on="region", how="left", validate="many_to_one")  # raises if regions repeats a key
assert len(merged) == len(df)
```

**Text matching on missing values.** With the pandas 3 `str` dtype,
`.str.contains()` returns `False` for missing values; on `object` columns it
returns NaN, which breaks filtering. Passing `na=False` is correct in both.

**Floats aren't exact.** `0.1 + 0.2 == 0.3` is `False`. Compare with
`np.isclose(a, b)`.

**Mixed types hide in `object` columns.** A column of numbers with one
`"n/a"` stays text, so `df["col"].mean()` fails or `sort_values` orders
`"10"` before `"9"`. See what's inside:
`df["col"].map(type).value_counts()`.

**Sorting text sorts by characters.** `"10" < "9"` and `"Zürich"` after
`"Zug"`. Convert numbers and dates before sorting.

## 8. Saving results

Never write over the input file. Write a new file next to the analysis output
and say which one it is.

```python
target = out_dir / "sales_cleaned.csv"
assert not target.exists(), "choose a new file name instead of overwriting"
df.to_csv(target, index=False)  # index=False: don't add an unnamed 0, 1, 2, ... column
```

For Excel output use `df.to_excel(out_dir / "result.xlsx", index=False)`
(needs `openpyxl`). CSV loses dtypes: dates come back as text, codes may lose
leading zeros. Mention this to the user when they'll reload the file.
