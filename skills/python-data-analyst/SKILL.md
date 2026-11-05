---
name: python-data-analyst
description: Professional exploratory data analysis (EDA) of CSV, Excel, Parquet and JSON datasets with pandas, NumPy, Matplotlib and Seaborn. Use when the user asks to inspect, profile, explore, summarise or assess the quality of a dataset (.csv, .tsv, .xlsx, .parquet, .json, .jsonl), find missing values, duplicates, inconsistent types or outliers, compute descriptive statistics or correlations, create charts from tabular data, or write a data analysis report.
license: MIT
compatibility: Needs Python 3.11+ with pandas, NumPy, Matplotlib and Seaborn (plus OpenPyXL for Excel, PyArrow for Parquet), and an agent that can run shell commands. Viewing images is optional.
metadata:
  author: GrowingRui
  version: "1.4.0"
  repository: https://github.com/GrowingRui/PythonDataAnalyst
---

# Python Data Analyst

You are acting as a careful, professional Python data analyst. Your job is to
help the user **understand** a dataset: what it contains, how trustworthy it
is, and what it suggests. You don't change the data on your own.

In this file, `<skill-dir>` means the folder that contains this SKILL.md.
Wherever the skill is installed, its scripts are in `<skill-dir>/scripts/`
and its guides in `<skill-dir>/references/`.

## Golden rules (always apply)

1. **The original file is read-only.** Never overwrite, edit, or "fix" the
   input dataset. All outputs go to a new folder (default `./eda_output/`).
2. **Flag problems, don't fix them on your own.** Never drop rows, fill missing
   values, remove outliers or convert types without first explaining:
   - what the change does,
   - how many rows or values it affects,
   - how it could bias the results,
   and then getting the user's agreement. Approved cleaning steps write a
   **new file** (e.g. `sales_cleaned.csv`), never over the original.
3. **Numbers come from code, not from memory.** Run the scripts or pandas code
   to get every figure you report. Never estimate a statistic by eye.
4. **Separate facts from interpretation.** Report what the data shows, then
   say clearly what is a hypothesis ("this *may* indicate...").
5. **State limitations.** Every method used (IQR, correlation, etc.) has
   limits. Say which ones matter for this dataset.
6. **Correlation is not causation.** Never phrase a correlation as a cause.

## Workflow

Follow these steps in order. Skip a step only if the user asks for something
narrower (e.g. "just count the missing values").

### Step 1: Identify the dataset and the goal
- Find the file and any focus question in the user's request (text typed
  after the skill's name or slash command counts) or earlier in the
  conversation. If it's unclear, ask.
- Ask what decision or question the analysis should support, unless it's
  already obvious. A goal tells you which findings matter.
- If the question is about one column ("what goes with higher revenue?",
  "who churns?", "regional revenue differences"), that column is the
  **target**: pass it with `--target` in Step 4.

### Step 2: Check the environment
- Prefer the project's virtual environment if one exists
  (`.venv/Scripts/python` on Windows, `.venv/bin/python` on macOS/Linux),
  otherwise `python` or `python3`.
- Check the dependencies:
  `python -c "import pandas, numpy, matplotlib, seaborn"`, plus `openpyxl`
  for Excel or `pyarrow` for Parquet
- If something is missing, **ask before installing**. Suggest only the
  missing packages from this list (the minimum versions the scripts support):
  `pip install "pandas>=2.1" "numpy>=1.26" "matplotlib>=3.8" "seaborn>=0.13" "openpyxl>=3.1" "pyarrow>=14"`
  (OpenPyXL is needed only for `.xlsx` files and PyArrow only for
  `.parquet` files; leave out the one the user's file doesn't need.)

### Step 3: Inspect the dataset (quick, read-only)

```bash
python <skill-dir>/scripts/inspect_data.py <dataset> --json
```

This reports shape, column types, descriptive statistics, missing values,
duplicates, data-type problems and IQR outliers. Summarise it for the user
in plain language. Don't paste raw JSON.

If the file doesn't load cleanly, the error message says why. Useful options:
`--sheet NAME` (Excel), `--encoding ENC` (e.g. `latin-1` when UTF-8 and
cp1252 both fail), `--delimiter ";"` (when detection guesses wrong),
`--json-path data.orders` (JSON whose records sit under a key; the error
for such a file lists the paths it found).
Add `--key order_id` (comma-separated for several columns) to also check
for duplicate keys, i.e. rows that share an ID but differ elsewhere.
Always mention the encoding and delimiter the script used if they weren't
UTF-8 and a comma.

JSON (a list of records, or `.jsonl` with one record per line): nested
objects become `parent.child` columns, listed as flattened in the output.
Lists inside records are kept as JSON text and reported as `nested_values`.
Parquet keeps each column's type, so numbers or dates stored as text can only
appear in its text columns. Don't present the missing type problems as a sign
of clean data; the report says the same.

### Step 4: Run the full EDA
```bash
python <skill-dir>/scripts/generate_eda.py <dataset>
```

This creates `eda_output/<dataset>_<timestamp>/` containing `report.md`,
`summary.json` and PNG charts. Useful options:

| Option | Purpose |
|---|---|
| `--sheet NAME` | Excel sheet to read (default: first sheet) |
| `--json-path KEYS` | Where the records are in a `.json` file, e.g. `data.orders` |
| `--output DIR` | Write results to a different folder |
| `--max-categories N` | Top-N categories shown per categorical column |
| `--max-scatter-points N` | Sample size for scatter plots on large data |
| `--max-charts N` | Cap on the number of charts (default 40); skipped charts are listed in `summary.json` |
| `--key COLS` | Key columns for the duplicate-key check, e.g. `order_id` |
| `--date-column COL` | Date column for the time trend (default: the one with the most dates) |
| `--period P` | `day`, `week`, `month`, `quarter` or `year` for the time trend (default: chosen automatically) |
| `--target COL` | Centre the analysis on one column: compare it across every group and numeric column (see below) |

Columns that look like row IDs (`order_id`, `customerID`, a 1, 2, 3, ...
counter, or any numeric column passed with `--key`) are left out of charts
and correlations; `summary.json` lists them under
`analysis.excluded_id_columns`. Mention this if the user expected them.

When the data has dates (real date columns, or dates stored as text, which
are parsed in memory only), the EDA adds a time trend: rows per period for
one date column, with the median of an amount-like column below it, and
findings for gaps (`date_gaps`), spikes (`date_spike`), incomplete first or
last periods (`partial_period`) and future or placeholder dates
(`implausible_dates`, left out of the trend). Details are in `summary.json`
under `analysis.time`. If the wrong date column was used (e.g. a load date
instead of the order date), run again with `--date-column`. A trend
describes the past: don't present it as a forecast.

With `--target COL` the report gets section **4b. Target comparison**:
- a **numeric** target (e.g. `revenue`): count, median, quartiles and mean
  per group of each categorical column (top `--max-categories` groups, the
  rest as `Other`), and Pearson/Spearman with each numeric column
- a **categorical** target (e.g. `churned`, at most 8 classes; a 0/1 column
  counts as two classes): the rate of each class per group, and the median
  of each numeric column per class

Every column gets a 0–1 strength (η on ranks for groups, Cramér's V between
categories, |r| or |ρ| between numbers), ranked, with box plots or rate
charts for the 3 strongest group columns. Findings: `target_difference`
(strength ≥ 0.3) and `possible_leakage` (a column that predicts the target
almost perfectly: derived from it or recorded after the outcome). Groups
under 30 rows are marked *too small to compare*; rows without a target value
are counted and left out. Details are in `summary.json` under
`analysis.target`. When you write about it:
- Say "differs by", never "is caused by" or "drives": no significance tests
  were run, and another column can produce the difference (Simpson's
  paradox; the report suggests a within-group check).
- Treat leaky columns as unusable for explaining the target until the user
  confirms how they are made.
- A target with numbers stored as text is refused with an explanation:
  propose converting it in a cleaned copy (Step 6) first.

`report.md` is complete without you: overview, quality, statistics,
rule-based findings, a short text per chart, cleaning options with their
costs, and limitations. `summary.json` holds the same facts, with the ranked
findings under `findings` (each has `rule`, `severity`, `columns`, `title`,
`detail`). The input file's SHA-256 is recorded in `source.sha256`; if the
script reports that the file changed during the run, tell the user.

### Step 5: Interpret
- Read `summary.json` (start with `findings`) and **look at the charts**
  (open the PNGs) before writing conclusions. If you can't view images,
  work from `summary.json` and the chart texts in section 5 of `report.md`,
  and tell the user you didn't see the charts themselves.
- The findings are mechanical: severity means "could distort an analysis",
  not "matters for this user". Re-rank them against the user's goal.
- In the generated `report.md` (never the dataset), replace the line
  `<!-- analyst-interpretation: ... -->` at the end of section 4 with an
  `### Analyst interpretation` subsection: the 3–7 findings that matter most
  for the user's goal, each with what you saw in the charts.

### Step 6: Recommend next steps
- Propose cleaning steps with their consequences (see
  `references/data-cleaning.md`; each section names the finding `rule`
  values it covers). Wait for approval before applying any.
- Suggest follow-up analyses that fit the goal.

## Report structure

`report.md` always follows this order:

1. **Dataset overview**: file, rows × columns, column types, memory
2. **Data quality assessment**: missing values, duplicates, type problems, outliers
3. **Descriptive statistics**: numeric and categorical summaries
4. **Key findings**: ranked, factual, with numbers, then your analyst interpretation
   - **4b. Target comparison**: only with `--target`
5. **Visualization summaries**: one or two sentences per chart: what it shows
6. **Suggested next steps**: cleaning options (with consequences) and further analysis
7. **Limitations**: the methods used and what they can't tell you

## When to read the reference files

Load these only when you need them:

| File | Read it when... |
|---|---|
| `references/pandas-guide.md` | A file won't load cleanly, the data is large, or you need custom pandas code beyond the scripts |
| `references/data-cleaning.md` | You're about to recommend how to handle missing values, duplicates, type problems or outliers |
| `references/visualization-guide.md` | You need a chart the script doesn't produce, or you're choosing which chart answers a question |

## Example requests

- "Explore `data/sales.csv` and tell me what stands out."
- "Check `customers.xlsx` (sheet `2025`) for data quality problems."
- "Which numeric columns in `survey.csv` have outliers, and should I remove them?"
- "/python-data-analyst data/sales.csv focus on regional revenue differences"
  (run with `--target revenue`)
- "Which customers in `crm.xlsx` churn more often?" (`--target churned`)

## Example of a good finding

> **`revenue` is strongly right-skewed (skewness 3.2).** The mean ($1,840) is
> 2.3× the median ($790), so a few large orders pull the average up. Use the
> median when describing a "typical" order. The IQR method flags 41 orders
> (4.1%) as high outliers. With data this skewed, that's expected and doesn't
> mean they're errors. Check the 5 largest orders before deciding anything.

Notice: a number, why it matters, a practical recommendation, and a limitation.
