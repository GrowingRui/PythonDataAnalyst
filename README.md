# Python Data Analyst: a Claude Code skill for exploratory data analysis

[![tests](https://github.com/GrowingRui/PythonDataAnalyst/actions/workflows/tests.yml/badge.svg)](https://github.com/GrowingRui/PythonDataAnalyst/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

Ask Claude Code *"What's wrong with the data in sales.csv?"* and get a careful
analyst's answer, backed by tested code: data quality problems, descriptive
statistics, correlations, charts and a written report. Your file is never
changed.

The skill works on CSV, TSV and Excel files with pandas, NumPy, Matplotlib and
Seaborn. Claude runs two scripts that compute every number, then reads the
results and the charts and writes the interpretation.

<p>
  <img src="docs/images/missing_values.png" alt="Bar chart of missing values per column" width="49%">
  <img src="docs/images/revenue_log.png" alt="Histogram of revenue on a log scale with median and mean marked" width="49%">
</p>

## Contents

- [What it does](#what-it-does)
- [Install](#install)
- [Use it](#use-it)
- [What you get](#what-you-get)
- [Try it on the sample data](#try-it-on-the-sample-data)
- [How it works](#how-it-works)
- [Methods and limitations](#methods-and-limitations)
- [Development](#development)
- [License](#license)

## What it does

- **Loads messy files robustly:** detects the delimiter, falls back from
  UTF-8 to Windows-1252, picks Excel sheets and explains clearly when a file
  can't be read.
- **Checks data quality:**
  - missing values, including placeholders such as `-` or `unknown`
  - exact duplicates and repeated keys
  - numbers or dates stored as text, mixed types
  - inconsistent labels (`North` / `north `)
  - IQR outliers, and numeric codes such as `999`
- **Describes the data:** statistics per column type, skewness, top
  categories.
- **Finds relationships:** Pearson and Spearman correlations, with a warning
  when they disagree.
- **Draws charts:** histograms with box plots, log-scale histograms for skewed
  data, top categories, a missing-values chart, a correlation heatmap and
  scatter plots of the strongest pairs. They use one consistent,
  colourblind-checked palette.
- **Writes a report:** `report.md` with ranked findings, cleaning options and
  what each one costs, suggested next steps and limitations. `summary.json`
  holds the same facts in machine-readable form.
- **Never changes your data.** Problems are flagged, not fixed. Claude
  proposes cleaning steps with their consequences and waits for your
  approval; approved steps write a new file.

## Install

You need [Claude Code](https://claude.com/claude-code), Python 3.11 or newer,
and the packages in `requirements.txt` (pandas, NumPy, Matplotlib, Seaborn,
openpyxl).

**1. Get the code.**

```bash
git clone https://github.com/GrowingRui/PythonDataAnalyst.git
```

**2. Install the skill** in one of two places:

| Where | Folder | Available in |
|---|---|---|
| A project | `<your-project>/.claude/skills/python-data-analyst/` | That project only |
| Your user profile | `~/.claude/skills/python-data-analyst/` | Every project |

Install it per project if you only want it where you work with data. Until
it's used, a skill costs Claude only its one-line description.

```bash
# macOS / Linux: install into a project
mkdir -p /path/to/your-project/.claude/skills
cp -r PythonDataAnalyst/.claude/skills/python-data-analyst /path/to/your-project/.claude/skills/
```

```powershell
# Windows PowerShell: install for your user (all projects)
New-Item -ItemType Directory -Force "$HOME\.claude\skills" | Out-Null
Copy-Item -Recurse PythonDataAnalyst\.claude\skills\python-data-analyst "$HOME\.claude\skills\"
```

**3. Install the Python packages** in the environment Claude will use. A
project virtual environment (`.venv`) is picked up automatically.

```bash
pip install -r PythonDataAnalyst/requirements.txt
```

If a package is missing, the skill tells Claude to ask you before installing
anything.

## Use it

Start a new Claude Code session in your project and ask in plain language:

- *"Explore `data/sales.csv` and tell me what stands out."*
- *"Check `customers.xlsx` (sheet `2025`) for data quality problems."*
- *"Which numeric columns in `survey.csv` have outliers, and should I remove them?"*

Or call the skill by name, with an optional focus:

```text
/python-data-analyst data/sales.csv focus on regional revenue differences
```

You can also run the scripts yourself, without Claude:

```bash
python .claude/skills/python-data-analyst/scripts/inspect_data.py data/sales.csv         # quick text summary
python .claude/skills/python-data-analyst/scripts/inspect_data.py data/sales.csv --json  # the same as JSON
python .claude/skills/python-data-analyst/scripts/generate_eda.py data/sales.csv         # full report + charts
```

| Option | Script | Purpose |
|---|---|---|
| `--sheet NAME` | both | Excel sheet to read (default: the first) |
| `--encoding ENC` | both | Force a text encoding, e.g. `latin-1` |
| `--delimiter ";"` | both | Force a delimiter when detection guesses wrong |
| `--key COLS` | both | Columns that should be unique, e.g. `order_id`; also checks for repeated keys |
| `--top N` | inspect | Top values shown per categorical column |
| `--output DIR` | EDA | Output folder (default `eda_output/<file>_<timestamp>/`, must be new or empty) |
| `--max-categories N` | EDA | Top-N categories per chart; the rest become "Other" |
| `--max-scatter-points N` | EDA | Random sample size for scatter plots on large data |
| `--max-charts N` | EDA | Cap on the number of charts (default 40) |

## What you get

```text
eda_output/sales_20261105-092532/
├── report.md       # the report: read this first
├── summary.json    # every computed fact, plus the ranked findings
└── charts/
    ├── 01_missing_values.png
    ├── 02_correlation_heatmap.png
    ├── 03_scatter_unit_price__revenue.png
    └── ...
```

`report.md` always has the same seven sections:

1. **Dataset overview**: file, size, column types
2. **Data quality assessment**: missing values, duplicates, type problems, outliers
3. **Descriptive statistics**: numeric, true/false, date and categorical columns
4. **Key findings**: ranked rule-based findings, then Claude's *Analyst interpretation*
5. **Visualization summaries**: one or two sentences per chart, with the chart
6. **Suggested next steps**: cleaning options with their costs, further analysis
7. **Limitations**: what the methods used can't tell you

An excerpt of the findings for the sample data:

> 1. **[high] 15 rows repeat a key on `order_id` (1.2%).** These rows share a
>    key that should be unique; where they differ in other columns it isn't
>    clear which version is correct.
> 2. **[high] 1 column is completely empty: `legacy_code`.**
> 3. **[medium] `order_date` holds dates stored as text.** 1215 of 1215 text
>    values (100.0%) parse as dates; values look day-first.

The input file's SHA-256 fingerprint is recorded before the run and checked
after it, so the report itself shows that the data wasn't modified.

## Try it on the sample data

`sample_data/sample_sales.csv` is a synthetic, deliberately messy dataset of
1,215 orders. The problems planted in it are listed at the top of
[`sample_data/make_sample_data.py`](sample_data/make_sample_data.py), so you
can check what the skill finds against the answer key.

```bash
python .claude/skills/python-data-analyst/scripts/generate_eda.py sample_data/sample_sales.csv --key order_id
```

Or, in Claude Code inside this repository: *"What's wrong with the data in
sample_data/sample_sales.csv?"*

## How it works

```text
.claude/skills/python-data-analyst/
├── SKILL.md                    # instructions Claude follows: workflow and rules
├── scripts/
│   ├── inspect_data.py         # loading, column types, statistics, quality checks
│   ├── generate_eda.py         # correlations, categories, charts; writes the output folder
│   └── report.py               # rule-based findings and report.md
└── references/                 # loaded only when needed
    ├── pandas-guide.md         # reading tricky files, groupby, memory, pandas pitfalls
    ├── data-cleaning.md        # options per problem and what each one costs
    └── visualization-guide.md  # which chart answers which question, misleading charts
```

The design follows a few rules:

- **The scripts compute the facts and Claude interprets them.** Every number
  comes from tested code, never from Claude estimating. The interpretation is
  written for your goal and dataset.
- **Read-only by default.** Removing outliers or filling gaps changes the
  results, so every cleaning step needs an explanation and your agreement.
- **Progressive disclosure.** Claude reads `SKILL.md` when the skill is used,
  and the reference guides only when a question calls for them.
- **Few dependencies.** Only pandas, NumPy, Matplotlib, Seaborn and openpyxl.

## Methods and limitations

| Method | What it tells you | What it can't |
|---|---|---|
| IQR outliers (Q1 − 1.5·IQR, Q3 + 1.5·IQR) | Which values are unusual | Whether they're wrong. On skewed data many legitimate values are flagged |
| Pearson correlation | Straight-line relationships | Curved relationships; very sensitive to a few extreme points |
| Spearman correlation | Any steadily rising or falling relationship | U-shaped relationships |
| Skewness (adjusted Fisher–Pearson) | Whether a few large or small values pull the mean | The cause |
| Column type detection | numeric / boolean / datetime / categorical / free text, by rules | It can misclassify; the report shows the result for every column |
| Placeholder detection | A fixed list (`-`, `?`, `n/a`, `unknown`, ...) and numeric codes like `999` that are outliers | Codes inside the normal range (`0` meaning "no answer") |

Correlations are computed between numeric columns only, and none of them
shows cause and effect. Severity labels on findings mean *how much this could
distort an analysis if ignored*, not *how much it matters for your question*.
That judgement is the Analyst interpretation's job.

## Development

```bash
git clone https://github.com/GrowingRui/PythonDataAnalyst.git
cd PythonDataAnalyst
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest
```

The test suite covers:
- loading and quality checks against known answers,
- the charts and the report,
- every Python example in the reference guides,
- an end-to-end run on the sample data that must find every planted problem
  and leave the input file byte-for-byte unchanged.

On every push, GitHub Actions runs the tests:
- on Python 3.11 and 3.12 (Linux) and 3.12 (Windows),
- and with the oldest supported versions (pandas 2.1, NumPy 1.26,
  Matplotlib 3.8).

To change the sample data, edit `sample_data/make_sample_data.py`, run it,
and commit the regenerated CSV. A test checks that the two match.

## License

[MIT](LICENSE) © 2026 GrowingRui
