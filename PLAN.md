# Development Plan: Python Data Analyst Skill

- **Repository:** https://github.com/GrowingRui/PythonDataAnalyst
- **Author:** GrowingRui (xiarui091@gmail.com)
- **Status legend:** ⬜ not started · 🟡 in progress · ✅ done

We build this in small phases. Each phase ends with a review: you check the result, ask questions, and approve before the next phase starts.

---

## Approved decisions

| # | Decision | Why |
|---|---|---|
| D1 | The skill lives in `.claude/skills/python-data-analyst/`. Tests, sample data, and the README sit at the repo root, outside the skill folder. | Anyone who installs the skill gets only what Claude needs. Contributors still get the tests. |
| D2 | A local `.venv` with packages from `requirements.txt`. | Dependencies stay isolated and can be reproduced. |
| D3 | Reports go to `./eda_output/<dataset>_<timestamp>/`, and `--output` overrides that. | The original dataset is never touched, and every run gets its own folder. |
| D4 | Scripts output facts (JSON/Markdown), and Claude writes the conclusions. | The numbers come from tested code. The commentary fits each dataset. |
| D5 | Read-only by default. Problems are flagged, never fixed automatically. | Removing outliers or missing values changes the results, so each change needs an explanation and the user's consent. |
| D6 | No extra dependencies beyond pandas, NumPy, Matplotlib, Seaborn, OpenPyXL and pytest. For example, we write our own Markdown table function so we don't need `tabulate`. | Easier for other people to install. |
| D7 | Charts use one consistent, colorblind-checked palette: one color for single-series charts and a blue↔red diverging scale with a gray midpoint for correlations. | Charts stay readable and look consistent. |
| D8 | Personal use is optional. Users install the skill only in the projects where they want it. | You don't need it all the time. A skill costs only its one-line description until it's used. |

---

## Phase 0: Environment ✅

**Goal:** a working Python environment and a clean repo.

- ✅ Create `.venv` (Python 3.12) and install pandas 3.1, numpy 2.5, matplotlib 3.11, seaborn 0.13, openpyxl 3.1, pytest 9.2
- ✅ `requirements.txt` (runtime) and `requirements-dev.txt` (adds pytest), so users who only want the skill skip pytest
- ✅ `.gitignore` (`.venv/`, `eda_output/`, `__pycache__/`, `.pytest_cache/`)
- ✅ `git init` (branch `main`), local git user GrowingRui, remote `origin` → GrowingRui/PythonDataAnalyst. Nothing committed or pushed yet.

**Done when:** `python -c "import pandas, seaborn, openpyxl"` works inside the venv.

---

## Phase 1: Skill skeleton and SKILL.md ✅

**Goal:** Claude Code can discover the skill, even before any script exists.

- Create the folder structure (empty `references/` and `scripts/` placeholders).
- Write `SKILL.md`:
  - YAML frontmatter: `name`, `description`, `argument-hint`
  - The workflow: inspect → quality → EDA → report → next steps
  - Hard rules: read-only, never auto-clean, explain the consequences of each cleaning step
  - How to run each script, and what to do if a dependency is missing (ask before installing)

**You will learn:** how progressive disclosure works, and why the `description` decides when the skill triggers.

**Done when:** in a new Claude Code session, `/python-data-analyst` appears and Claude can describe the skill.

---

## Phase 2: `inspect_data.py`, loading and overview ✅

**Goal:** read any CSV or Excel file robustly and describe it.

- `load_dataset()`: CSV/TSV/XLSX, delimiter sniffing, an encoding fallback (UTF-8 → cp1252), Excel sheet selection
- Clear errors: file not found, empty file, unsupported format, wrong sheet name (lists the available sheets)
- `classify_columns()`: numeric / boolean / datetime / categorical / text
- `get_overview()`: rows, columns, dtypes, memory, unique counts
- `descriptive_stats()`: count, mean, std, quartiles, skewness; top values for categorical columns
- CLI: readable text output, or `--json`
- Tests: `tests/conftest.py` and `tests/test_inspect_data.py` (loading and overview)

**You will learn:** pandas I/O pitfalls, dtypes, mean vs. median, and what skewness means.

**Done when:** all tests pass and `python inspect_data.py some.csv` prints a summary.

---

## Phase 3: `inspect_data.py`, data quality ✅

**Goal:** detect problems without changing anything.

- `missing_values()`: per column and in total, plus disguised missing values (`"-"`, `"?"`, `"unknown"`, whitespace)
- `find_duplicates()`: exact duplicate rows, with an optional subset of key columns
- `detect_type_issues()`: numbers stored as text, dates stored as text, mixed Python types, inconsistent labels (`"North"` vs `"north "`)
- `iqr_outliers()`: Q1 − 1.5·IQR / Q3 + 1.5·IQR, with handling for small samples and zero IQR
- Tests that check against known answers (e.g., values 1–10 plus 1000 give exactly one outlier)

**You will learn:** missing-data types (MCAR/MAR/MNAR), the IQR method and when it misleads, exact vs. key-based duplicates.

**Done when:** every check finds the problems planted in the test data, and nothing more.

---

## Phase 4: `generate_eda.py`, analysis and charts ✅

**Goal:** go deeper than inspection and visualise the data.

- Reuses the functions from `inspect_data.py` (no duplicated code)
- `correlations()`: Pearson and Spearman, plus the strongest pairs
- `categorical_summary()`: top-N categories, with the rest folded into "Other"
- Charts (headless Matplotlib backend, saved as PNG):
  - histogram and box plot for each numeric column
  - bar chart of missing values per column
  - top-category bar charts
  - correlation heatmap
  - scatter plots for the 3 most correlated pairs
  - log-scale histogram for highly right-skewed, all-positive columns that span at least 20×
- Limits for large or wide data: `--max-categories`, `--max-scatter-points`, a cap on the number of charts
- Tests: chart files are created, and edge cases (no numeric columns, a single column) don't crash

**You will learn:** Pearson vs. Spearman, why correlation ≠ causation, choosing a chart for each question.

---

## Phase 5: Report generation ⬜

**Goal:** a structured Markdown report plus machine-readable JSON.

- `report.md`: overview → data quality → descriptive statistics → key findings → chart summaries → suggested next steps → limitations
- `summary.json`: all computed facts, for Claude to read
- Rule-based findings (high missing rates, duplicates, skewed or constant columns, strong correlations)
- Tests: every section exists, and the input file's hash is **unchanged** after a run

**Done when:** one command produces a readable report folder.

---

## Phase 6: Reference documentation ⬜

**Goal:** the knowledge Claude (and you) can look up when needed.

- `references/pandas-guide.md`: reading files, inspection, groupby/pivot, memory, pitfalls, with examples
- `references/data-cleaning.md`: options for each problem type and **what each one costs**
- `references/visualization-guide.md`: question → chart, readability rules, misleading-chart pitfalls
- Link each file from `SKILL.md` so Claude loads it only when relevant

---

## Phase 7: Sample data and end-to-end demo ⬜

**Goal:** prove the whole skill works in Claude Code.

- `sample_data/make_sample_data.py` builds a reproducible, deliberately messy `sample_sales.csv`
- Run both scripts on it and review the report and charts together
- Test the skill in Claude Code:
  - automatic trigger ("What's wrong with the data in sample_sales.csv?")
  - manual trigger (`/python-data-analyst sample_data/sample_sales.csv`)
  - negative check (unrelated questions should not trigger it)

---

## Phase 8: Publish to GitHub ⬜

**Goal:** a repo other people can use.

- `README.md`: what it does, install (project or personal), usage examples, methods and limitations, running the tests
- `LICENSE` (MIT, © 2026 GrowingRui)
- `.github/workflows/tests.yml`: run pytest on Python 3.11 and 3.12 for every push
- Final commit and push to `GrowingRui/PythonDataAnalyst`, **only after you confirm**

---

## Change log

- 2026-11-04: Plan created. Phase 0 done. Phase 1 built: SKILL.md (129 lines, 412-char description) and folder skeleton.
- 2026-11-05: Phase 1 approved. Phase 2 built: `inspect_data.py` (load, classify, overview, stats, CLI) and 37 passing tests. Additions beyond the plan: an `empty` column kind for all-missing columns, `--encoding`/`--delimiter`/`--top` options, and a `skew_label` rule of thumb. `.xls` is rejected with instructions to re-save as `.xlsx`.
- 2026-11-05: Phase 2 approved, committed and pushed. Phase 0-1 gaps filled: `.claude/settings.local.json` ignored, `pytest.ini` added, SKILL.md documents the inspect options.
- 2026-11-05: Phase 3 built: `missing_values`, `find_duplicates`, `detect_type_issues`, `iqr_outliers`, `data_quality`, `--key` option, 71 passing tests. Additions beyond the plan: `untrimmed_whitespace` as its own issue, day/month order detection for text dates, currency/%/decimal-comma handling for numbers-as-text, and sampling for date detection on columns with more than 5,000 distinct values.
- 2026-11-05: Phase 3 approved, committed and pushed. Phase 4 built: `generate_eda.py` (correlations, categorical summary, 5 chart types, `--max-categories`/`--max-scatter-points`/`--max-charts`, writes `summary.json`), 97 passing tests. Additions beyond the plan: likely-ID columns are excluded from charts and correlations, Pearson-vs-Spearman gap and low-n notes on each pair, the output folder must be new or empty (never overwrites). Histogram and box plot share one PNG per column.
- 2026-11-05: Phase 4 approved with one addition: a log-scale histogram for highly skewed positive columns. 99 passing tests. Committed and pushed.
