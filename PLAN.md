# Development Plan: Python Data Analyst Skill

- **Repository:** https://github.com/GrowingRui/PythonDataAnalyst
- **Author:** GrowingRui (xiarui091@gmail.com)
- **Status legend:** ⬜ not started · 🟡 in progress · ✅ done

We build this in small phases. Each phase ends with a review: you check the result, ask questions, and approve before the next phase starts.

---

## Approved decisions

| # | Decision | Why |
|---|---|---|
| D1 | ~~The skill lives in `.claude/skills/python-data-analyst/`.~~ Replaced by D9 (the folder is now `skills/python-data-analyst/`). Tests, sample data, and the README sit at the repo root, outside the skill folder. | Anyone who installs the skill gets only what Claude needs. Contributors still get the tests. |
| D2 | A local `.venv` with packages from `requirements.txt`. | Dependencies stay isolated and can be reproduced. |
| D3 | Reports go to `./eda_output/<dataset>_<timestamp>/`, and `--output` overrides that. | The original dataset is never touched, and every run gets its own folder. |
| D4 | Scripts output facts (JSON/Markdown), and Claude writes the conclusions. | The numbers come from tested code. The commentary fits each dataset. |
| D5 | Read-only by default. Problems are flagged, never fixed automatically. | Removing outliers or missing values changes the results, so each change needs an explanation and the user's consent. |
| D6 | No extra dependencies beyond pandas, NumPy, Matplotlib, Seaborn, OpenPyXL and pytest. For example, we write our own Markdown table function so we don't need `tabulate`. Extended in Phase 11: PyArrow as an **optional** dependency, needed only for Parquet (like OpenPyXL for Excel). | Easier for other people to install. |
| D7 | Charts use one consistent, colorblind-checked palette: one color for single-series charts and a blue↔red diverging scale with a gray midpoint for correlations. Extended in Phase 6: a fixed 8-color order for unordered groups and a light→dark blue scale for ordered groups (`references/visualization-guide.md`). | Charts stay readable and look consistent. |
| D8 | Personal use is optional. Users install the skill only in the projects where they want it. | You don't need it all the time. A skill costs only its one-line description until it's used. |
| D9 | The skill lives in a neutral `skills/python-data-analyst/` folder (moved from `.claude/skills/` in Phase 9). The repo root is also a Claude Code plugin and marketplace (`.claude-plugin/`). | One copy works in every agent; Claude Code users install it as a plugin, others copy the folder. |

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

## Phase 5: Report generation ✅

**Goal:** a structured Markdown report plus machine-readable JSON.

- `report.md`: overview → data quality → descriptive statistics → key findings → chart summaries → suggested next steps → limitations
- `summary.json`: all computed facts, for Claude to read
- Rule-based findings (high missing rates, duplicates, skewed or constant columns, strong correlations)
- Tests: every section exists, and the input file's hash is **unchanged** after a run

**Done when:** one command produces a readable report folder.

---

## Phase 6: Reference documentation ✅

**Goal:** the knowledge Claude (and you) can look up when needed.

- `references/pandas-guide.md`: reading files, inspection, groupby/pivot, memory, pitfalls, with examples
- `references/data-cleaning.md`: options for each problem type and **what each one costs**
- `references/visualization-guide.md`: question → chart, readability rules, misleading-chart pitfalls
- Link each file from `SKILL.md` so Claude loads it only when relevant

---

## Phase 7: Sample data and end-to-end demo ✅

**Goal:** prove the whole skill works in Claude Code.

- `sample_data/make_sample_data.py` builds a reproducible, deliberately messy `sample_sales.csv`
- Run both scripts on it and review the report and charts together
- Test the skill in Claude Code:
  - automatic trigger ("What's wrong with the data in sample_sales.csv?")
  - manual trigger (`/python-data-analyst sample_data/sample_sales.csv`)
  - negative check (unrelated questions should not trigger it)

---

## Phase 8: Publish to GitHub ✅

**Goal:** a repo other people can use.

- `README.md`: what it does, install (project or personal), usage examples, methods and limitations, running the tests
- `LICENSE` (MIT, © 2026 GrowingRui)
- `.github/workflows/tests.yml`: run pytest on Python 3.11 and 3.12 for every push
- Final commit and push to `GrowingRui/PythonDataAnalyst`, **only after you confirm**

---

## Phase 9: Works in any agent, not only Claude Code ✅

**Goal:** the skill installs and runs in any agent that supports the open Agent Skills format (agentskills.io), and still helps in agents that don't.

What already works everywhere: the scripts are plain Python command-line tools, and the folder layout (`SKILL.md` + `scripts/` + `references/`) is the standard one. What only works in Claude Code:

| Today | Problem in other agents | Change |
|---|---|---|
| `argument-hint` in the frontmatter | Not in the standard; strict validators may reject the file | Keep only standard fields (`name`, `description`, `license`, `compatibility`, `metadata`); add `compatibility: Python 3.11+ with pandas, NumPy, Matplotlib, Seaborn, OpenPyXL` |
| `$ARGUMENTS` in SKILL.md | Only Claude Code fills it in; other agents see the literal text | Phrase step 1 as "the file and question from the user's request"; mention `$ARGUMENTS` only as a Claude Code extra |
| `<skill-dir>` placeholder | Fine, but each agent finds its skill folder differently | Say "the folder that contains this SKILL.md" and give relative paths |
| "Look at the charts (open the PNGs)" | Some agents can't view images | Add a fallback: rely on `summary.json` and the chart texts in `report.md`, and say the charts weren't viewed |
| "Claude" in `report.py` and the docs | Wrong name in other agents | Neutral wording: "the agent" / "the analyst" |
| Skill only in `.claude/skills/` | Other agents look in other folders (e.g. `.agents/skills/`) | See decision D9 below |
| README install steps are Claude Code only | Other users don't know where to put it | An install table per agent, plus one generic "copy this folder" rule |

Also:
- `AGENTS.md` at the repo root for agents without skill support: a short pointer that tells them to read `SKILL.md` and follow it.
- A test that checks `SKILL.md` against the standard (allowed fields, `name` matches the folder, description length), and the official validator in CI if it installs cleanly.
- Hands-on trial in at least one other agent (same three checks as Phase 7: automatic trigger, manual request, negative check).

**Decisions (approved 2026-11-05):**
- **D9:** move to `skills/python-data-analyst/`, and also make the repo a Claude Code plugin, like the frontend-design skill.
- **Agents to test:** only Claude Code is available. The second-agent trial is replaced by the official format validator (`skills-ref`) and a test that copies the bare skill folder elsewhere and runs it.

**Done when:** SKILL.md passes the standard check, the plugin installs in Claude Code, and all tests pass.

### Later candidates (not planned yet)
A trial in a second agent when one is available. (Plugin packaging was done in Phase 9; the other candidates are now Phases 10–14 below.)

---

## Phases 10–14: approved 2026-11-05 (order and recommended decisions)

Order: trigger tests come first, because every later phase changes `SKILL.md`, and a changed description can change when the skill triggers. Then the small change to input formats, then two analysis features, then the cleaning script, the biggest change to how the skill behaves. Each phase raises the version by one minor step (1.2.0, 1.3.0, ...) in both `SKILL.md` and `plugin.json`, so plugin users get the update.

## Phase 10: Repeatable trigger tests ✅

**Goal:** the Phase 7 trigger checks become a suite you can re-run after every change to `SKILL.md`.

- `evals/triggers.json`: about 24 prompts, each with the expected result:
  - **should trigger** (~12): plain requests ("what's wrong with sales.csv?"), indirect ones ("can I trust these numbers?" with a CSV attached), Excel and sheet names, a non-English request
  - **should not trigger** (~12), mostly near misses: "write a SQL query", "plot a sine wave", "fix this pandas error", "convert this CSV to JSON", "explain what a p-value is", a request about a Python web app
- A runner that starts a headless Claude Code session per prompt in a throwaway config with the plugin installed and records whether the skill was loaded. First check whether `claude plugin eval` already does this; write our own small script (`evals/run_triggers.py`) only if it doesn't.
- The result goes to `evals/results/<date>.md` (pass rate per group plus every miss), so changes to the description can be compared.
- If results vary between runs, each prompt runs 3 times and counts as passed only on a majority.

**You will learn:** how skill descriptions are matched to requests, and false positives vs. false negatives.

**Decision (approved):** run it **manually**, before each release, not in CI.

**Done when:** the suite runs with one command and the current description passes at least 11 of 12 in each group.

## Phase 11: Parquet and JSON input ✅

**Goal:** the skill reads `.parquet`, `.json` and `.jsonl` files as well as CSV and Excel.

- `load_dataset()` gains:
  - **Parquet** via `pd.read_parquet`. That needs `pyarrow`. Like openpyxl for Excel, it's needed only for that format, and the error explains how to install it.
  - **JSON:** an array of records, or JSON Lines (`.jsonl`, one record per line). Nested objects are flattened into `parent.child` columns with `pd.json_normalize`, and the overview says which columns came from flattening. Lists inside records stay as they are and are reported as a type problem (a column of lists can't be summarised).
  - Clear errors for JSON that isn't table-shaped (e.g. a single object of settings).
- `--json-path` for records nested under a key (e.g. `{"data": {"orders": [...]}}` → `--json-path data.orders`).
- Parquet keeps its stored types, so fewer "numbers stored as text" findings appear. Tests check that this is reported correctly, not misread as "no problems".
- SKILL.md description, README, `requirements*.txt` (pyarrow as optional) and the CI minimum-versions job updated.

**You will learn:** columnar vs. row-based formats, why Parquet keeps types and CSV doesn't, and nested data.

**Decision (approved, changes D6):** add `pyarrow` as an **optional** dependency.

**Done when:** the same sample data saved as Parquet, JSON and JSONL gives the same findings as the CSV, apart from the type problems that Parquet keeps from happening at all.

## Phase 12: Time trends for date columns ✅

**Goal:** for data with dates, show how it changes over time and find problems that only show up in time order.

- `time_summary()` for every datetime column, and for dates stored as text (parsed in memory only, with the day/month order the quality check detected, and labelled as such):
  - a period size chosen automatically (day / week / month / quarter / year) so the chart has roughly 12–120 points
  - row count per period, plus the median of the most relevant numeric column (the `--target` column once Phase 13 exists)
- New chart: a line chart of rows per period, with empty periods marked and partial first and last periods drawn as dashed lines (a half-finished month looks like a drop).
- New finding rules:
  - `date_gaps`: empty periods inside the range (possibly missing data)
  - `partial_period`: the first or last period is incomplete
  - `implausible_dates`: dates in the future, or placeholder dates such as `1900-01-01` and `1970-01-01`
  - `date_spike`: a period with far more rows than usual (e.g. a duplicate import)
- `--date-column` to choose the column when there are several, and `--period` to override the automatic size.
- Sample data: add a planted gap and a spike to `make_sample_data.py`, and add both to its answer key.
- A section on trends and seasonality in `visualization-guide.md`, including why a trend isn't a forecast.

**You will learn:** resampling, partial periods, and why seasonality makes month-on-month comparisons misleading.

**Done when:** the planted gap, spike and future date in the sample data are found, and data without dates produces no time section and no error.

## Phase 13: `--target` for group comparisons ✅

**Goal:** answer "what goes with higher X?". The user names one column, and the analysis is centred on it.

- `--target COL`:
  - **numeric target** (e.g. `revenue`): for each categorical column, the count, median, mean and IQR per group (top N groups, the rest as "Other"); for each numeric column, Spearman and Pearson correlation with the target, ranked
  - **categorical target** (e.g. `churned` yes/no, at most ~10 classes): the rate of each class per group, and the numeric columns compared across classes (medians)
- New charts: box plots of the target by group (sorted by median, with n per group) for the strongest 3 categorical columns; for a categorical target, a bar chart of rates per group.
- Safeguards in the report:
  - groups with fewer than 30 rows are marked "too small to compare"
  - a `possible_leakage` finding when a column predicts the target almost perfectly (e.g. `revenue` vs. `revenue_eur`, or a status set after the outcome)
  - rows where the target is missing are counted and left out, and the report says so
  - the limitations section states that group differences aren't causes and that no significance tests were run
- The report gets a new section 4b, "Target comparison", only when `--target` is used. The 7 standard sections stay as they are.

**You will learn:** comparing groups fairly, why medians beat means for skewed targets, small-group noise, Simpson's paradox and target leakage.

**Decision (approved):** **descriptive only**: no new dependency, and nobody reads a p-value as proof. (Significance tests would need SciPy or our own implementation.)

**Done when:** on the sample data `--target revenue` ranks the region and product differences planted in `make_sample_data.py` correctly, and flags a planted leaky column.

## Phase 14: Cleaning script with a change log ✅

**Goal:** approved cleaning steps are applied by tested code, not ad-hoc pandas, and every change is recorded. This finishes golden rule 2: flag → explain → approve → apply to a **new** file → check.

- `scripts/clean_data.py <dataset> --plan plan.json --output cleaned.csv`
- `plan.json` is a list of steps from a fixed set, each tied to the finding `rule` it fixes:

  | Step | Fixes |
  |---|---|
  | `drop_exact_duplicates` (keep first) | exact duplicate rows |
  | `drop_duplicate_keys` (keep first / last / most complete) | repeated keys |
  | `placeholders_to_missing` | `-`, `unknown`, `999`, ... |
  | `strip_whitespace`, `normalize_labels` (an explicit mapping, e.g. `north` → `North`) | untrimmed and inconsistent labels |
  | `to_numeric` (currency, %, decimal comma), `to_datetime` (day/month order) | numbers and dates stored as text |
  | `drop_columns` | empty or constant columns |
  | `drop_rows_missing` (in given columns) | missing values |
  | `fill_missing` (median / mode / constant, always adds a `<col>_was_missing` column) | missing values |
  | `cap_outliers` (to the IQR fences) or `flag_outliers` (adds a column, changes nothing) | outliers |

- **Dry run by default:** it prints what each step would change (rows and values affected, examples) and writes nothing. `--apply` writes the file.
- Safeguards: it refuses to write over the input or into a non-empty folder, checks the input's SHA-256 before and after, and stops with a clear error if a step names a column that doesn't exist.
- Writes `cleaning_log.md` (readable) and `cleaning_log.json` next to the output: every step with counts before and after, the SHA-256 of input and output, and the plan itself, so the cleaned file can be rebuilt exactly.
- SKILL.md Step 6 becomes: write `plan.json` from the options the user approved → show the dry run → apply after a second "yes" → run the EDA on the cleaned file and report what changed (findings that went away, findings that are new).
- `data-cleaning.md`: each section points to its step name.
- Tests: every step against known answers, a full plan on the sample data that fixes the planted problems and nothing else, a dry run that writes no files, and the same plan run twice gives byte-identical output.

**You will learn:** reproducible cleaning, why imputation needs an indicator column, and how capping outliers changes the mean and standard deviation.

**Decisions (approved 2026-11-05 with Phases 10-14):**
- **Output format:** CSV, or Parquet when `--output` ends in `.parquet` (the default for Parquet input, so stored types survive). No Excel or JSON output.
- **Free-form steps:** only the fixed set above; every step is tested. A one-off fix outside it is done with pandas by hand, as before, and the user is told it isn't in the log.

**Done when:** the full flag → approve → dry run → apply → re-check cycle works on the sample data in a real Claude Code session, and the input file is unchanged.

---

## Phase 15: Reliable triggering, cleaning included 🟡

**Goal:** the skill loads whenever someone asks for what it does, including the cleaning added in Phase 14, and stays out of the way for near misses. People who don't know the skill is installed only benefit if it triggers on its own.

Why now: the trigger suite was last run on version 1.1.0. The description changed in Phase 11 (Parquet and JSON added) and hasn't been tested since, and in the Phase 14 trial a plain "What's wrong with the data in sample_sales.csv? Tell me what you'd clean." did not load the skill. The description also says nothing about cleaning, so "clean this file and log what you changed" probably won't trigger it.

1. **More cases** in `evals/make_cases.py` (the folders are generated from it, and `tests/test_evals.py` keeps them in step):
   - **should trigger** (+7): the exact Phase 14 miss, four cleaning requests ("clean up sales.csv: remove duplicates and fix the dates, and keep a record of what you changed", "fill in the missing values in survey.csv sensibly", "prepare customers.xlsx for analysis, fix whatever is broken", "standardise the region labels in sales.csv"), one Parquet and one JSON Lines request
   - **should not trigger** (+4), near misses for cleaning: "drop the notes column from sales.csv", "write a pandas function that removes duplicate rows from a DataFrame", "merge sales.csv and customers.xlsx on customer_id", "read orders.parquet and save it as CSV"
   - two new fixtures in `evals/_fixtures/`: `orders.parquet` and `events.jsonl`
   - total 19 + 16 = 35 cases, 3 runs each
2. **New description:** add cleaning without inviting simple edits. Draft of the added sentence: "…or to clean a dataset, applying approved fixes (duplicates, placeholder values, inconsistent labels, numbers or dates stored as text, missing values, outliers) to a new copy with a change log." It stays under the 1,024-character limit, and the `plugin.json` description is already fine.
3. Version **1.6.0** in `SKILL.md` and `plugin.json`, the README trigger section updated once the numbers are in.

**Spreading the usage (the user doesn't want to use the whole limit at once).** Steps 1-3 use no Claude usage: they're local edits and pytest. The eval runs are split into small stages. Each stage stops with a short report, and the next one starts only when you say so, on the same day or a later one.

| Stage | What runs | Sessions | Approx. usage |
|---|---|---|---|
| A. Baseline | all 35 cases on the **old** 1.5.0 description, 1 run each (shows whether the Phase 11 change already hurt plain requests) | 35 | ~$3 |
| B. New wording | all 35 cases on the new description, 1 run each | 35 | ~$3 |
| C. Fixes (only if needed) | changed wording, only the missed cases and their nearest opposites (`--case`), 1 run each, at most 3 rounds | ~5-10 per round | ~$0.5-1 per round |
| D1. Confirm, should trigger | `--case 'pos*'`, 3 runs each (majority counts, as in Phase 10) | 57 | ~$4.5 |
| D2. Confirm, should not trigger | `--case 'neg*'`, 3 runs each | 48 | ~$4 |

About $15 in total, but no stage takes more than about $4.5. Concurrency (`-j`) only changes speed, not usage, so it stays at 4. The dollar figures are what the eval tool estimates at API prices. On Pro they come out of the usage limit and aren't billed. Each stage's result goes to `evals/results/` so a later session can pick up where the last one stopped.

**Done when:** in stages D1 and D2, at most one case per group is missed (≥ 18 of 19 should trigger, ≥ 15 of 16 should not), all tests pass, the skill validator and `claude plugin validate` pass, and the version is 1.6.0. Afterwards you update your installed plugin (still 1.1.0).

---

## Change log

- 2026-11-04: Plan created. Phase 0 done. Phase 1 built: SKILL.md (129 lines, 412-char description) and folder skeleton.
- 2026-11-05: Phase 1 approved. Phase 2 built: `inspect_data.py` (load, classify, overview, stats, CLI) and 37 passing tests. Additions beyond the plan: an `empty` column kind for all-missing columns, `--encoding`/`--delimiter`/`--top` options, and a `skew_label` rule of thumb. `.xls` is rejected with instructions to re-save as `.xlsx`.
- 2026-11-05: Phase 2 approved, committed and pushed. Phase 0-1 gaps filled: `.claude/settings.local.json` ignored, `pytest.ini` added, SKILL.md documents the inspect options.
- 2026-11-05: Phase 3 built: `missing_values`, `find_duplicates`, `detect_type_issues`, `iqr_outliers`, `data_quality`, `--key` option, 71 passing tests. Additions beyond the plan: `untrimmed_whitespace` as its own issue, day/month order detection for text dates, currency/%/decimal-comma handling for numbers-as-text, and sampling for date detection on columns with more than 5,000 distinct values.
- 2026-11-05: Phase 3 approved, committed and pushed. Phase 4 built: `generate_eda.py` (correlations, categorical summary, 5 chart types, `--max-categories`/`--max-scatter-points`/`--max-charts`, writes `summary.json`), 97 passing tests. Additions beyond the plan: likely-ID columns are excluded from charts and correlations, Pearson-vs-Spearman gap and low-n notes on each pair, the output folder must be new or empty (never overwrites). Histogram and box plot share one PNG per column.
- 2026-11-05: Phase 4 approved with one addition: a log-scale histogram for highly skewed positive columns. 99 passing tests. Committed and pushed.
- 2026-11-05: Phase 5 built: `report.py` (rule-based findings with high/medium/low severity, `report.md` in 7 sections, cleaning options with their costs, analysis suggestions, limitations), called at the end of `generate_eda.py`; `summary.json` gains `findings`. 121 passing tests. Additions beyond the plan: the input file's SHA-256 is recorded before and checked after every run (the CLI exits with an error if it changed), `--key` columns are treated as IDs, findings also cover duplicate keys, placeholder values, type problems, nearly constant columns, outliers and near-duplicate columns (|r| ≥ 0.98), and the report leaves a marked spot for Claude's "Analyst interpretation".
- 2026-11-05: Phase 5 approved, committed and pushed. Phase 6 built: `pandas-guide.md` (reading, inspection, filtering, dates, groupby, memory, pandas 3 pitfalls), `data-cleaning.md` (options and costs per problem, MCAR/MAR/MNAR, proposal template, cleaning log) and `visualization-guide.md` (question → chart, readability, colors, 5 chart recipes in the skill's style, misleading-chart pitfalls). 131 passing tests. Additions beyond the plan: `tests/test_references.py` runs every Python example in the guides against a messy sales file and checks that the links resolve and that every finding rule has a cleaning section; D7 extended with a categorical order and an ordered blue scale (validated for color-vision deficiencies).
- 2026-11-05: Phase 6 approved, committed and pushed. Phase 7 built: `sample_data/make_sample_data.py` (seed 2025, 1,215 rows x 16 columns, answer key of 12 planted problems in its docstring) and the committed `sample_sales.csv`; `.gitattributes` keeps CSVs byte-identical across OSes. The demo review led to fixes: complete-row counts that ignore all-empty columns, numeric codes such as 999 among outliers reported as `possible_missing_code` (and no longer called "probably legitimate"), correlation titles that say when only the ranks agree, no "most common values" chart for dates stored as text, and the missing-values axis capped at 100%. `tests/test_sample_data.py` checks reproducibility and that every planted problem is found. 163 passing tests. Still open: the trigger tests in a new Claude Code session.
- 2026-11-05: Phase 7 trigger tests done in Claude Code. Automatic trigger and `/python-data-analyst` both used the skill, left the input unchanged (SHA-256 verified) and proposed cleaning steps with costs without applying them; the full run added an Analyst interpretation (stating its assumed goal rather than asking). The unrelated question didn't load the skill.
- 2026-11-05: Phase 7 committed and pushed. Phase 8 built: `README.md` (what it does, install per project or per user, usage, output, sample data, design, methods and limitations, development) with three chart images in `docs/images/`, `LICENSE` (MIT, © 2026 GrowingRui) and `.github/workflows/tests.yml`. Additions beyond the plan: a Windows test run, and a job that tests the oldest versions `requirements.txt` promises (pandas 2.1, NumPy 1.26, Matplotlib 3.8). Running those locally found a real difference: pandas 2 reports skewness 0 for a constant column, so `inspect_data.py` now decides that case itself; the guide examples, written for pandas 3 and Matplotlib 3.10+, are skipped on older versions. 164 passing tests (159 + 4 skipped on minimum versions). Approved, committed and pushed.
- 2026-11-05: Phase 9 approved (D9: neutral `skills/` folder plus a Claude Code plugin; only Claude Code available for trials). Built: skill moved to `skills/python-data-analyst/`; SKILL.md frontmatter reduced to standard fields (`argument-hint` removed, `license`, `compatibility` and `metadata` added, version 1.1.0) and its text made agent-neutral (no `$ARGUMENTS`, a fallback for agents that can't view images); `.claude-plugin/plugin.json` and `marketplace.json` (marketplace `growingrui-data-skills`); `AGENTS.md`; README install section per agent; CI job `skill-format` running `skills-ref` and `claude plugin validate`; `tests/test_portability.py` (8 tests). Found and fixed on the way: SKILL.md told agents to `pip install -r requirements.txt`, a file that isn't part of an installed skill; it now lists the packages, and a test keeps them in step with `requirements.txt`. Checked: the official skill validator passes (it rejected `argument-hint` before the change), `claude plugin validate .` passes, the plugin installs into a throwaway Claude Code config and lists 1 skill (~110 tokens per session), and the EDA runs from the installed copy. 172 passing tests. Not tested: other agents and the third-party `npx skills` installer (blocked from running here).
- 2026-11-05: Phase 9 approved, committed and pushed.
- 2026-11-05: Phases 10-14 proposed (trigger tests, Parquet/JSON input, time trends, `--target`, cleaning script). Waiting for approval and the open decisions.
- 2026-11-05: Phases 10-14 approved with the recommended decisions. Phase 10 built: `claude plugin eval` (Claude Code 2.1.318) already runs this kind of test, so there's no own runner, only `evals/make_cases.py` (one list of 12 should-trigger and 12 should-not prompts, generating one `case.yaml` per case with a `tool_used: Skill` grader) and `evals/run_triggers.py` (runs the suite and writes `evals/results/<date>-triggers.md`). Prompts that name a file get it in their sandbox: the eval tool only runs a scaffold script inside the case folder, so the generator copies `setup.sh` and the file from `evals/_fixtures/` into each case, and `tests/test_evals.py` (30 tests, no API needed) checks the copies and cases match the generator. Runs are capped at 3 turns with read-only tools, and `--ablation none` (no without-plugin arm, it can't trigger anyway). First full run: 12 of 12 in both groups, 72 of 72 runs right, 5 minutes, priced at $5.78 (counted against the Pro subscription, not billed). Fixed on the way: the summary crashed when printed on a Windows console (the ≥ sign). The version wasn't raised: nothing in the skill changed. 202 passing tests.
- 2026-11-05: Phase 10 approved, committed and pushed. Phase 11 postponed by the user.
- 2026-11-05: Phase 11 built (version 1.2.0). `load_dataset()` reads `.parquet`/`.pq` (pyarrow, with an install hint when it's missing), `.json` (a list of records) and `.jsonl`/`.ndjson`; nested objects are flattened into `parent.child` columns (listed as `flattened_columns`), and `--json-path data.orders` reaches records under a key. JSON that isn't a table gets a specific error: a settings object (with the paths where lists of records were found, ready to pass to `--json-path`), a list of plain values, records mixed with other values, an empty list, a bad JSON Lines line (by line number). Lists and objects inside cells (JSON lists, Parquet lists and structs) are kept as JSON text so pandas can count them, reported as the new type problem `nested_values` with its own finding, cleaning advice and guide section, and left out of the category charts. For Parquet the report explains that typed columns can't hold numbers or dates as text, so few type problems is a property of the format, not proof the values are right. Additions beyond the plan: JSON Lines saved as `.json` is detected, Parquet DECIMAL columns are analysed as floats (and say so), `--encoding` applies to JSON, the pandas guide has Parquet/JSON reading examples. Checked: the sample data saved as Parquet, JSON and JSONL gives exactly the CSV's findings, and a typed Parquet version loses only the numbers/dates-as-text findings. `tests/test_formats.py` (30 tests); 232 passing tests, and 228 + 4 skipped on the minimum versions (pandas 2.1, NumPy 1.26, pyarrow 14, now in the CI job). Skill validator and `claude plugin validate` pass. The SKILL.md description changed (mentions Parquet and JSON); the trigger suite has not been re-run yet.
- 2026-11-05: Phase 11 approved, committed and pushed. The trigger suite re-run was postponed to save session usage (small description change: Parquet and JSON added); run `python evals/run_triggers.py` before the next release.
- 2026-11-05: Phase 12 built (version 1.3.0), waiting for review. New module `scripts/time_trends.py` (pandas only, no charts): `time_summary()` checks every date column (real datetimes, and text the quality check found to be dates, parsed in memory with the detected day/month order; ambiguous text is read month-first and the report says so) for future dates and placeholder dates (1899-12-30/31, 1900-01-01, 1970-01-01), and builds the trend for one column: `--date-column`, else the one with the most dates. Future dates are left out of the trend when they are at most 5% of the column (more look intended: due dates), placeholders always, so one 2052 typo can't stretch the axis. Period: the finest of day/week/month/quarter/year with at most 120 points, moving coarser while it averages under 5 rows per point and the coarser one still has 12 points; `--period` overrides. Rules: `date_gaps` = an empty run with e^-(typical rows x length) < 0.001 (Poisson rule of thumb; empty weekend days on daily data are ignored), `date_spike` = at least 3x the typical period and 10 rows more, with the busiest day and the exact duplicates in it, `partial_period` = first/last period less than 90% covered (not for days), `implausible_dates`. Typical = median rows over full, non-empty periods; gap and spike checks need 3 of them. The value shown per period is the median of an amount-like column (`revenue`, `sales`, ...; else the fullest numeric non-ID column); Phase 13's `--target` can pass its column. Chart `NN_time_<col>_by_<period>.png` second in priority: rows per period plus a median panel on the same x-axis (no dual axes), dashed partial ends, red bands for gaps, rings for spikes. Report: a "Trend over time" table in section 3, the chart text in section 5, cleaning advice for all four rules, a seasonality suggestion (year-on-year from two years of data, otherwise "can't separate seasonality from trend"), limitations (one column, not a forecast, Poisson assumption, how text dates were parsed). The 7 sections are unchanged. Wrong `--date-column`/`--period` fails before the output folder is created. Sample data: only `order_date` changed (own random generator, so every other number stays the same): no orders 4-31 August, all of them stamped with the load date 15 September (104 rows that week, 4.5x the typical 23; 93 on that day), one date in 2052, one 01/01/1900; partial first and last weeks come from 2025 starting and ending on a Wednesday; the busy November/December (1.8x) is not flagged. Guides: data-cleaning.md section 8 "Dates over time" (sections 8-9 renumbered to 9-10), visualization-guide.md section 5 "Trends and seasonality" (calendar effects, rolling medians, why a trend isn't a forecast; 5-6 renumbered to 6-7). Fixed on the way: "Further analysis" suggested comparing groups of `order_date` (dates, numbers and lists stored as text are no longer treated as groups). `tests/test_time_trends.py` (34 tests) and 5 more sample-data checks: 271 passing tests, and 267 + 4 skipped on the minimum versions. 300,000 text timestamps take about 2 seconds. Skill validator and `claude plugin validate` pass. The SKILL.md description did not change, so the trigger suite needs no re-run for this phase (still owed for Phase 11).
- 2026-11-05: Phase 12 approved, committed and pushed.
- 2026-11-05: Phase 13 built (version 1.4.0), waiting for review. New module `scripts/target_analysis.py` (pandas/NumPy only, no charts) and `--target COL` on `generate_eda.py`. A numeric target gets per-group count, median, quartiles and mean for every categorical column (top `--max-categories` groups, the rest `Other`, missing values as `(missing)`) and Pearson/Spearman with every numeric column; a categorical target (at most **8** classes instead of ~10, one per palette slot; a 0/1 column counts as two classes) gets class rates per group (the `yes`-like or rarer class when there are two) and numeric medians per class. Every comparison has one 0-1 strength on the |r| scale: rank-based eta corrected for the number of groups like adjusted R^2 (robust to skew; many tiny groups no longer 'explain' the target by chance), bias-corrected Cramer's V (Bergsma) between categories, max(|r|, |rho|) between numbers. `possible_leakage` (high) when a column predicts the target almost perfectly (>= 0.95): correlation, a chance-corrected 'separation' (groups occupy non-overlapping ranges of the target, kappa-style) or Goodman-Kruskal lambda; the leaky pair is no longer also reported as `near_duplicate_columns`. `target_difference` (low from 0.3, medium from 0.5). Groups under 30 rows are marked 'too small to compare' and never named highest/lowest. Rows without a target are counted and left out. Unusable targets (missing, ID, free text, numbers or dates stored as text, too many classes, constant) fail with a reason before the output folder is created. Charts right after the missing-values chart: box plots of a numeric target for the 3 strongest group columns (sorted by median, n per group, small/Other/missing groups in gray; a skewed target gets a log axis with the boxes computed on log values, so the whiskers stay symmetric), or rate bars with the overall rate as a dashed line (two classes) / 100% stacked bars (more). A numeric target is also the value shown per period in the time chart. Report: section '4b. Target comparison' (ranking table, per-group tables for the top 3 columns, numeric medians per class) between sections 4 and 5, only with `--target`; the 7 sections are unchanged. Next steps suggest a within-group (Simpson's paradox) check with the exact pandas line; limitations say it's descriptive, no significance tests, the small-group and Other rules, leakage only when almost perfect, and which columns weren't compared and why. Without `--target` the report suggests it. Sample data: unit prices x1.5 West, x1.1 North, x0.9 South, x0.7 East (no extra random draws, so every other column is unchanged), and a new column `order_size` (small/medium/large from revenue: the planted leak), 17 columns. `--target revenue` ranks category (eta 0.60) far ahead of region (0.18, West > North > South > East), channel and returning_customer negligible, and flags `order_size` first. Guides: pandas-guide.md 'Comparing groups fairly' (medians with n, Simpson's check, range-overlap leakage check; runs in the tests), data-cleaning.md covers `possible_leakage`, visualization-guide.md a row for rates per group. README images refreshed from the new sample data. `tests/test_target_analysis.py` (30 tests) and 2 more sample-data checks: 303 passing, and 299 + 4 skipped on the minimum versions. Skill validator and `claude plugin validate` pass. The SKILL.md description did not change (trigger suite still owed from Phase 11).
- 2026-11-05: Phase 13 approved, committed and pushed.
- 2026-11-05: Phase 14 built (version 1.5.0), waiting for review. New `scripts/clean_data.py`: a plan of steps from the fixed set of 12 (each validated before any data is read: unknown steps, unknown options with a typo, missing required options and a `fixes` rule the step can't fix are refused), a dry run by default, `--apply` to write a CSV or Parquet copy plus `cleaning_log.md` and `cleaning_log.json` into a new or empty folder. Each step logs rows, columns and missing cells before and after, the values changed per column with examples, type changes and warnings; the log holds both SHA-256 fingerprints and the plan, and passing it back as `--plan` rebuilds the file byte for byte (warning when the input differs). Refuses the input as output, a non-empty folder, other extensions, a column that doesn't exist at that step (naming the step and the columns there), and leaves no folder behind when a step fails. Details that came up while building: `to_datetime` parses ISO values apart from `day_first` ones (pandas' `dayfirst` reads 2025-01-02 as 1 February) and refuses ambiguous day/month order; `to_numeric` lists unparseable values before they become missing (or stops with `"unparseable": "error"`), `map` turns words into numbers (`"free"` -> 0); `cap_outliers` keeps integer columns whole; `drop_duplicate_keys` never drops rows without a key and lists the columns where repeats disagree; CSV columns of codes with leading zeros or more than 15 digits are kept as text so `01234` isn't written as `1234`. Additions beyond the plan: `scripts/compare_runs.py` (findings gone / new / changed between two EDA runs, a lone finding whose columns changed counts as changed), and CSV/JSON columns that hold only strict ISO dates are now read as dates by `load_dataset` (a cleaned CSV can't store dates any other way, so the re-check would otherwise report `dates_as_text` again; one other value keeps the column text and flagged). The report's cleaning options name their step (`report.CLEANING_STEPS`, which `clean_data.py` uses for its `fixes` check). `sample_data/cleaning_plan.json` fixes the planted errors and nothing else: 1,215 -> 1,200 rows, 17 -> 15 columns, the re-run EDA loses 15 findings, the only new one is `rating` missing 10.8% (the cost of `-` -> missing), the 2052 typo, gaps, spike, skew and leakage are left alone; the untouched columns are identical to the input. SKILL.md Step 6 is the plan -> dry run -> apply after a second yes -> re-check cycle; data-cleaning.md names the step in every section and section 10 is rewritten around the script; README updated. `tests/test_clean_data.py` (58 tests): 361 passing, and 357 + 4 skipped on the minimum versions. 300,000 rows clean in about 10 seconds. Skill validator and `claude plugin validate` pass. The SKILL.md description did not change. Not done yet: the hands-on cycle in a real Claude Code session.
- 2026-11-05: Phase 14 trial in a real Claude Code session (headless, 2.1.318, in a throwaway folder with its own `.venv`; this repo loaded with `--plugin-dir`, the installed 1.1.0 plugin switched off for the session with `--settings`). Cycle with `/python-data-analyst:python-data-analyst sample_sales.csv`: the EDA with `--key order_id`, charts viewed, Analyst interpretation written, cleaning proposed with costs and three questions, nothing changed. After "go ahead, keep the last row per ID": a `plan.json` of 12 steps, each with `fixes` and `reason`, in the ground-rule order, next to the EDA output; dry run only (no file written), summarised correctly (1,215 -> 1,200 rows, 157 not 160 discounts filled because of removed duplicates). After "yes, apply it": cleaned copy plus both logs, EDA re-run, `compare_runs.py` (14 findings gone, 0 new), input SHA-256 unchanged. Cost about $2.90 of subscription usage over 4 turns. **Trigger miss:** the first try, a plain "What's wrong with the data in sample_sales.csv? Tell me what you'd clean.", didn't load the skill (the agent profiled the file with the `csv` module), although the very similar trigger-suite prompt passed 12/12 in Phase 10. One run can't separate chance from the Phase 11 description change or the other plugins in the session; the trigger-suite re-run (owed since Phase 11) should settle it before release.
- 2026-11-05: Phase 14 approved, committed and pushed.
- 2026-11-05: Phase 15 proposed (trigger suite extended with cleaning, Parquet and JSON cases; description mentions cleaning; version 1.6.0). Waiting for approval and the baseline decision.
- 2026-11-05: Phase 15 changed at the user's request so it doesn't use the whole usage limit at once: eval runs split into stages A-D (1 run per case for baseline and wording, 3 runs only for the final confirmation, split into pos and neg), each stage started only on the user's go.
- 2026-11-05: Phase 15 approved with the staged runs. Free part built: 11 new cases (19 should trigger, 16 should not) in `make_cases.py`, fixtures `sample_sales.csv` (a copy of the sample data, checked by a test), `orders.parquet` (the `sales.csv` fixture as Parquet) and `events.jsonl` (61 app events with nested `user`, UTC timestamps, missing durations, an outlier, a blank country, one duplicate line); file names in prompts now match whole, so `sample_sales.csv` doesn't also pull in `sales.csv`. `run_triggers.py`: the target is at most one missed case per group (all of them for groups under 12 cases, e.g. a `--case` subset), and a group that wasn't run is left out of the table instead of counting as 0 of 0. **Bug found by the new JSONL fixture and fixed:** timestamps with a time zone (`2025-10-01T10:00:00Z`, `+02:00`) crashed the type check in `inspect_data.py` (tz-aware values put into a tz-naive Series), so the whole EDA failed on such a file; the check now parses with `utc=True`. The `to_datetime` cleaning step failed on them too: it now converts values that all carry a zone to UTC and drops the zone (noted in the log, `time_zones_converted_to_utc` per column), parses zoned ISO values apart from day-first ones, and refuses a column mixing zoned and plain times. Time trends already handled zones. data-cleaning.md says what the conversion costs (local clock times). 381 passing tests, 377 + 4 skipped on the minimum versions; `claude plugin validate` passes. The SKILL.md description and the version are unchanged on purpose: stage A (baseline) must run on the old description. Next: stage A, on the user's go.
