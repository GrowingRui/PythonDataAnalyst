# Trigger tests

These tests check *when* the skill loads, not how good its analysis is (the
pytest suite covers that). An agent decides to load a skill from its one-line
`description` alone, so any change to that line, or to how Claude Code picks
skills, can make it load too rarely (users don't get it) or too often (it
takes over unrelated requests).

- **19 should-trigger cases** (`pos*`): plain and indirect requests, Excel
  with a sheet name, charts, a report, a German request, cleaning requests
  (with a change log, filling missing values, fixing labels), Parquet and
  JSON Lines files.
- **16 should-not-trigger cases** (`neg*`), mostly near misses: SQL, a
  matplotlib sine wave, a pandas error, converting or renaming a column in a
  CSV, a Python function that reads a CSV or drops duplicates, dropping a
  column, merging two files, Parquet to CSV.

Each case runs 3 times and passes when the majority of its runs did the right
thing. The target is **at most one missed case per group**. A run of only a
few cases (fewer than 12 in a group, e.g. a `--case` subset) must get all of
them right.

## Run them

You need Claude Code (`claude`, or `npx @anthropic-ai/claude-code`) and a
login. From the repository root:

```bash
python evals/run_triggers.py                         # the full suite: 105 short runs
python evals/run_triggers.py --runs 1                # a quick check: 35 runs
python evals/run_triggers.py --case "pos*"           # one group (57 runs)
python evals/run_triggers.py --case "pos1[3-9]*" --runs 1   # a few cases
```

It runs `claude plugin eval` on this repository's plugin and writes
`evals/results/<date>-<time>-triggers.md`: pass rates, the description that
was tested, and every miss. Commit that file when the run is worth keeping
(e.g. before a release); the raw JSON and traces next to it are ignored by git.

**Cost:** it uses your own Claude usage. A full run is priced at about $8-9 in API terms in
November 2026 (a should-trigger run about $0.15, a should-not run about
$0.05), and takes about 7 minutes with 4 runs at a time. To spread it out,
run one group at a time or `--runs 1`; every run writes its own result file. With a claude.ai
subscription (Pro, Max) that counts against your usage limits instead of being
billed; with an API key it is billed. It isn't run in CI.

## How it works

- Each `pos*`/`neg*` folder holds a `case.yaml` for `claude plugin eval`: the
  prompt, a short turn limit (3) and one grader, "the Skill tool was called
  with `python-data-analyst`" (at least once, or never).
- The agent may use only `Skill`, `Read`, `Glob` and `Grep`, so it can load
  the skill and look at the file but not run the analysis. That keeps runs
  short and cheap.
- Prompts that name one of the files in `_fixtures/` (`sales.csv`,
  `survey.csv`, `customers.xlsx`, `sample_sales.csv`, `orders.parquet`,
  `events.jsonl`) get that file in their sandbox, copied by the case's
  `setup.sh`. Names match whole: `sample_sales.csv` doesn't also bring
  `sales.csv`. `sample_sales.csv` is the repository's sample data, and
  `orders.parquet` is `sales.csv` saved as Parquet.
- `--ablation none`: there is no "without the plugin" comparison, because
  without it the skill can't load at all.

## Change the cases

Edit the lists in [`make_cases.py`](make_cases.py), then run
`python evals/make_cases.py`. Don't edit the case folders by hand:
`tests/test_evals.py` checks they match the generator.
