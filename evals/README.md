# Trigger tests

These tests check *when* the skill loads, not how good its analysis is (the
pytest suite covers that). An agent decides to load a skill from its one-line
`description` alone, so any change to that line, or to how Claude Code picks
skills, can make it load too rarely (users don't get it) or too often (it
takes over unrelated requests).

- **12 should-trigger cases** (`pos*`): plain and indirect requests, Excel
  with a sheet name, charts, a report, a German request.
- **12 should-not-trigger cases** (`neg*`), mostly near misses: SQL, a
  matplotlib sine wave, a pandas error, converting or renaming a column in a
  CSV, a Python function that reads a CSV.

Each case runs 3 times and passes when the majority of its runs did the right
thing. The target is at least **11 of 12 in each group**.

## Run them

You need Claude Code (`claude`, or `npx @anthropic-ai/claude-code`) and a
login. From the repository root:

```bash
python evals/run_triggers.py              # the full suite: 72 short runs
python evals/run_triggers.py --runs 1     # a quick check: 24 runs
python evals/run_triggers.py --case "neg*"
```

It runs `claude plugin eval` on this repository's plugin and writes
`evals/results/<date>-<time>-triggers.md`: pass rates, the description that
was tested, and every miss. Commit that file when the run is worth keeping
(e.g. before a release); the raw JSON and traces next to it are ignored by git.

**Cost:** it uses your own Claude usage. A full run is priced at about $6 in API terms in
November 2026 (a should-trigger run about $0.15, a should-not run about
$0.05), and takes about 5 minutes with 4 runs at a time. With a claude.ai
subscription (Pro, Max) that counts against your usage limits instead of being
billed; with an API key it is billed. It isn't run in CI.

## How it works

- Each `pos*`/`neg*` folder holds a `case.yaml` for `claude plugin eval`: the
  prompt, a short turn limit (3) and one grader, "the Skill tool was called
  with `python-data-analyst`" (at least once, or never).
- The agent may use only `Skill`, `Read`, `Glob` and `Grep`, so it can load
  the skill and look at the file but not run the analysis. That keeps runs
  short and cheap.
- Prompts that name `sales.csv`, `survey.csv` or `customers.xlsx` get that
  file in their sandbox, copied from `_fixtures/` by the case's `setup.sh`.
- `--ablation none`: there is no "without the plugin" comparison, because
  without it the skill can't load at all.

## Change the cases

Edit the lists in [`make_cases.py`](make_cases.py), then run
`python evals/make_cases.py`. Don't edit the case folders by hand:
`tests/test_evals.py` checks they match the generator.
