# AGENTS.md

This repository holds one agent skill: `skills/python-data-analyst/`.

## Analysing a dataset

When the user asks to inspect, profile, explore, clean up or chart a CSV, TSV
or Excel file, or to write a data analysis report, read
`skills/python-data-analyst/SKILL.md` and follow it exactly, including its
golden rules: the input file is read-only, and no cleaning step is applied
without the user's agreement. In that file, `<skill-dir>` means
`skills/python-data-analyst`.

## Working on the skill itself

- Python 3.11+. Set up with `pip install -r requirements-dev.txt`, test with
  `python -m pytest`.
- Everything an installed skill needs lives inside `skills/python-data-analyst/`.
  Don't make it depend on files outside that folder.
- Keep `SKILL.md` to the standard Agent Skills fields and agent-neutral
  wording; `tests/test_portability.py` checks this.
- The plan and decisions are in `PLAN.md`.
