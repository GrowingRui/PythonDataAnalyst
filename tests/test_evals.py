"""The trigger-test suite in evals/ is complete and in step with its generator.

Running the suite itself needs a Claude login and costs usage, so CI only
checks its shape: the case folders match evals/make_cases.py, both groups are
balanced, every case can pass with the tools it gets, and the summary counts
results correctly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EVALS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


make_cases = _load("make_cases")
run_triggers = _load("run_triggers")
CASES = make_cases.case_names()


def test_groups_are_balanced_and_names_unique():
    pos = [c for c in CASES if c[2]]
    neg = [c for c in CASES if not c[2]]
    assert len(pos) == len(neg) == 12
    assert len({c[0] for c in CASES}) == len(CASES)
    assert len({c[1] for c in CASES}) == len(CASES), "two cases share a prompt"


def test_case_folders_match_the_generator():
    on_disk = {p.name for p in EVALS.iterdir()
               if p.is_dir() and (p / "case.yaml").exists()}
    assert on_disk == {c[0] for c in CASES}, "run `python evals/make_cases.py`"


@pytest.mark.parametrize("name,prompt,should", CASES, ids=[c[0] for c in CASES])
def test_case_content_is_generated(name, prompt, should):
    d = EVALS / name
    needed = make_cases.files_for(prompt)
    expected = make_cases.case_yaml(name, prompt, should, bool(needed))
    assert (d / "case.yaml").read_text(encoding="utf-8") == expected, "run `python evals/make_cases.py`"
    if needed:
        assert (d / "setup.sh").read_text(encoding="utf-8") == make_cases.SETUP_SH
        copied = sorted(p.name for p in (d / "files").iterdir())
        assert copied == sorted(needed)
        for f in needed:
            assert (d / "files" / f).read_bytes() == (make_cases.FIXTURES / f).read_bytes()
    else:
        assert not (d / "setup.sh").exists() and not (d / "files").exists()


def test_every_should_trigger_prompt_has_its_file():
    # Real users ask about a file they have; a missing file could change what the agent does first.
    for name, prompt, should in CASES:
        if should:
            assert make_cases.files_for(prompt), f"{name} names no example file"


def test_cases_can_load_the_skill():
    # A "should not" case only means something if the Skill tool was available.
    for name, _, _ in CASES:
        text = (EVALS / name / "case.yaml").read_text(encoding="utf-8")
        assert "Skill" in text.split("allowed_tools:")[1].splitlines()[0]
        assert f"input_match: {make_cases.SKILL}" in text


def _result(pos_passes: list[list[bool]], neg_passes: list[list[bool]]) -> dict:
    cases = []
    for prefix, groups in (("pos", pos_passes), ("neg", neg_passes)):
        for i, runs in enumerate(groups, 1):
            cases.append({"name": f"{prefix}{i:02d}-x", "promptMarkdown": f"prompt {prefix}{i}",
                          "arms": {"with": [{"passed": p, "error": None} for p in runs]}})
    return {"startedAt": "2026-11-05T10:00:00Z", "claudeVersion": "test", "costUsd": 1.5,
            "durationSeconds": 120, "cases": cases}


def test_summary_counts_majority_per_case():
    pos = [[True] * 3] * 11 + [[True, False, False]]   # 11 of 12 cases pass
    neg = [[True] * 3] * 11 + [[True, True, False]]    # 12 of 12, one miss on majority
    text, met = run_triggers.summarize(_result(pos, neg))
    assert met
    assert "| Should trigger | 11 of 12 | 34 of 36 |" in text
    assert "| Should not trigger | 12 of 12 | 35 of 36 |" in text
    assert "**pos12-x** (1 of 3 runs right, case failed)" in text
    assert "**neg12-x** (2 of 3 runs right, case passed on majority)" in text


def test_summary_reports_a_missed_target():
    pos = [[True] * 3] * 10 + [[False] * 3] * 2
    neg = [[True] * 3] * 12
    text, met = run_triggers.summarize(_result(pos, neg))
    assert not met
    assert "missed" in text
