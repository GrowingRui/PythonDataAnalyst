"""The skill follows the open Agent Skills format and works outside this repo.

These checks stand in for agents we can't run in CI: if SKILL.md uses only
standard fields, names no Claude Code-only features, and the skill folder runs
on its own after a plain copy, any agent that supports the format can use it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "python-data-analyst"
PLUGIN_DIR = ROOT / ".claude-plugin"

# From the Agent Skills specification (agentskills.io) and its reference validator.
ALLOWED_FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def _frontmatter() -> dict:
    """Parse the simple YAML subset SKILL.md uses: `key: value` plus one nested map."""
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    assert match, "SKILL.md must start with a --- frontmatter block"
    data: dict = {}
    parent = None
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        key, _, value = line.strip().partition(":")
        value = value.strip().strip('"')
        if line.startswith((" ", "\t")):
            assert parent is not None, f"indented line without a parent: {line!r}"
            data[parent][key] = value
        elif value:
            data[key], parent = value, None
        else:
            data[key], parent = {}, key
    return data


def _skill_files() -> list[Path]:
    return [p for p in SKILL_DIR.rglob("*") if p.is_file() and p.suffix in {".md", ".py"}]


def test_frontmatter_uses_only_standard_fields():
    fm = _frontmatter()
    assert set(fm) <= ALLOWED_FIELDS, f"non-standard fields: {set(fm) - ALLOWED_FIELDS}"
    assert {"name", "description"} <= set(fm)


def test_name_matches_folder_and_rules():
    name = _frontmatter()["name"]
    assert name == SKILL_DIR.name
    assert len(name) <= 64 and NAME_PATTERN.match(name)


def test_field_lengths():
    fm = _frontmatter()
    assert 0 < len(fm["description"]) <= 1024
    assert len(fm.get("compatibility", "")) <= 500
    assert all(isinstance(v, str) for v in fm.get("metadata", {}).values())


def test_no_claude_code_only_features_in_the_skill():
    """$ARGUMENTS, argument-hint and agent names only work, or only fit, in one agent."""
    for path in _skill_files():
        text = path.read_text(encoding="utf-8")
        for word in ("$ARGUMENTS", "argument-hint", "Claude"):
            assert word not in text, f"{path.relative_to(ROOT)} mentions {word!r}"


def test_skill_does_not_point_outside_its_folder():
    """An installed skill is only this folder: no requirements.txt, tests or sample data."""
    for path in _skill_files():
        text = path.read_text(encoding="utf-8")
        for outside in ("requirements.txt", "sample_data/", "tests/"):
            assert outside not in text, f"{path.relative_to(ROOT)} refers to {outside!r}"


def test_install_line_matches_requirements():
    """SKILL.md lists the packages itself; keep it in step with requirements.txt."""
    wanted = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            pkg, _, version = line.partition(">=")
            wanted[pkg.strip()] = version.strip()
    skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    listed = dict(re.findall(r'"([a-z]+)>=([\d.]+)"', skill))
    assert listed == wanted


def test_plugin_manifests_agree():
    plugin = json.loads((PLUGIN_DIR / "plugin.json").read_text(encoding="utf-8"))
    market = json.loads((PLUGIN_DIR / "marketplace.json").read_text(encoding="utf-8"))
    fm = _frontmatter()
    [entry] = market["plugins"]
    assert entry["name"] == plugin["name"] == fm["name"]
    assert plugin["version"] == fm["metadata"]["version"]
    # The repo root is the plugin, and Claude Code finds skills in its skills/ folder.
    assert entry["source"] == "./"
    assert (ROOT / "skills" / fm["name"] / "SKILL.md").is_file()


def test_copied_skill_folder_runs_on_its_own(tmp_path):
    """What every agent installer does: copy the folder somewhere else and run it."""
    installed = tmp_path / ".agents" / "skills" / SKILL_DIR.name
    shutil.copytree(SKILL_DIR, installed, ignore=shutil.ignore_patterns("__pycache__"))
    data = tmp_path / "data.csv"
    data.write_text("region,revenue\nNorth,10\nSouth,20\nNorth,\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(installed / "scripts" / "inspect_data.py"), str(data), "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["overview"]["rows"] == 3
