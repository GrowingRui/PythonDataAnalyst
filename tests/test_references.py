"""The reference guides: links resolve, they match the code, and every Python example runs."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from generate_eda import COLOR_SERIES
from report import ADVICE_ALIAS, CLEANING_ADVICE

matplotlib.use("Agg")

SKILL_DIR = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "python-data-analyst"
GUIDES = ["pandas-guide.md", "data-cleaning.md", "visualization-guide.md"]


def _text(name: str) -> str:
    return (SKILL_DIR / "references" / name).read_text(encoding="utf-8")


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", text, flags=re.MULTILINE | re.DOTALL)


def _github_anchor(heading: str) -> str:
    """How GitHub turns a heading into a link target."""
    slug = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return slug.replace(" ", "-")


@pytest.fixture
def sales_files(tmp_path):
    """A messy sales file with every column the guides use."""
    rng = np.random.default_rng(20)
    n = 120
    qty = rng.integers(1, 6, n)
    price = rng.lognormal(3, 0.8, n).round(2)
    df = pd.DataFrame(
        {
            "order_id": np.arange(1, n + 1),
            "order_date": pd.date_range("2025-01-01", periods=n, freq="3D").strftime("%d/%m/%Y"),
            "region": rng.choice(["North", "South", "East", "north "], n),
            "customer_id": [f"{i:05d}" for i in rng.integers(1, 400, n)],
            "quantity": qty,
            "unit_price": price,
            "revenue": (qty * price).round(2),
            "discount": np.where(rng.random(n) < 0.3, np.nan, rng.uniform(0, 0.3, n).round(3)),
            "rating": rng.choice(["1", "2", "3", "4", "5", "-"], n),
        }
    )
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)  # exact duplicates and repeated keys
    path = tmp_path / "sales.csv"
    df.to_csv(path, index=False)
    xlsx_path = tmp_path / "sales.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="2025", index=False)
        df.head(10).to_excel(xw, sheet_name="2024", index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    return path, xlsx_path, out_dir


def test_skill_links_every_guide():
    skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    for name in GUIDES:
        assert f"references/{name}" in skill
        assert len(_text(name)) > 2000


@pytest.mark.parametrize("name", GUIDES)
def test_internal_links_resolve(name):
    text = _text(name)
    anchors = {_github_anchor(h) for h in re.findall(r"^#{1,4} (.+)$", text, flags=re.MULTILINE)}
    links = re.findall(r"\]\(#([^)]+)\)", text)
    assert links, "each guide starts with a linked table of contents"
    assert set(links) <= anchors, set(links) - anchors


@pytest.mark.parametrize("name", GUIDES)
def test_python_examples_run(name, sales_files, scripts_dir):
    path, xlsx_path, out_dir = sales_files
    namespace = {"path": path, "xlsx_path": xlsx_path, "out_dir": out_dir, "scripts_dir": scripts_dir}
    before = path.read_bytes()
    blocks = _python_blocks(_text(name))
    assert blocks
    for i, block in enumerate(blocks, 1):
        try:
            exec(compile(block, f"{name}, python block {i}", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - the message is the point
            pytest.fail(f"{name}, python block {i} failed: {type(exc).__name__}: {exc}\n\n{block}")
    assert path.read_bytes() == before  # the examples never write to the input


def test_visualization_examples_save_charts(sales_files, scripts_dir):
    path, xlsx_path, out_dir = sales_files
    namespace = {"path": path, "xlsx_path": xlsx_path, "out_dir": out_dir, "scripts_dir": scripts_dir}
    for block in _python_blocks(_text("visualization-guide.md")):
        exec(block, namespace)
    pngs = sorted(p.name for p in out_dir.glob("*.png"))
    assert len(pngs) == 5
    assert all((out_dir / p).read_bytes()[:4] == b"\x89PNG" for p in pngs)


def test_cleaning_guide_covers_every_finding_rule():
    text = _text("data-cleaning.md")
    for rule in [*CLEANING_ADVICE, *ADVICE_ALIAS]:
        assert f"`{rule}`" in text, rule


def test_palette_matches_the_scripts():
    text = _text("visualization-guide.md")
    assert f'CATEGORICAL = ["{COLOR_SERIES}"' in text  # slot 1 is the single-series blue
