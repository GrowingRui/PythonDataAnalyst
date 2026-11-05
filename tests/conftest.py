"""Shared fixtures: small datasets written to a temporary folder for each test."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "skills" / "python-data-analyst" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def scripts_dir() -> Path:
    return SCRIPTS_DIR


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """One column of each kind, with known statistics."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],  # mean 30, median 30, symmetric
            "region": ["North", "South", "North", "East", "North"],
            "active": [True, False, True, True, None],
            "joined": pd.to_datetime(["2024-01-01", "2024-01-11", "2024-02-01", "2024-03-01", "2024-01-01"]),
            "notes": [None, None, None, None, None],
        }
    )


def _write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_bytes(text.encode(encoding))
    return path


@pytest.fixture
def csv_comma(tmp_path) -> Path:
    return _write(tmp_path / "comma.csv", "a,b,c\n1,x,2.5\n2,y,3.5\n3,z,4.5\n")


@pytest.fixture
def csv_semicolon(tmp_path) -> Path:
    # European-style export: ';' separates fields, ',' is the decimal mark.
    return _write(tmp_path / "semi.csv", "a;b;c\n1;x;2,5\n2;y;3,5\n3;z;4,5\n")


@pytest.fixture
def tsv_file(tmp_path) -> Path:
    return _write(tmp_path / "tabs.tsv", "a\tb\n1\tx\n2\ty\n")


@pytest.fixture
def csv_utf8_bom(tmp_path) -> Path:
    return _write(tmp_path / "bom.csv", "name,city\nAnna,Zürich\n", encoding="utf-8-sig")


@pytest.fixture
def csv_cp1252(tmp_path) -> Path:
    # 'é' and '€' are single bytes in cp1252 and invalid as UTF-8.
    return _write(tmp_path / "legacy.csv", "item,price\ncafé,€3\nthé,€2\n", encoding="cp1252")


@pytest.fixture
def empty_file(tmp_path) -> Path:
    return _write(tmp_path / "empty.csv", "")


@pytest.fixture
def header_only(tmp_path) -> Path:
    return _write(tmp_path / "header.csv", "a,b,c\n")


@pytest.fixture
def excel_two_sheets(tmp_path) -> Path:
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"x": [1, 2, 3]}).to_excel(xw, sheet_name="first", index=False)
        pd.DataFrame({"y": ["a", "b"], "z": [1.5, 2.5]}).to_excel(xw, sheet_name="2025", index=False)
    return path
