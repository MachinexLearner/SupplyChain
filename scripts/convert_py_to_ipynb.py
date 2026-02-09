#!/usr/bin/env python3
"""
Convert Databricks source-format .py notebooks (# COMMAND ----------, # MAGIC %md)
to Jupyter .ipynb for direct upload to Databricks (Workspace → Import → Jupyter Notebook).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _split_cells(content: str) -> list[str]:
    """Split by # COMMAND ---------- and return list of cell contents (strip)."""
    parts = re.split(r"\n# COMMAND ----------\s*\n", content)
    return [p.strip() for p in parts if p.strip()]


def _is_markdown_cell(cell_text: str) -> bool:
    """True if cell is a MAGIC %md cell."""
    first = cell_text.lstrip()
    return first.startswith("# MAGIC %md") or first.startswith("# MAGIC %md\n")


def _magic_md_to_markdown(cell_text: str) -> str:
    """Convert # MAGIC %md ... and # MAGIC ... lines to plain markdown."""
    lines = cell_text.splitlines()
    out = []
    started = False
    for line in lines:
        if line.strip().startswith("# MAGIC %md"):
            started = True
            rest = line.strip()[len("# MAGIC %md") :].strip()
            if rest:
                out.append(rest)
            continue
        if started and line.strip().startswith("# MAGIC "):
            out.append(line.strip()[len("# MAGIC ") :])
            continue
        if started and line.strip().startswith("# MAGIC"):
            out.append("")
            continue
        if started:
            out.append(line)
    return "\n".join(out)


def _strip_databricks_header(cell_text: str) -> str:
    """Remove first line if it is '# Databricks notebook source'."""
    lines = cell_text.splitlines()
    if lines and lines[0].strip() == "# Databricks notebook source":
        return "\n".join(lines[1:]).strip()
    return cell_text


def py_to_cells(py_path: Path) -> list[dict]:
    """Parse a Databricks .py notebook and return list of Jupyter cell dicts."""
    text = py_path.read_text(encoding="utf-8")
    raw_cells = _split_cells(text)
    cells = []
    for raw in raw_cells:
        raw = _strip_databricks_header(raw)
        if not raw:
            continue
        if _is_markdown_cell(raw):
            source = _magic_md_to_markdown(raw)
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _to_source(source)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "source": _to_source(raw), "outputs": [], "execution_count": None})
    return cells


def _to_source(s: str) -> list[str]:
    """Jupyter source: list of lines, each line including newline except last."""
    if not s:
        return []
    lines = s.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1] + "\n"
    return lines


def build_nb(cells: list[dict], language: str = "python") -> dict:
    """Build nbformat v4 notebook dict."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "language_info": {"name": language},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "cells": cells,
    }


def convert_file(py_path: Path, out_path: Path | None = None) -> Path:
    """Convert one .py notebook to .ipynb. Returns path to .ipynb."""
    out_path = out_path or py_path.with_suffix(".ipynb")
    cells = py_to_cells(py_path)
    nb = build_nb(cells)
    out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out_path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    # Folders that contain Databricks .py notebooks (exclude scripts/ and non-notebook .py)
    skip_names = {"ingestion_utils.py", "convert_py_to_ipynb.py", "__init__.py"}
    dirs = ["ingestion", "transformation", "forecasting", "agents"]
    if len(sys.argv) > 1:
        paths = [(root / p).resolve() if not Path(p).is_absolute() else Path(p).resolve() for p in sys.argv[1:]]
    else:
        paths = []
        for d in dirs:
            folder = root / d
            if folder.is_dir():
                for f in sorted(folder.glob("*.py")):
                    if f.name not in skip_names:
                        paths.append(f)
    for py_path in paths:
        if not py_path.is_file():
            continue
        out = convert_file(py_path)
        print(f"  {py_path.relative_to(root)} -> {out.relative_to(root)}")
    if not paths:
        print("No .py notebooks found. Usage: python convert_py_to_ipynb.py [file1.py ...]")


if __name__ == "__main__":
    main()
