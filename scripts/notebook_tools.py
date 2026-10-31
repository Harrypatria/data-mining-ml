#!/usr/bin/env python3
"""Notebook tools on the standard library: build paired .ipynb files from jupytext percent .py files,
check pairing, validate notebook structure and clear outputs.

Usage:
    python notebook_tools.py build PATH...      write <name>.ipynb next to each percent-format <name>.py
    python notebook_tools.py check PATH...      exit 1 when a paired .ipynb is missing or out of date
    python notebook_tools.py validate PATH...   nbformat 4 structure checks and ast.parse of every code cell
    python notebook_tools.py clear PATH...      remove outputs and execution counts in place
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}
MAGIC_COMMENT = re.compile(r"^(\s*)# ([%!].*)$")


def _cell_id(name: str, index: int) -> str:
    return hashlib.sha1(f"{name}:{index}".encode()).hexdigest()[:12]


def _lines(text: str) -> list[str]:
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def py_to_nb(text: str, name: str) -> dict:
    """Convert a jupytext percent script to an nbformat 4.5 notebook with empty outputs."""
    lines = text.split("\n")
    if lines and lines[0].strip() == "# ---":
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "# ---")
        lines = lines[end + 1:]
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        while buf and not buf[-1].strip():
            buf.pop()
        while buf and not buf[0].strip():
            buf.pop(0)
        if kind == "markdown":
            src = "\n".join(re.sub(r"^# ?", "", ln) for ln in buf)
            cells.append({"cell_type": "markdown", "id": _cell_id(name, len(cells)), "metadata": {}, "source": _lines(src)})
        else:
            src = "\n".join(MAGIC_COMMENT.sub(r"\1\2", ln) for ln in buf)
            cells.append({"cell_type": "code", "execution_count": None, "id": _cell_id(name, len(cells)),
                          "metadata": {}, "outputs": [], "source": _lines(src)})

    for line in lines:
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            buf.append(line)
    flush()
    return {"cells": cells,
            "metadata": {"jupytext": {"formats": "ipynb,py:percent"}, "kernelspec": KERNELSPEC,
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def dumps(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def validate_nb(nb: dict) -> list[str]:
    problems = []
    if nb.get("nbformat") != 4:
        problems.append("nbformat is not 4")
    if not isinstance(nb.get("metadata"), dict):
        problems.append("metadata missing")
    ids = set()
    for i, cell in enumerate(nb.get("cells", [])):
        kind = cell.get("cell_type")
        if kind not in ("code", "markdown", "raw"):
            problems.append(f"cell {i}: unknown cell_type {kind}")
            continue
        for key in ("metadata", "source"):
            if key not in cell:
                problems.append(f"cell {i}: missing {key}")
        if nb.get("nbformat_minor", 0) >= 5:
            cid = cell.get("id")
            if not cid or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cid) or cid in ids:
                problems.append(f"cell {i}: missing, invalid or duplicate id")
            ids.add(cid)
        if kind == "code":
            if "outputs" not in cell or "execution_count" not in cell:
                problems.append(f"cell {i}: code cell lacks outputs or execution_count")
            src = "".join(cell.get("source", []))
            code = "\n".join(("pass" if ln.lstrip().startswith(("%", "!")) else ln) for ln in src.split("\n"))
            try:
                ast.parse(code)
            except SyntaxError as exc:
                problems.append(f"cell {i}: syntax error line {exc.lineno}: {exc.msg}")
    return problems


def _expand(paths: list[str], suffix: str) -> list[Path]:
    out = []
    for raw in paths:
        p = Path(raw)
        out += sorted(p.rglob(f"*{suffix}")) if p.is_dir() else [p]
    return [p for p in out if ".ipynb_checkpoints" not in p.parts]


def _is_percent(path: Path) -> bool:
    """A percent script that is paired with a notebook (its jupytext header lists ipynb, or it has no header)."""
    if path.suffix != ".py":
        return False
    text = path.read_text(encoding="utf-8")
    if "# %%" not in text:
        return False
    header = re.search(r"#\s+formats:\s*(\S+)", text[:600])
    return header is None or "ipynb" in header.group(1)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("build", "check", "validate", "clear"):
        print(__doc__)
        return 2
    cmd, paths = argv[0], argv[1:]
    status = 0
    if cmd in ("build", "check"):
        for py in filter(_is_percent, _expand(paths, ".py")):
            target = py.with_suffix(".ipynb")
            text = dumps(py_to_nb(py.read_text(encoding="utf-8"), py.stem))
            if cmd == "build":
                target.write_text(text, encoding="utf-8", newline="\n")
                print(f"built {target}")
            elif not target.is_file() or target.read_text(encoding="utf-8") != text:
                print(f"out of date: {target} (run notebook_tools.py build)")
                status = 1
    elif cmd == "validate":
        for nb_path in _expand(paths, ".ipynb"):
            try:
                problems = validate_nb(json.loads(nb_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:
                problems = [f"invalid JSON: {exc}"]
            for p in problems:
                print(f"{nb_path}: {p}")
            status |= bool(problems)
        if not status:
            print("all notebooks valid")
    else:
        for nb_path in _expand(paths, ".ipynb"):
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    cell["outputs"], cell["execution_count"] = [], None
            nb_path.write_text(dumps(nb), encoding="utf-8", newline="\n")
            print(f"cleared {nb_path}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
