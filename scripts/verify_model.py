#!/usr/bin/env python3
"""Verify a model bundle without unpickling it. Standard library only.

Checks: model.pkl or model.sav exists; model.sha256 matches; metadata.json and schema.json parse;
required metadata keys and library versions are recorded; schema features match metadata columns;
background.csv has no identifier-like columns.

Usage: python verify_model.py models/v1.2.0   (or a models/ root that holds CURRENT)
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

REQUIRED_META = ("model_version", "created_utc", "seed", "task", "target", "threshold", "metrics", "columns",
                 "libraries", "model_sha256")
REQUIRED_LIBS = ("python", "numpy", "pandas", "scikit-learn")
ID_HINTS = ("id", "email", "phone", "name", "address", "ssn", "dob")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path) -> tuple[list[str], dict]:
    d = path
    if (d / "CURRENT").is_file():
        d = d / (d / "CURRENT").read_text(encoding="utf-8").strip()
    problems: list[str] = []
    model = next((d / n for n in ("model.pkl", "model.sav") if (d / n).is_file()), None)
    if model is None:
        return [f"{d}: no model.pkl or model.sav"], {}
    for name in ("model.sha256", "metadata.json", "schema.json"):
        if not (d / name).is_file():
            problems.append(f"missing {name}")
    if problems:
        return problems, {}
    expected = (d / "model.sha256").read_text(encoding="utf-8").split()
    actual = sha256(model)
    if not expected or expected[0] != actual:
        problems.append("checksum mismatch between the model file and model.sha256")
    try:
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        schema = json.loads((d / "schema.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return problems + [f"invalid JSON: {exc}"], {}
    problems += [f"metadata.json lacks {k}" for k in REQUIRED_META if k not in meta]
    problems += [f"metadata.libraries lacks {k}" for k in REQUIRED_LIBS if k not in meta.get("libraries", {})]
    if meta.get("model_sha256") and meta["model_sha256"] != actual:
        problems.append("metadata.model_sha256 does not match the model file")
    names = [f.get("name") for f in schema.get("features", [])]
    if names != meta.get("columns"):
        problems.append("schema.json features do not match metadata.columns")
    for f in schema.get("features", []):
        if f.get("type") in ("number", "integer") and not (f.get("min") is not None and f.get("max") is not None and f["min"] <= f.get("default", f["min"]) <= f["max"]):
            problems.append(f"schema feature {f.get('name')} lacks a valid min, max and default")
    bg = d / "background.csv"
    if bg.is_file():
        with bg.open(encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
        leaked = [c for c in header if any(c.lower() == h or c.lower().endswith("_" + h) for h in ID_HINTS)]
        if leaked:
            problems.append(f"background.csv contains identifier-like columns {leaked}")
    return problems, {"path": str(d), "version": meta.get("model_version"), "libraries": meta.get("libraries", {}),
                      "threshold": meta.get("threshold"), "sha256": actual[:16]}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    problems, info = verify(Path(argv[0]))
    if info:
        print(f"bundle {info['path']}  version {info['version']}  threshold {info['threshold']}  sha256 {info['sha256']}...")
        print("libraries: " + ", ".join(f"{k}=={v}" for k, v in info["libraries"].items()))
    for p in problems:
        print(f"FAIL {p}")
    print("verify-model: passed" if not problems else f"verify-model: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
