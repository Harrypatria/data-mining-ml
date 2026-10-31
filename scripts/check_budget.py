#!/usr/bin/env python3
"""Check every skills/*/SKILL.md: valid frontmatter, name matches the folder, trigger-rich description,
body under the token budget (characters / 4 when no tokenizer is available). Standard library only.

Usage: python check_budget.py [--budget 1800] [--root PATH]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse(text: str) -> tuple[dict, str] | None:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return None
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, m.group(2)


def check(root: Path, budget: int) -> list[str]:
    problems, rows = [], []
    for skill in sorted((root / "skills").glob("*/SKILL.md")):
        folder = skill.parent.name
        parsed = parse(skill.read_text(encoding="utf-8"))
        if parsed is None:
            problems.append(f"{folder}: missing or malformed frontmatter")
            continue
        meta, body = parsed
        tokens = len(body) // 4
        rows.append((folder, tokens))
        if meta.get("name") != folder:
            problems.append(f"{folder}: frontmatter name {meta.get('name')!r} does not match the folder")
        desc = meta.get("description", "")
        if len(desc) < 120:
            problems.append(f"{folder}: description too short to trigger reliably ({len(desc)} chars)")
        if len(desc) > 1024:
            problems.append(f"{folder}: description longer than 1024 chars")
        if tokens > budget:
            problems.append(f"{folder}: body about {tokens} tokens, budget {budget}")
        for ref in re.findall(r"`(references/[\w.-]+\.md)`", body):
            if not (skill.parent / ref).is_file():
                problems.append(f"{folder}: referenced file {ref} does not exist")
    for folder, tokens in rows:
        print(f"{folder:<32} {tokens:>5} tokens")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1800)
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    problems = check(args.root, args.budget)
    for p in problems:
        print(f"FAIL {p}")
    print("budget: passed" if not problems else f"budget: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
