#!/usr/bin/env python3
"""Claude Code PostToolUse hook: scan the file just written and report slop back to the agent.

Mode from DMML_HOOK_MODE: block (default; errors are fed back so the agent fixes them), warn (findings added
as context, never blocking) or off. For files under a web/ or ml-web/ folder it also runs the Tastemaker
scanner when that tool is installed. Standard library only; never fails the tool call on its own errors.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ml_scan  # noqa: E402

SCANNED = {".py", ".ipynb", ".r", ".sql"}
WEB = {".ts", ".tsx", ".js", ".jsx", ".css"}


def target_path(payload: dict) -> Path | None:
    tool_input = payload.get("tool_input") or {}
    raw = tool_input.get("file_path") or tool_input.get("notebook_path") or tool_input.get("path")
    return Path(raw) if raw else None


def tastemaker(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    if path.suffix.lower() not in WEB or not ({"web", "ml-web"} & parts):
        return ""
    npx = shutil.which("npx")
    if not npx:
        return ""
    try:
        out = subprocess.run([npx, "--no-install", "tastemaker", "scan", str(path)], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode != 0 and out.stdout else ""


def main() -> int:
    mode = os.environ.get("DMML_HOOK_MODE", "block").lower()
    if mode == "off":
        return 0
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    path = target_path(payload)
    if path is None or not path.is_file():
        return 0
    messages = []
    if path.suffix.lower() in SCANNED:
        findings = ml_scan.run([str(path)])
        errors = [f for f in findings if f.severity == "error"]
        warnings = [f for f in findings if f.severity == "warning"]
        lines = [f"{f.loc} {f.severity} {f.rule} {f.message}" for f in errors + warnings][:25]
        if lines:
            messages.append(f"data-mining-ml scan of {path.name}: {len(errors)} errors, {len(warnings)} warnings, "
                            f"score {ml_scan.score(findings)}\n" + "\n".join(lines))
        blocking = bool(errors)
    else:
        blocking = False
    taste = tastemaker(path)
    if taste:
        messages.append("Tastemaker scan:\n" + taste[:3000])
        blocking = True
    if not messages:
        return 0
    text = "\n\n".join(messages) + "\nFix at the cause, or add '# ml-ignore <ID> <reason>' when intentional."
    if mode == "block" and blocking:
        print(json.dumps({"decision": "block", "reason": text}))
    else:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": text}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
