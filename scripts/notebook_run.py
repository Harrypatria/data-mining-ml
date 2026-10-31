#!/usr/bin/env python3
"""Run notebooks top to bottom from a clean kernel (the Restart and Run All check).

Uses `python -m nbconvert --execute` when nbconvert is installed in the running interpreter,
so the kernel is this interpreter. Otherwise converts code cells to a script (magics dropped)
and runs it with this interpreter. Standard library only.

Usage:
    python notebook_run.py NOTEBOOK_OR_DIR... [--out DIR] [--timeout 900] [--cwd DIR] [--script]
Exit code 1 when any notebook fails.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def to_script(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    chunks = ["import matplotlib\nmatplotlib.use('Agg')\ntry:\n    from IPython.display import display, Markdown\n"
              "except ImportError:\n    display = print\n    Markdown = str\n"]
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        lines = [("pass  # " + ln.strip()) if ln.lstrip().startswith(("%", "!")) else ln for ln in src.split("\n")]
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) + "\n"


def run_one(nb_path: Path, out_dir: Path, timeout: int, cwd: Path, force_script: bool) -> tuple[bool, str, float]:
    start = time.time()
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    if not force_script and importlib.util.find_spec("nbconvert") and importlib.util.find_spec("ipykernel"):
        # A private kernelspec pins the kernel to this interpreter; a global "python3" spec could point elsewhere.
        spec_dir = out_dir / "kernels" / "dmml-run"
        spec_dir.mkdir(parents=True, exist_ok=True)
        (spec_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "dmml-run", "language": "python"}), encoding="utf-8")
        env["JUPYTER_PATH"] = os.pathsep.join([str(out_dir.resolve()), env.get("JUPYTER_PATH", "")])
        cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", str(nb_path.resolve()),
               "--output-dir", str(out_dir.resolve()), f"--ExecutePreprocessor.timeout={timeout}",
               "--ExecutePreprocessor.kernel_name=dmml-run"]
    else:
        script = out_dir / f"{nb_path.stem}.py"
        script.write_text(to_script(nb_path), encoding="utf-8")
        cmd = [sys.executable, str(script.resolve())]
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout + 60)
    tail = (proc.stderr or proc.stdout or "")[-3000:]
    return proc.returncode == 0, tail, time.time() - start


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", help="directory for executed copies (default: a temporary directory)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--cwd", help="working directory for the kernel (default: the notebook's folder)")
    ap.add_argument("--script", action="store_true", help="force the plain script fallback")
    args = ap.parse_args(argv)
    notebooks = []
    for raw in args.paths:
        p = Path(raw)
        notebooks += sorted(p.glob("*.ipynb")) if p.is_dir() else [p]
    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="nbrun-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for nb in notebooks:
        ok, tail, secs = run_one(nb, out_dir, args.timeout, Path(args.cwd) if args.cwd else nb.parent, args.script)
        print(f"{'passed' if ok else 'FAILED'}  {nb}  ({secs:.0f} s)")
        if not ok:
            failed += 1
            print(tail)
    print(f"run: {'passed' if not failed else 'failed'} ({len(notebooks) - failed}/{len(notebooks)} notebooks), outputs in {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
