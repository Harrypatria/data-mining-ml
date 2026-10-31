# Environments

One notebook source runs in all five targets. Detect once, branch only where behavior differs.

```python
import os, sys
IN_COLAB = "google.colab" in sys.modules
IN_VSCODE = "VSCODE_PID" in os.environ
```

## Jupyter Notebook 7

- Linear cells, `%matplotlib inline`, ipywidgets 8 core widgets.
- Clean-run check: `python scripts/notebook_run.py notebook.ipynb` (wraps `nbconvert --execute` with a kernel pinned to the current interpreter).
- Clear outputs before committing: `python scripts/notebook_tools.py clear notebooks/`.

## JupyterLab 4

- Same notebooks. The widget manager ships with ipywidgets 8; no extension install needed.
- Table of contents is built from headings, so findings-first headings make the TOC a summary.
- jupytext pairing: `jupytext --set-formats ipynb,py:percent notebook.ipynb`; edit either file, jupytext keeps them in sync. Without jupytext, `scripts/notebook_tools.py build` regenerates the `.ipynb` from the percent file.

## Google Colab with Gemini

- Install with `%pip install -q` and pinned versions; the bootstrap cell installs `mlkit` from the repository tag.
- Secrets: `from google.colab import userdata; userdata.get("OPENAI_API_KEY")`. Never paste keys into cells.
- Drive only on request (`drive.mount("/content/drive")`); the runtime is ephemeral, so save models and reports to Drive or download them.
- `from google.colab import output; output.enable_custom_widget_manager()` only for third-party widgets; core ipywidgets work without it.
- `%load_ext google.colab.data_table` for large frame previews.
- GPU or high-RAM runtimes: Runtime menu, Change runtime type; tabular scikit-learn work rarely needs a GPU.
- Paste-ready Gemini instructions: `dist/gemini-colab.md`.

## Visual Studio Code

- Jupyter extension: `.ipynb` editing, or `# %%` percent `.py` files with the Interactive Window (Shift+Enter runs a cell).
- Data Wrangler for frame inspection without printing.
- `.vscode/tasks.json`: run the scanner, the clean-execute check, the API and the web dev server. `.vscode/launch.json` debugs FastAPI. `.vscode/extensions.json` recommends extensions.

## Claude Code

- Prefer editing the paired percent `.py` file; jupytext (or `notebook_tools.py build`) syncs the `.ipynb`. Edit `.ipynb` directly with the notebook edit tool only when no pair exists.
- The hook scans each edited `.py`, `.ipynb`, `.R` or `.sql` file and reports errors back; `DMML_HOOK_MODE=block|warn|off`.
- Slash commands: `/data-mining-ml:build :audit :scan :fix :train :serve :web :explain :story :report`.

## Secondary targets

`bin/data-mining-ml init --target <name>` writes instructions generated from `adapters/core-prompt.md`:

| Target | File |
|---|---|
| Codex and other agents | `AGENTS.md` |
| Cursor | `.cursor/rules/data-mining-ml.mdc` |
| Replit | `replit.md` |
| Claude.ai Projects | `claude-project-instructions.md` |
| Google AI Studio | `ai-studio-system-instructions.md` |
