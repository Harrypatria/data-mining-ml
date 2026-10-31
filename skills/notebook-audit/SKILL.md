---
name: notebook-audit
description: Notebook Audit. Score and fix an existing Jupyter notebook, Colab notebook, Python script, ML repository or model service - leakage, statistics, imbalance, reproducibility, explainability, communication and production readiness - with a ranked findings list, RIGOR score and minimal fixes. Use for review my notebook, audit this analysis, check this model, code review for data science, is this production ready, or before sharing or deploying someone else's work.
---

# Notebook Audit

## Procedure

1. **Inventory.** List notebooks, scripts, model files, API code, prompts and data files. Note the environment (Jupyter, JupyterLab, Colab, VS Code) and Python version.
2. **Scan.** `python "${CLAUDE_PLUGIN_ROOT}/scripts/ml_scan.py" <paths> --format text`. Add `--imbalanced` when the target is rare.
3. **Run.** `python "${CLAUDE_PLUGIN_ROOT}/scripts/notebook_run.py" <notebook>` from a clean kernel. A notebook that does not run scores 0 on Reproducibility.
4. **Read for what the scanner cannot see** (`references/audit-rubric.md`): target definition and timing, split versus data structure, metric versus cost, threshold source, test-set reuse, causal language, chart honesty, limitations.
5. **Score** the seven RIGOR gates with evidence (cell or line).
6. **Report** (at most 30 lines): status line, RIGOR table, findings ranked P0 (wrong results), P1 (misleading or unsafe), P2 (quality), each with location, rule ID, why it matters and the exact fix.
7. **Fix** on request: P0 first with minimal diffs, rerun the notebook and the scanner, report before and after. At most 3 cycles.

## Priorities

| Priority | Examples |
|---|---|
| P0 | Leakage (L01 to L10), test set used for tuning or threshold, fabricated metrics (C01), unpickling untrusted input (P01) |
| P1 | Accuracy alone on imbalance (M02), ROC-AUC alone (M03), no baseline, no intervals, causal claims, secrets in code, LLM without grounding |
| P2 | Missing feature dictionary, unlabeled charts, oversized outputs, unpinned installs |

## Rules

- Evidence for every finding: file, cell and line, or the scanner rule.
- Do not rewrite working code to match taste; change only what a finding requires.
- If the notebook cannot run (missing data or library), say which and score what can be scored.
- For a second opinion without builder bias, run the `ml-critic` agent.
