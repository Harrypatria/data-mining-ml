---
name: ml-quality-guard
description: ML Quality Guard. Rigor harness for data science and machine learning work: scan notebooks, scripts, APIs and prompts for leakage, statistical errors, imbalance mistakes, reproducibility gaps, production and LLM explainer slop, then fix and score. Use when asked to check, verify, lint, review, validate, score or "is this right" for a notebook, model, pipeline, analysis, FastAPI service or explainer, and before declaring any ML deliverable done.
---

# ML Quality Guard

Nothing reaches the user until it passes this loop.

## Run the scanner

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ml_scan.py" <paths> [--format text|json] [--imbalanced] [--fail-on warning]
```

Scans `.py` (including `# %%` percent files), `.ipynb`, `.R`, `.sql`, `.md` prompts, `Dockerfile`, `.env*`, `.ts`, `.tsx`, `.js`. Output per finding: rule ID, severity, location (`line N` or `cell C line N`), message. Ends with `Rigor score 0-100`. Exit code 1 on any error. The catalog lives in `references/slop-catalog.md`; read a rule there only when a finding needs explaining.

Escape hatch with a reason: `# ml-ignore M04 threshold fixed by regulator at 0.5`. No reason, no ignore.

## Think, Make, Check (max 3 fix cycles)

1. **Compile and Run.** Execute from a clean kernel: `python "${CLAUDE_PLUGIN_ROOT}/scripts/notebook_run.py" <notebook>` (nbconvert or a plain script run). Confirm shapes and row counts at every join and split. Confirm the notebook finishes.
2. **Statistical and Leakage Self-Correction.** Run the scanner. Then check by hand what it cannot see:
   - Split logic matches the data (stratified, group-aware, forward-chaining). Test set touched once.
   - All fitting inside pipelines on training folds; resampling in-fold only.
   - Every test states question, estimand, assumptions; effect size and CI next to p-values.
   - Baseline comparison, intervals on headline metrics, threshold chosen on validation, calibration checked after resampling.
   - Every widget callback runs once without error (call the update function directly in a cell).
3. **Refinement.** Remove redundant code and repeated output, recompute the RIGOR score, print the status line.

Fix errors at the cause, not by suppressing. After 3 cycles, report what remains.

## RIGOR gates (0 or 1 each)

| Gate | Passes when |
|---|---|
| Reproducibility | Seed constant, pinned versions, relative paths, Restart and Run All succeeds |
| Integrity | No leakage findings, fitting in pipelines, labels verified, resampling in-fold |
| Generalization | Honest validation, baseline beaten, intervals, slices, threshold from validation |
| Explainability | Global and local explanations on held-out data, feature dictionary, caveats |
| Observability | Logging, data checks, monitoring plan, model card |
| Communication | Decision-first narrative, labeled charts, limitations, what-if with static fallback |
| Deployability | Reload parity, schema-validated API, health checks and tests pass, secrets out of code and browser, explainer grounded |

Score each gate from evidence, not intent. Gates that do not apply to the deliverable (for example Deployability for a one-off A/B readout) are reported as `n/a` and excluded from the denominator.

## Status line

`Rigor: <errors> errors, <warnings> warnings, score <0-100>, RIGOR <passed>/<applicable>, run: passed|failed|not run`

## Checks beyond the scanner

- Notebook state: execution counts in order, no outputs over 100 KB, no full data frames printed.
- Leakage: list each feature with its availability time; anything after prediction time is removed.
- Imbalance: positive count stated next to every rare-class metric; PR-AUC shown; accuracy never alone.
- Production: `bin/data-mining-ml verify-model models/<version>` passes; API tests pass.
- Explainer: grounding test rejects a number absent from the context.

Details of each gate: `references/rigor-gates.md`.
