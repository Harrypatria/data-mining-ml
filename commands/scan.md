---
description: Run the analysis and production slop scanner and summarize findings
argument-hint: [paths] [--imbalanced]
---

Run: `python "${CLAUDE_PLUGIN_ROOT}/scripts/ml_scan.py" $ARGUMENTS` (if no paths: the project root).

Summarize by severity and family (L, S, M, F, X, D, E, C, P, G), quote the three most important findings with their fix from `skills/ml-quality-guard/references/slop-catalog.md`, and print the Rigor score. Do not edit files.
