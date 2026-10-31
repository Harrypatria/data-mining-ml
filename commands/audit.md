---
description: Score and fix an existing notebook, script or ML repository
argument-hint: [path] [fix]
---

Use the `notebook-audit` skill on: $ARGUMENTS (if empty: the current project).

Scan, run from a clean kernel, read for what the scanner cannot see, score the RIGOR gates, and report P0, P1 and P2 findings with locations and exact fixes. If the arguments include "fix", apply P0 then P1 fixes with minimal diffs, rerun, and report before and after.
