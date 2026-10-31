---
description: Fix every scanner error (and unexplained warnings) in the given paths, looping until clean
argument-hint: [paths]
---

Use the `ml-quality-guard` skill on: $ARGUMENTS

Scan, fix each error at its cause (never by suppressing), rerun the scanner and the notebook, and repeat at most 3 cycles. Warnings are fixed or given an `# ml-ignore <ID> <reason>` with a real reason. Report the before and after scores and anything left.
