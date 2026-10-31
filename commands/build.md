---
description: End-to-end build: brief to notebook, model bundle, API, web app and explainer
argument-hint: [brief, dataset path or PRD]
---

Use the `data-mining-machine-learning` hub skill on: $ARGUMENTS

1. Run `problem-framing` and write ANALYSIS_MANIFEST.md (state assumptions in one line each; ask only if the answer changes the architecture).
2. Scaffold the project when none exists: `python "${CLAUDE_PLUGIN_ROOT}/bin/data-mining-ml" init --project <dir> --package <name>`.
3. Work through the CRISP-DM phases in order, loading each skill only when its phase starts: data-profiling, data-preparation, feature-engineering, model-training, hyperparameter-tuning, model-evaluation, model-explainability, data-storytelling, model-deployment, ml-web-application, ai-prediction-explainer.
4. After each phase run the Think, Make, Check loop from `ml-quality-guard` (clean run, scanner, refinement; at most 3 fix cycles).
5. Finish with the artifact list, the RIGOR table and the status line.
