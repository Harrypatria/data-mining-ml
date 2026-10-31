---
description: Train, test once, refit, save the model bundle and verify it
argument-hint: [notebook or task description]
---

Use the `model-training` skill on: $ARGUMENTS

Follow the protocol: one split with a recorded seed, one pipeline, baseline first, CV on train, threshold on validation, test scored once with intervals, final refit policy stated, learning curve, `save_bundle`, fresh-process `parity_check`, then `python "${CLAUDE_PLUGIN_ROOT}/bin/data-mining-ml" verify-model models/<version>`. Report the bundle path, test metrics with intervals and the status line.
