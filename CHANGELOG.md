# Changelog

All notable changes follow semantic versioning: added skills or rules are minor, renamed or removed ones are major.

## 1.0.0 - 2026-10-30

### Added
- 17 skills from problem framing to the grounded LLM prediction explainer, each under the 1,800-token body budget.
- Slop catalog with 85 rules in ten families; `scripts/ml_scan.py` detects 79 of them with no dependencies.
- PostToolUse hook that scans edited Python, notebook, R and SQL files (`DMML_HOOK_MODE=block|warn|off`) and calls Tastemaker for web files.
- Ten slash commands and the `ml-critic` reviewer agent.
- `mlkit` package: simulated data, transformers, features, metrics with intervals, model bundle persistence with checksum, version guard and fresh-process parity check, interventional Shapley values, counterfactuals, RTCC context and numeric grounding.
- Six runnable notebook templates with jupytext-paired percent files, plus a percent skeleton.
- FastAPI service scaffold with 34 tests, Dockerfile and compose file.
- React what-if workbench built with the Tastemaker workflow, with 7 tests.
- Platform prompts generated from one core prompt: system prompt, Gemini in Colab, AI Studio, notebook cell; `init` targets for Codex, Cursor, Replit and Claude.ai Projects.
