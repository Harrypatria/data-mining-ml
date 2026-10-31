# Contributing

Thank you for improving the Data Mining and Machine Learning Skill. Changes are small, tested and explained.

## Set up

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r templates/project/requirements-notebooks.txt -r templates/ml-service/requirements-dev.txt -e templates
cd templates/ml-web && npm ci && cd ../..
```

## Before opening a pull request

```bash
python -m unittest discover -s tests          # scanner rule tests, standard library only
python bin/data-mining-ml check               # manifests, notebooks, pairing, budgets, prompts, fixtures, secrets, style
python scripts/notebook_run.py templates/notebooks
(cd templates/ml-service && python -m pytest)
(cd templates/ml-web && npm run typecheck && npm test && npm run build)
```

## Adding a scanner rule

1. Add the row to `skills/ml-quality-guard/references/slop-catalog.md` (ID, severity, `auto`, rule, why, fix). Keep the catalog between 70 and 85 rules.
2. Add the rule to `RULES` in `scripts/ml_scan.py` and implement it. Heuristic rules are warnings; certain rules are errors.
3. Add a positive case to `CASES` in `tests/test_rules.py` and, when a false positive is likely, a negative case.
4. Confirm `tests/fixtures/clean` still has no errors.

## Editing notebooks

Edit the percent `.py` file, then run `python scripts/notebook_tools.py build templates/notebooks`. Never commit outputs; the builder writes empty outputs. Every number shown in a notebook must be computed when it runs.

## Editing prompts

Edit `adapters/core-prompt.md` or `adapters/overlays/*.md`, or `templates/prompts/*`, then run `python scripts/gen_prompts.py`. Never edit `dist/` or `templates/src/mlkit/prompts/` by hand.

## Style

Plain, direct language. No emojis, no em dashes, no placeholder logic or unfinished markers. Accessible docs: alt text on images, descriptive link text, no region-specific idioms.
