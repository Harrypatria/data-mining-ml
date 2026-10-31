# Templates

Source templates copied into a project by `bin/data-mining-ml init --project <dir> --package <name>`.

| Folder | Becomes | Contents |
|---|---|---|
| `src/mlkit/` | `src/<package>/` | Importable transformers, features, metrics, persistence, explanations, RTCC explainer |
| `notebooks/` | `notebooks/` | Six runnable notebooks with jupytext-paired percent `.py` files, plus `percent_template.py` |
| `ml-service/` | `api/` | FastAPI prediction, what-if and explanation service with tests and Dockerfile |
| `ml-web/` | `web/` | React what-if workbench with the AI explainer chat |
| `prompts/` | `api/app/prompts/` | RTCC system prompt, context schema, placeholder examples |
| `project/` | project root | `docker-compose.yml`, `.env.example`, `README.md`, `requirements-notebooks.txt` |

`pip install -e templates` (or `pip install "git+https://github.com/Harrypatria/data-mining-ml@v1.0.0#subdirectory=templates"` in Colab) installs `mlkit`. Files under `src/mlkit/prompts/` are generated from `prompts/` by `scripts/gen_prompts.py`; edit `prompts/` only.
