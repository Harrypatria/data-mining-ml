# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Replace with the takeaway, stated as a finding
# **Decision.** One sentence: who decides what, using this analysis.
#
# Skeleton for VS Code Interactive Window, JupyterLab with jupytext pairing, and Claude Code. Edit this `.py` file;
# jupytext (or `scripts/notebook_tools.py build`) keeps the paired `.ipynb` in sync.

# %% [markdown]
# ## Environment, seed and paths

# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
MLKIT_SPEC = "git+https://github.com/Harrypatria/data-mining-ml@v1.0.0#subdirectory=templates"
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", MLKIT_SPEC], check=True)
    ROOT = Path("/content")
else:
    ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "mlkit").is_dir()), None)
    if ROOT is None:
        raise FileNotFoundError("src/mlkit not found in this folder or its parents; open the notebook inside the project.")
    sys.path.insert(0, str(ROOT / "src"))
OUT = Path(os.environ.get("DMML_OUTPUT_DIR", ROOT))

import numpy as np
import pandas as pd

from mlkit.persistence import library_versions

SEED = 42
print(library_versions())

# %% [markdown]
# ## Data load with schema and row-count checks

# %%
from mlkit.data import make_customer_churn
from mlkit.features import PREDICTORS, TARGET

df = make_customer_churn(2000, 0.15, SEED)
assert {"customer_id", *PREDICTORS, TARGET} <= set(df.columns)
assert df["customer_id"].is_unique
print(df.shape, f"missing cells: {int(df.isna().sum().sum())}", f"duplicated rows: {int(df.duplicated().sum())}")
df.head()

# %% [markdown]
# ## Next cells, in order
# Profiling, preparation pipeline, baseline, candidates, tuning, evaluation with threshold and calibration, final refit,
# persistence with parity check, explainability, what-if dashboard, limitations, artifact export. Copy each block
# from `crisp_template.py`; keep one finding per heading.
#
# ## Limitations
# State them here before sharing results.
