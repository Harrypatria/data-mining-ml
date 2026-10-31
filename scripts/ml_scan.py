#!/usr/bin/env python3
# ml-scan: skip-file the scanner's own rule patterns would match themselves
"""Zero-dependency scanner for analysis slop and production slop.

Scans Python scripts, `# %%` percent files, Jupyter notebooks, R, SQL, explainer
prompts, Dockerfiles, compose files, env files and front-end sources. Prints one
line per finding and a Rigor score from 0 to 100. Exit code 1 when any error is
found (or any warning with --fail-on warning).

Usage:
    python ml_scan.py PATH [PATH ...] [--format text|json] [--imbalanced]
                      [--fail-on error|warning] [--max-output-kb 100] [--quiet]

Escape hatch: `# ml-ignore L02 reason` on the flagged line or the line above.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Rule table: id -> (severity, title, fix). Only rules detected here are listed;
# the full catalog (including manual rules) is skills/ml-quality-guard/references/slop-catalog.md.
RULES: dict[str, tuple[str, str, str]] = {
    "L01": ("error", "Preprocessor fitted before the train and test split", "Fit transformers inside a Pipeline after splitting."),
    "L02": ("error", "Fitted on test or validation data", "Fit on training data only; transform or predict held-out data."),
    "L03": ("error", "Resampling outside the training fold", "Resample inside an imbalanced-learn Pipeline so it only touches training folds."),
    "L05": ("warning", "Future data in a time-series feature", "Use positive shifts and trailing windows only."),
    "L06": ("warning", "Target statistics without out-of-fold fitting", "Use sklearn TargetEncoder (cross-fitted) inside the pipeline."),
    "L07": ("error", "Early stopping evaluated on the test set", "Early stop on a validation fold carved from the training data."),
    "L08": ("warning", "Decision threshold chosen on the test set", "Choose the threshold on validation data, then score the test set once."),
    "L09": ("warning", "No duplicate check before splitting", "Check duplicated() and use group-aware splits for repeated entities."),
    "L10": ("warning", "Entity identifiers present but the split is not group-aware", "Use GroupKFold, StratifiedGroupKFold or GroupShuffleSplit, or assert ids are unique."),
    "S01": ("warning", "p-value without effect size or confidence interval", "Report the effect size and its interval next to the p-value."),
    "S02": ("warning", "Parametric test without assumption checks", "Check normality and variance first (shapiro, levene) or use a rank test."),
    "S03": ("warning", "Multiple tests without correction", "Apply Holm, Bonferroni or Benjamini-Hochberg via multipletests."),
    "S05": ("warning", "Causal language without a causal design", "Say 'is associated with', or name the causal design and its assumptions."),
    "S07": ("warning", "Regression coefficients read without diagnostics", "Run VIF, residual plots and a heteroscedasticity test before reading coefficients."),
    "M01": ("warning", "No baseline model", "Fit a DummyClassifier, DummyRegressor or simple linear baseline first."),
    "M02": ("error", "Accuracy is the only metric", "Add recall, precision, PR-AUC, MCC or a cost-based metric."),
    "M03": ("error", "ROC-AUC without a precision-recall metric on imbalanced data", "Report average precision (PR-AUC) and precision and recall at the threshold."),
    "M04": ("warning", "Hardcoded 0.5 decision threshold", "Tune the threshold on validation data with a cost matrix or constraint."),
    "M05": ("warning", "Shuffled split on time-ordered data", "Use TimeSeriesSplit or a cutoff date with shuffle=False."),
    "M06": ("warning", "Missing random_state", "Pass a single SEED constant to every stochastic object."),
    "M07": ("warning", "Classification split without stratify", "Pass stratify=y to keep class ratios equal across splits."),
    "M08": ("warning", "Single split with no uncertainty estimate", "Use cross-validation or bootstrap confidence intervals."),
    "M09": ("error", "Resampler used outside an imbalanced-learn Pipeline", "Import Pipeline from imblearn.pipeline and put the sampler inside it."),
    "M10": ("warning", "Default hyperparameters presented as tuned", "Run a search with cross-validation or say 'default configuration'."),
    "M11": ("warning", "Distance-based method on unscaled features", "Scale inside the pipeline before k-means, k-NN, SVM, DBSCAN or PCA."),
    "M12": ("warning", "Number of clusters chosen without validation", "Compare elbow, silhouette, gap statistic and bootstrap stability."),
    "M13": ("warning", "No calibration check after reweighting or resampling", "Plot a calibration curve, report the Brier score, recalibrate on validation data."),
    "F01": ("warning", "Engineered features without a feature dictionary", "Record name, source, transform, rationale and availability time."),
    "F02": ("error", "Feature selection fitted outside the pipeline", "Put the selector inside the Pipeline so it is fitted per training fold."),
    "F03": ("warning", "Target encoder without internal cross-fitting", "Use sklearn.preprocessing.TargetEncoder or out-of-fold encoding."),
    "F04": ("warning", "Manual search loop without cross-validation", "Use GridSearchCV, RandomizedSearchCV, successive halving or Optuna with CV."),
    "F05": ("error", "Tuning objective reads the test set", "Score trials on cross-validation folds of the training data."),
    "F06": ("warning", "Best score reported without spread across folds", "Report mean and standard deviation from cv_results_."),
    "F07": ("warning", "Search scored on accuracy with imbalanced data", "Score on average_precision, F-beta or neg_log_loss."),
    "X01": ("warning", "Explanations computed on training data only", "Compute permutation importance and SHAP on held-out data."),
    "X02": ("warning", "Impurity importance used with no permutation check", "Add permutation_importance on held-out data."),
    "X03": ("warning", "SHAP used without a correlated-feature caveat", "State that correlated features share or distort attribution."),
    "X04": ("warning", "Importance described as causation", "Say 'the model associates' instead of causal verbs."),
    "D01": ("warning", "Silent row drop", "Print or assert row counts before and after dropping."),
    "D02": ("warning", "Merge without validate or row-count check", "Pass validate='one_to_one' or 'many_to_one' and assert counts."),
    "D04": ("warning", "Missingness never examined", "Profile isna() by column and by target."),
    "D05": ("warning", "Personal data printed or exported", "Drop or hash identifiers before display or export."),
    "D06": ("warning", "No schema or range checks after loading data", "Assert columns, dtypes and value ranges after loading."),
    "E01": ("error", "Absolute local path", "Use paths relative to the project root."),
    "E02": ("error", "Hardcoded secret or API key", "Read secrets from environment variables or google.colab.userdata."),
    "E03": ("error", "Secret literal assigned to os.environ", "Set secrets outside code (shell, .env not committed, Colab secrets)."),
    "E04": ("error", "Wildcard import", "Import names explicitly."),
    "E05": ("warning", "Global warnings suppression", "Filter a specific category and message only."),
    "E06": ("warning", "Out-of-order or missing execution counts", "Restart the kernel and run all cells, or clear all outputs."),
    "E07": ("warning", "Oversized stored output", "Clear or summarize large outputs before saving."),
    "E08": ("error", "Unfinished marker or stub function", "Finish the logic or remove it."),
    "E09": ("warning", "Unpinned pip install", "Pin versions, for example scikit-learn==1.9.0."),
    "E10": ("warning", "Printing a full data frame", "Print shape, dtypes and head() instead."),
    "C01": ("error", "Placeholder or literal metric value", "Print metrics computed by code that ran in this session."),
    "C02": ("warning", "Chart without axis labels", "Label both axes, with units."),
    "C03": ("warning", "Misleading chart geometry", "Start bar axes at zero; avoid 3D and dual axes."),
    "C04": ("warning", "Notebook without a limitations section", "Add limitations, risks, open questions and next steps."),
    "C05": ("warning", "Widgets without a static fallback", "Add a cell that prints the key scenarios as tables or static plots."),
    "P01": ("error", "Unpickling a path from user input or an upload", "Load only an operator-set path; never accept model uploads."),
    "P02": ("error", "Unpickling without a checksum check", "Verify model.sha256 before loading."),
    "P03": ("error", "Model loaded inside a request handler", "Load once in the FastAPI lifespan handler and keep it on app.state."),
    "P04": ("warning", "Preprocessing saved separately from the model", "Save one fitted Pipeline as the single artifact."),
    "P05": ("error", "Custom transformer defined in a file that pickles it", "Move the class into src/<package>/transformers.py."),
    "P06": ("error", "Blocking model call inside an async route", "Use a plain def route or await run_in_threadpool."),
    "P07": ("error", "CORS wildcard together with credentials", "Use an explicit origin allowlist."),
    "P08": ("error", "Auto-reload in a production command", "Remove --reload and reload=True from production commands."),
    "P09": ("error", "Container runs as root", "Add a non-root USER instruction."),
    "P10": ("warning", "FastAPI app without a health endpoint", "Add /health/live and /health/ready."),
    "P11": ("warning", "Model saved with no reload parity test", "Reload in a fresh process and assert numpy.allclose on predictions."),
    "P12": ("error", "Model saved with no metadata", "Write metadata.json and schema.json beside the pickle."),
    "P13": ("warning", "Dockerfile without HEALTHCHECK", "Add HEALTHCHECK against /health/live."),
    "G01": ("error", "Explainer prompt without constraints or causation guard", "Use the RTCC prompt with a CONSTRAINTS section that forbids causal claims."),
    "G02": ("error", "Request accepts client-supplied explanation values", "Recompute SHAP and predictions on the server."),
    "G03": ("warning", "LLM output used without a numeric-grounding check", "Run the grounding checker before showing the answer."),
    "G04": ("error", "LLM key or SDK in front-end code", "Call the LLM from the backend only; the browser calls the API."),
    "G05": ("warning", "LLM call without timeout or token cap", "Set a request timeout and max_tokens."),
    "G06": ("warning", "LLM call without fallback", "Catch provider errors and return the template explanation."),
    "G07": ("error", "User text placed in the system prompt", "Put user text in the user message only."),
    "G08": ("warning", "Hardcoded LLM model name", "Read the model name from LLM_MODEL in the environment."),
}

UNFINISHED = re.compile(r"\b(" + "TO" + "DO|" + "FIX" + "ME|" + "XX" + r"X)\b")

TRANSFORMERS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer", "QuantileTransformer",
    "PowerTransformer", "SimpleImputer", "KNNImputer", "IterativeImputer", "OneHotEncoder", "OrdinalEncoder",
    "TargetEncoder", "PCA", "TruncatedSVD", "KBinsDiscretizer", "TfidfVectorizer", "CountVectorizer",
    "ColumnTransformer", "SelectKBest", "SelectPercentile", "VarianceThreshold", "RFE", "RFECV",
    "SelectFromModel", "SequentialFeatureSelector", "GenericUnivariateSelect", "FastICA", "NMF", "KernelPCA",
}
SELECTORS = {"SelectKBest", "SelectPercentile", "VarianceThreshold", "RFE", "RFECV", "SelectFromModel",
             "SequentialFeatureSelector", "GenericUnivariateSelect", "SelectFpr", "SelectFdr", "SelectFwe"}
SAMPLERS = {
    "SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE", "KMeansSMOTE", "SMOTENC", "SMOTEN", "RandomOverSampler",
    "RandomUnderSampler", "NearMiss", "TomekLinks", "EditedNearestNeighbours", "SMOTETomek", "SMOTEENN",
    "ClusterCentroids", "InstanceHardnessThreshold", "CondensedNearestNeighbour", "OneSidedSelection",
}
SCALERS = {"StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "QuantileTransformer",
           "PowerTransformer", "Normalizer", "scale", "minmax_scale", "robust_scale"}
DISTANCE = {"KMeans", "MiniBatchKMeans", "DBSCAN", "HDBSCAN", "KNeighborsClassifier", "KNeighborsRegressor",
            "SVC", "SVR", "AgglomerativeClustering", "LocalOutlierFactor", "SpectralClustering", "OneClassSVM",
            "KMedoids", "OPTICS"}
CLUSTERERS = {"KMeans", "MiniBatchKMeans", "GaussianMixture", "AgglomerativeClustering", "SpectralClustering",
              "KMedoids", "BayesianGaussianMixture"}
SPLIT_FUNCS = {"train_test_split", "cross_val_score", "cross_validate", "cross_val_predict",
               "learning_curve", "validation_curve"}
SPLITTERS = {"KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold", "TimeSeriesSplit",
             "ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit", "RepeatedStratifiedKFold",
             "RepeatedKFold", "LeaveOneGroupOut"}
RANDOM_CTORS = {
    "train_test_split", "ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit",
    "RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor", "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor", "DecisionTreeClassifier", "DecisionTreeRegressor", "RandomizedSearchCV",
    "HalvingRandomSearchCV", "KMeans", "MiniBatchKMeans", "IsolationForest", "GaussianMixture", "TSNE",
    "MLPClassifier", "MLPRegressor", "SGDClassifier", "SGDRegressor", "make_classification", "make_regression",
    "make_blobs", "make_moons", "permutation_importance", "BalancedRandomForestClassifier", "XGBClassifier",
    "XGBRegressor", "LGBMClassifier", "LGBMRegressor", "BaggingClassifier", "AdaBoostClassifier",
} | SAMPLERS - {"TomekLinks", "EditedNearestNeighbours", "NearMiss", "CondensedNearestNeighbour"}
SHUFFLE_SPLITTERS = {"KFold", "StratifiedKFold", "RepeatedKFold", "RepeatedStratifiedKFold"}
TEST_FUNCS = {"ttest_ind", "ttest_rel", "ttest_1samp", "mannwhitneyu", "wilcoxon", "kruskal", "f_oneway",
              "chi2_contingency", "chisquare", "fisher_exact", "pearsonr", "spearmanr", "kendalltau", "ks_2samp",
              "kstest", "binomtest", "proportions_ztest", "ttest_ind_from_stats", "anova_lm", "mcnemar",
              "friedmanchisquare"}
PARAMETRIC = {"ttest_ind", "ttest_rel", "ttest_1samp", "f_oneway", "pearsonr", "anova_lm", "ttest_ind_from_stats"}
LOADERS = {"read_csv", "read_parquet", "read_excel", "read_sql", "read_sql_query", "read_json", "read_feather",
           "read_table", "fetch_openml", "load_dataset"}
PLOTS = {"plot", "scatter", "bar", "barh", "hist", "boxplot", "violinplot", "lineplot", "scatterplot", "barplot",
         "histplot", "kdeplot", "fill_between", "step", "errorbar", "stackplot", "ecdfplot"}
EXPLAINERS = {"Explainer", "TreeExplainer", "KernelExplainer", "LinearExplainer", "DeepExplainer",
              "PermutationExplainer", "GradientExplainer"}
CALIBRATORS = {"CalibratedClassifierCV", "IsotonicRegression", "FrozenEstimator"}
ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "api_route", "websocket", "options", "head"}
PICKLE_LOADS = ("pickle.load", "pickle.loads", "joblib.load", "cloudpickle.load", "dill.load", "torch.load", "read_pickle")
PICKLE_DUMPS = ("pickle.dump", "joblib.dump", "cloudpickle.dump", "dill.dump")
LLM_CALLS = ("chat.completions.create", "messages.create", "messages.stream", "responses.create",
             "generate_content", "completions.create")
PROVIDER_HOSTS = ("api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com")

IMBALANCE_HINT = re.compile(r"imbalanc|prevalence|class_weight|scale_pos_weight|minority|rare (?:class|event)|"
                            r"SMOTE|ADASYN|OverSampler|UnderSampler|is_unbalance|ml-config imbalanced", re.I)
CAUSAL = re.compile(r"\b(causes|caused by|leads to|led to|results in|drives|driven by|because of)\b", re.I)
CAUSAL_DESIGN = re.compile(r"randomi[sz]|difference-in-differences|diff-in-diff|instrumental variable|"
                           r"propensity|regression discontinuity|causal design", re.I)
NEGATION = re.compile(r"\b(not|never|no|nor|without|cannot|isn't|aren't|doesn't)\b[^.]*$", re.I)
METRIC_WORD = (r"(?:accuracy|roc[- ]?auc|pr[- ]?auc|auc|f1(?:[- ]score)?|precision|recall|rmse|mae|mape|"
               r"r-squared|log ?loss|brier(?: score)?|mcc)")
METRIC_NUM = r"(?:\d{1,3}(?:\.\d+)?\s?%|0?\.\d+|1\.0+)"
LITERAL_METRIC = re.compile(rf"\b{METRIC_WORD}\b\s*(?:of|=|:|is|was|reached|scored)?\s*{METRIC_NUM}"
                            rf"|{METRIC_NUM}\s*{METRIC_WORD}\b|\b0\.XX\b|\bXX(?:\.X+)?\s?%", re.I)
COMPARATOR = re.compile(r"(>=|<=|>|<|at least|at most|above|below|minimum|maximum|target|floor|constraint)\s*$", re.I)
SECRET_TEXT = [
    re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"""(?i)\b(?:api_?key|secret|password|passwd|token|access_key)\b\s*[:=]\s*["'][^"'\s]{8,}["']"""),
]
SECRET_NAME = re.compile(r"(?i)^(?:.*_)?(api_?key|secret|secret_key|password|passwd|token|access_key|auth_token)$")
ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/](?:Users|Documents and Settings|home)[\\/]|/(?:home|Users)/[^/\s]+/)")
PII = re.compile(r"(?i)^(email|e_mail|phone|phone_number|ssn|social_security|full_name|first_name|last_name|"
                 r"address|street|passport|national_id|dob|date_of_birth|credit_card|iban)$")
ENTITY_ID = re.compile(r"(?i)^(customer|user|patient|account|client|member|subject|session|device|household|"
                       r"person|employee|store)_?id$")
IGNORE = re.compile(r"ml-ignore\s+([A-Z]\d{2}(?:\s*,\s*[A-Z]\d{2})*)(?:\s+(?!-->)(\S[^\n]*?))?\s*(?:-->)?\s*$")
SKIP_FILE = re.compile(r"ml-scan:\s*skip-file\s+\S+")
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "env", "__pycache__", ".ipynb_checkpoints", "dist", "build",
             ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages"}
WEB_EXT = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".vue", ".svelte"}


@dataclass
class Finding:
    rule: str
    severity: str
    path: str
    loc: str
    message: str
    line: int | None = None
    cell: int | None = field(default=None, repr=False)


@dataclass
class Cell:
    index: int
    kind: str
    source: str
    start: int = 0
    n_lines: int = 0
    exec_count: object = None
    output_bytes: int = 0


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return f"{dotted(node.func)}()"
    if isinstance(node, ast.Subscript):
        return f"{dotted(node.value)}[]"
    return "?"


def last(name: str) -> str:
    return name.rsplit(".", 1)[-1].replace("()", "")


def ctor_name(call: ast.Call) -> str:
    func = call.func
    while isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call):
        func = func.value.func
    return last(dotted(func))


def text_of(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # unparse can fail on exotic nodes; empty text means "no evidence"
        return ""


def kw(call: ast.Call, name: str) -> ast.AST | None:
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def is_const(node: ast.AST | None, value: object) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


class Doc:
    def __init__(self, path: Path, display: str, text: str):
        self.path, self.display, self.text = path, display, text
        self.cells: list[Cell] = []
        self.code_lines: list[str] = []
        self.markdown: list[tuple[str, int]] = []
        self.notebook = False
        self.percent = False
        self.tree: ast.Module | None = None
        suffix = path.suffix.lower()
        name = path.name.lower()
        if suffix == ".ipynb":
            self.kind = "python"
            self._load_ipynb()
        elif suffix == ".py":
            self.kind = "python"
            self._load_py()
        elif suffix in (".r",):
            self.kind = "r"
        elif suffix == ".sql":
            self.kind = "sql"
        elif name.startswith("dockerfile") or suffix == ".dockerfile":
            self.kind = "docker"
        elif name.startswith(".env") or suffix == ".env":
            self.kind = "env"
        elif suffix in WEB_EXT:
            self.kind = "web"
        elif suffix in (".yml", ".yaml"):
            self.kind = "yaml"
        elif suffix == ".md":
            self.kind = "prompt"
        else:
            self.kind = "other"
        self.code = "\n".join(self.code_lines)
        self.md_text = "\n".join(t for t, _ in self.markdown)
        if self.tree is not None:
            for node in ast.walk(self.tree):
                for child in ast.iter_child_nodes(node):
                    child._parent = node  # type: ignore[attr-defined]

    def _load_ipynb(self) -> None:
        self.notebook = True
        try:
            nb = json.loads(self.text)
        except json.JSONDecodeError:
            return
        body: list[ast.stmt] = []
        for i, c in enumerate(nb.get("cells", [])):
            src = c.get("source", "")
            src = "".join(src) if isinstance(src, list) else src
            kind = c.get("cell_type", "code")
            cell = Cell(i, kind, src)
            if kind == "code":
                lines = []
                for line in src.split("\n"):
                    stripped = line.lstrip()
                    if stripped.startswith(("%", "!")):
                        indent = line[: len(line) - len(stripped)]
                        line = f"{indent}pass  # {stripped}"
                    lines.append(line)
                cell.start = len(self.code_lines) + 1
                cell.n_lines = len(lines)
                cell.exec_count = c.get("execution_count")
                cell.output_bytes = len(json.dumps(c.get("outputs", [])))
                self.code_lines.extend(lines)
                try:
                    tree = ast.parse("\n".join(lines))
                    ast.increment_lineno(tree, cell.start - 1)
                    body.extend(tree.body)
                except SyntaxError:
                    pass
            elif kind == "markdown":
                self.markdown.append((src, i))
            self.cells.append(cell)
        self.tree = ast.Module(body=body, type_ignores=[])

    def _load_py(self) -> None:
        self.code_lines = self.text.split("\n")
        self.percent = any(line.startswith("# %%") for line in self.code_lines)
        if self.percent:
            idx, current, kind, start = 0, [], "code", 1
            for n, line in enumerate(self.code_lines + ["# %%"], start=1):
                if line.startswith("# %%"):
                    if current or idx:
                        src = "\n".join(current)
                        self.cells.append(Cell(idx, kind, src, start=start, n_lines=len(current)))
                        if kind == "markdown":
                            self.markdown.append((re.sub(r"(?m)^# ?", "", src), idx))
                        idx += 1
                    kind = "markdown" if "[markdown]" in line else "code"
                    current, start = [], n + 1
                else:
                    current.append(line)
        try:
            self.tree = ast.parse(self.text)
        except SyntaxError:
            self.tree = None

    def loc(self, line: int | None, cell: int | None = None) -> str:
        if line is None:
            return f"cell {cell}" if cell is not None else "file"
        if self.notebook:
            for c in self.cells:
                if c.kind == "code" and c.start <= line < c.start + c.n_lines:
                    return f"cell {c.index} line {line - c.start + 1}"
        return f"line {line}"

    def cell_of(self, line: int) -> int:
        for c in self.cells:
            if c.kind == "code" and c.start <= line < c.start + max(c.n_lines, 1):
                return c.index
        return 0

    def cell_text(self, line: int) -> str:
        if not self.cells:
            return self.code
        for c in self.cells:
            if c.kind == "code" and c.start <= line < c.start + max(c.n_lines, 1):
                return "\n".join(self.code_lines[c.start - 1: c.start - 1 + c.n_lines])
        return self.code

    def window(self, line: int, before: int = 1, after: int = 3) -> str:
        lo, hi = max(0, line - 1 - before), min(len(self.code_lines), line + after)
        return "\n".join(self.code_lines[lo:hi])

    def line_text(self, line: int) -> str:
        return self.code_lines[line - 1] if 0 < line <= len(self.code_lines) else ""


class Facts:
    """One AST pass that collects what the rules need."""

    def __init__(self, doc: Doc):
        self.calls: list[tuple[str, ast.Call]] = []
        self.assigns: dict[str, str] = {}
        self.imports: set[str] = set()
        self.strings: list[tuple[str, int]] = []
        self.funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.classes: list[ast.ClassDef] = []
        self.compares: list[ast.Compare] = []
        self.assign_nodes: list[ast.Assign] = []
        self.star_imports: list[ast.ImportFrom] = []
        tree = doc.tree
        if tree is None:
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self.calls.append((dotted(node.func), node))
            elif isinstance(node, ast.Assign):
                self.assign_nodes.append(node)
                if isinstance(node.value, ast.Call):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            self.assigns[t.id] = ctor_name(node.value)
            elif isinstance(node, ast.Import):
                self.imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                self.imports.add(mod)
                self.imports.update(f"{mod}.{a.name}" for a in node.names)
                if any(a.name == "*" for a in node.names):
                    self.star_imports.append(node)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.strings.append((node.value, getattr(node, "lineno", 0)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs.append(node)
            elif isinstance(node, ast.ClassDef):
                self.classes.append(node)
            elif isinstance(node, ast.Compare):
                self.compares.append(node)
        self.calls.sort(key=lambda c: c[1].lineno)

    def named(self, *names: str) -> list[ast.Call]:
        wanted = set(names)
        return [c for n, c in self.calls if last(n) in wanted]

    def receiver_class(self, call: ast.Call) -> str:
        func = call.func
        if not isinstance(func, ast.Attribute):
            return ""
        value = func.value
        if isinstance(value, ast.Name):
            return self.assigns.get(value.id, "")
        if isinstance(value, ast.Call):
            return ctor_name(value)
        return ""

    def ctors(self) -> set[str]:
        return {ctor_name(c) for _, c in self.calls}


def parents(node: ast.AST):
    node = getattr(node, "_parent", None)
    while node is not None:
        yield node
        node = getattr(node, "_parent", None)


def enclosing_func(node: ast.AST):
    for p in parents(node):
        if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return p
    return None


def is_route(fn: ast.AST) -> bool:
    for d in getattr(fn, "decorator_list", []):
        parts = dotted(d.func if isinstance(d, ast.Call) else d).split(".")
        if len(parts) >= 2 and parts[-1] in ROUTE_METHODS:
            return True
    return False


def first_arg(call: ast.Call, index: int = 0, name: str = "X") -> str:
    if len(call.args) > index:
        return text_of(call.args[index])
    return text_of(kw(call, name))


def heldout(text: str) -> bool:
    t = text.lower()
    if re.search(r"test|holdout", t):
        return True
    return bool(re.search(r"(?<![a-z])val(id|idation)?(?![a-z])", t)) and "train" not in t


def negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 40): start]
    return bool(NEGATION.search(prefix))


class Scanner:
    def __init__(self, imbalanced: bool = False, max_output_kb: int = 100):
        self.force_imbalanced = imbalanced
        self.max_output_kb = max_output_kb
        self.global_text = ""

    # ---------- entry points ----------
    def scan(self, docs: list[Doc]) -> list[Finding]:
        self.global_text = "\n".join(d.text for d in docs)
        out: list[Finding] = []
        for doc in docs:
            if SKIP_FILE.search(doc.text[:400]):
                continue
            found = list(self.scan_doc(doc))
            out.extend(self.apply_ignores(doc, found))
        return out

    def scan_doc(self, doc: Doc):
        self.doc = doc
        common = [self.r_secrets, self.r_unfinished_text]
        kind_rules = {
            "python": [self.r_python],
            "r": [self.r_abs_path_text],
            "sql": [self.r_abs_path_text],
            "docker": [self.r_docker],
            "yaml": [self.r_reload_text],
            "env": [self.r_env],
            "web": [self.r_web, self.r_abs_path_text],
            "prompt": [self.r_prompt],
            "other": [],
        }[doc.kind]
        if doc.kind == "prompt" and not self.is_prompt_file(doc):
            return
        for rule in common + kind_rules:
            yield from rule(doc)

    def f(self, rule: str, line: int | None = None, detail: str = "", cell: int | None = None) -> Finding:
        sev, title, fix = RULES[rule]
        msg = f"{title}{': ' + detail if detail else ''}. Fix: {fix}"
        return Finding(rule, sev, self.doc.display, self.doc.loc(line, cell), msg, line, cell)

    def apply_ignores(self, doc: Doc, found: list[Finding]) -> list[Finding]:
        kept = []
        for fd in found:
            if fd.line is not None:
                scope = [doc.line_text(fd.line), doc.line_text(fd.line - 1)]
                if doc.kind != "python":
                    raw = doc.text.split("\n")
                    scope = raw[max(0, fd.line - 2): fd.line]
            elif fd.cell is not None:
                scope = [t for t, i in doc.markdown if i == fd.cell]
            else:
                scope = doc.text.split("\n")
            decision = None
            for text in scope:
                for line in text.split("\n"):
                    m = IGNORE.search(line)
                    if m and fd.rule in re.split(r"\s*,\s*", m.group(1)):
                        decision = bool(m.group(2) and m.group(2).strip())
            if decision is True:
                continue
            if decision is False:
                fd.message += " (ml-ignore found without a reason, not applied)"
            kept.append(fd)
        return kept

    # ---------- shared helpers ----------
    def imbalanced(self, doc: Doc) -> bool:
        return self.force_imbalanced or bool(IMBALANCE_HINT.search(doc.code + "\n" + doc.md_text))

    def prose(self, doc: Doc, facts: Facts):
        """Markdown cells plus string literals of three or more words: (text, line, cell)."""
        for text, idx in doc.markdown:
            yield text, None, idx
        for s, line in facts.strings:
            if len(s.split()) >= 3:
                yield s, line, None

    @staticmethod
    def is_prompt_file(doc: Doc) -> bool:
        name = doc.path.name.lower()
        return name.endswith(".md") and "system" in name

    # ---------- text rules (all file types) ----------
    def r_secrets(self, doc: Doc):
        for n, line in enumerate(doc.text.split("\n"), start=1):
            for pat in SECRET_TEXT:
                if pat.search(line) and "os.environ" not in line and "getenv" not in line:
                    yield self.f("E02", self._line_for(doc, n), "literal credential in source")
                    break

    def _line_for(self, doc: Doc, raw_line: int) -> int | None:
        """Map a raw-text line to the code coordinate system (notebooks need the virtual index)."""
        if not doc.notebook:
            return raw_line
        needle = doc.text.split("\n")[raw_line - 1].strip().strip('",').replace('\\"', '"').replace("\\n", "")
        for i, line in enumerate(doc.code_lines, start=1):
            if needle and needle[:40] in line:
                return i
        return None

    def r_unfinished_text(self, doc: Doc):
        if doc.kind == "python":
            for n, line in enumerate(doc.code_lines, start=1):
                hash_at = line.find("#")
                if hash_at >= 0 and UNFINISHED.search(line[hash_at:]):
                    yield self.f("E08", n, "unfinished marker in a comment")
            for text, idx in doc.markdown:
                if UNFINISHED.search(text):
                    yield self.f("E08", None, "unfinished marker in Markdown", cell=idx)
        else:
            for n, line in enumerate(doc.text.split("\n"), start=1):
                if UNFINISHED.search(line):
                    yield self.f("E08", n, "unfinished marker")

    def r_abs_path_text(self, doc: Doc):
        for n, line in enumerate(doc.text.split("\n"), start=1):
            for m in re.finditer(r"""["']([^"']+)["']""", line):
                if ABS_PATH.search(m.group(1)):
                    yield self.f("E01", n, m.group(1)[:60])

    def r_reload_text(self, doc: Doc):
        for n, line in enumerate(doc.text.split("\n"), start=1):
            if "--reload" in line and not line.lstrip().startswith("#"):
                yield self.f("P08", n)

    def r_docker(self, doc: Doc):
        yield from self.r_reload_text(doc)
        users = re.findall(r"(?im)^\s*USER\s+(\S+)", doc.text)
        if not users or users[-1].lower() in ("root", "0"):
            yield self.f("P09", None, "no non-root USER instruction")
        if not re.search(r"(?im)^\s*HEALTHCHECK\b", doc.text):
            yield self.f("P13")

    def r_env(self, doc: Doc):
        for n, line in enumerate(doc.text.split("\n"), start=1):
            m = re.match(r"\s*([A-Za-z0-9_]+)\s*=\s*(.*)$", line)
            if not m:
                continue
            key, value = m.group(1), m.group(2).strip().strip('"').strip("'")
            if re.search(r"(?i)(api_?key|secret|password|passwd|access_key|auth_token|(^|_)token$)", key):
                if key.upper().startswith("VITE_"):
                    yield self.f("G04", n, f"{key} would be bundled into the browser")
                elif value:
                    yield self.f("E02", n, f"{key} has a value in a committed env file")

    def r_web(self, doc: Doc):
        for n, line in enumerate(doc.text.split("\n"), start=1):
            if re.search(r"VITE_[A-Z0-9_]*(API_?KEY|SECRET|PASSWORD|ACCESS_KEY|_TOKEN)", line):
                yield self.f("G04", n, "secret-like VITE_ variable")
            if re.search(r"""(from\s+|require\(\s*|import\(\s*)["'](openai|@anthropic-ai/sdk|@google/generative-ai)["']""", line):
                yield self.f("G04", n, "LLM SDK imported in front-end code")
            if "dangerouslyAllowBrowser" in line:
                yield self.f("G04", n, "browser-side LLM client")

    def r_prompt(self, doc: Doc):
        text = doc.text
        if not re.search(r"(?i)\bconstraints?\b", text):
            yield self.f("G01", None, "no CONSTRAINTS section")
        elif not re.search(r"(?i)caus", text):
            yield self.f("G01", None, "constraints do not forbid causal explanations")

    # ---------- Python rules ----------
    def r_python(self, doc: Doc):
        facts = Facts(doc)
        for group in (self.leakage, self.statistics, self.modeling, self.features, self.explain, self.data,
                      self.engineering, self.communication, self.production, self.genai):
            yield from group(doc, facts)

    def split_line(self, facts: Facts) -> int | None:
        lines = [c.lineno for n, c in facts.calls if last(n) in SPLIT_FUNCS]
        lines += [c.lineno for n, c in facts.calls if last(n) == "split" and facts.receiver_class(c) in SPLITTERS]
        lines += [c.lineno for n, c in facts.calls if last(n) == "fit" and facts.receiver_class(c).endswith("SearchCV")]
        return min(lines) if lines else None

    def leakage(self, doc: Doc, facts: Facts):
        split = self.split_line(facts)
        for name, call in facts.calls:
            method = last(name)
            cls = facts.receiver_class(call)
            arg = first_arg(call)
            if method in ("fit", "fit_transform") and cls in TRANSFORMERS and split and call.lineno < split:
                yield self.f("L01", call.lineno, f"{cls}.{method} at {doc.loc(call.lineno)}, split at {doc.loc(split)}")
            if method in ("fit", "fit_transform", "fit_resample", "partial_fit") and arg and heldout(arg):
                receiver = text_of(call.func.value).lower() if isinstance(call.func, ast.Attribute) else ""
                if cls not in CALIBRATORS and "calib" not in receiver:
                    rule = "L03" if method == "fit_resample" else "L02"
                    yield self.f(rule, call.lineno, f"{method}({arg})")
            if method == "fit_resample" and split and call.lineno < split:
                yield self.f("L03", call.lineno, "resampling happens before the split")
            if method == "shift":
                periods = call.args[0] if call.args else kw(call, "periods")
                if isinstance(periods, ast.UnaryOp) and isinstance(periods.op, ast.USub):
                    target = self._assign_target(call)
                    if not re.search(r"target|label|^y$|^y_|future|horizon", target, re.I):
                        yield self.f("L05", call.lineno, "negative shift reads future rows")
            if method == "rolling" and is_const(kw(call, "center"), True):
                yield self.f("L05", call.lineno, "centered window reads future rows")
            eval_set = kw(call, "eval_set")
            if eval_set is not None and "test" in text_of(eval_set).lower():
                yield self.f("L07", call.lineno)
        targets = self._target_names(facts)
        for node in (ast.walk(doc.tree) if doc.tree else []):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call) and last(dotted(node.value.func)) == "groupby":
                key = node.slice
                key_text = key.value if isinstance(key, ast.Constant) else text_of(key)
                if key_text in targets:
                    yield self.f("L06", node.lineno, f"groupby(...)[{key_text!r}]")
        for a in facts.assign_nodes:
            names = " ".join(text_of(t) for t in a.targets).lower()
            value = text_of(a.value).lower()
            if re.search(r"thresh|cutoff", names) and "test" in value and re.search(r"argmax|argmin|idxmax|idxmin|max\(|min\(|choose|select|optimal|best", value):
                yield self.f("L08", a.lineno)
        tts = facts.named("train_test_split")
        if tts and not re.search(r"duplicated|drop_duplicates", doc.code):
            yield self.f("L09", tts[0].lineno)
        if tts and any(ENTITY_ID.match(s) for s, _ in facts.strings):
            if not re.search(r"Group(K|Shuffle)|StratifiedGroupKFold|LeaveOneGroupOut|groups\s*=|is_unique|nunique", doc.code):
                yield self.f("L10", tts[0].lineno)

    @staticmethod
    def _assign_target(node: ast.AST) -> str:
        for p in parents(node):
            if isinstance(p, ast.Assign):
                return " ".join(text_of(t) for t in p.targets)
        return ""

    @staticmethod
    def _target_names(facts: Facts) -> set[str]:
        names: set[str] = set()
        for a in facts.assign_nodes:
            for t in a.targets:
                tname = text_of(t)
                if re.fullmatch(r"(?i)target(_col(umn)?)?|label(_col(umn)?)?", tname) and isinstance(a.value, ast.Constant):
                    names.add(str(a.value.value))
                    names.add(tname)
                if tname == "y" and isinstance(a.value, ast.Subscript) and isinstance(a.value.slice, ast.Constant):
                    names.add(str(a.value.slice.value))
        return names

    def statistics(self, doc: Doc, facts: Facts):
        tests = [c for n, c in facts.calls if last(n) in TEST_FUNCS]
        if tests:
            text = doc.code + "\n" + doc.md_text
            if not re.search(r"(?i)effect|cohen|hedges|cramer|odds_ratio|confidence|conf_int|bootstrap|\bci\b|_ci\b|\bci_|interval", text):
                yield self.f("S01", tests[0].lineno)
            para = [c for c in tests if last(dotted(c.func)) in PARAMETRIC]
            if para and not re.search(r"shapiro|normaltest|anderson|levene|bartlett|jarque_bera|probplot|qqplot", text):
                yield self.f("S02", para[0].lineno)
            looped = any(isinstance(p, (ast.For, ast.While, ast.comprehension, ast.ListComp, ast.DictComp, ast.GeneratorExp))
                         for c in tests for p in parents(c))
            if (len(tests) >= 3 or looped) and not re.search(r"(?i)multipletests|bonferroni|holm|fdr|benjamini", text):
                yield self.f("S03", tests[0].lineno)
        if any(i.startswith("statsmodels") for i in facts.imports) and re.search(r"\.summary\(|\.params\b", doc.code):
            if not re.search(r"variance_inflation_factor|het_breuschpagan|het_white|resid|influence|cooks|durbin", doc.code):
                yield self.f("S07", None, "statsmodels results read without diagnostics")
        if not CAUSAL_DESIGN.search(doc.code + doc.md_text):
            reported = 0
            for text, line, cell in self.prose(doc, facts):
                for m in CAUSAL.finditer(text):
                    if not negated(text, m.start()) and reported < 3:
                        reported += 1
                        yield self.f("S05", line, f"'{m.group(0)}'", cell=cell)

    def modeling(self, doc: Doc, facts: Facts):
        ctors = facts.ctors()
        code = doc.code
        estimators = {c for c in ctors if re.search(r"Classifier$|Regressor$|Regression$|^SVC$|^SVR$|^LinearSVC$|NB$|^XGB|^LGBM|^CatBoost", c)}
        fits = facts.named("fit")
        if estimators and fits and not re.search(r"(?i)dummy|baseline", code):
            yield self.f("M01", fits[0].lineno)
        acc = facts.named("accuracy_score")
        if acc and not re.search(r"precision|recall|f1|fbeta|roc_auc|average_precision|matthews|balanced_accuracy|log_loss|brier|classification_report|cohen_kappa|confusion_matrix", code):
            yield self.f("M02", acc[0].lineno)
        imb = self.imbalanced(doc)
        roc = [c for n, c in facts.calls if last(n) in ("roc_auc_score", "roc_curve") or "RocCurveDisplay" in n
               or is_const(kw(c, "scoring"), "roc_auc")]
        if imb and roc and not re.search(r"average_precision|precision_recall_curve|PrecisionRecallDisplay|pr_auc", code):
            yield self.f("M03", roc[0].lineno)
        for cmp in facts.compares:
            sides = [cmp.left, *cmp.comparators]
            if any(is_const(s, 0.5) for s in sides) and any(isinstance(o, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)) for o in cmp.ops):
                other = " ".join(text_of(s) for s in sides if not is_const(s, 0.5)).lower()
                if re.search(r"proba|prob|score|pred|p_", other):
                    yield self.f("M04", cmp.lineno)
        for a in facts.assign_nodes:
            if is_const(a.value, 0.5) and any(re.search(r"(?i)thresh|cutoff", text_of(t)) for t in a.targets):
                yield self.f("M04", a.lineno)
        dated = re.search(r"to_datetime|parse_dates|datetime64|\.dt\.", code)
        for call in facts.named("train_test_split"):
            if dated and not is_const(kw(call, "shuffle"), False):
                yield self.f("M05", call.lineno)
        for name, call in facts.calls:
            cname = last(name)
            if cname in RANDOM_CTORS and kw(call, "random_state") is None and kw(call, "seed") is None:
                if cname == "train_test_split" and is_const(kw(call, "shuffle"), False):
                    continue
                yield self.f("M06", call.lineno, cname)
            if cname in SHUFFLE_SPLITTERS and is_const(kw(call, "shuffle"), True) and kw(call, "random_state") is None:
                yield self.f("M06", call.lineno, cname)
            if cname == "default_rng" and not call.args and not call.keywords:
                yield self.f("M06", call.lineno, "default_rng() without a seed")
        classification = any(re.search(r"Classifier$|LogisticRegression|SVC$|NB$", c) for c in ctors) or "predict_proba" in code
        if classification:
            for call in facts.named("train_test_split"):
                if kw(call, "stratify") is None and not is_const(kw(call, "shuffle"), False):
                    yield self.f("M07", call.lineno)
        if estimators and fits and not re.search(r"(?i)cross_val|cross_validate|KFold|bootstrap|resample\(|confidence|_ci\b|\bci_|interval|SearchCV|wilson", code):
            yield self.f("M08", fits[0].lineno)
        samplers = [c for n, c in facts.calls if last(n) in SAMPLERS]
        if samplers and not any(i.startswith("imblearn.pipeline") for i in facts.imports):
            yield self.f("M09", samplers[0].lineno)
        if not re.search(r"SearchCV|optuna|Halving|hyperopt|skopt|BayesSearch", code):
            for text, line, cell in self.prose(doc, facts):
                m = re.search(r"(?i)\btuned\b|\boptimi[sz]ed (hyper)?parameters\b|\bbest hyperparameters\b", text)
                if m and not negated(text, m.start()):
                    yield self.f("M10", line, cell=cell)
                    break
        distance = [c for n, c in facts.calls if last(n) in DISTANCE]
        if distance and not (ctors & SCALERS):
            yield self.f("M11", distance[0].lineno, ctor_name(distance[0]))
        clusters = [c for n, c in facts.calls if last(n) in CLUSTERERS]
        if clusters and not re.search(r"(?i)silhouette|inertia_|elbow|gap_stat|calinski|davies|\.bic\(|\.aic\(|stability", code):
            yield self.f("M12", clusters[0].lineno)
        reweighted = samplers or [c for _, c in facts.calls if (kw(c, "class_weight") is not None and not is_const(kw(c, "class_weight"), None)) or kw(c, "scale_pos_weight") is not None or kw(c, "is_unbalance") is not None]
        if reweighted and not re.search(r"(?i)calibrat|brier", code):
            yield self.f("M13", reweighted[0].lineno)

    def features(self, doc: Doc, facts: Facts):
        code = doc.code
        new_cols = []
        for a in facts.assign_nodes:
            for t in a.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str)                         and isinstance(t.value, ast.Name) and re.match(r"^(df|data|frame|X|train|test)\w*$|\w*_df$", t.value.id):
                    new_cols.append(a.lineno)
        new_cols += [c.lineno for c in facts.named("assign") for _ in c.keywords]
        if len(new_cols) >= 3 and not re.search(r"(?i)feature[_ ]dictionary|feature_dict", code + doc.md_text):
            yield self.f("F01", min(new_cols), f"{len(new_cols)} engineered columns")
        for name, call in facts.calls:
            cls = facts.receiver_class(call)
            if last(name) in ("fit", "fit_transform") and cls in SELECTORS and "train" not in first_arg(call).lower():
                yield self.f("F02", call.lineno, cls)
        if any(i.startswith("category_encoders") for i in facts.imports) and re.search(r"TargetEncoder|LeaveOneOutEncoder|MEstimateEncoder", code):
            yield self.f("F03", None, "category_encoders target encoders are fitted in-sample")
        grid = facts.named("ParameterGrid", "ParameterSampler")
        if grid and not re.search(r"cross_val|cross_validate|KFold", code):
            yield self.f("F04", grid[0].lineno)
        for fn in facts.funcs:
            body = text_of(fn)
            if re.search(r"\.suggest_\w+\(", body) and re.search(r"test", body, re.I):
                yield self.f("F05", fn.lineno, f"objective {fn.name}()")
        if "best_score_" in code and "cv_results_" not in code:
            yield self.f("F06", next(i for i, l in enumerate(doc.code_lines, 1) if "best_score_" in l))
        if self.imbalanced(doc):
            for _, call in facts.calls:
                if is_const(kw(call, "scoring"), "accuracy"):
                    yield self.f("F07", call.lineno)

    def explain(self, doc: Doc, facts: Facts):
        code = doc.code
        perm = facts.named("permutation_importance")
        if perm:
            args = [first_arg(c, 1, "X").lower() for c in perm]
            if any("train" in a for a in args) and not any(heldout(a) for a in args):
                yield self.f("X01", perm[0].lineno, "permutation importance on training rows")
        shap_calls = [c for n, c in facts.calls if last(n) == "shap_values" or (isinstance(c.func, ast.Name) and facts.assigns.get(c.func.id) in EXPLAINERS)]
        if shap_calls:
            args = [first_arg(c).lower() for c in shap_calls]
            if any("train" in a for a in args) and not any(heldout(a) for a in args):
                yield self.f("X01", shap_calls[0].lineno, "SHAP on training rows")
        if "feature_importances_" in code and "permutation_importance" not in code:
            yield self.f("X02", next(i for i, l in enumerate(doc.code_lines, 1) if "feature_importances_" in l))
        if "shap" in facts.imports and not re.search(r"(?i)correlat", code + doc.md_text):
            yield self.f("X03", None)
        for text, line, cell in self.prose(doc, facts):
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                m = CAUSAL.search(sentence)
                if m and re.search(r"(?i)importan|shap|attribution|contribut", sentence) and not negated(sentence, m.start()):
                    yield self.f("X04", line, cell=cell)
                    break

    def data(self, doc: Doc, facts: Facts):
        code = doc.code
        for name, call in facts.calls:
            method = last(name)
            parent = getattr(call, "_parent", None)
            reassigns = isinstance(parent, ast.Assign) and isinstance(call.func, ast.Attribute) and                 isinstance(call.func.value, ast.Name) and any(text_of(t) == call.func.value.id for t in parent.targets)
            if method in ("dropna", "drop_duplicates") and (reassigns or is_const(kw(call, "inplace"), True)):
                if not re.search(r"shape|len\(|assert|print\(|rows|count", doc.window(call.lineno, 1, 3)):
                    yield self.f("D01", call.lineno, method)
            if method == "merge" or (method == "join" and (kw(call, "on") is not None or kw(call, "how") is not None)):
                if kw(call, "validate") is None and not re.search(r"assert|shape|len\(", doc.window(call.lineno, 1, 3)):
                    yield self.f("D02", call.lineno, method)
        loads = [c for n, c in facts.calls if last(n) in LOADERS]
        if loads:
            if not re.search(r"isna|isnull|notna|missing", code):
                yield self.f("D04", loads[0].lineno)
            if not re.search(r"assert|\.dtypes|validate|schema|pandera|great_expectations|\.between\(|\.describe\(", code):
                yield self.f("D06", loads[0].lineno)
        for s, line in facts.strings:
            if PII.match(s) and re.search(r"print\(|display\(|\.head\(|\.to_csv\(|\.sample\(", doc.line_text(line)):
                yield self.f("D05", line, s)

    def engineering(self, doc: Doc, facts: Facts):
        for s, line in facts.strings:
            if ABS_PATH.search(s):
                yield self.f("E01", line, s[:60])
        for a in facts.assign_nodes:
            for t in a.targets:
                if isinstance(t, ast.Name) and SECRET_NAME.match(t.id) and isinstance(a.value, ast.Constant) \
                        and isinstance(a.value.value, str) and len(a.value.value) >= 8 and " " not in a.value.value:
                    yield self.f("E02", a.lineno, t.id)
                if isinstance(t, ast.Subscript) and text_of(t.value) == "os.environ" and isinstance(a.value, ast.Constant) \
                        and re.search(r"(?i)key|secret|token|password", text_of(t.slice)) and a.value.value:
                    yield self.f("E03", a.lineno)
        for call in facts.named("setdefault", "putenv"):
            if "environ" in text_of(call.func) or last(dotted(call.func)) == "putenv":
                if len(call.args) == 2 and isinstance(call.args[1], ast.Constant) and re.search(r"(?i)key|secret|token|password", text_of(call.args[0])):
                    yield self.f("E03", call.lineno)
        for n, line in enumerate(doc.code_lines, start=1):
            if re.search(r"%env\s+\w*(KEY|SECRET|TOKEN|PASSWORD)\w*\s*=?\s*\S+", line, re.I):
                yield self.f("E03", n, "%env magic")
        for node in facts.star_imports:
            yield self.f("E04", node.lineno, f"from {node.module} import *")
        for call in facts.named("filterwarnings", "simplefilter"):
            if call.args and is_const(call.args[0], "ignore") and len(call.args) == 1 and not call.keywords:
                yield self.f("E05", call.lineno)
        if doc.notebook:
            counts = [c.exec_count for c in doc.cells if c.kind == "code" and c.source.strip()]
            ints = [c for c in counts if isinstance(c, int)]
            if ints and len(ints) < len(counts):
                yield self.f("E06", None, "some code cells were never executed")
            if any(b <= a for a, b in zip(ints, ints[1:])):
                yield self.f("E06", None, "execution counts are out of order")
            for c in doc.cells:
                if c.output_bytes > self.max_output_kb * 1024:
                    yield self.f("E07", c.start or None, f"{c.output_bytes // 1024} KB in cell {c.index}", cell=c.index)
        for fn in facts.funcs:
            body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
            decorators = " ".join(text_of(d) for d in fn.decorator_list)
            stubby = len(body) == 1 and (isinstance(body[0], ast.Pass) or (isinstance(body[0], ast.Expr) and is_const(body[0].value, Ellipsis))
                                         or (isinstance(body[0], ast.Raise) and "NotImplementedError" in text_of(body[0])))
            if stubby and not re.search(r"abstractmethod|overload", decorators):
                owner = next((p for p in parents(fn) if isinstance(p, ast.ClassDef)), None)
                if owner is None or not re.search(r"Protocol|ABC", " ".join(text_of(b) for b in owner.bases)):
                    yield self.f("E08", fn.lineno, f"stub function {fn.name}()")
        pip_lines = doc.code_lines if doc.notebook else doc.text.split("\n")
        for n, line in enumerate(pip_lines, start=1):
            m = re.match(r"\s*(?:pass\s+#\s*|#\s*)?[%!]\s*pip3?\s+install\s+(.*)$", line)
            if m:
                tokens, skip = m.group(1).split(), False
                for tok in tokens:
                    if skip:
                        skip = False
                        continue
                    if tok in ("-r", "-e", "-c", "--requirement", "--constraint", "--index-url", "-i", "--extra-index-url", "-f", "--find-links"):
                        skip = True
                        continue
                    if tok.startswith("-") or tok.startswith("#"):
                        if tok.startswith("#"):
                            break
                        continue
                    if not re.search(r"==|@|\.whl$|\.tar\.gz$|^git\+|^https?:|^\.|^/|^\$|^\{", tok):
                        yield self.f("E09", n, tok)
                        break
        frames = {k for k, v in facts.assigns.items() if v in LOADERS}
        for call in facts.named("print", "display"):
            if len(call.args) == 1:
                arg = call.args[0]
                if isinstance(arg, ast.Name) and (arg.id in frames or re.match(r"^(df|frame)\w*$|^data$|\w*_df$", arg.id)):
                    yield self.f("E10", call.lineno, arg.id)
                elif isinstance(arg, ast.Call) and last(dotted(arg.func)) == "to_string" and isinstance(arg.func, ast.Attribute)                         and isinstance(arg.func.value, ast.Name):
                    root = arg.func.value.id
                    if root in frames or re.match(r"^(df|frame)\w*$|^data$|\w*_df$", root):
                        yield self.f("E10", call.lineno, f"{root}.to_string()")
        for call in facts.named("set_option"):
            if len(call.args) >= 2 and isinstance(call.args[0], ast.Constant) and "max_rows" in str(call.args[0].value) and is_const(call.args[1], None):
                yield self.f("E10", call.lineno, "display.max_rows=None")

    def communication(self, doc: Doc, facts: Facts):
        for text, line, cell in self.prose(doc, facts):
            for m in LITERAL_METRIC.finditer(text):
                if not COMPARATOR.search(text[max(0, m.start() - 15): m.start()]):
                    yield self.f("C01", line, f"'{m.group(0).strip()}'", cell=cell)
                    break
        for a in facts.assign_nodes:
            value = a.value
            if isinstance(value, ast.Constant) and isinstance(value.value, float) and 0 < value.value < 1:
                for t in a.targets:
                    name = text_of(t).lower()
                    if re.search(r"accuracy|auc|f1|precision|recall|rmse|mae|r2|brier|mcc", name) and not re.search(r"min|max|target|thresh|floor|goal|required|alpha|beta|weight", name):
                        yield self.f("C01", a.lineno, f"{text_of(t)} = {value.value}")
            if isinstance(value, ast.Dict):
                for k, v in zip(value.keys, value.values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str) and re.fullmatch(r"(?i)accuracy|auc|roc_auc|pr_auc|f1|precision|recall|rmse|mae|r2", k.value) \
                            and isinstance(v, ast.Constant) and isinstance(v.value, float):
                        yield self.f("C01", a.lineno, f"{k.value}: {v.value}")
        plotted: dict[int, int] = {}
        for name, call in facts.calls:
            if last(name) in PLOTS and "Display" not in name + facts.receiver_class(call):
                plotted.setdefault(doc.cell_of(call.lineno), call.lineno)
        for cell, line in plotted.items():
            scope = doc.cell_text(line) if doc.cells else doc.code
            if not re.search(r"xlabel|ylabel|\.set\(|labels\s*=|xaxis_title|label_axes", scope):
                yield self.f("C02", line)
        for name, call in facts.calls:
            method = last(name)
            if method in ("bar", "barh"):
                scope = doc.cell_text(call.lineno) if doc.cells else doc.code
                m = re.search(r"[xy]lim\(\s*(?:bottom\s*=\s*|left\s*=\s*)?([0-9]+(?:\.[0-9]+)?)", scope)
                if m and float(m.group(1)) != 0.0:
                    yield self.f("C03", call.lineno, "bar axis does not start at zero")
            if method in ("twinx", "twiny"):
                yield self.f("C03", call.lineno, "dual axis")
            if is_const(kw(call, "projection"), "3d"):
                yield self.f("C03", call.lineno, "3D projection")
        if (doc.notebook or doc.percent) and not re.search(r"(?i)limitation", doc.md_text + doc.code):
            yield self.f("C04", None)
        if any(i.startswith("ipywidgets") for i in facts.imports) and not re.search(r"(?i)fallback|static", doc.code + doc.md_text):
            yield self.f("C05", None)

    def production(self, doc: Doc, facts: Facts):
        code = doc.code
        loads = [c for n, c in facts.calls if any(n.endswith(p) for p in PICKLE_LOADS)]
        dumps = [c for n, c in facts.calls if any(n.endswith(p) for p in PICKLE_DUMPS)]
        for call in loads:
            if re.search(r"request|upload|\.file\b|form|query|body|user_|input\(", text_of(call), re.I) or "UploadFile" in code:
                yield self.f("P01", call.lineno)
            fn = enclosing_func(call)
            if fn is not None and is_route(fn):
                yield self.f("P03", call.lineno, f"inside {fn.name}()")
        if loads and not re.search(r"(?i)sha256|hashlib|checksum|verify", code):
            yield self.f("P02", loads[0].lineno)
        if len(dumps) >= 2 or any(c.args and isinstance(c.args[0], ast.Name) and facts.assigns.get(c.args[0].id) in TRANSFORMERS for c in dumps):
            yield self.f("P04", dumps[0].lineno)
        if dumps:
            for cls in facts.classes:
                if re.search(r"BaseEstimator|TransformerMixin|ClassifierMixin|RegressorMixin", " ".join(text_of(b) for b in cls.bases)):
                    yield self.f("P05", cls.lineno, cls.name)
            if not re.search(r"allclose|array_equal|assert_array|parity|assert_frame_equal", self.global_text):
                yield self.f("P11", dumps[0].lineno)
            if "metadata" not in code.lower():
                yield self.f("P12", dumps[0].lineno)
        for fn in facts.funcs:
            if isinstance(fn, ast.AsyncFunctionDef) and is_route(fn):
                for node in ast.walk(fn):
                    if isinstance(node, ast.Call) and last(dotted(node.func)) in ("predict", "predict_proba", "decision_function", "transform", "fit", "score_samples"):
                        wrapped = any(isinstance(p, ast.Call) and last(dotted(p.func)) in ("run_in_threadpool", "to_thread", "run_in_executor", "run_sync") for p in parents(node))
                        if not wrapped:
                            yield self.f("P06", node.lineno, f"{fn.name}()")
                            break
        for name, call in facts.calls:
            if last(name) in ("add_middleware", "CORSMiddleware"):
                origins = text_of(kw(call, "allow_origins"))
                if "'*'" in origins or '"*"' in origins:
                    if is_const(kw(call, "allow_credentials"), True):
                        yield self.f("P07", call.lineno)
            if name.endswith("uvicorn.run") and is_const(kw(call, "reload"), True):
                yield self.f("P08", call.lineno, "uvicorn.run(reload=True)")
        if facts.named("FastAPI") and "health" not in self.global_text.lower():
            yield self.f("P10", facts.named("FastAPI")[0].lineno)

    def genai(self, doc: Doc, facts: Facts):
        code = doc.code
        provider_http = any(h in code for h in PROVIDER_HOSTS)
        llm = [c for n, c in facts.calls if any(n.endswith(p) for p in LLM_CALLS) or (provider_http and last(n) in ("post", "stream"))]
        for cls in facts.classes:
            if re.search(r"BaseModel", " ".join(text_of(b) for b in cls.bases)) and re.search(r"Request|Input|Payload|Body|Query", cls.name):
                for stmt in cls.body:
                    if isinstance(stmt, ast.AnnAssign) and re.search(r"(?i)shap|contribution|attribution|explanation_values", text_of(stmt.target)):
                        yield self.f("G02", stmt.lineno, f"{cls.name}.{text_of(stmt.target)}")
        if llm:
            if "ground" not in self.global_text.lower():
                yield self.f("G03", llm[0].lineno)
            if not (re.search(r"timeout", code) and re.search(r"max_tokens|max_output_tokens|max_completion_tokens", code)):
                yield self.f("G05", llm[0].lineno)
            for call in llm:
                in_try = any(isinstance(p, ast.Try) for p in parents(call))
                if not in_try and "fallback" not in code.lower():
                    yield self.f("G06", call.lineno)
                    break
            for call in llm:
                model = kw(call, "model")
                if isinstance(model, ast.Constant) and isinstance(model.value, str):
                    yield self.f("G08", call.lineno, model.value)
        user_names = re.compile(r"(?i)question|user|message|query|input|text|prompt_text|chat")

        def dynamic_user(node: ast.AST | None) -> bool:
            if isinstance(node, (ast.JoinedStr, ast.BinOp)) or (isinstance(node, ast.Call) and last(dotted(node.func)) == "format"):
                return any(isinstance(n, ast.Name) and user_names.search(n.id) or isinstance(n, ast.Attribute) and user_names.search(n.attr) for n in ast.walk(node))
            return False

        for _, call in facts.calls:
            if dynamic_user(kw(call, "system")):
                yield self.f("G07", call.lineno)
        for node in (ast.walk(doc.tree) if doc.tree else []):
            if isinstance(node, ast.Dict):
                pairs = {k.value: v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)}
                if is_const(pairs.get("role"), "system") and dynamic_user(pairs.get("content")):
                    yield self.f("G07", node.lineno)
        for s, line in facts.strings:
            if len(s) > 200 and re.search(r"\bYou are\b", s) and not (re.search(r"(?i)constraint", s) and re.search(r"(?i)caus", s)):
                yield self.f("G01", line, "inline system prompt")


def collect(paths: list[str]) -> list[Path]:
    exts = {".py", ".ipynb", ".r", ".sql", ".md", ".yml", ".yaml", ".env", ".dockerfile"} | WEB_EXT
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            files.append(p)
            continue
        if not p.is_dir():
            continue
        for f in sorted(p.rglob("*")):
            if not f.is_file() or any(part in SKIP_DIRS for part in f.relative_to(p).parts[:-1]):
                continue
            name = f.name.lower()
            if f.suffix.lower() in exts or name.startswith("dockerfile") or name.startswith(".env"):
                if f.suffix.lower() == ".md" and "system" not in name:
                    continue
                if f.suffix.lower() in (".yml", ".yaml") and "compose" not in name:
                    continue
                if f.name.endswith(".d.ts") or ".min." in f.name:
                    continue
                files.append(f)
    return files


def score(findings: list[Finding]) -> int:
    errors = sum(1 for f in findings if f.severity == "error")
    warn_rules = {f.rule for f in findings if f.severity == "warning"}
    return max(0, 100 - 20 * errors - 6 * len(warn_rules))


def run(paths: list[str], imbalanced: bool = False, max_output_kb: int = 100) -> list[Finding]:
    docs = []
    for f in collect(paths):
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        docs.append(Doc(f, f.as_posix(), text))
    return Scanner(imbalanced, max_output_kb).scan(docs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scan notebooks, scripts, services and prompts for analysis and production slop.")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--format", choices=("text", "json"), default="text")
    ap.add_argument("--imbalanced", action="store_true", help="treat the target as imbalanced in every file")
    ap.add_argument("--fail-on", choices=("error", "warning"), default="error")
    ap.add_argument("--max-output-kb", type=int, default=100)
    ap.add_argument("--quiet", action="store_true", help="print only the summary line")
    ap.add_argument("--list-rules", action="store_true", help="print the rule table and exit")
    args = ap.parse_args(argv)
    if args.list_rules:
        for rid, (sev, title, _) in RULES.items():
            print(f"{rid}  {sev:<7}  {title}")
        return 0
    if not args.paths:
        ap.error("at least one path is required")
    findings = run(args.paths, args.imbalanced, args.max_output_kb)
    errors = sum(1 for f in findings if f.severity == "error")
    warnings_ = len(findings) - errors
    total = score(findings)
    if args.format == "json":
        print(json.dumps({"score": total, "errors": errors, "warnings": warnings_,
                          "findings": [{k: v for k, v in asdict(f).items() if k != "cell"} for f in findings]}, indent=2))
    else:
        if not args.quiet:
            current = None
            for f in sorted(findings, key=lambda x: (x.path, x.line or 0)):
                if f.path != current:
                    current = f.path
                    print(current)
                print(f"  {f.loc:<18} {f.severity:<7} {f.rule}  {f.message}")
        print(f"Rigor score {total}/100: {errors} errors, {warnings_} warnings")
    failing = errors or (args.fail_on == "warning" and warnings_)
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
