# ml-scan: skip-file this module holds deliberately sloppy snippets as test inputs
"""Rule tests for scripts/ml_scan.py. Standard library only: python -m unittest discover tests."""
from __future__ import annotations

import json
import re
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ml_scan  # noqa: E402


def nb(cells: list[tuple[str, str, int | None]], outputs: str = "") -> str:
    """Build a minimal notebook: cells are (kind, source, execution_count)."""
    out = []
    for kind, src, count in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update(execution_count=count, outputs=[{"output_type": "stream", "name": "stdout", "text": outputs}] if outputs else [])
        out.append(cell)
    return json.dumps({"cells": out, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


SK = "sk-" + "a1B2c3D4" * 4  # built at run time so no key-shaped literal exists on disk

CASES: dict[str, tuple[str, str]] = {
    "L01": ("a.py", """
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=0)
    """),
    "L02": ("a.py", "model.fit(X_test, y_test)"),
    "L03": ("a.py", """
        X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
        X_train, X_test = train_test_split(X_res, random_state=0)
    """),
    "L05": ("a.py", 'df["lag_sales"] = df["sales"].shift(-1)'),
    "L06": ("a.py", """
        TARGET = "churn"
        df["plan_rate"] = df.groupby("plan")[TARGET].transform("mean")
    """),
    "L07": ("a.py", "model.fit(X_train, y_train, eval_set=[(X_test, y_test)])"),
    "L08": ("a.py", "best_threshold = thresholds[np.argmax(f1_test)]"),
    "L09": ("a.py", "X_train, X_test = train_test_split(X, random_state=0)"),
    "L10": ("a.py", """
        ids = df["customer_id"]
        X_train, X_test = train_test_split(X, random_state=0)
    """),
    "S01": ("a.py", "t, p = stats.ttest_ind(a, b, equal_var=False)"),
    "S02": ("a.py", "t, p = stats.ttest_ind(a, b, equal_var=False)"),
    "S03": ("a.py", """
        for col in cols:
            stats.mannwhitneyu(a[col], b[col])
    """),
    "S05": ("a.py", 'print("Higher spend causes churn in this sample")'),
    "S07": ("a.py", """
        import statsmodels.api as sm
        res = sm.OLS(y, X).fit()
        print(res.summary())
    """),
    "M01": ("a.py", """
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(random_state=0)
        m.fit(X_train, y_train)
    """),
    "M02": ("a.py", "print(accuracy_score(y_test, pred))"),
    "M03": ("a.py", """
        # rare positives, prevalence near two percent
        print(roc_auc_score(y_test, p))
    """),
    "M04": ("a.py", "pred = (proba > 0.5).astype(int)"),
    "M05": ("a.py", """
        df["d"] = pd.to_datetime(df["d"])
        X_train, X_test = train_test_split(X, random_state=0)
    """),
    "M06": ("a.py", "parts = train_test_split(X, y)"),
    "M07": ("a.py", """
        LogisticRegression().fit(X_train, y_train)
        parts = train_test_split(X, y, random_state=0)
    """),
    "M08": ("a.py", """
        m = LogisticRegression()
        m.fit(X_train, y_train)
    """),
    "M09": ("a.py", """
        from imblearn.over_sampling import SMOTE
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("s", SMOTE(random_state=0)), ("m", model)])
    """),
    "M10": ("a.py", 'print("The final model uses tuned hyperparameters throughout")'),
    "M11": ("a.py", "KMeans(n_clusters=3, random_state=0).fit(X)"),
    "M12": ("a.py", "KMeans(n_clusters=3, random_state=0).fit(X)"),
    "M13": ("a.py", 'LogisticRegression(class_weight="balanced").fit(X_train, y_train)'),
    "F01": ("a.py", """
        df["a"] = df.x / df.y
        df["b"] = df.x * 2
        df["c"] = df.y - 1
    """),
    "F02": ("a.py", "Xs = SelectKBest(k=5).fit_transform(X, y)"),
    "F03": ("a.py", """
        import category_encoders as ce
        enc = ce.TargetEncoder(cols=["city"])
    """),
    "F04": ("a.py", """
        for params in ParameterGrid({"C": [1, 10]}):
            model.set_params(**params)
    """),
    "F05": ("a.py", """
        def objective(trial):
            c = trial.suggest_float("C", 0.01, 10, log=True)
            return evaluate(c, X_test, y_test)
    """),
    "F06": ("a.py", "print(search.best_score_)"),
    "F07": ("a.py", """
        # imbalanced target
        search = GridSearchCV(model, grid, scoring="accuracy")
    """),
    "X01": ("a.py", "r = permutation_importance(model, X_train, y_train, random_state=0)"),
    "X02": ("a.py", "print(model.feature_importances_)"),
    "X03": ("a.py", """
        import shap
        explainer = shap.TreeExplainer(model)
    """),
    "X04": ("a.py", 'print("The SHAP plot shows tenure drives cancellations.")'),
    "D01": ("a.py", """
        df = df.dropna()
        means = df.mean()
    """),
    "D02": ("a.py", 'out = left.merge(right, on="id")'),
    "D04": ("a.py", """
        df = pd.read_csv("data/raw.csv")
        assert set(df.columns) >= {"a"}
    """),
    "D05": ("a.py", 'print(df["email"].head())'),
    "D06": ("a.py", """
        df = pd.read_csv("data/raw.csv")
        print(df.isna().mean())
    """),
    "E01": ("a.py", 'df = pd.read_csv("C:/Users/alex/data.csv")'),
    "E02": ("a.py", 'password = "hunter2hunter2"'),
    "E03": ("a.py", 'os.environ["OPENAI_API_KEY"] = "abc123def456"'),
    "E04": ("a.py", "from numpy import *"),
    "E05": ("a.py", 'warnings.filterwarnings("ignore")'),
    "E06": ("a.ipynb", nb([("code", "x = 1", 2), ("code", "y = 2", 1)])),
    "E07": ("a.ipynb", nb([("code", "print(1)", 1)], outputs="z" * 150_000)),
    "E08": ("a.py", """
        def train():
            pass
    """),
    "E09": ("a.py", "# %pip install scikit-learn"),
    "E10": ("a.py", "print(df)"),
    "C01": ("a.py", 'print("Final model accuracy: 0.97 on the holdout")'),
    "C02": ("a.py", """
        plt.plot([1, 2], [3, 4])
        plt.title("Trend")
    """),
    "C03": ("a.py", "ax2 = ax.twinx()"),
    "C04": ("a.ipynb", nb([("markdown", "# Result", None), ("code", "x = 1", None)])),
    "C05": ("a.py", """
        import ipywidgets as widgets
        w = widgets.IntSlider()
    """),
    "P01": ("a.py", "model = joblib.load(upload.file)"),
    "P02": ("a.py", """
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)
    """),
    "P03": ("a.py", """
        @app.get("/predict")
        def predict():
            model = joblib.load("models/model.pkl")
            return {}
    """),
    "P04": ("a.py", """
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(model, "model.pkl")
    """),
    "P05": ("a.py", """
        class Clip(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                return X
        pickle.dump(Clip(), f)
    """),
    "P06": ("a.py", """
        @app.post("/predict")
        async def predict(req: Req):
            return {"p": model.predict_proba(X)[0, 1]}
    """),
    "P07": ("a.py", 'app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True)'),
    "P08": ("Dockerfile", 'FROM python:3.11-slim\nUSER app\nHEALTHCHECK CMD true\nCMD ["uvicorn", "app.main:app", "--reload"]'),
    "P09": ("Dockerfile", 'FROM python:3.11-slim\nHEALTHCHECK CMD true\nCMD ["uvicorn", "app.main:app"]'),
    "P10": ("a.py", "app = FastAPI()"),
    "P11": ("a.py", """
        json.dump(metadata, g)
        pickle.dump(pipe, f)
    """),
    "P12": ("a.py", "pickle.dump(pipe, f)"),
    "P13": ("Dockerfile", 'FROM python:3.11-slim\nUSER app\nCMD ["uvicorn", "app.main:app"]'),
    "G01": ("rtcc_system.md", "ROLE\nYou are helpful.\nTASK\nExplain the prediction."),
    "G02": ("a.py", """
        class ExplainRequest(BaseModel):
            shap_values: list[float]
    """),
    "G03": ("a.py", """
        try:
            r = client.chat.completions.create(model=cfg.model, messages=m, timeout=10, max_tokens=200)
        except TimeoutError:
            r = None
    """),
    "G04": ("api.ts", "const key = import.meta.env.VITE_OPENAI_API_KEY;"),
    "G05": ("a.py", "r = client.messages.create(model=cfg.model, messages=m)"),
    "G06": ("a.py", "r = client.messages.create(model=cfg.model, messages=m)"),
    "G07": ("a.py", 'r = client.messages.create(model=cfg.model, system=f"Rules. User asked: {question}", messages=m)'),
    "G08": ("a.py", 'r = client.chat.completions.create(model="gpt-4o", messages=m)'),
}


def scan_text(name: str, text: str, **kwargs) -> list[ml_scan.Finding]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
        return ml_scan.run([str(path)], **kwargs)


class RulePositives(unittest.TestCase):
    def test_every_rule_has_a_case(self):
        self.assertEqual(set(CASES), set(ml_scan.RULES))

    def test_each_rule_fires(self):
        for rule, (name, text) in CASES.items():
            with self.subTest(rule=rule):
                kwargs = {"max_output_kb": 100}
                rules = {f.rule for f in scan_text(name, text, **kwargs)}
                self.assertIn(rule, rules, f"{rule} not detected; got {sorted(rules)}")

    def test_key_shaped_string_is_an_error(self):
        findings = scan_text("a.py", f'client = make_client("{SK}")')
        self.assertIn("E02", {f.rule for f in findings})

    def test_env_file_secret_value(self):
        findings = scan_text(".env", "OPENAI_API_KEY=abc123def456\nLLM_MODEL=\n")
        self.assertEqual([f.rule for f in findings], ["E02"])


class RuleFamilies(unittest.TestCase):
    """At least one positive per family, run as separate tests for readable CI output."""

    def family(self, prefix: str):
        rules = [r for r in CASES if r.startswith(prefix)]
        self.assertTrue(rules)
        for rule in rules:
            name, text = CASES[rule]
            self.assertIn(rule, {f.rule for f in scan_text(name, text)})

    def test_leakage(self): self.family("L")
    def test_statistics(self): self.family("S")
    def test_modeling(self): self.family("M")
    def test_features(self): self.family("F")
    def test_explainability(self): self.family("X")
    def test_data(self): self.family("D")
    def test_engineering(self): self.family("E")
    def test_communication(self): self.family("C")
    def test_production(self): self.family("P")
    def test_genai(self): self.family("G")


class Negatives(unittest.TestCase):
    def rules(self, name, text, **kw):
        return {f.rule for f in scan_text(name, text, **kw)}

    def test_ignore_with_reason(self):
        self.assertNotIn("M04", self.rules("a.py", "pred = (proba > 0.5).astype(int)  # ml-ignore M04 threshold fixed by regulator"))

    def test_ignore_without_reason_is_not_applied(self):
        findings = scan_text("a.py", "pred = (proba > 0.5).astype(int)  # ml-ignore M04")
        m04 = [f for f in findings if f.rule == "M04"]
        self.assertTrue(m04 and "without a reason" in m04[0].message)

    def test_ignore_on_previous_line(self):
        self.assertNotIn("E05", self.rules("a.py", '# ml-ignore E05 third-party library spams a known warning\nwarnings.filterwarnings("ignore")'))

    def test_negated_causal_language(self):
        self.assertNotIn("S05", self.rules("a.py", 'note = "Explanations describe the model, not causes in the world"'))

    def test_calibration_on_validation_is_allowed(self):
        self.assertNotIn("L02", self.rules("a.py", 'calibrator = CalibratedClassifierCV(FrozenEstimator(m), method="isotonic")\ncalibrator.fit(X_val, y_val)'))

    def test_final_refit_name_is_not_heldout(self):
        self.assertNotIn("L02", self.rules("a.py", "pipe.fit(X_trval, y_trval)"))

    def test_threshold_constraint_is_not_a_metric_claim(self):
        self.assertNotIn("C01", self.rules("a.py", 'goal = "keep recall high while precision >= 0.40 for the queue"'))

    def test_transform_after_split_is_clean(self):
        code = """
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
            scaler = StandardScaler().fit(X_train)
        """
        self.assertNotIn("L01", self.rules("a.py", code))

    def test_skip_file_directive(self):
        self.assertEqual(self.rules("a.py", "# ml-scan: skip-file generated fixture\nfrom numpy import *"), set())

    def test_route_with_threadpool_is_clean(self):
        code = """
            @app.post("/predict")
            async def predict(req: Req):
                return await run_in_threadpool(model.predict_proba, X)
        """
        self.assertNotIn("P06", self.rules("a.py", code))


class Fixtures(unittest.TestCase):
    def test_sloppy_fixtures_have_errors(self):
        findings = ml_scan.run([str(ROOT / "tests" / "fixtures" / "sloppy")])
        errors = {f.rule for f in findings if f.severity == "error"}
        self.assertGreaterEqual(len(errors), 10, sorted(errors))
        families = {r[0] for r in errors}
        self.assertTrue({"L", "M", "E", "P", "G"} <= families, families)

    def test_clean_fixtures_have_no_errors(self):
        findings = ml_scan.run([str(ROOT / "tests" / "fixtures" / "clean")])
        errors = [f"{f.path} {f.loc} {f.rule}" for f in findings if f.severity == "error"]
        self.assertEqual(errors, [])

    def test_score_formula(self):
        f = lambda r, s: ml_scan.Finding(r, s, "p", "file", "m")
        self.assertEqual(ml_scan.score([f("M04", "warning")]), 94)
        self.assertEqual(ml_scan.score([f("L01", "error"), f("M04", "warning"), f("M04", "warning")]), 74)


class Catalog(unittest.TestCase):
    def test_catalog_lists_every_scanner_rule(self):
        text = (ROOT / "skills" / "ml-quality-guard" / "references" / "slop-catalog.md").read_text(encoding="utf-8")
        rows = re.findall(r"^\| ([A-Z]\d{2}) \| (error|warning) \| (auto|manual) \|", text, re.M)
        self.assertTrue(70 <= len(rows) <= 85, len(rows))
        auto = {rid for rid, _, det in rows if det == "auto"}
        self.assertEqual(auto, set(ml_scan.RULES))
        for rid, sev, det in rows:
            if det == "auto":
                self.assertEqual(sev, ml_scan.RULES[rid][0], rid)


if __name__ == "__main__":
    unittest.main()
