"""Model bundle persistence: pickle the fitted pipeline with checksum, metadata, schema and background sample.

Security: unpickling runs arbitrary code. load_bundle only reads an operator-controlled
directory and refuses a model.pkl whose SHA-256 does not match model.sha256. Never pass a
path that came from a request, an upload or any other user input.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_FILES = ("model.pkl", "model.sav")
TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "imbalanced-learn", "xgboost", "lightgbm",
                    "shap", "joblib")
PII_HINTS = ("id", "email", "phone", "name", "address", "ssn", "dob")


class BundleError(RuntimeError):
    """The bundle is incomplete, tampered with or unreadable."""


class VersionMismatchError(RuntimeError):
    """Installed libraries differ from the training environment in a major or minor version."""


@dataclass
class Bundle:
    pipeline: object
    metadata: dict
    schema: dict
    background: pd.DataFrame
    path: Path


CORE_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy")
MODULE_TO_DIST = {"sklearn": "scikit-learn", "imblearn": "imbalanced-learn", "xgboost": "xgboost", "lightgbm": "lightgbm"}


def library_versions(packages=None) -> dict[str, str]:
    """Installed versions of `packages` (default: every tracked package) plus the Python version."""
    versions = {"python": platform.python_version()}
    for pkg in packages or TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def used_packages(obj) -> set[str]:
    """Distributions whose classes appear in a fitted estimator graph, so the guard checks only what unpickling needs."""
    found, stack, seen = set(), [obj], set()
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        dist = MODULE_TO_DIST.get(type(item).__module__.split(".")[0])
        if dist:
            found.add(dist)
        if hasattr(item, "get_params"):
            stack.extend(item.get_params(deep=True).values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return found


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_hash(frame: pd.DataFrame) -> str:
    """Order-sensitive SHA-256 of the training frame, recorded so retraining can prove which data it used."""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _nice_step(span: float) -> float:
    if span <= 0:
        return 1.0
    raw = span / 100
    exp = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * exp:
            return float(m * exp)
    return float(10 * exp)


def build_schema(X_train: pd.DataFrame, feature_info: dict | None = None, adjustable: list[str] | None = None) -> dict:
    """Predictor definitions for API validation and UI controls, taken from the training data ranges."""
    feature_info = feature_info or {}
    features = []
    for col in X_train.columns:
        s = X_train[col]
        info = feature_info.get(col, {})
        base = {"name": col, "label": info.get("label", col.replace("_", " ").capitalize()),
                "unit": info.get("unit", ""), "description": info.get("description", ""),
                "adjustable": adjustable is None or col in adjustable}
        if pd.api.types.is_bool_dtype(s):
            base.update(type="boolean", default=bool(s.mode().iloc[0]))
        elif pd.api.types.is_numeric_dtype(s):
            is_int = pd.api.types.is_integer_dtype(s)
            lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
            median = float(np.nanmedian(s))
            base.update(type="integer" if is_int else "number", min=int(lo) if is_int else lo, max=int(hi) if is_int else hi,
                        default=round(median) if is_int else round(median, 4),
                        step=1.0 if is_int else _nice_step(hi - lo),
                        missing_rate=round(float(s.isna().mean()), 4))
        else:
            allowed = sorted(str(v) for v in s.dropna().unique())
            base.update(type="category", allowed=allowed, default=str(s.mode().iloc[0]))
        features.append(base)
    return {"version": 1, "columns": list(X_train.columns), "features": features}


def _background(X_train: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    sample = X_train.sample(n=min(rows, len(X_train)), random_state=seed)
    pii = [c for c in sample.columns if any(h == c.lower() or c.lower().endswith("_" + h) for h in PII_HINTS)]
    if pii:
        raise BundleError(f"background sample would contain identifier columns {pii}; drop them from X_train")
    return sample.reset_index(drop=True)


def save_bundle(pipeline, out_dir, X_train: pd.DataFrame, y_train, *, version: str, task: str, target: str,
                classes: list | None, threshold: float | None, metrics: dict, seed: int,
                cost_matrix: dict | None = None, feature_info: dict | None = None,
                model_purpose: str = "", audience: str = "", calibration_note: str = "",
                background_rows: int = 200, model_card: str | None = None,
                feature_dictionary: pd.DataFrame | None = None, filename: str = "model.pkl",
                set_current: bool = True, extra: dict | None = None) -> Path:
    """Write models/v<version>/ with model.pkl, model.sha256, metadata.json, schema.json, background.csv."""
    if filename not in MODEL_FILES:
        raise ValueError(f"filename must be one of {MODEL_FILES}")
    out = Path(out_dir) / f"v{version}"
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / filename
    with model_path.open("wb") as f:
        pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
    digest = sha256_file(model_path)
    (out / "model.sha256").write_text(f"{digest}  {filename}\n", encoding="utf-8")
    meta = {
        "model_version": version,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seed": seed,
        "task": task,
        "target": target,
        "classes": classes,
        "threshold": threshold,
        "cost_matrix": cost_matrix,
        "metrics": metrics,
        "positive_count": int(np.asarray(y_train).sum()) if task == "binary_classification" else None,
        "training_rows": int(len(X_train)),
        "training_data_sha256": data_hash(X_train),
        "model_file": filename,
        "model_sha256": digest,
        "columns": list(X_train.columns),
        "dtypes": {c: str(t) for c, t in X_train.dtypes.items()},
        "libraries": library_versions(sorted(set(CORE_PACKAGES) | used_packages(pipeline))),
        "environment": library_versions(),
        "model_purpose": model_purpose,
        "audience": audience,
        "calibration_note": calibration_note,
        **(extra or {}),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    (out / "schema.json").write_text(json.dumps(build_schema(X_train, feature_info), indent=2), encoding="utf-8")
    _background(X_train, background_rows, seed).to_csv(out / "background.csv", index=False)
    if model_card:
        (out / "model_card.md").write_text(model_card, encoding="utf-8")
    if feature_dictionary is not None:
        feature_dictionary.to_csv(out / "feature_dictionary.csv", index=False)
    if set_current:
        (Path(out_dir) / "CURRENT").write_text(f"v{version}\n", encoding="utf-8")
    return out


def resolve_model_dir(path) -> Path:
    """Accept either a version folder or a models/ root that holds a CURRENT file."""
    path = Path(path)
    current = path / "CURRENT"
    if current.is_file():
        return path / current.read_text(encoding="utf-8").strip()
    return path


def verify_bundle(model_dir) -> list[str]:
    """Static checks that never unpickle: files exist, checksum matches, JSON parses, versions recorded."""
    d = resolve_model_dir(model_dir)
    problems = []
    model_file = next((d / n for n in MODEL_FILES if (d / n).is_file()), None)
    if model_file is None:
        return [f"{d}: no model.pkl or model.sav"]
    for name in ("metadata.json", "schema.json", "model.sha256"):
        if not (d / name).is_file():
            problems.append(f"missing {name}")
    if problems:
        return problems
    expected = (d / "model.sha256").read_text(encoding="utf-8").split()[0]
    if sha256_file(model_file) != expected:
        problems.append("checksum mismatch: model file does not match model.sha256")
    try:
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        schema = json.loads((d / "schema.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return problems + [f"invalid JSON: {exc}"]
    libs = meta.get("libraries", {})
    for key in ("python", "scikit-learn", "numpy", "pandas"):
        if key not in libs:
            problems.append(f"metadata.libraries lacks {key}")
    if [f["name"] for f in schema.get("features", [])] != meta.get("columns"):
        problems.append("schema.json features do not match metadata.columns")
    return problems


def _major_minor(v: str) -> tuple[str, ...]:
    return tuple(v.split(".")[:2])


def check_versions(recorded: dict, mode: str = "production") -> list[str]:
    """Compare recorded and installed versions; raise in production, warn in development."""
    installed = library_versions()
    mismatches = [f"{pkg}: trained {ver}, installed {installed[pkg]}" for pkg, ver in recorded.items()
                  if pkg in installed and _major_minor(ver) != _major_minor(installed[pkg])]
    missing = [pkg for pkg in recorded if pkg not in installed and pkg != "python"]
    if missing:
        mismatches += [f"{pkg}: trained {recorded[pkg]}, not installed" for pkg in missing]
    if mismatches:
        msg = "library versions differ from training: " + "; ".join(mismatches)
        if mode == "production":
            raise VersionMismatchError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return mismatches


def load_bundle(model_dir, mode: str = "production") -> Bundle:
    """Verify then load. Only call with an operator-set path, never with user input."""
    d = resolve_model_dir(model_dir)
    problems = verify_bundle(d)
    if problems:
        raise BundleError(f"{d}: " + "; ".join(problems))
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    schema = json.loads((d / "schema.json").read_text(encoding="utf-8"))
    check_versions(meta.get("libraries", {}), mode)
    with (d / meta.get("model_file", "model.pkl")).open("rb") as f:
        pipeline = pickle.load(f)
    background = read_frame(d / "background.csv", schema) if (d / "background.csv").is_file() else pd.DataFrame()
    return Bundle(pipeline, meta, schema, background, d)


def coerce_frame(frame: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Enforce column order and dtypes from schema.json before predict."""
    out = pd.DataFrame(index=frame.index)
    for feat in schema["features"]:
        s = frame[feat["name"]]
        if feat["type"] == "boolean":
            out[feat["name"]] = s.map(lambda v: str(v).lower() in ("true", "1")).astype(bool)
        elif feat["type"] == "integer":
            out[feat["name"]] = pd.to_numeric(s).astype("float64") if s.isna().any() else pd.to_numeric(s).astype("int64")
        elif feat["type"] == "number":
            out[feat["name"]] = pd.to_numeric(s).astype("float64")
        else:
            out[feat["name"]] = s.astype(object).where(s.notna(), None)
    return out


def read_frame(path, schema: dict) -> pd.DataFrame:
    return coerce_frame(pd.read_csv(path), schema)


def parity_check(model_dir, X_sample: pd.DataFrame, pipeline, mode: str = "development") -> dict:
    """Reload the bundle in a fresh Python process and compare predictions with the in-memory pipeline."""
    d = resolve_model_dir(model_dir)
    src_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        sample_path, out_path = Path(tmp) / "sample.csv", Path(tmp) / "out.npz"
        X_sample.to_csv(sample_path, index=False)
        code = (
            "import sys, numpy as np\n"
            "from mlkit.persistence import load_bundle, read_frame\n"
            f"b = load_bundle({str(d)!r}, mode={mode!r})\n"
            f"X = read_frame({str(sample_path)!r}, b.schema)\n"
            "proba = b.pipeline.predict_proba(X) if hasattr(b.pipeline, 'predict_proba') else b.pipeline.predict(X)\n"
            f"np.savez({str(out_path)!r}, proba=proba, labels=b.pipeline.predict(X))\n"
        )
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(src_root), os.environ.get("PYTHONPATH", "")])}
        subprocess.run([sys.executable, "-c", code], check=True, env=env, capture_output=True, text=True, timeout=300)
        with np.load(out_path) as fresh:
            fresh_proba, fresh_labels = fresh["proba"], fresh["labels"]
    X_local = read_frame_from_memory(X_sample, json.loads((d / "schema.json").read_text(encoding="utf-8")))
    local_proba = pipeline.predict_proba(X_local) if hasattr(pipeline, "predict_proba") else pipeline.predict(X_local)
    local_labels = pipeline.predict(X_local)
    result = {"rows": int(len(X_sample)), "proba_allclose": bool(np.allclose(local_proba, fresh_proba)),
              "labels_equal": bool(np.array_equal(local_labels, fresh_labels)),
              "max_abs_diff": float(np.max(np.abs(np.asarray(local_proba, float) - np.asarray(fresh_proba, float))))}
    result["passed"] = result["proba_allclose"] and result["labels_equal"]
    return result


def read_frame_from_memory(frame: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Round-trip through CSV text so the local and fresh-process predictions see identical inputs."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.csv"
        frame.to_csv(p, index=False)
        return read_frame(p, schema)
