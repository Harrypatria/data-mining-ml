"""Load the model bundle once, verify it, and expose prediction helpers.

The path comes from MODEL_DIR (operator configuration). load_bundle checks model.sha256 and
the recorded library versions before unpickling. There is no upload endpoint.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mlkit.explainer import extrapolation_flags, median_profile, predict_frame
from mlkit.persistence import Bundle, BundleError, VersionMismatchError, coerce_frame, load_bundle, resolve_model_dir

from .config import Settings

log = logging.getLogger("app.model_store")


class ModelStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_dir: Path = resolve_model_dir(settings.model_dir)
        schema_path = self.model_dir / "schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError(f"{schema_path} not found; set MODEL_DIR to a bundle folder or a models/ root with CURRENT")
        # schema.json is plain JSON, read at startup to build request models; the pickle loads in the lifespan handler.
        self.schema: dict = json.loads(schema_path.read_text(encoding="utf-8"))
        self.bundle: Bundle | None = None
        self.error: str | None = None

    def load(self) -> None:
        try:
            self.bundle = load_bundle(self.model_dir, mode=self.settings.app_env)
            self.error = None
            log.info("model loaded", extra={"model_version": self.bundle.metadata.get("model_version"), "path": str(self.model_dir)})
        except (BundleError, VersionMismatchError, OSError, ValueError) as exc:
            self.bundle, self.error = None, str(exc)
            log.error("model not loaded", extra={"reason": str(exc)})

    @property
    def ready(self) -> bool:
        return self.bundle is not None

    @property
    def metadata(self) -> dict:
        return self.bundle.metadata if self.bundle else {}

    @property
    def classification(self) -> bool:
        return self.metadata.get("task") == "binary_classification"

    def median_profile(self) -> dict:
        return median_profile(self.schema)

    def flags(self, profile: dict) -> dict[str, bool]:
        return extrapolation_flags(self.schema, profile)

    def frame(self, profiles: list[dict]) -> pd.DataFrame:
        """Column order and dtypes come from schema.json, never from the request."""
        return coerce_frame(pd.DataFrame(profiles, columns=self.schema["columns"]), self.schema)

    def predict(self, profiles: list[dict]) -> np.ndarray:
        return predict_frame(self.bundle, pd.DataFrame(profiles, columns=self.schema["columns"]))

    def record(self, value: float, flags: dict[str, bool], request_id: str) -> dict:
        meta = self.metadata
        out = {"model_version": str(meta.get("model_version")), "request_id": request_id,
               "extrapolation": flags, "extrapolated": any(flags.values()), "threshold": meta.get("threshold")}
        if self.classification:
            out.update(probability=float(value), value=None,
                       label="above threshold" if value >= meta["threshold"] else "below threshold")
        else:
            out.update(probability=None, value=float(value), label=None)
        return out
