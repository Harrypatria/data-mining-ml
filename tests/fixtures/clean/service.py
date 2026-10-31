"""Clean FastAPI service fixture: verified load at startup, schema validation, threadpool inference."""
import hashlib
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

MODEL_DIR = Path("models") / "v1.0.0"


def load_verified(model_dir: Path):
    expected = (model_dir / "model.sha256").read_text(encoding="utf-8").split()[0]
    payload = (model_dir / "model.pkl").read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError("model.pkl does not match model.sha256")
    return pickle.loads(payload)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_verified(MODEL_DIR)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["https://workbench.example.org"], allow_credentials=False)


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenure_months: float = Field(ge=0, le=72)
    plan: str


@app.get("/health/live")
def live() -> dict:
    return {"status": "ok"}


@app.post("/v1/predict")
def predict(profile: Profile, request: Request) -> dict:
    frame = pd.DataFrame([profile.model_dump()])
    return {"probability": float(request.app.state.model.predict_proba(frame)[0, 1])}
