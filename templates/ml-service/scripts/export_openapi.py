"""Write the OpenAPI document for the typed web client: python scripts/export_openapi.py MODEL_DIR OUT.json"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import mlkit  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    app = create_app(Settings(_env_file=None, model_dir=Path(sys.argv[1]), app_env="development"))
    Path(sys.argv[2]).write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
    print(f"wrote {sys.argv[2]}")
