"""Routers. Model routes are plain `def` so FastAPI runs blocking inference in its threadpool."""
import hmac

from fastapi import HTTPException, Request, status


def require_ready(request: Request):
    store = request.app.state.store
    if not store.ready:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"model not ready: {store.error or 'loading'}")
    return store


def require_api_key(request: Request) -> None:
    expected = request.app.state.settings.api_key.get_secret_value()
    if not expected:
        return
    supplied = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing or invalid API key")


def check_ranges(store, profile: dict, allow_extrapolation: bool) -> dict[str, bool]:
    """Reject out-of-range numerics unless the caller opts in; then flag them per field."""
    flags = store.flags(profile)
    outside = [k for k, v in flags.items() if v]
    if outside and not allow_extrapolation:
        detail = [{"loc": ["body", k], "msg": f"outside training range [{f['min']}, {f['max']}]", "type": "out_of_range"}
                  for f in store.schema["features"] for k in outside if f["name"] == k]
        raise HTTPException(422, detail=detail)
    return flags
