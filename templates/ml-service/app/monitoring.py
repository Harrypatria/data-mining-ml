"""JSON logging, request context, size and rate limits, Prometheus metrics and a PSI drift monitor."""
import contextvars
import json
import logging
import math
import re
import threading
import time
import uuid
from collections import defaultdict, deque

import numpy as np
from starlette.types import ASGIApp, Receive, Scope, Send

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Records carry ids, routes and timings, never request bodies or profiles."""

    _skip = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)), "level": record.levelname,
                   "logger": record.name, "msg": record.getMessage(), "request_id": request_id_var.get()}
        payload.update({k: v for k, v in vars(record).items() if k not in self._skip})
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


class DriftMonitor:
    """Population stability index per numeric predictor: live values against the training background."""

    def __init__(self, window: int = 2000, bins: int = 10):
        self.window, self.bins = window, bins
        self.edges: dict[str, np.ndarray] = {}
        self.expected: dict[str, np.ndarray] = {}
        self.live: dict[str, deque] = {}
        self._lock = threading.Lock()

    def set_reference(self, background, numeric: list[str]) -> None:
        for col in numeric:
            values = background[col].dropna().to_numpy(dtype=float)
            if len(values) < self.bins:
                continue
            edges = np.unique(np.quantile(values, np.linspace(0, 1, self.bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            self.edges[col] = edges
            self.expected[col] = np.histogram(values, edges)[0] / len(values)
            self.live[col] = deque(maxlen=self.window)

    def observe(self, profile: dict) -> None:
        with self._lock:
            for col, buf in self.live.items():
                v = profile.get(col)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    buf.append(float(v))

    def psi(self, min_samples: int = 100) -> dict[str, float]:
        out = {}
        with self._lock:
            for col, buf in self.live.items():
                if len(buf) < min_samples:
                    continue
                actual = np.histogram(np.asarray(buf), self.edges[col])[0] / len(buf)
                e, a = np.clip(self.expected[col], 1e-4, None), np.clip(actual, 1e-4, None)
                out[col] = float(np.sum((a - e) * np.log(a / e)))
        return out


class Metrics:
    def __init__(self):
        self.counters: dict[tuple, float] = defaultdict(float)
        self.latency: dict[str, list[int]] = defaultdict(lambda: [0] * (len(LATENCY_BUCKETS) + 1))
        self.latency_sum: dict[str, float] = defaultdict(float)
        self.predictions = [0] * 10
        self.drift = DriftMonitor()
        self._lock = threading.Lock()

    def count(self, name: str, labels: dict, value: float = 1.0) -> None:
        with self._lock:
            self.counters[(name, tuple(sorted(labels.items())))] += value

    def observe_latency(self, route: str, seconds: float) -> None:
        with self._lock:
            i = next((k for k, b in enumerate(LATENCY_BUCKETS) if seconds <= b), len(LATENCY_BUCKETS))
            self.latency[route][i] += 1
            self.latency_sum[route] += seconds

    def observe_prediction(self, value: float, flags: dict[str, bool]) -> None:
        with self._lock:
            if 0.0 <= value <= 1.0:
                self.predictions[min(int(value * 10), 9)] += 1
        for name, out in flags.items():
            if out:
                self.count("input_range_violations_total", {"feature": name})

    def render(self) -> str:
        lines = []
        with self._lock:
            for (name, labels), v in sorted(self.counters.items()):
                lab = ",".join(f'{k}="{val}"' for k, val in labels)
                lines.append(f"{name}{{{lab}}} {v}")
            for route, counts in self.latency.items():
                cum = 0
                for b, c in zip((*LATENCY_BUCKETS, math.inf), counts):
                    cum += c
                    le = "+Inf" if b == math.inf else b
                    lines.append(f'request_latency_seconds_bucket{{route="{route}",le="{le}"}} {cum}')
                lines.append(f'request_latency_seconds_count{{route="{route}"}} {cum}')
                lines.append(f'request_latency_seconds_sum{{route="{route}"}} {self.latency_sum[route]:.6f}')
            for i, c in enumerate(self.predictions):
                lines.append(f'prediction_probability_bin{{bin="{i / 10:.1f}-{(i + 1) / 10:.1f}"}} {c}')
        for col, value in self.drift.psi().items():
            lines.append(f'feature_drift_psi{{feature="{col}"}} {value:.6f}')
        return "\n".join(lines) + "\n"


class RequestContextMiddleware:
    """Request id, JSON access log, latency metric. Pure ASGI so streaming responses are not buffered."""

    def __init__(self, app: ASGIApp, metrics: Metrics):
        self.app, self.metrics = app, metrics
        self.log = logging.getLogger("app.access")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        rid = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(rid)
        scope.setdefault("state", {})["request_id"] = rid
        start, status = time.perf_counter(), {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", rid.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            route = scope.get("route").path if scope.get("route") is not None else scope["path"]
            self.metrics.observe_latency(route, elapsed)
            self.metrics.count("http_requests_total", {"route": route, "method": scope["method"], "status": str(status["code"])})
            self.log.info("request", extra={"method": scope["method"], "route": route, "status": status["code"],
                                            "duration_ms": round(elapsed * 1000, 2)})
            request_id_var.reset(token)


class BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and int(declared) > self.max_bytes:
            await _plain(send, 413, "request body too large")
            return
        state = {"n": 0, "started": False}

        async def limited_receive():
            message = await receive()
            if message["type"] == "http.request":
                state["n"] += len(message.get("body", b""))
                if state["n"] > self.max_bytes:
                    raise BodyTooLarge()
            return message

        async def tracking_send(message):
            if message["type"] == "http.response.start":
                state["started"] = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BodyTooLarge:
            if not state["started"]:
                await _plain(send, 413, "request body too large")


class RateLimitMiddleware:
    """Token bucket per client address for /v1 routes."""

    def __init__(self, app: ASGIApp, per_minute: int):
        self.app, self.rate = app, per_minute / 60.0
        self.capacity = float(per_minute)
        self.buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/v1"):
            await self.app(scope, receive, send)
            return
        client = (scope.get("client") or ("unknown", 0))[0]
        now = time.monotonic()
        with self._lock:
            tokens, last = self.buckets.get(client, [self.capacity, now])
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            allowed = tokens >= 1.0
            self.buckets[client] = [tokens - 1.0 if allowed else tokens, now]
        if not allowed:
            await _plain(send, 429, "rate limit exceeded", [(b"retry-after", b"60")])
            return
        await self.app(scope, receive, send)


async def _plain(send: Send, status: int, detail: str, headers: list | None = None) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *(headers or [])]})
    await send({"type": "http.response.body", "body": body})
