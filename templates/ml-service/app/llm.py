"""LLM provider adapter for the explainer: OpenAI or Anthropic over HTTPS, or none.

Server-side only. Keys from OPENAI_API_KEY or ANTHROPIC_API_KEY, model from LLM_MODEL. Every call
has a timeout, a max token cap and bounded retries with exponential backoff. Answers are cached by
the hash of (context, question). A per-session limit caps cost. The context is validated against
rtcc_context.schema.json before every call, and every answer passes the numeric grounding check
(grounding.guarded) or is replaced by the template explanation.
"""
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

from mlkit import llm as provider
from mlkit import narrative
from mlkit.explainer import packaged_prompt

from . import grounding
from .config import Settings

log = logging.getLogger("app.llm")


class SessionLimitExceeded(RuntimeError):
    pass


class ContextInvalid(ValueError):
    pass


def _prompt_text(settings: Settings, name: str) -> str:
    folder = settings.prompts_dir or Path(__file__).parent / "prompts"
    path = Path(folder) / name
    return path.read_text(encoding="utf-8") if path.is_file() else packaged_prompt(name)


class ExplainerService:
    def __init__(self, settings: Settings, model_purpose: str, audience: str, metrics=None):
        self.settings = settings
        self.metrics = metrics
        self.system_prompt = (_prompt_text(settings, "rtcc_system.md")
                              .replace("[MODEL_PURPOSE]", model_purpose or "a predictive model")
                              .replace("[AUDIENCE]", audience or "business users"))
        self.context_schema = json.loads(_prompt_text(settings, "rtcc_context.schema.json"))
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._sessions: dict[str, deque] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        s = self.settings
        return s.llm_provider != "none" and bool(s.llm_model) and bool(s.llm_api_key)

    def check_session(self, session_id: str) -> None:
        now, window = time.monotonic(), 3600.0
        with self._lock:
            hits = self._sessions.setdefault(session_id, deque())
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= self.settings.llm_session_limit_per_hour:
                raise SessionLimitExceeded(f"explanation limit of {self.settings.llm_session_limit_per_hour} per hour reached")
            hits.append(now)

    def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """One answer with retries and backoff. Raises provider.ProviderError when the provider stays unavailable."""
        if not self.enabled:
            raise provider.ProviderError("LLM provider disabled (LLM_PROVIDER=none or missing model or key)")
        s = self.settings
        kwargs = dict(api_key=s.llm_api_key, max_tokens=s.llm_max_tokens, timeout=s.llm_timeout_s, base_url=s.llm_base_url or None)
        for attempt in range(s.llm_max_retries + 1):
            try:
                if s.llm_stream:
                    return "".join(provider.stream_text(s.llm_provider, s.llm_model, system_prompt, messages, **kwargs))
                return provider.complete(s.llm_provider, s.llm_model, system_prompt, messages, **kwargs)
            except provider.ProviderError as exc:
                if not exc.retryable or attempt == s.llm_max_retries:
                    raise
                time.sleep(min(8.0, 0.5 * 2 ** attempt))
        raise provider.ProviderError("retries exhausted")

    def answer(self, context: dict, question: str, session_id: str) -> dict:
        """Validated, cached, grounded answer. The route enforces the per-session limit before calling."""
        problems = narrative.validate(context, self.context_schema)
        if problems:
            raise ContextInvalid("; ".join(problems[:5]))
        key = hashlib.sha256(json.dumps([context, question], sort_keys=True, default=str).encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return {**self._cache[key], "cached": True}
        result = grounding.guarded(context, question, self.system_prompt, self.generate, self.metrics)
        if result["source"] == "llm" and self.settings.llm_cache_size:
            with self._lock:
                self._cache[key] = result
                while len(self._cache) > self.settings.llm_cache_size:
                    self._cache.popitem(last=False)
        return {**result, "cached": False}
