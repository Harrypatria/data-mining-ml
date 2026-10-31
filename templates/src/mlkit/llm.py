"""Minimal LLM provider adapter on the standard library (OpenAI Chat Completions, Anthropic Messages).

Keys come from the caller (read from environment variables or google.colab.userdata), the model
name from LLM_MODEL. Every request has a timeout and a token cap. Callers must run
narrative.check_grounding on the text and fall back to narrative.template_explanation on failure.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator

DEFAULT_BASE = {"openai": "https://api.openai.com", "anthropic": "https://api.anthropic.com"}


class ProviderError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def settings_from_env() -> dict:
    provider = os.environ.get("LLM_PROVIDER", "none").lower()
    key_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider, "")
    return {"provider": provider, "model": os.environ.get("LLM_MODEL", ""),
            "api_key": os.environ.get(key_var, "") if key_var else "",
            "base_url": os.environ.get("LLM_BASE_URL", "") or DEFAULT_BASE.get(provider, "")}


def _request(provider: str, model: str, system: str, messages: list[dict], api_key: str, max_tokens: int,
             base_url: str | None, stream: bool, temperature: float | None) -> urllib.request.Request:
    if not model:
        raise ProviderError("LLM_MODEL is not set")
    if not api_key:
        raise ProviderError(f"no API key configured for provider {provider}")
    base = (base_url or DEFAULT_BASE[provider]).rstrip("/")
    if provider == "openai":
        body = {"model": model, "messages": [{"role": "system", "content": system}, *messages],
                "max_completion_tokens": max_tokens, "stream": stream}
        headers = {"Authorization": f"Bearer {api_key}"}
        url = f"{base}/v1/chat/completions"
    elif provider == "anthropic":
        body = {"model": model, "system": system, "messages": messages, "max_tokens": max_tokens, "stream": stream}
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        url = f"{base}/v1/messages"
    else:
        raise ProviderError(f"unsupported provider {provider}")
    if temperature is not None:
        body["temperature"] = temperature
    headers["content-type"] = "application/json"
    return urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")


def _open(req: urllib.request.Request, timeout: float):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        retryable = exc.code in (408, 409, 429) or exc.code >= 500
        raise ProviderError(f"provider returned HTTP {exc.code}", retryable=retryable) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderError(f"provider unreachable: {exc}", retryable=True) from exc


def complete(provider: str, model: str, system: str, messages: list[dict], *, api_key: str, max_tokens: int = 400,
             timeout: float = 20.0, base_url: str | None = None, temperature: float | None = None) -> str:
    """One non-streaming call. Raises ProviderError; never returns partial text."""
    req = _request(provider, model, system, messages, api_key, max_tokens, base_url, False, temperature)
    with _open(req, timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if provider == "openai":
        return data["choices"][0]["message"]["content"] or ""
    return "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")


def stream_text(provider: str, model: str, system: str, messages: list[dict], *, api_key: str, max_tokens: int = 400,
                timeout: float = 20.0, base_url: str | None = None, temperature: float | None = None) -> Iterator[str]:
    """Yield text deltas from the provider's server-sent events stream."""
    req = _request(provider, model, system, messages, api_key, max_tokens, base_url, True, temperature)
    with _open(req, timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if provider == "openai":
                choices = event.get("choices") or [{}]
                delta = choices[0].get("delta", {}).get("content")
            else:
                delta = event.get("delta", {}).get("text") if event.get("type") == "content_block_delta" else None
            if delta:
                yield delta
