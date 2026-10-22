from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Iterator

import requests

PROVIDERS: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
}

DEFAULT_CHAIN = (
    "groq:openai/gpt-oss-120b,"
    "gemini:gemini-3.5-flash"
)


@dataclass(frozen=True)
class Target:
    provider: str
    model: str
    base_url: str
    api_key: str

    @property
    def label(self) -> str:
        return f"{self.provider} · {self.model}"


class AllProvidersFailed(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors) or "no providers configured")


def load_chain(spec: str | None, get_secret: Callable[[str], str | None]) -> list[Target]:
    targets: list[Target] = []
    for item in (spec or DEFAULT_CHAIN).split(","):
        item = item.strip()
        if ":" not in item:
            continue
        provider, model = item.split(":", 1)
        provider = provider.strip().lower()
        if provider not in PROVIDERS:
            continue
        base_url, key_name = PROVIDERS[provider]
        key = (get_secret(key_name) or "").strip()
        if key:
            targets.append(Target(provider, model.strip(), base_url, key))
    return targets


class ChatStream:
    def __init__(
        self,
        messages: list[dict],
        chain: list[Target],
        *,
        temperature: float = 0.3,
        max_tokens: int = 700,
        timeout: float = 30.0,
    ):
        self.messages = messages
        self.chain = chain
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.used: Target | None = None
        self.errors: list[str] = []

    def tokens(self) -> Iterator[str]:
        for target in self.chain:
            yielded = False
            try:
                for piece in self._stream_one(target):
                    yielded = True
                    self.used = target
                    yield piece
                if yielded:
                    return
                self.errors.append(f"{target.label}: empty response")
            except _Retryable as e:
                self.errors.append(f"{target.label}: {e}")
                if yielded:
                    return
        raise AllProvidersFailed(self.errors)

    def _stream_one(self, t: Target) -> Iterator[str]:
        try:
            resp = requests.post(
                f"{t.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {t.api_key}", "Content-Type": "application/json"},
                json={
                    "model": t.model,
                    "messages": self.messages,
                    "stream": True,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                stream=True,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise _Retryable(f"network error ({type(e).__name__})") from e

        with resp:
            if resp.status_code != 200:
                raise _Retryable(_describe(resp.status_code))
            resp.encoding = "utf-8"
            try:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        choices = json.loads(payload).get("choices") or []
                    except json.JSONDecodeError:
                        continue
                    if choices:
                        piece = (choices[0].get("delta") or {}).get("content")
                        if piece:
                            yield piece
            except requests.RequestException as e:
                raise _Retryable(f"stream interrupted ({type(e).__name__})") from e


class _Retryable(Exception):
    pass


def _describe(status: int) -> str:
    return {
        401: "invalid API key",
        403: "access denied",
        404: "model not found",
        429: "rate limit reached",
    }.get(status, f"HTTP {status}")


def list_models(target_provider: str, api_key: str, timeout: float = 20.0) -> list[str]:
    base_url, _ = PROVIDERS[target_provider]
    r = requests.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []))
