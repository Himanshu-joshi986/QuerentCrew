"""LLM access layer.

`LLMClient` is a small interface the agents depend on. `LangChainLLM` is the
real implementation: it uses LangChain's provider-agnostic `init_chat_model`,
so switching between Groq, Gemini, OpenAI, Anthropic or a local Ollama model is
a one-line change in `.env`. It adds structured output, retries, an optional
fallback model, and token/latency tracking per call.

Tests use `FakeLLM` from tests/fakes.py, which implements the same interface,
so the whole agent graph can be tested without an API key.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class UsageTracker:
    """Accumulates token usage and latency across LLM calls (per question or per eval run)."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    by_agent: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(self, agent: str, input_tokens: int, output_tokens: int, latency_ms: float) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.latency_ms += latency_ms
        slot = self.by_agent.setdefault(agent, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0})
        slot["calls"] += 1
        slot["input_tokens"] += input_tokens
        slot["output_tokens"] += output_tokens
        slot["latency_ms"] += latency_ms

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "by_agent": {k: {**v, "latency_ms": round(v["latency_ms"], 1)} for k, v in self.by_agent.items()},
        }

    def reset(self) -> None:
        self.calls = self.input_tokens = self.output_tokens = 0
        self.latency_ms = 0.0
        self.by_agent = {}


class LLMClient(ABC):
    """Interface used by every agent."""

    def __init__(self) -> None:
        self.usage = UsageTracker()

    @abstractmethod
    def structured(self, agent: str, system: str, user: str, schema: type[T]) -> T:
        """Return an instance of `schema` produced by the model."""

    @abstractmethod
    def text(self, agent: str, system: str, user: str) -> str:
        """Return a free-text completion."""

    @property
    def label(self) -> str:
        return getattr(self, "model_name", type(self).__name__)

    def ping(self) -> tuple[bool, str]:
        """Cheap connectivity check. Returns (ok, message)."""
        try:
            self.text("healthcheck", "Reply with the single word: ok", "ping")
            return True, "connected"
        except Exception as exc:
            return False, describe_llm_error(exc, self.label) or f"{type(exc).__name__}: {exc}"


_CONNECTION_HINTS = ("connection error", "connecterror", "connection refused", "name resolution",
                     "getaddrinfo", "network is unreachable", "apiconnectionerror", "connection reset")
_TIMEOUT_HINTS = ("timeout", "timed out")
_AUTH_HINTS = ("authenticationerror", "invalid api key", "invalid_api_key", "incorrect api key",
               "error code: 401", "unauthorized", "permissiondenied", "error code: 403")
_RATE_HINTS = ("ratelimit", "rate limit", "rate_limit", "error code: 429", "quota")
_SERVER_HINTS = ("internalservererror", "error code: 500", "error code: 502", "error code: 503",
                 "service unavailable", "overloaded")


def classify_llm_error(exc: BaseException) -> str | None:
    """Return a short category if `exc` means the LLM provider can't be used right now.

    Categories: connection, timeout, auth, rate_limit, server. Returns None for ordinary
    model mistakes (e.g. invalid structured output), which the agents handle by retrying.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    for category, hints in (("auth", _AUTH_HINTS), ("rate_limit", _RATE_HINTS), ("timeout", _TIMEOUT_HINTS),
                            ("connection", _CONNECTION_HINTS), ("server", _SERVER_HINTS)):
        if any(h in text for h in hints):
            return category
    return None


def describe_llm_error(exc: BaseException, model: str = "") -> str | None:
    """User-facing explanation for a provider failure, or None if it isn't one."""
    category = classify_llm_error(exc)
    if category is None:
        return None
    name = f" (`{model}`)" if model else ""
    return {
        "connection": f"Couldn't reach the language model{name}. Check your internet connection, VPN or "
                      "proxy, then try again.",
        "timeout": f"The language model{name} took too long to respond. Try again in a moment.",
        "auth": f"The language model provider rejected the API key{name}. Check the API key in your `.env` file.",
        "rate_limit": f"The language model provider's rate limit or quota was hit{name}. Wait a minute and "
                      "try again, or set `LLM_FALLBACK_MODEL` in `.env`.",
        "server": f"The language model provider{name} is having problems right now. Try again shortly.",
    }[category]


def _usage_from_message(message) -> tuple[int, int]:
    meta = getattr(message, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0) or 0), int(meta.get("output_tokens", 0) or 0)


class LangChainLLM(LLMClient):
    """LangChain-backed client: `init_chat_model` + structured output + retry + fallback."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        fallback_model: str | None = None,
        max_retries: int = 2,
    ) -> None:
        super().__init__()
        from langchain.chat_models import init_chat_model  # imported lazily

        self.model_name = model
        self.fallback_name = fallback_model
        self.max_retries = max_retries
        self._primary = init_chat_model(model, temperature=temperature)
        self._fallback = init_chat_model(fallback_model, temperature=temperature) if fallback_model else None

    def _with_resilience(self, primary, fallback):
        runnable = primary.with_retry(stop_after_attempt=self.max_retries + 1)
        if fallback is not None:
            runnable = runnable.with_fallbacks([fallback.with_retry(stop_after_attempt=self.max_retries + 1)])
        return runnable

    def structured(self, agent: str, system: str, user: str, schema: type[T]) -> T:
        def build(model):
            return model.with_structured_output(schema, include_raw=True)

        runnable = self._with_resilience(
            build(self._primary), build(self._fallback) if self._fallback is not None else None
        )
        start = time.perf_counter()
        out = runnable.invoke(
            [("system", system), ("human", user)],
            config={"run_name": f"querentcrew.{agent}", "tags": ["querentcrew", agent]},
        )
        latency = (time.perf_counter() - start) * 1000
        raw, parsed = out.get("raw"), out.get("parsed")
        tokens_in, tokens_out = _usage_from_message(raw)
        self.usage.record(agent, tokens_in, tokens_out, latency)
        if parsed is None:
            error = out.get("parsing_error")
            raise ValueError(f"{agent}: model did not return valid {schema.__name__}: {error}")
        return parsed

    def text(self, agent: str, system: str, user: str) -> str:
        runnable = self._with_resilience(self._primary, self._fallback)
        start = time.perf_counter()
        message = runnable.invoke(
            [("system", system), ("human", user)],
            config={"run_name": f"querentcrew.{agent}", "tags": ["querentcrew", agent]},
        )
        latency = (time.perf_counter() - start) * 1000
        tokens_in, tokens_out = _usage_from_message(message)
        self.usage.record(agent, tokens_in, tokens_out, latency)
        content = message.content
        if isinstance(content, list):  # some providers return content blocks
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return str(content).strip()
