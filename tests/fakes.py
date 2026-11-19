"""Test doubles: a scripted LLM that implements the same interface as LangChainLLM."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from querentcrew.llm import LLMClient


class FakeLLM(LLMClient):
    """Returns queued responses per agent.

    `responses` maps an agent name to a list of responses (consumed in order) or a single
    response/callable reused forever. A callable receives (system, user, schema) and
    returns the response, which lets a test react to the prompt (e.g. error feedback).
    """

    def __init__(self, responses: dict[str, Any]):
        super().__init__()
        self.queues: dict[str, deque] = {}
        self.fixed: dict[str, Any] = {}
        for agent, value in responses.items():
            if isinstance(value, list):
                self.queues[agent] = deque(value)
            else:
                self.fixed[agent] = value
        self.prompts: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def _next(self, agent: str, system: str, user: str, schema=None):
        self.prompts[agent].append((system, user))
        if agent in self.queues and self.queues[agent]:
            value = self.queues[agent].popleft()
        elif agent in self.fixed:
            value = self.fixed[agent]
        else:
            raise AssertionError(f"FakeLLM has no response for agent '{agent}'")
        if isinstance(value, Exception):
            raise value
        if callable(value) and not isinstance(value, type):
            value = value(system, user, schema)
        self.usage.record(agent, 100, 20, 1.0)
        return value

    def structured(self, agent, system, user, schema):
        value = self._next(agent, system, user, schema)
        return value if isinstance(value, schema) else schema(**value)

    def text(self, agent, system, user):
        return str(self._next(agent, system, user))
