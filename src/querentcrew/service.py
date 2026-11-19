"""High-level facade used by the Streamlit app, the FastAPI server and the CLI."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import IO, Any

from langgraph.types import Command

from .config import Settings, get_settings
from .database import DatabaseManager
from .graph import build_graph
from .llm import LangChainLLM, LLMClient
from .nodes import AgentNodes


StepCallback = Callable[[str, dict], None]


@dataclass
class AskResponse:
    thread_id: str
    status: str  # "completed" | "awaiting_approval"
    answer: str | None = None
    approval_request: dict | None = None
    intent: str | None = None
    plan: list[str] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    insights: dict = field(default_factory=dict)
    chart: dict | None = None
    trace: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0
    llm_error: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _turn_reset(question: str) -> dict[str, Any]:
    """Per-question fields reset at the start of every turn (history is kept by the checkpointer)."""
    return {
        "question": question,
        "intent": "",
        "plan_reasoning": "",
        "sub_questions": [],
        "current_index": 0,
        "attempts": 0,
        "sql_reasoning": "",
        "draft_sql": "",
        "feedback": "",
        "guard": {},
        "approval": None,
        "next_action": "",
        "results": [],
        "insights": {},
        "chart": None,
        "answer": "",
        "trace": [],
        "llm_error": "",
    }


class QuerentCrew:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None, checkpointer=None):
        self.settings = settings or get_settings()
        self.db = DatabaseManager(
            self.settings.database_path,
            timeout_seconds=self.settings.query_timeout_seconds,
            max_rows=self.settings.max_result_rows,
        )
        self.llm = llm or LangChainLLM(
            self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            fallback_model=self.settings.llm_fallback_model,
            max_retries=self.settings.llm_max_retries,
        )
        self.nodes = AgentNodes(self.llm, self.db, self.settings)
        self.graph = build_graph(self.nodes, checkpointer)

    @staticmethod
    def new_thread_id() -> str:
        return uuid.uuid4().hex[:12]

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 60,
                "run_name": "querentcrew.ask", "tags": ["querentcrew"]}

    def _respond(self, thread_id: str, output: dict, started: float) -> AskResponse:
        interrupts = output.get("__interrupt__") or []
        if not interrupts:  # fall back to the checkpoint (works across langgraph versions)
            snapshot = self.graph.get_state(self._config(thread_id))
            for task in getattr(snapshot, "tasks", ()) or ():
                interrupts.extend(getattr(task, "interrupts", ()) or ())
        state = self.graph.get_state(self._config(thread_id)).values or output
        response = AskResponse(
            thread_id=thread_id,
            status="awaiting_approval" if interrupts else "completed",
            answer=None if interrupts else state.get("answer"),
            approval_request=getattr(interrupts[0], "value", interrupts[0]) if interrupts else None,
            intent=state.get("intent"),
            plan=state.get("sub_questions", []),
            results=state.get("results", []),
            insights=state.get("insights", {}),
            chart=None if interrupts else state.get("chart"),
            trace=state.get("trace", []),
            usage=self.llm.usage.snapshot(),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            llm_error=state.get("llm_error") or None,
        )
        return response

    def _run(self, payload: Any, thread_id: str, on_step: StepCallback | None) -> dict:
        """Invoke the graph; with `on_step`, stream node updates so callers can show live progress."""
        config = self._config(thread_id)
        if on_step is None:
            return self.graph.invoke(payload, config)
        for chunk in self.graph.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node.startswith("__"):
                    continue
                trace = (update or {}).get("trace") or []
                on_step(node, trace[-1] if trace else {"agent": node, "message": ""})
        return {}

    def ask(self, question: str, thread_id: str | None = None, on_step: StepCallback | None = None) -> AskResponse:
        thread_id = thread_id or self.new_thread_id()
        self.llm.usage.reset()
        started = time.perf_counter()
        output = self._run(_turn_reset(question.strip()), thread_id, on_step)
        return self._respond(thread_id, output, started)

    def resume(self, thread_id: str, approved: bool, comment: str = "",
               on_step: StepCallback | None = None) -> AskResponse:
        """Continue a run that paused for human approval."""
        started = time.perf_counter()
        output = self._run(Command(resume={"approved": approved, "comment": comment}), thread_id, on_step)
        return self._respond(thread_id, output, started)

    def check_llm(self) -> tuple[bool, str]:
        """Return (ok, message) after a tiny test call to the configured model."""
        return self.llm.ping()

    def preview(self, table: str, limit: int = 20) -> tuple[list[str], list[list]]:
        if table not in self.db.table_names():
            raise ValueError(f"unknown table: {table}")
        result = self.db.execute(f'SELECT * FROM "{table}" LIMIT {int(limit)}')
        return result.columns, result.rows

    def history(self, thread_id: str) -> list[dict]:
        snapshot = self.graph.get_state(self._config(thread_id))
        return (snapshot.values or {}).get("history", [])

    def import_csv(self, file: IO | str, table_name: str) -> tuple[str, int]:
        table, rows = self.db.import_csv(file, table_name)
        self.nodes.refresh_schema()
        return table, rows

    def schema(self) -> dict[str, Any]:
        return {"tables": self.db.table_columns(), "row_counts": self.db.row_counts()}
