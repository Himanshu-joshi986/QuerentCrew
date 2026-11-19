"""Shared state passed between the agents in the LangGraph workflow.

All values are plain JSON-friendly types so the state can be checkpointed
(conversation memory + pause/resume for human approval).
"""

from __future__ import annotations

from typing import Any, TypedDict


class QueryRecord(TypedDict, total=False):
    sub_question: str
    sql: str
    status: str  # ok | failed | blocked | rejected | llm_error
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    masked_columns: list[str]
    attempts: int
    error: str | None
    elapsed_ms: float


class AgentState(TypedDict, total=False):
    # conversation
    question: str
    history: list[dict[str, str]]  # [{"role": "user"|"assistant", "content": ...}]

    # routing / planning
    intent: str
    plan_reasoning: str
    sub_questions: list[str]
    current_index: int

    # SQL loop for the current sub-question
    attempts: int
    sql_reasoning: str
    draft_sql: str
    feedback: str
    guard: dict[str, Any]
    approval: dict[str, Any] | None
    next_action: str  # execute | approve | retry | advance

    # outputs
    results: list[QueryRecord]
    insights: dict[str, Any]
    chart: dict[str, Any] | None
    answer: str
    trace: list[dict[str, Any]]  # step-by-step log shown in the UI
    llm_error: str  # set when the LLM provider is unusable; short-circuits to the reporter
