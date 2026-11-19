"""Pydantic models used for structured LLM output.

Each agent asks the LLM for one of these models instead of free text, so the
graph can route on typed fields rather than parsing prose.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Router output: what kind of message is this?"""

    intent: Literal["data_question", "general", "out_of_scope"] = Field(
        description="data_question = needs a database query; general = about the assistant or the "
        "available data/schema; out_of_scope = unrelated to business data."
    )
    reason: str = Field(description="One short sentence explaining the decision.")


class AnalysisPlan(BaseModel):
    """Planner output: the question broken into SQL-sized sub-questions."""

    reasoning: str = Field(description="Brief explanation of how the question will be answered.")
    sub_questions: list[str] = Field(
        description="1 to 3 self-contained questions, each answerable with ONE SQL query. "
        "Resolve references to earlier turns (e.g. 'same for North') into explicit questions."
    )


class SQLDraft(BaseModel):
    """SQL writer output."""

    reasoning: str = Field(description="Which tables, joins and filters are needed, in 1-3 sentences.")
    sql: str = Field(description="A single read-only SQLite SELECT statement. No markdown fences.")


class Insights(BaseModel):
    """Analyst output."""

    key_findings: list[str] = Field(description="3-6 concise findings, each citing concrete numbers.")
    caveats: list[str] = Field(default_factory=list, description="Data limitations or assumptions.")


class ChartSpec(BaseModel):
    """Chart agent output."""

    chart_type: Literal["bar", "line", "pie", "none"] = Field(
        description="bar for category comparisons, line for time series, pie for shares of a whole "
        "(max 8 slices), none if a chart would not help."
    )
    result_index: int = Field(default=0, description="Which query result to plot (0-based).")
    x: str | None = Field(default=None, description="Column for the x-axis / labels.")
    y: str | None = Field(default=None, description="Numeric column for the y-axis / values.")
    series: str | None = Field(default=None, description="Optional column to split into multiple series.")
    title: str = Field(default="", description="Short chart title.")
