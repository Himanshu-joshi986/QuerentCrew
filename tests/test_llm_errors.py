"""The assistant must fail fast and explain itself when the LLM provider is unusable."""

import pytest

pytest.importorskip("langgraph")

from querentcrew.llm import classify_llm_error, describe_llm_error  # noqa: E402
from querentcrew.schemas import AnalysisPlan, RouteDecision  # noqa: E402
from querentcrew.service import QuerentCrew  # noqa: E402

from .fakes import FakeLLM  # noqa: E402
from .scenarios import why_drop_script  # noqa: E402


class APIConnectionError(Exception):
    """Mimics openai/groq's APIConnectionError."""


@pytest.mark.parametrize(
    ("exc", "category"),
    [
        (APIConnectionError("Connection error."), "connection"),
        (Exception("Error code: 401 - invalid api key"), "auth"),
        (Exception("Error code: 429 - rate limit reached"), "rate_limit"),
        (TimeoutError("Request timed out."), "timeout"),
        (Exception("Error code: 503 - service unavailable"), "server"),
        (ValueError("sql_writer: model did not return valid SQLDraft"), None),
    ],
)
def test_classify_llm_error(exc, category):
    assert classify_llm_error(exc) == category
    assert (describe_llm_error(exc) is None) == (category is None)


def test_unreachable_llm_short_circuits_with_clear_message(settings):
    llm = FakeLLM({"router": APIConnectionError("Connection error.")})
    resp = QuerentCrew(settings, llm=llm).ask("Why did revenue drop in the South region in Q3 2025?")
    assert resp.status == "completed"
    assert resp.llm_error and "Couldn't reach the language model" in resp.answer
    assert "empty query" not in resp.answer
    agents = [t["agent"] for t in resp.trace]
    assert agents == ["router", "reporter"]  # no pointless SQL-guard retry loop


def test_outage_mid_run_stops_sql_loop(settings):
    llm = FakeLLM({
        "router": RouteDecision(intent="data_question", reason="r"),
        "planner": AnalysisPlan(reasoning="r", sub_questions=["revenue by region"]),
        "sql_writer": APIConnectionError("Connection error."),
    })
    resp = QuerentCrew(settings, llm=llm).ask("revenue?")
    assert [r["status"] for r in resp.results] == ["llm_error"]
    assert "sql_guard" not in [t["agent"] for t in resp.trace]
    assert len(llm.prompts["sql_writer"]) == 1


def test_streaming_reports_each_agent(settings):
    seen = []
    pilot = QuerentCrew(settings, llm=FakeLLM(why_drop_script()))
    resp = pilot.ask("Why did revenue drop in the South region in Q3?", on_step=lambda node, _: seen.append(node))
    assert resp.status == "completed" and "South" in resp.answer
    assert seen[0] == "router" and seen[-1] == "reporter" and "sql_executor" in seen
