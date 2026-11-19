"""End-to-end tests of the compiled LangGraph workflow (memory + human-in-the-loop)."""

import pytest

pytest.importorskip("langgraph")

from querentcrew.schemas import AnalysisPlan, RouteDecision, SQLDraft  # noqa: E402
from querentcrew.service import QuerentCrew  # noqa: E402

from .fakes import FakeLLM  # noqa: E402
from .scenarios import pii_script, why_drop_script  # noqa: E402


def test_full_pipeline_with_self_correction(settings):
    pilot = QuerentCrew(settings, llm=FakeLLM(why_drop_script()))
    resp = pilot.ask("Why did revenue drop in the South region in Q3?")
    assert resp.status == "completed"
    assert len(resp.plan) == 2
    assert [r["status"] for r in resp.results] == ["ok", "ok"]
    assert resp.results[0]["attempts"] == 2  # first SQL failed, agent fixed it
    assert resp.chart and resp.chart["image_b64"]
    assert "South" in resp.answer
    agents = [t["agent"] for t in resp.trace]
    assert agents[0] == "router" and agents[-1] == "reporter"


def test_human_in_the_loop_pause_and_resume(settings):
    pilot = QuerentCrew(settings, llm=FakeLLM(pii_script()))
    first = pilot.ask("Emails of Enterprise customers")
    assert first.status == "awaiting_approval"
    assert first.approval_request["pii_columns"] == ["email"]
    done = pilot.resume(first.thread_id, approved=True)
    assert done.status == "completed"
    assert done.results[0]["masked_columns"] == ["email"]


def test_rejection_path(settings):
    pilot = QuerentCrew(settings, llm=FakeLLM(pii_script()))
    first = pilot.ask("Emails of Enterprise customers")
    done = pilot.resume(first.thread_id, approved=False)
    assert done.results[0]["status"] == "rejected"
    assert "couldn't answer" in done.answer


def test_memory_across_turns_and_general_route(settings):
    script = {
        "router": [RouteDecision(intent="data_question", reason="r"), RouteDecision(intent="data_question", reason="r"),
                   RouteDecision(intent="out_of_scope", reason="joke")],
        "planner": lambda s, u, sc: AnalysisPlan(reasoning="r", sub_questions=["count"]),
        "sql_writer": SQLDraft(reasoning="r", sql="SELECT COUNT(*) AS n FROM orders"),
        "analyst": {"key_findings": ["n orders"], "caveats": []},
        "reporter": "There are many orders.",
        "responder": "I only answer questions about your business data.",
    }
    llm = FakeLLM(script)
    pilot = QuerentCrew(settings, llm=llm)
    thread = pilot.new_thread_id()
    pilot.ask("How many orders?", thread)
    pilot.ask("Now the same for North", thread)
    # the second planner call saw the first turn in its history
    assert "How many orders?" in llm.prompts["planner"][1][1]
    resp = pilot.ask("Tell me a joke", thread)
    assert resp.intent == "out_of_scope" and resp.results == []
    assert len(pilot.history(thread)) == 6
