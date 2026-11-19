from querentcrew.nodes import AgentNodes
from querentcrew.schemas import AnalysisPlan, ChartSpec, SQLDraft

from .fakes import FakeLLM
from .scenarios import pii_script, why_drop_script


def make_nodes(settings, db, script, decision=None):
    return AgentNodes(FakeLLM(script), db, settings,
                      interrupt_fn=lambda payload: decision if decision is not None else {"approved": True})


def base_state(question="Why did South revenue drop in Q3?"):
    return {"question": question, "history": [], "results": [], "trace": [], "attempts": 0,
            "current_index": 0, "feedback": ""}


def apply(state, update):
    state.update(update)
    return state


def test_router_falls_back_to_data_path_on_llm_error(settings, db):
    nodes = make_nodes(settings, db, {"router": RuntimeError("provider down")})
    out = nodes.router(base_state())
    assert out["intent"] == "data_question"


def test_planner_caps_and_defaults(settings, db):
    many = AnalysisPlan(reasoning="r", sub_questions=["a", "b", "c", "d", "e"])
    nodes = make_nodes(settings, db, {"planner": [many, AnalysisPlan(reasoning="r", sub_questions=[])]})
    assert len(nodes.planner(base_state())["sub_questions"]) == settings.max_sub_questions
    assert nodes.planner(base_state("q?"))["sub_questions"] == ["q?"]


def test_execution_error_triggers_retry_with_feedback(settings, db):
    nodes = make_nodes(settings, db, {"sql_writer": SQLDraft(reasoning="r", sql="SELECT nope FROM orders")})
    state = apply(base_state(), {"sub_questions": ["x"]})
    apply(state, nodes.sql_writer(state))
    apply(state, nodes.sql_guard(state))
    out = nodes.sql_executor(state)
    assert out["next_action"] == "retry"
    assert "no such column" in out["feedback"]


def test_retries_exhausted_records_failure(settings, db):
    nodes = make_nodes(settings, db, {"sql_writer": SQLDraft(reasoning="r", sql="SELECT nope FROM orders")})
    state = apply(base_state(), {"sub_questions": ["x"]})
    for _ in range(settings.max_sql_attempts):
        apply(state, nodes.sql_writer(state))
        apply(state, nodes.sql_guard(state))
        apply(state, nodes.sql_executor(state))
    assert state["next_action"] == "advance"
    assert state["results"][-1]["status"] == "failed"


def test_blocked_sql_is_rewritten(settings, db):
    script = {"sql_writer": [SQLDraft(reasoning="r", sql="DELETE FROM orders"),
                             SQLDraft(reasoning="r", sql="SELECT COUNT(*) AS n FROM orders")]}
    nodes = make_nodes(settings, db, script)
    state = apply(base_state(), {"sub_questions": ["x"]})
    apply(state, nodes.sql_writer(state))
    apply(state, nodes.sql_guard(state))
    assert state["next_action"] == "retry" and "safety layer" in state["feedback"]
    apply(state, nodes.sql_writer(state))
    apply(state, nodes.sql_guard(state))
    assert state["next_action"] == "execute"
    # the rejection reason was passed to the SQL agent's prompt
    assert "safety layer" in nodes.llm.prompts["sql_writer"][1][1]


def test_pii_rejection_and_masking(settings, db):
    rejected = make_nodes(settings, db, pii_script(), decision={"approved": False, "comment": "no"})
    state = apply(base_state("emails?"), {"sub_questions": ["emails"]})
    apply(state, rejected.sql_writer(state))
    apply(state, rejected.sql_guard(state))
    assert state["next_action"] == "approve"
    apply(state, rejected.human_approval(state))
    assert state["results"][-1]["status"] == "rejected"

    approved = make_nodes(settings, db, pii_script())
    state = apply(base_state("emails?"), {"sub_questions": ["emails"]})
    for step in (approved.sql_writer, approved.sql_guard, approved.human_approval, approved.sql_executor):
        apply(state, step(state))
    rec = state["results"][-1]
    assert rec["status"] == "ok" and rec["masked_columns"] == ["email"]
    assert all("***@" in row[1] for row in rec["rows"])


def test_analyst_charter_reporter(settings, db):
    nodes = make_nodes(settings, db, why_drop_script())
    state = apply(base_state(), {"sub_questions": ["a"]})
    rec = db.execute("SELECT r.name AS region, COUNT(*) AS customers FROM customers c "
                     "JOIN regions r ON r.region_id = c.region_id GROUP BY r.name")
    state["results"] = [{"sub_question": "a", "sql": "", "status": "ok", "columns": rec.columns, "rows": rec.rows,
                         "row_count": rec.row_count, "truncated": False, "masked_columns": [], "error": None}]
    apply(state, nodes.analyst(state))
    assert state["insights"]["key_findings"]
    # the analyst prompt contains Pandas statistics
    assert "share_of_total" in nodes.llm.prompts["analyst"][0][1]
    apply(state, nodes.charter(state))
    assert state["chart"]["image_b64"]
    apply(state, nodes.reporter(state))
    assert state["answer"].startswith("**South")
    assert state["history"][-1]["role"] == "assistant"


def _result(db, sql):
    rec = db.execute(sql)
    return {"sub_question": "a", "sql": sql, "status": "ok", "columns": rec.columns, "rows": rec.rows,
            "row_count": rec.row_count, "truncated": False, "masked_columns": [], "error": None}


def test_charter_falls_back_when_model_declines_chartable_data(settings, db):
    nodes = make_nodes(settings, db, {"charter": ChartSpec(chart_type="none")})
    state = base_state()
    state["results"] = [_result(db, "SELECT r.name AS region, COUNT(*) AS customers FROM customers c "
                                    "JOIN regions r ON r.region_id = c.region_id GROUP BY r.name")]
    out = nodes.charter(state)
    assert out["chart"] and out["chart"]["image_b64"]
    assert out["chart"]["spec"]["chart_type"] == "bar"


def test_charter_skips_text_only_results(settings, db):
    nodes = make_nodes(settings, db, {"charter": ChartSpec(chart_type="none")})
    state = base_state()
    state["results"] = [_result(db, "SELECT name, segment FROM customers LIMIT 5")]
    assert nodes.charter(state)["chart"] is None


def test_reporter_without_results_explains(settings, db):
    nodes = make_nodes(settings, db, {})
    state = base_state()
    state["results"] = [{"sub_question": "emails", "status": "rejected", "error": "rejected by user"}]
    out = nodes.reporter(state)
    assert "couldn't answer" in out["answer"] and "rejected" in out["answer"]
