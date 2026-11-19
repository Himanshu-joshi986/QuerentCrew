import json
from pathlib import Path

from querentcrew.evaluation import SQLEvaluator, results_match
from querentcrew.nodes import AgentNodes
from querentcrew.schemas import SQLDraft

from .fakes import FakeLLM

DATASET = json.loads((Path(__file__).resolve().parents[1] / "evals" / "dataset.json").read_text())


def test_results_match_ignores_names_order_and_rounding():
    assert results_match(["a", "b"], [["x", 1.004], ["y", 2]], ["rev", "name"], [[2.0, "y"], [1.0, "x"]])
    assert not results_match(["a"], [[1], [2]], ["a"], [[2], [1]], ordered=True)
    assert not results_match(["a"], [[1]], ["a"], [[1], [2]])
    assert results_match(["n"], [[5]], ["name", "n"], [["extra column", 5]])


def test_all_gold_queries_run(db):
    for item in DATASET:
        db.execute(item["gold_sql"])


def test_evaluator_with_perfect_and_broken_agent(settings, db, tmp_path):
    golds = iter(item["gold_sql"] for item in DATASET[:3])
    perfect = AgentNodes(FakeLLM({"sql_writer": lambda s, u, sc: SQLDraft(reasoning="", sql=next(golds))}),
                         db, settings)
    evaluator = SQLEvaluator(perfect, db)
    records = evaluator.run(DATASET[:3], ["v3"], progress=None)
    summary = evaluator.summarize(records)
    assert summary[0]["final_accuracy_pct"] == 100.0

    broken = AgentNodes(FakeLLM({"sql_writer": SQLDraft(reasoning="", sql="SELECT nope FROM orders")}),
                        db, settings)
    rec = SQLEvaluator(broken, db).run(DATASET[:1], ["v1"], progress=None)[0]
    assert rec.status == "error" and rec.attempts == settings.max_sql_attempts

    report = SQLEvaluator.write_reports(records, summary, tmp_path, "fake:model")
    assert "final_accuracy_pct" in report.read_text()


def test_evaluator_waits_out_rate_limits_instead_of_scoring_them(settings, db):
    gold = DATASET[0]["gold_sql"]
    rate_limit = Exception("Error code: 429 - rate limit reached")
    llm = FakeLLM({"sql_writer": [rate_limit, rate_limit, SQLDraft(reasoning="", sql=gold)]})
    waits = []
    evaluator = SQLEvaluator(AgentNodes(llm, db, settings), db, sleep=waits.append)
    rec = evaluator.run(DATASET[:1], ["v3"], progress=None)[0]
    assert rec.status == "correct" and rec.first_try_correct and rec.attempts == 1
    assert len(waits) == 2


def test_evaluator_reports_persistent_provider_failure_separately(settings, db):
    llm = FakeLLM({"sql_writer": Exception("Error code: 503 - service unavailable")})
    evaluator = SQLEvaluator(AgentNodes(llm, db, settings), db, provider_retries=1, sleep=lambda s: None)
    records = evaluator.run(DATASET[:1], ["v3"], progress=None)
    assert records[0].status == "llm_error"
    summary = evaluator.summarize(records)[0]
    assert summary["provider_failures"] == 1


HOLDOUT = json.loads((Path(__file__).resolve().parents[1] / "evals" / "holdout.json").read_text())
PROMPT = (Path(__file__).resolve().parents[1] / "src" / "querentcrew" / "prompts" / "sql_writer.yaml").read_text()


def _norm_q(text):
    import re
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).replace(" in 2025", " ").split())


def test_all_holdout_gold_queries_run_and_are_not_empty(db):
    for item in HOLDOUT:
        result = db.execute(item["gold_sql"])
        assert result.rows, item["id"]
        assert result.rows != [(0,)] and result.rows != [[0]], f"{item['id']}: a zero answer is too easy to hit"


def test_no_test_question_leaks_into_prompt_examples():
    """Few-shot examples must not be (near-)copies of any evaluation question."""
    import difflib
    examples = [line.split("Q:", 1)[1].strip() for line in PROMPT.splitlines() if line.strip().startswith("Q:")]
    assert examples
    for item in DATASET + HOLDOUT:
        for ex in examples:
            ratio = difflib.SequenceMatcher(None, _norm_q(ex), _norm_q(item["question"])).ratio()
            assert ratio < 0.75, f"{item['id']} '{item['question']}' is too close to prompt example '{ex}'"


def test_dev_and_holdout_do_not_overlap():
    import difflib
    for h in HOLDOUT:
        for d in DATASET:
            assert difflib.SequenceMatcher(None, _norm_q(h["question"]), _norm_q(d["question"])).ratio() < 0.8, \
                (h["id"], d["id"])


def test_evaluator_stops_on_daily_quota_instead_of_waiting(settings, db):
    import pytest
    from querentcrew.evaluation import DailyQuotaReached
    llm = FakeLLM({"sql_writer": Exception("Error code: 429 - Rate limit reached on tokens per day (TPD): "
                                           "Limit 200000, Used 199578")})
    waits = []
    evaluator = SQLEvaluator(AgentNodes(llm, db, settings), db, sleep=waits.append)
    with pytest.raises(DailyQuotaReached):
        evaluator.run(DATASET[:1], ["v3"], progress=None)
    assert waits == []
