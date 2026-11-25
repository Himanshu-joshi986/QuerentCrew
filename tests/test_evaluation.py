import json
from pathlib import Path

from querentcrew.evaluation import (SQLEvaluator, fingerprint_variants, fingerprints, numbers_close, results_match,
                                   results_match_round1, same_inputs, sign_test, wilson_interval)
from querentcrew.nodes import AgentNodes
from querentcrew.schemas import SQLDraft

from .fakes import FakeLLM

DATASET = json.loads((Path(__file__).resolve().parents[1] / "evals" / "dataset.json").read_text())
HOLDOUT = json.loads((Path(__file__).resolve().parents[1] / "evals" / "holdout.json").read_text())


def test_original_round1_grader_behaviour_is_preserved():
    assert results_match_round1(["a", "b"], [["x", 1.004], ["y", 2]], ["rev", "name"], [[2.0, "y"], [1.0, "x"]])
    # its known flaw, kept so the original results can be reproduced exactly
    assert not results_match_round1(["r"], [[0.046]], ["r"], [[0.05]])
    assert results_match_round1(["r"], [[0.05]], ["r"], [[0.14]])


def test_results_match_ignores_names_order_and_extra_columns():
    assert results_match(["a", "b"], [["x", 1.00004], ["y", 2]], ["rev", "name"], [[2.0, "y"], [1.0, "x"]])
    assert not results_match(["a"], [[1], [2]], ["a"], [[2], [1]], ordered=True)
    assert results_match(["a"], [[1], [2]], ["a"], [[2], [1]], ordered=False)
    assert not results_match(["a"], [[1]], ["a"], [[1], [2]])
    assert results_match(["n"], [[5]], ["name", "n"], [["extra column", 5]])
    assert results_match(["c", "r"], [["Online", 0.046], ["Partner", 0.0518]],
                         ["r", "c"], [[0.05, "Partner"], [0.05, "Online"]])
    assert not results_match(["c", "r"], [["Online", 0.046]], ["c", "r"], [["Retail", 0.046]])


def test_numbers_close_allows_rounding_but_not_different_answers():
    # rounding by either side
    assert numbers_close(0.046, 0.05)              # rate rounded to 2 places
    assert numbers_close(1234.5678, 1234.57)
    assert numbers_close(31.196066, 31.2)          # 31.20 prints as 31.2
    assert numbers_close(7173382.4, 7173382.0)     # large amount rounded to whole units (0.01% rule)
    assert numbers_close(0.30000000000000004, 0.3)
    # different answers
    assert not numbers_close(0.05, 0.14)           # the round1 grader accepted this
    assert not numbers_close(0.0504, 0.1)          # one decimal place gets at most 0.005 allowance
    assert not numbers_close(5.0, 6.0)             # whole numbers must be equal
    assert not numbers_close(1000.0, 1001.0)
    assert not numbers_close(0.046, 4.6)           # fraction vs percent is a different value
    assert not numbers_close(0.0, 0.004)
    assert not numbers_close(2394.97, 2390.03)     # AOV with and without the completed filter


def test_grader_rejects_every_broken_query_and_accepts_harmless_rewrites(db):
    """Mutation testing of the grader on all 40 gold queries (see grader_validation.py)."""
    from querentcrew.grader_validation import summarize, validate
    checks = validate(DATASET + HOLDOUT, db, results_match)
    failures = [(c.id, c.kind, c.name) for c in checks if not c.ok]
    assert failures == []
    s = summarize(checks)
    assert s["mutants_sql_error"] == 0
    rejected, total = map(int, s["mutants_rejected"].split("/"))
    assert rejected == total and total >= 150


def test_original_grader_misjudges_some_rewrites_and_mutants(db):
    """Documents why the grader was corrected: round1 accepts broken queries and rejects correct ones."""
    from querentcrew.grader_validation import validate
    checks = validate(DATASET + HOLDOUT, db, results_match_round1)
    assert any(c.kind == "mutant" and not c.ok for c in checks)
    assert any(c.kind == "rewrite" and not c.ok for c in checks)


def test_fingerprints_ignore_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.yaml", tmp_path / "crlf.yaml"
    lf.write_bytes(b"a: 1\nb: 2\n")
    crlf.write_bytes(b"a: 1\r\nb: 2\r\n")
    assert fingerprints({"f": lf}) == fingerprints({"f": crlf})
    assert fingerprint_variants(lf) == fingerprint_variants(crlf)
    import hashlib
    old_style = {"f": hashlib.sha256(crlf.read_bytes()).hexdigest()}  # raw-bytes hash from a Windows run
    assert same_inputs(old_style, {"f": lf})
    lf.write_bytes(b"a: 1\nb: 3\n")
    assert not same_inputs(old_style, {"f": lf})


def test_wilson_interval_and_sign_test():
    lo, hi = wilson_interval(1.0, 20)
    assert round(lo, 3) == 0.839 and hi == 1.0
    lo, hi = wilson_interval(0.9, 20)
    assert round(lo, 3) == 0.699 and round(hi, 3) == 0.972
    t = sign_test({"a": 0, "b": 0, "c": 1}, {"a": 1, "b": 1, "c": 1})
    assert (t["better"], t["worse"], t["same"]) == (2, 0, 1) and t["p_value"] == 0.5
    assert sign_test({"a": 1}, {"a": 1})["p_value"] == 1.0


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
