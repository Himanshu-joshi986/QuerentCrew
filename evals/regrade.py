"""Re-grade the saved answers of an evaluation run, without calling any model.

Every answer's SQL is stored in evals/results/checkpoint.jsonl, so grading can be repeated and
checked by anyone, for free:

  1. Checks that the prompts, glossary, test sets and database are the ones the run used.
  2. Re-grades every answer with the original grader (`round1`). This must reproduce the original
     report exactly; it proves the saved answers and the grader are what produced those numbers.
  3. Re-grades with the current grader (`tolerance`).
  4. Test-suite check (Zhong et al., 2020): an answer marked correct is also run against 5 extra
     databases generated with different random seeds, together with the gold query. It stays correct
     only if it matches on all of them, so a query that was right only by luck on one dataset is
     caught.
  5. Applies the manual review (evals/manual_review.json) and measures how often the grader agrees
     with it.
  6. Reports accuracy with 95% intervals over questions and paired comparisons between prompt
     versions, a per-question table and every verdict that changed, with the actual result rows.

Usage:
    python evals/regrade.py               # writes evals/results/regrade_report.md
    python evals/regrade.py --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evals"))

from querentcrew.config import get_settings  # noqa: E402
from querentcrew.database import DatabaseManager, QueryError  # noqa: E402
from querentcrew.evaluation import (  # noqa: E402
    EvalRecord, fingerprints, results_match, results_match_round1, same_inputs, sign_test, wilson_interval)
from querentcrew.grader_validation import summarize as summarize_checks, validate  # noqa: E402
from run_eval import SUITES, input_files  # noqa: E402
from seed_database import seed  # noqa: E402

VERSIONS = ["v1", "v2", "v3"]
GRADES = ["recorded", "round1", "tolerance", "test_suite", "manual_strict", "manual_lenient"]
GRADE_LABEL = {
    "recorded": "as recorded in the run",
    "round1": "original grader, re-run",
    "tolerance": "corrected grader",
    "test_suite": "corrected grader + 5 extra databases",
    "manual_strict": "manual review, ambiguous = wrong",
    "manual_lenient": "manual review, ambiguous = correct",
}
MANUAL = ROOT / "evals" / "manual_review.json"


def preview(result, limit: int = 6) -> str:
    if result is None:
        return "(no result)"
    rows = [tuple(round(v, 4) if isinstance(v, float) else v for v in r) for r in result.rows[:limit]]
    more = f" ... ({len(result.rows)} rows)" if len(result.rows) > limit else ""
    return f"{list(result.columns)} {rows}{more}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(ROOT / "evals" / "results" / "checkpoint.jsonl"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--out", default=str(ROOT / "evals" / "results"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    settings = get_settings()
    lines = Path(args.checkpoint).read_text(encoding="utf-8").splitlines()
    config = json.loads(lines[0])["config"]
    records = [EvalRecord(**json.loads(line)) for line in lines[1:]]
    records = [r for r in records if r.status != "llm_error"]
    if not same_inputs(config["fingerprints"], input_files(settings)):
        sys.exit("The input files on disk differ from the ones this checkpoint was made with.")
    print(f"{len(records)} saved answers; input files match the run.")

    items = {(suite, q["id"]): q for suite, path in SUITES.items()
             for q in json.loads(path.read_text(encoding="utf-8"))}
    main_db = DatabaseManager(settings.database_path, settings.query_timeout_seconds, settings.max_result_rows)

    alt_dir = Path(args.out) / "testsuite_dbs"
    alt_dbs = {}
    for s in args.seeds:
        path = alt_dir / f"sales_seed{s}.db"
        if not path.exists():
            print(f"Generating extra database with seed {s} ...")
            seed(path, s)
        alt_dbs[s] = DatabaseManager(path, settings.query_timeout_seconds, settings.max_result_rows)

    def run(db, sql):
        try:
            return db.execute(sql)
        except QueryError:
            return None

    gold_cache, pred_cache, ts_cache = {}, {}, {}
    rows = []
    for r in records:
        item = items[(r.suite, r.id)]
        ordered = item.get("ordered", False)
        key = (r.suite, r.id)
        if key not in gold_cache:
            gold_cache[key] = main_db.execute(item["gold_sql"])
        gold = gold_cache[key]
        pred = None
        if r.sql and r.status in {"correct", "wrong"}:
            pred = pred_cache.setdefault((key, r.sql), run(main_db, r.sql))
        g_round1 = bool(pred) and results_match_round1(gold.columns, gold.rows, pred.columns, pred.rows, ordered)
        g_tol = bool(pred) and results_match(gold.columns, gold.rows, pred.columns, pred.rows, ordered)
        failed_on = []
        if g_tol:
            if (key, r.sql) not in ts_cache:
                fails = []
                for s, db in alt_dbs.items():
                    g_alt, p_alt = db.execute(item["gold_sql"]), run(db, r.sql)
                    if p_alt is None or not results_match(g_alt.columns, g_alt.rows, p_alt.columns,
                                                          p_alt.rows, ordered):
                        fails.append(s)
                ts_cache[(key, r.sql)] = fails
            failed_on = ts_cache[(key, r.sql)]
        rows.append({"rec": r, "gold": gold, "pred": pred, "failed_on": failed_on,
                     "recorded": r.final_correct, "round1": g_round1, "tolerance": g_tol,
                     "test_suite": g_tol and not failed_on})

    # ---- manual review ---------------------------------------------------------------------
    labels = {}
    for entry in json.loads(MANUAL.read_text(encoding="utf-8"))["labels"]:
        for run_no in entry["runs"]:
            labels[(entry["suite"], run_no, entry["version"], entry["id"])] = entry
    rng = random.Random(7)
    sample_ids = {id(x) for x in rng.sample([x for x in rows if x["test_suite"]],
                                            min(20, sum(x["test_suite"] for x in rows)))}
    for x in rows:
        r = x["rec"]
        entry = labels.get((r.suite, r.run, r.prompt_version, r.id))
        manual = entry["manual"] if entry else ("correct" if x["test_suite"] else "wrong")
        x["manual"] = manual
        x["manual_strict"] = manual == "correct"
        x["manual_lenient"] = manual in {"correct", "ambiguous"}
        x["reviewed"] = (not x["test_suite"]) or x["round1"] != x["tolerance"] or id(x) in sample_ids             or entry is not None

    # ---- summaries -------------------------------------------------------------------------
    def per_question(suite, version, grade):
        scores = defaultdict(list)
        for x in rows:
            r = x["rec"]
            if r.suite == suite and r.prompt_version == version:
                scores[r.id].append(float(x[grade]))
        return {k: statistics.mean(v) for k, v in scores.items()}

    def accuracy_row(suite, version, grade):
        group = [x for x in rows if x["rec"].suite == suite and x["rec"].prompt_version == version]
        runs = sorted({x["rec"].run for x in group})
        per_run = [100 * statistics.mean(float(x[grade]) for x in group if x["rec"].run == k) for k in runs]
        pq = per_question(suite, version, grade)
        mean = statistics.mean(pq.values())
        lo, hi = wilson_interval(mean, len(pq))
        first = 100 * statistics.mean(float(x[grade] and x["rec"].attempts == 1) for x in group)
        return {"mean": 100 * mean, "lo": 100 * lo, "hi": 100 * hi, "min": min(per_run), "max": max(per_run),
                "first": first, "n_q": len(pq), "n": len(group)}

    out = []
    out += ["# Re-grading of the saved answers", "",
            f"Model: `{config['model']}`, temperature {config['temperature']}, "
            f"{len(records)} saved answers (`evals/results/checkpoint.jsonl`). No model was called.", "",
            "Command: `python evals/regrade.py`", "",
            "Input files on disk match the fingerprints stored in the checkpoint (line endings ignored).", ""]

    # reproducibility check
    mismatch = [x for x in rows if x["recorded"] != x["round1"]]
    out += ["## 1. Reproducibility check", "",
            f"Re-running the **original grader** on the saved SQL gives the recorded verdict for "
            f"**{len(rows) - len(mismatch)} of {len(rows)}** answers.", ""]

    # grader validation
    dataset = [q for (_s, _i), q in items.items()]
    out += ["## 2. Grader validation (mutation testing)", "",
            "Each gold query is changed automatically. *Broken queries* change the meaning and must be "
            "rejected; *harmless rewrites* keep the meaning and must be accepted "
            "(`src/querentcrew/grader_validation.py`).", "",
            "| grader | gold vs itself | broken queries rejected | harmless rewrites accepted | "
            "broken queries with an identical result (no grader can tell) |", "|---|---|---|---|---|"]
    for name, fn in (("original (round1)", results_match_round1), ("corrected (tolerance)", results_match)):
        sm = summarize_checks(validate(dataset, main_db, fn))
        out.append(f"| {name} | {sm['gold_self_match']} | {sm['mutants_rejected']} | {sm['rewrites_accepted']} "
                   f"| {sm['mutants_same_result']} |")
    out.append("")

    # accuracy tables
    out += ["## 3. Final accuracy", "",
            "Mean over 3 runs. *95% interval* is a Wilson interval over the 20 questions of the suite "
            "(each question scored as its mean over the runs), so repeated runs are not treated as extra "
            "questions. *Run range* is the lowest and highest single run.", ""]
    for suite in SUITES:
        out += [f"### Suite: {suite}", "",
                "| prompt | " + " | ".join(GRADE_LABEL[g] for g in GRADES) + " | 95% interval (corrected grader + 5 extra databases) "
                "| run range (same) |", "|---|" + "---|" * (len(GRADES) + 2)]
        for v in VERSIONS:
            cells = [f"{accuracy_row(suite, v, g)['mean']:.1f}%" for g in GRADES]
            final = accuracy_row(suite, v, "test_suite")
            out.append(f"| {v} | " + " | ".join(cells) + f" | {final['lo']:.1f}–{final['hi']:.1f}% "
                       f"| {final['min']:.0f}–{final['max']:.0f}% |")
        out.append("")

    out += ["### First-try accuracy (corrected grader + 5 extra databases)", "",
            "| suite | v1 | v2 | v3 |", "|---|---|---|---|"]
    for suite in SUITES:
        out.append(f"| {suite} | " + " | ".join(f"{accuracy_row(suite, v, 'test_suite')['first']:.1f}%"
                                                for v in VERSIONS) + " |")
    out.append("")

    # agreement with manual review
    reviewed = [x for x in rows if x["reviewed"]]
    decided = [x for x in reviewed if x["manual"] != "ambiguous"]
    fn = [x for x in decided if x["manual"] == "correct" and not x["test_suite"]]
    fp = [x for x in decided if x["manual"] == "wrong" and x["test_suite"]]
    amb = [x for x in reviewed if x["manual"] == "ambiguous"]
    out += ["## Agreement between the grader and the manual review", "",
            f"Answers reviewed by hand: **{len(reviewed)}** of {len(rows)} (every answer graded wrong, every "
            "answer whose verdict changed between graders, and a random sample of 20 graded correct). "
            "Labels and reasons: `evals/manual_review.json`.", "",
            "| | answers |", "|---|---|",
            f"| grader and reviewer agree | {len(decided) - len(fn) - len(fp)} |",
            f"| grader says wrong, reviewer says correct (false negative) | {len(fn)} |",
            f"| grader says correct, reviewer says wrong (false positive) | {len(fp)} |",
            f"| question allows two readings (ambiguous) | {len(amb)} |",
            f"| **agreement on unambiguous answers** | **{len(decided) - len(fn) - len(fp)}/{len(decided)} "
            f"({100 * (len(decided) - len(fn) - len(fp)) / max(1, len(decided)):.1f}%)** |", ""]

    # paired comparisons
    out += ["## 4. Paired comparison of prompt versions", "",
            "Same questions, compared one by one (corrected grader + 5 extra databases). *Better / worse* "
            "count questions whose score (correct runs out of 3) went up or down. *p* is the exact "
            "two-sided sign test: the chance of a split at least this lopsided if the two versions were "
            "equally good.", "", "| suite | comparison | better | worse | same | p |", "|---|---|---|---|---|---|"]
    for suite in SUITES:
        for a, b in (("v1", "v2"), ("v2", "v3"), ("v1", "v3")):
            t = sign_test(per_question(suite, a, "test_suite"), per_question(suite, b, "test_suite"))
            out.append(f"| {suite} | {a} → {b} | {t['better']} | {t['worse']} | {t['same']} | {t['p_value']:.4f} |")
    out.append("")

    # per-question table
    out += ["## 5. Per-question results", "",
            "Correct runs out of 3 (corrected grader + 5 extra databases).", "",
            "| suite | id | difficulty | v1 | v2 | v3 | question |", "|---|---|---|---|---|---|---|"]
    for (suite, qid), item in items.items():
        cells = []
        for v in VERSIONS:
            grp = [x for x in rows if x["rec"].suite == suite and x["rec"].id == qid and x["rec"].prompt_version == v]
            cells.append(f"{sum(x['test_suite'] for x in grp)}/{len(grp)}")
        out.append(f"| {suite} | {qid} | {item.get('difficulty', '')} | " + " | ".join(cells)
                   + f" | {item['question']} |")
    out.append("")

    # changed verdicts
    changed = [x for x in rows if x["round1"] != x["tolerance"] or x["tolerance"] != x["test_suite"]]
    out += ["## 6. Every verdict that changed", "",
            f"{len(changed)} answers. *Gold* is the result of the hand-written query, *model* the result of "
            "the saved SQL (first rows shown).", ""]
    for x in changed:
        r = x["rec"]
        why = ("original → corrected grader" if x["round1"] != x["tolerance"]
               else f"failed on extra database(s) with seed {x['failed_on']}")
        verdict = (f"{'correct' if x['round1'] else 'wrong'} → {'correct' if x['tolerance'] else 'wrong'}"
                   if x["round1"] != x["tolerance"] else "correct → wrong")
        out += [f"- **{r.suite} run {r.run} {r.prompt_version} {r.id}** ({why}): {verdict}",
                f"  - question: {r.question}",
                f"  - gold: `{preview(x['gold'])}`",
                f"  - model: `{preview(x['pred'])}`",
                f"  - SQL: `{' '.join(r.sql.split())}`"]
    out.append("")
    report = Path(args.out) / "regrade_report.md"
    report.write_text("\n".join(out) + "\n", encoding="utf-8")

    # ---- evidence for the manual audit -------------------------------------------------------
    ev = ["# Evidence for the manual audit", "",
          "Generated by `python evals/regrade.py`. For every question/prompt pair with at least one answer "
          "graded wrong (corrected grader + 5 extra databases), each distinct SQL the model wrote is shown "
          "with its result next to the gold result. Then a random sample of answers graded correct.", ""]
    groups = defaultdict(list)
    for x in rows:
        if not x["test_suite"]:
            groups[(x["rec"].suite, x["rec"].prompt_version, x["rec"].id)].append(x)
    for (suite, v, qid), xs in groups.items():
        item = items[(suite, qid)]
        ev += [f"## {suite} {v} {qid}: {item['question']}", "",
               f"- gold SQL: `{' '.join(item['gold_sql'].split())}`",
               f"- gold result: `{preview(xs[0]['gold'], 12)}`"]
        for sql in dict.fromkeys(x["rec"].sql for x in xs):
            runs = [x["rec"].run for x in xs if x["rec"].sql == sql]
            x0 = next(x for x in xs if x["rec"].sql == sql)
            ev += [f"- runs {runs}: `{' '.join(sql.split())}`",
                   f"  - result: `{preview(x0['pred'], 12)}`"
                   + (f" (matched here, failed on extra databases {x0['failed_on']})" if x0["failed_on"] else "")]
        ev.append("")
    ev += ["## Random sample of 20 answers graded correct (seed 7)", ""]
    for x in [x for x in rows if id(x) in sample_ids]:
        r = x["rec"]
        ev += [f"- **{r.suite} run {r.run} {r.prompt_version} {r.id}**: {r.question}",
               f"  - gold: `{preview(x['gold'])}`", f"  - model: `{preview(x['pred'])}`",
               f"  - SQL: `{' '.join(r.sql.split())}`"]
    (Path(args.out) / "audit_evidence.md").write_text("\n".join(ev) + "\n", encoding="utf-8")

    summary = {suite: {v: {g: round(accuracy_row(suite, v, g)["mean"], 1) for g in GRADES}
                       for v in VERSIONS} for suite in SUITES}
    (Path(args.out) / "regrade_summary.json").write_text(
        json.dumps({"model": config["model"], "fingerprints": fingerprints(input_files(settings)),
                    "seeds": args.seeds, "accuracy_pct": summary}, indent=2), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
