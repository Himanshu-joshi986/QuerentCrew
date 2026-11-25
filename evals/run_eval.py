"""Measure the SQL agent's execution accuracy for each prompt version.

Two test suites:
  dev      evals/dataset.json   20 questions the prompts were developed against
  holdout  evals/holdout.json   20 questions written after the prompts were frozen (never used for tuning)

Usage:
    python evals/run_eval.py                          # both suites, v1 v2 v3, 1 run
    python evals/run_eval.py --repeats 3              # 3 runs, reports mean and min-max range
    python evals/run_eval.py --suites holdout --versions v3 --limit 5   # quick smoke run
    python evals/run_eval.py --fingerprints           # print SHA-256 of every input file and exit

Results are saved after every question to evals/results/checkpoint.jsonl. If a run stops (for example
on a provider's daily token limit), run the same command again and it continues where it left off.
Use --fresh to start over. Writes evals/results/latest_report.md plus timestamped CSV/JSON details.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from querentcrew.config import get_settings  # noqa: E402
from querentcrew.database import DatabaseManager  # noqa: E402
from querentcrew.evaluation import (  # noqa: E402
    DailyQuotaReached, EvalRecord, SQLEvaluator, fingerprints, same_inputs)
from querentcrew.llm import LangChainLLM  # noqa: E402
from querentcrew.nodes import AgentNodes  # noqa: E402

SUITES = {"dev": ROOT / "evals" / "dataset.json", "holdout": ROOT / "evals" / "holdout.json"}


def input_files(settings) -> dict[str, Path]:
    prompts = ROOT / "src" / "querentcrew" / "prompts"
    return {
        "prompts/sql_writer.yaml": prompts / "sql_writer.yaml",
        "data/business_glossary.md": settings.glossary_path,
        "evals/dataset.json": SUITES["dev"],
        "evals/holdout.json": SUITES["holdout"],
        "data/sales.db": settings.database_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    parser.add_argument("--suites", nargs="+", default=["dev", "holdout"], choices=list(SUITES))
    parser.add_argument("--repeats", type=int, default=1, help="Independent runs (reports mean and range)")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions per suite")
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds to wait between questions (rate limits)")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved progress and start over")
    parser.add_argument("--fingerprints", action="store_true", help="Print input file hashes and exit")
    parser.add_argument("--out", default=str(ROOT / "evals" / "results"))
    args = parser.parse_args()

    settings = get_settings()
    prints = fingerprints(input_files(settings))
    if args.fingerprints:
        for name, digest in prints.items():
            print(f"{digest}  {name}")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.jsonl"
    config = {"model": settings.llm_model, "temperature": settings.llm_temperature,
              "max_sql_retries": settings.max_sql_retries, "grader": "tolerance", "fingerprints": prints}

    records: list[EvalRecord] = []
    if checkpoint.exists() and not args.fresh:
        lines = checkpoint.read_text(encoding="utf-8").splitlines()
        saved = json.loads(lines[0]) if lines else {}
        saved_cfg = {"grader": "round1", **saved.get("config", {})}  # checkpoints without a grader used round1
        settings_match = {k: v for k, v in saved_cfg.items() if k != "fingerprints"} ==             {k: v for k, v in config.items() if k != "fingerprints"}
        if not settings_match or not same_inputs(saved_cfg.get("fingerprints", {}), input_files(settings)):
            sys.exit("Saved progress in evals/results/checkpoint.jsonl was made with a different model, settings, "
                     "grader or input files. Rerun with --fresh to start over, or re-grade the saved answers "
                     "without any API calls: python evals/regrade.py")
        # provider failures are not kept: they are retried on resume
        records = [EvalRecord(**json.loads(line)) for line in lines[1:]]
        records = [r for r in records if r.status != "llm_error"]
        print(f"Resuming: {len(records)} answers already graded.")

    # rewrite the checkpoint: settings header + graded answers (provider failures dropped, so retried)
    checkpoint.write_text(json.dumps({"config": config}) + "\n"
                          + "".join(json.dumps(asdict(r)) + "\n" for r in records), encoding="utf-8")

    def save(rec: EvalRecord) -> None:
        with checkpoint.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec)) + "\n")

    db = DatabaseManager(settings.database_path, settings.query_timeout_seconds, settings.max_result_rows)
    llm = LangChainLLM(settings.llm_model, settings.llm_temperature, settings.llm_fallback_model,
                       settings.llm_max_retries)
    nodes = AgentNodes(llm, db, settings, interrupt_fn=lambda _: {"approved": True})
    evaluator = SQLEvaluator(nodes, db)

    done = {(r.suite, r.run, r.prompt_version, r.id) for r in records}
    try:
        for run in range(1, args.repeats + 1):
            for suite in args.suites:
                dataset = json.loads(SUITES[suite].read_text(encoding="utf-8"))[: args.limit]
                records += evaluator.run(dataset, args.versions, suite=suite, run=run, done=done,
                                         on_record=save, pause_seconds=args.pause,
                                         progress=lambda msg: print(msg, flush=True))
    except DailyQuotaReached as exc:
        graded = len(checkpoint.read_text(encoding="utf-8").splitlines()) - 1
        sys.exit(f"\nStopped: the provider's daily token quota is used up.\n{exc}\n"
                 f"{graded} answers are saved. Run the same command again later to continue; "
                 "no report is written until every question has been graded.")

    wanted = [r for r in records if r.suite in args.suites and r.prompt_version in args.versions
              and r.run <= args.repeats]
    summary = evaluator.summarize(wanted)
    command = "python evals/run_eval.py " + " ".join(sys.argv[1:])
    report = evaluator.write_reports(wanted, summary, out, settings.llm_model,
                                     meta={**config, "command": command.strip()})
    print()
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
