"""Evaluation harness for the SQL agent and its prompt versions.

Metric: **execution accuracy**. The SQL the agent writes is executed and its
result is compared with the result of a hand-written gold query. Column names
and column order may differ; values must match (numbers compared after
rounding to 1 decimal place). Row order matters only for questions marked
`ordered` (rankings, time series).

For each question we record:
  * first_try_correct  - correct without any retry
  * final_correct      - correct after the self-correction loop (errors fed back)
  * attempts, blocked/failed status, tokens and latency
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .database import DatabaseManager, QueryError
from .nodes import AgentNodes


def _norm(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return round(float(value), 1)
    if value is None:
        return None
    text = str(value).strip()
    try:
        return round(float(text), 1)
    except ValueError:
        return text.lower()


def results_match(gold_cols, gold_rows, pred_cols, pred_rows, ordered: bool = False) -> bool:
    """True if every gold column can be matched to a predicted column with identical values."""
    if len(gold_rows) != len(pred_rows):
        return False
    if not gold_rows:
        return True
    gold_t = [[_norm(r[i]) for r in gold_rows] for i in range(len(gold_cols))]
    pred_t = [[_norm(r[i]) for r in pred_rows] for i in range(len(pred_cols))]

    def key(col):
        return Counter(map(str, col))

    mapping, used = [], set()
    for g in gold_t:
        match = next((j for j, p in enumerate(pred_t) if j not in used and key(p) == key(g)), None)
        if match is None:
            return False
        mapping.append(match)
        used.add(match)

    gold_rows_n = [tuple(str(c[k]) for c in gold_t) for k in range(len(gold_rows))]
    pred_rows_n = [tuple(str(pred_t[j][k]) for j in mapping) for k in range(len(pred_rows))]
    return gold_rows_n == pred_rows_n if ordered else sorted(gold_rows_n) == sorted(pred_rows_n)


@dataclass
class EvalRecord:
    id: str
    difficulty: str
    question: str
    prompt_version: str
    sql: str
    status: str  # correct | wrong | error | blocked
    first_try_correct: bool
    final_correct: bool
    attempts: int
    error: str | None
    tokens: int
    latency_ms: float
    suite: str = "dev"
    run: int = 1


class DailyQuotaReached(RuntimeError):
    """The provider's daily token quota is used up; waiting minutes will not help."""


def _is_daily_quota(update: dict) -> bool:
    raw = " ".join(str(t.get("details", {}).get("error", "")) for t in update.get("trace", [])[-1:])
    return "per day" in raw.lower() or "(tpd)" in raw.lower()


class SQLEvaluator:
    def __init__(self, nodes: AgentNodes, db: DatabaseManager, provider_retries: int = 4,
                 backoff_seconds: float = 20.0, sleep=time.sleep):
        self.nodes = nodes
        self.db = db
        self.provider_retries = provider_retries
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep

    def _write(self, state: dict) -> dict:
        """Run the SQL writer, waiting out provider errors (rate limits, outages).

        A provider error says nothing about prompt quality, so it is retried rather than scored.
        Returns the node update; it still contains `llm_error` if the provider never recovered.
        """
        for retry in range(self.provider_retries + 1):
            update = self.nodes.sql_writer({**state, "attempts": state.get("attempts", 0)})
            if "llm_error" not in update:
                return update
            if _is_daily_quota(update):
                raise DailyQuotaReached(update["llm_error"])
            if retry < self.provider_retries:
                self.sleep(self.backoff_seconds * (retry + 1))
        return update

    def _run_one(self, item: dict, version: str) -> EvalRecord:
        self.nodes.prompt_versions["sql_writer"] = version
        usage_before = self.nodes.llm.usage.snapshot()
        start = time.perf_counter()
        gold = self.db.execute(item["gold_sql"])

        state = {"question": item["question"], "history": [], "sub_questions": [item["question"]],
                 "current_index": 0, "attempts": 0, "feedback": "", "results": [], "trace": []}
        first_try, final, status, error, sql = False, False, "error", None, ""
        for attempt in range(1, self.nodes.settings.max_sql_attempts + 1):
            update = self._write(state)
            if "llm_error" in update:
                status, error = "llm_error", update["llm_error"]
                state["attempts"] = update.get("attempts", state["attempts"])
                break
            state.update(update)
            guard = self.nodes.guard.check(state["draft_sql"])
            sql = guard.sql
            if guard.status == "blocked":
                status, error = "blocked", "; ".join(guard.reasons)
                state["feedback"] = f"The query was rejected by the safety layer: {error}"
                continue
            try:
                pred = self.db.execute(guard.sql)
            except QueryError as exc:
                status, error = "error", str(exc)
                state["feedback"] = f"SQLite error: {exc}\nFailed SQL:\n{guard.sql}"
                continue
            correct = results_match(gold.columns, gold.rows, pred.columns, pred.rows, item.get("ordered", False))
            first_try = correct and attempt == 1
            final = correct
            status, error = ("correct", None) if correct else ("wrong", None)
            break  # executed successfully: the pipeline would not retry a valid query

        usage_after = self.nodes.llm.usage.snapshot()
        return EvalRecord(
            id=item["id"], difficulty=item.get("difficulty", ""), question=item["question"],
            prompt_version=version, sql=sql, status=status, first_try_correct=first_try,
            final_correct=final, attempts=state["attempts"], error=error,
            tokens=usage_after["total_tokens"] - usage_before["total_tokens"],
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    def run(self, dataset: list[dict], versions: list[str], progress=print, suite: str = "dev", run: int = 1,
            done: set | None = None, on_record=None, pause_seconds: float = 0.0) -> list[EvalRecord]:
        """Evaluate every question for every prompt version.

        `done` holds (suite, run, version, id) keys already finished by an interrupted run; they are
        skipped. `on_record` is called after each question so results can be saved as they arrive.
        """
        records = []
        for version in versions:
            for i, item in enumerate(dataset, 1):
                if done and (suite, run, version, item["id"]) in done:
                    continue
                rec = self._run_one(item, version)
                rec.suite, rec.run = suite, run
                records.append(rec)
                if on_record:
                    on_record(rec)
                if progress:
                    progress(f"[{suite} run {run} {version}] {i:>2}/{len(dataset)} {rec.id} {rec.status:<8} "
                             f"attempts={rec.attempts}"
                             + (f"  ({rec.error[:120]})" if rec.status == "llm_error" and rec.error else ""))
                if pause_seconds:
                    self.sleep(pause_seconds)
        return records

    @staticmethod
    def summarize(records: list[EvalRecord]) -> list[dict]:
        """One row per (suite, prompt version). With several runs, accuracy is the mean over runs and
        the min-max range shows how much the model varies between identical runs."""
        rows = []
        for suite, version in dict.fromkeys((r.suite, r.prompt_version) for r in records):
            group = [r for r in records if r.suite == suite and r.prompt_version == version]
            per_run = []
            for run in sorted({r.run for r in group}):
                scored = [r for r in group if r.run == run and r.status != "llm_error"]
                if scored:
                    per_run.append(100 * sum(r.final_correct for r in scored) / len(scored))
            row = SQLEvaluator._summarize_group(group, version)
            row = {"suite": suite, **row, "runs": len(per_run)}
            if len(per_run) > 1:
                row["final_accuracy_pct"] = round(statistics.mean(per_run), 1)
                row["final_accuracy_range"] = f"{min(per_run):.0f}-{max(per_run):.0f}"
            rows.append(row)
        return rows

    @staticmethod
    def _summarize_group(every: list[EvalRecord], version: str) -> dict:
        # provider failures are reported, not scored: they measure the API, not the prompt
        recs = [r for r in every if r.status != "llm_error"] or every
        n = len(recs)
        row = {
            "prompt_version": version,
            "questions": n,
            "provider_failures": len(every) - len([r for r in every if r.status != "llm_error"]),
            "first_try_accuracy_pct": round(100 * sum(r.first_try_correct for r in recs) / n, 1),
            "final_accuracy_pct": round(100 * sum(r.final_correct for r in recs) / n, 1),
            "errors_or_blocked": sum(r.status in {"error", "blocked"} for r in recs),
            "avg_attempts": round(sum(r.attempts for r in recs) / n, 2),
            "avg_tokens": round(sum(r.tokens for r in recs) / n, 1),
            "avg_latency_ms": round(sum(r.latency_ms for r in recs) / n, 1),
        }
        for level in ("easy", "medium", "hard"):
            sub = [r for r in recs if r.difficulty == level]
            if sub:
                row[f"{level}_accuracy_pct"] = round(100 * sum(r.final_correct for r in sub) / len(sub), 1)
        return row

    @staticmethod
    def write_reports(records: list[EvalRecord], summary: list[dict], out_dir: str | Path, model: str,
                      meta: dict | None = None) -> Path:
        import pandas as pd

        meta = meta or {}
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        pd.DataFrame([asdict(r) for r in records]).to_csv(out / f"eval_details_{stamp}.csv", index=False)
        (out / f"eval_summary_{stamp}.json").write_text(
            json.dumps({"model": model, **meta, "summary": summary}, indent=2), encoding="utf-8")

        lines = [f"# SQL agent evaluation ({time.strftime('%Y-%m-%d %H:%M')})", "", f"Model: `{model}`"]
        if meta.get("command"):
            lines += ["", f"Command: `{meta['command']}`"]
        for suite in dict.fromkeys(row["suite"] for row in summary):
            rows = [row for row in summary if row["suite"] == suite]
            cols = list(dict.fromkeys(k for row in rows for k in row if k != "suite"))
            lines += ["", f"## Suite: {suite}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            lines += ["| " + " | ".join(str(row.get(c, "")) for c in cols) + " |" for row in rows]
        wrong = [r for r in records if not r.final_correct]
        if wrong:
            lines += ["", "## Every answer that was not correct", "",
                      "| suite | run | version | id | status | question | SQL written by the model |",
                      "|---|---|---|---|---|---|---|"]
            for r in wrong:
                sql = " ".join((r.sql or r.error or "").split()).replace("|", "\\|")
                lines.append(f"| {r.suite} | {r.run} | {r.prompt_version} | {r.id} | {r.status} | {r.question} "
                             f"| `{sql}` |")
        if meta.get("fingerprints"):
            lines += ["", "## File fingerprints (SHA-256)", "",
                      "Run `python evals/run_eval.py --fingerprints` to check your copy matches.", "",
                      "| file | sha256 |", "|---|---|"]
            lines += [f"| `{k}` | `{v}` |" for k, v in meta["fingerprints"].items()]
        report = out / "latest_report.md"
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report


def _db_content_hash(path: Path) -> str:
    """Hash a SQLite database by its data, not its bytes: the same data written by another SQLite
    version (or regenerated with scripts/seed_database.py) gives the same fingerprint."""
    import sqlite3

    digest = hashlib.sha256()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = [t for (t,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for table in tables:
            cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{table}")')]
            digest.update(json.dumps([table, cols]).encode())
            for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY {", ".join(map(str, range(1, len(cols) + 1)))}'):
                digest.update(json.dumps(row, default=str).encode())
    return digest.hexdigest()


def fingerprints(paths: dict[str, Path]) -> dict[str, str]:
    """SHA-256 of every input that determines the result, so a rerun can prove it tested the same inputs."""
    return {name: _db_content_hash(Path(p)) if str(p).endswith(".db") else hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for name, p in paths.items()}
