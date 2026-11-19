"""The agents (graph nodes) and the routing functions between them.

Every node is a method that takes the current `AgentState` and returns a
partial update. Dependencies (LLM, database, guard, prompts...) are injected
through the constructor, which keeps the nodes unit-testable with fakes.

Flow (see graph.py for the wiring):

    router ──► responder (general / out of scope)
       │
       ▼
    planner ──► sql_writer ──► sql_guard ──► [human_approval] ──► sql_executor ──► advance
                    ▲              │                                  │             │
                    └── retry ◄────┴──────────── retry on error ◄─────┘             │
                                                                                    ▼
                         reporter ◄── charter ◄── analyst ◄── (all sub-questions done)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from .analysis import DataProfiler
from .charts import ChartBuilder
from .config import Settings
from .database import DatabaseManager, QueryError
from .llm import LLMClient, describe_llm_error
from .prompts.registry import PromptRegistry
from .schemas import AnalysisPlan, ChartSpec, Insights, RouteDecision, SQLDraft
from .sql_safety import SQLGuard, mask_value
from .state import AgentState, QueryRecord

logger = logging.getLogger("querentcrew")


def _default_interrupt(payload: dict) -> Any:
    from langgraph.types import interrupt  # imported lazily so nodes work without langgraph in tests

    return interrupt(payload)


class AgentNodes:
    def __init__(
        self,
        llm: LLMClient,
        db: DatabaseManager,
        settings: Settings,
        prompts: PromptRegistry | None = None,
        guard: SQLGuard | None = None,
        profiler: DataProfiler | None = None,
        charts: ChartBuilder | None = None,
        glossary: str | None = None,
        interrupt_fn: Callable[[dict], Any] | None = None,
    ) -> None:
        self.llm = llm
        self.db = db
        self.settings = settings
        self.prompts = prompts or PromptRegistry()
        self.profiler = profiler or DataProfiler()
        self.charts = charts or ChartBuilder()
        self.interrupt_fn = interrupt_fn or _default_interrupt
        self.prompt_versions = dict(settings.prompt_versions)
        if glossary is None:
            path = settings.glossary_path
            glossary = path.read_text(encoding="utf-8") if path.exists() else "(no glossary provided)"
        self.glossary = glossary
        self.refresh_schema()
        self.guard = guard or self._build_guard()

    # ------------------------------------------------------------------ helpers
    def _build_guard(self) -> SQLGuard:
        return SQLGuard(
            self.db.table_columns(),
            pii_columns=self.settings.pii_columns,
            max_rows=self.settings.max_result_rows,
            require_approval_for_pii=self.settings.require_approval_for_pii,
            require_approval_for_all=self.settings.require_approval_for_all,
        )

    def refresh_schema(self) -> None:
        """Re-read the schema (call after importing a CSV)."""
        self.schema = self.db.schema_description()
        self.tables = ", ".join(self.db.table_names())
        if hasattr(self, "guard"):
            self.guard = self._build_guard()

    def _render(self, name: str, **variables) -> tuple[str, str]:
        return self.prompts.render(name, self.prompt_versions.get(name), **variables)

    def _history_text(self, state: AgentState) -> str:
        history = state.get("history", [])[-self.settings.history_turns * 2:]
        if not history:
            return "(no previous turns)"
        return "\n".join(f"{m['role']}: {m['content'][:600]}" for m in history)

    @staticmethod
    def _log(state: AgentState, agent: str, message: str, **details) -> list[dict]:
        entry = {"agent": agent, "message": message}
        if details:
            entry["details"] = details
        logger.info("[%s] %s", agent, message)
        return [*state.get("trace", []), entry]

    def _current_sub_question(self, state: AgentState) -> str:
        subs = state.get("sub_questions") or [state["question"]]
        return subs[min(state.get("current_index", 0), len(subs) - 1)]

    def _record(self, state: AgentState, **fields) -> list[QueryRecord]:
        record: QueryRecord = {
            "sub_question": self._current_sub_question(state),
            "sql": fields.pop("sql", state.get("draft_sql", "")),
            "attempts": state.get("attempts", 0),
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "masked_columns": [],
            "error": None,
            **fields,
        }
        return [*state.get("results", []), record]

    def _provider_error(self, exc: Exception) -> str | None:
        """Friendly message when the LLM provider itself is unusable (network, key, quota)."""
        return describe_llm_error(exc, self.llm.label)

    def _pii_columns_in(self, columns: list[str]) -> list[str]:
        return [c for c in columns if any(p in c.lower() for p in self.settings.pii_columns)]

    # ------------------------------------------------------------------ agents
    def router(self, state: AgentState) -> dict:
        system, user = self._render(
            "router", question=state["question"], history=self._history_text(state), tables=self.tables
        )
        try:
            decision = self.llm.structured("router", system, user, RouteDecision)
            intent, reason = decision.intent, decision.reason
        except Exception as exc:  # never fail the whole run because routing failed
            provider_error = self._provider_error(exc)
            if provider_error:
                return {"intent": "data_question", "llm_error": provider_error,
                        "trace": self._log(state, "router", "language model unavailable", error=str(exc))}
            intent, reason = "data_question", f"router error, defaulting to data path ({exc})"
        return {
            "intent": intent,
            "trace": self._log(state, "router", f"intent = {intent}", reason=reason),
        }

    def responder(self, state: AgentState) -> dict:
        system, user = self._render(
            "responder",
            question=state["question"],
            history=self._history_text(state),
            schema=self.schema,
            intent=state.get("intent", "general"),
        )
        try:
            answer = self.llm.text("responder", system, user)
        except Exception as exc:
            answer = "⚠️ " + (self._provider_error(exc) or f"The assistant failed to answer: {exc}")
        return {
            "answer": answer,
            "history": [*state.get("history", []),
                        {"role": "user", "content": state["question"]},
                        {"role": "assistant", "content": answer}],
            "trace": self._log(state, "responder", "answered without querying the database"),
        }

    def planner(self, state: AgentState) -> dict:
        system, user = self._render(
            "planner",
            question=state["question"],
            history=self._history_text(state),
            schema=self.schema,
            glossary=self.glossary,
            max_sub_questions=self.settings.max_sub_questions,
        )
        try:
            plan = self.llm.structured("planner", system, user, AnalysisPlan)
            subs = [s.strip() for s in plan.sub_questions if s and s.strip()][: self.settings.max_sub_questions]
            reasoning = plan.reasoning
        except Exception as exc:
            provider_error = self._provider_error(exc)
            if provider_error:
                return {"llm_error": provider_error,
                        "trace": self._log(state, "planner", "language model unavailable", error=str(exc))}
            subs, reasoning = [], f"planner error ({exc}); answering the question directly"
        if not subs:
            subs = [state["question"]]
        return {
            "plan_reasoning": reasoning,
            "sub_questions": subs,
            "current_index": 0,
            "attempts": 0,
            "feedback": "",
            "trace": self._log(state, "planner", f"{len(subs)} step plan", steps=subs, reasoning=reasoning),
        }

    def sql_writer(self, state: AgentState) -> dict:
        attempts = state.get("attempts", 0) + 1
        sub_q = self._current_sub_question(state)
        feedback = state.get("feedback") or ""
        system, user = self._render(
            "sql_writer",
            schema=self.schema,
            glossary=self.glossary,
            question=sub_q,
            main_question=state["question"],
            history=self._history_text(state),
            feedback=f"\nPrevious attempt failed. Fix this:\n{feedback}" if feedback else "",
            max_rows=self.settings.max_result_rows,
        )
        try:
            draft = self.llm.structured("sql_writer", system, user, SQLDraft)
            sql, reasoning = draft.sql, draft.reasoning
        except Exception as exc:
            provider_error = self._provider_error(exc)
            if provider_error:
                return {
                    "attempts": attempts,
                    "llm_error": provider_error,
                    "results": self._record(state, sql="", attempts=attempts, status="llm_error",
                                            error=provider_error),
                    "trace": self._log(state, "sql_writer", "language model unavailable", error=str(exc)),
                }
            sql, reasoning = "", f"sql writer error: {exc}"
        return {
            "attempts": attempts,
            "draft_sql": sql,
            "sql_reasoning": reasoning,
            "trace": self._log(state, "sql_writer", f"attempt {attempts}: wrote SQL", sql=sql, reasoning=reasoning),
        }

    def sql_guard(self, state: AgentState) -> dict:
        result = self.guard.check(state.get("draft_sql", ""))
        update: dict[str, Any] = {"guard": result.to_dict()}
        if result.status == "ok":
            update["next_action"] = "execute"
            msg = "SQL passed safety checks"
        elif result.status == "review":
            update["next_action"] = "approve"
            msg = "SQL needs human approval"
        elif state.get("attempts", 0) < self.settings.max_sql_attempts:
            update["next_action"] = "retry"
            update["feedback"] = f"The query was rejected by the safety layer: {'; '.join(result.reasons)}"
            msg = "SQL blocked, asking the SQL agent to rewrite it"
        else:
            update["next_action"] = "advance"
            update["results"] = self._record(state, sql=result.sql, status="blocked", error="; ".join(result.reasons))
            msg = "SQL blocked, retries exhausted"
        update["trace"] = self._log(state, "sql_guard", msg, status=result.status, reasons=result.reasons)
        return update

    def human_approval(self, state: AgentState) -> dict:
        guard = state.get("guard", {})
        decision = self.interrupt_fn({
            "type": "sql_approval",
            "sub_question": self._current_sub_question(state),
            "sql": guard.get("sql", state.get("draft_sql", "")),
            "reasons": guard.get("reasons", []),
            "pii_columns": guard.get("pii_columns", []),
        })
        if isinstance(decision, dict):
            approved, comment = bool(decision.get("approved")), decision.get("comment", "")
        else:
            approved, comment = bool(decision), ""
        update: dict[str, Any] = {"approval": {"approved": approved, "comment": comment}}
        if approved:
            update["next_action"] = "execute"
            update["trace"] = self._log(state, "human_approval", "approved by user", comment=comment)
        else:
            update["next_action"] = "advance"
            update["results"] = self._record(
                state, sql=guard.get("sql", ""), status="rejected",
                error=f"rejected by user{': ' + comment if comment else ''}",
            )
            update["trace"] = self._log(state, "human_approval", "rejected by user", comment=comment)
        return update

    def sql_executor(self, state: AgentState) -> dict:
        sql = state.get("guard", {}).get("sql") or state.get("draft_sql", "")
        try:
            result = self.db.execute(sql)
        except QueryError as exc:
            if state.get("attempts", 0) < self.settings.max_sql_attempts:
                return {
                    "next_action": "retry",
                    "feedback": f"SQLite error: {exc}\nFailed SQL:\n{sql}",
                    "trace": self._log(state, "sql_executor", "query failed, sending error back for self-correction",
                                       error=str(exc)),
                }
            return {
                "next_action": "advance",
                "results": self._record(state, sql=sql, status="failed", error=str(exc)),
                "trace": self._log(state, "sql_executor", "query failed, retries exhausted", error=str(exc)),
            }

        masked = self._pii_columns_in(result.columns)
        rows = result.rows
        if masked:
            idx = [result.columns.index(c) for c in masked]
            rows = [[mask_value(v) if i in idx else v for i, v in enumerate(row)] for row in rows]
        return {
            "next_action": "advance",
            "results": self._record(
                state, sql=sql, status="ok", columns=result.columns, rows=rows, row_count=result.row_count,
                truncated=result.truncated, masked_columns=masked, elapsed_ms=result.elapsed_ms,
            ),
            "trace": self._log(state, "sql_executor", f"returned {result.row_count} rows",
                               elapsed_ms=result.elapsed_ms, masked=masked),
        }

    def advance(self, state: AgentState) -> dict:
        return {
            "current_index": state.get("current_index", 0) + 1,
            "attempts": 0,
            "feedback": "",
            "draft_sql": "",
            "guard": {},
            "approval": None,
        }

    def _results_for_prompt(self, results: list[QueryRecord]) -> str:
        blocks = []
        for i, rec in enumerate(results):
            header = f"### Result {i}: {rec['sub_question']}\nstatus: {rec['status']}"
            if rec["status"] != "ok":
                blocks.append(f"{header}\nerror: {rec.get('error')}")
                continue
            df = pd.DataFrame(rec["rows"], columns=rec["columns"])
            profile = self.profiler.profile(df)
            block = (
                f"{header}\nrows: {rec['row_count']}{' (truncated)' if rec['truncated'] else ''}\n"
                f"data:\n{self.profiler.preview(df)}\n"
                f"pandas statistics: {json.dumps(profile.stats, default=str)}"
            )
            if profile.notes:
                block += f"\nnotes: {'; '.join(profile.notes)}"
            blocks.append(block)
        return "\n\n".join(blocks) if blocks else "(no results)"

    def analyst(self, state: AgentState) -> dict:
        results = state.get("results", [])
        ok = [r for r in results if r["status"] == "ok"]
        if not ok:
            insights = {"key_findings": [], "caveats": [f"{r['sub_question']}: {r['status']} ({r.get('error')})"
                                                        for r in results]}
            return {"insights": insights,
                    "trace": self._log(state, "analyst", "no successful results to analyse")}
        system, user = self._render("analyst", question=state["question"], results=self._results_for_prompt(results))
        try:
            insights = self.llm.structured("analyst", system, user, Insights).model_dump()
        except Exception as exc:
            provider_error = self._provider_error(exc)
            insights = {"key_findings": [], "caveats": [f"analysis step failed: {provider_error or exc}"]}
            if provider_error:
                return {"insights": insights, "llm_error": provider_error,
                        "trace": self._log(state, "analyst", "language model unavailable", error=str(exc))}
        return {"insights": insights,
                "trace": self._log(state, "analyst", f"{len(insights['key_findings'])} findings")}

    def charter(self, state: AgentState) -> dict:
        results = state.get("results", [])
        candidates = [i for i, r in enumerate(results) if r["status"] == "ok" and r["row_count"] >= 2]
        if not candidates:
            return {"chart": None, "trace": self._log(state, "charter", "no chartable result")}
        overview = "\n".join(
            f"result_index {i}: '{results[i]['sub_question']}' columns={results[i]['columns']} "
            f"rows={results[i]['row_count']} first_row={results[i]['rows'][0]}"
            for i in candidates
        )
        system, user = self._render("charter", question=state["question"], results_overview=overview)
        try:
            spec = self.llm.structured("charter", system, user, ChartSpec)
        except Exception:
            spec = ChartSpec(chart_type="bar", result_index=candidates[-1])
        index = spec.result_index if spec.result_index in candidates else candidates[-1]
        rec = results[index]
        df = pd.DataFrame(rec["rows"], columns=rec["columns"])
        spec = self.charts.validate(spec, df)
        if spec.chart_type == "none":
            # The model declined to chart; fall back to the heuristic so chartable data still gets a chart.
            for i in [index] + [c for c in candidates if c != index]:
                frame = pd.DataFrame(results[i]["rows"], columns=results[i]["columns"])
                auto = self.charts.auto_spec(frame, spec.title)
                if auto.chart_type != "none":
                    index, rec, df, spec = i, results[i], frame, auto
                    break
        image = self.charts.render(spec, df)
        chart = {"spec": spec.model_dump(), "result_index": index, "image_b64": image} if image else None
        return {"chart": chart,
                "trace": self._log(state, "charter", f"chart: {spec.chart_type}" if image else "no chart drawn")}

    def reporter(self, state: AgentState) -> dict:
        results = state.get("results", [])
        insights = state.get("insights", {})
        notes = [f"{r['sub_question']}: {r['status']} - {r.get('error')}" for r in results if r["status"] != "ok"]
        notes += [f"Result {i} was truncated to {r['row_count']} rows" for i, r in enumerate(results) if r.get("truncated")]
        notes += [f"Personal data masked in columns: {', '.join(r['masked_columns'])}"
                  for r in results if r.get("masked_columns")]

        llm_error = state.get("llm_error")
        if llm_error:
            answer = self._offline_answer(llm_error, results, insights)
        elif not any(r["status"] == "ok" for r in results):
            lines = ["I couldn't answer this from the database."]
            for r in results:
                lines.append(f"- **{r['sub_question']}** - {r['status']}: {r.get('error')}")
            lines.append("\nTry rephrasing the question, or approve the query if it needs personal data.")
            answer = "\n".join(lines)
        else:
            system, user = self._render(
                "reporter",
                question=state["question"],
                insights=json.dumps(insights, indent=2),
                results=self._results_for_prompt(results),
                notes="\n".join(notes) or "(none)",
            )
            try:
                answer = self.llm.text("reporter", system, user)
            except Exception as exc:
                llm_error = self._provider_error(exc) or f"The report writer failed: {exc}"
                answer = self._offline_answer(llm_error, results, insights)

        return {
            "answer": answer,
            "history": [*state.get("history", []),
                        {"role": "user", "content": state["question"]},
                        {"role": "assistant", "content": answer}],
            "trace": self._log(state, "reporter", "final answer written"),
        }

    @staticmethod
    def _offline_answer(error: str, results: list[QueryRecord], insights: dict) -> str:
        """Answer built without the LLM: explain the outage and show whatever was computed."""
        lines = [f"⚠️ **{error}**"]
        findings = insights.get("key_findings") or []
        ok = [r for r in results if r["status"] == "ok"]
        if findings:
            lines.append("\nFindings computed before the outage:")
            lines += [f"- {f}" for f in findings]
        elif ok:
            noun = "query" if len(ok) == 1 else "queries"
            lines.append(f"\n{len(ok)} {noun} ran successfully; the raw results are in the Data tab.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ routing
    @staticmethod
    def route_intent(state: AgentState) -> str:
        if state.get("llm_error"):
            return "reporter"
        return "planner" if state.get("intent") == "data_question" else "responder"

    @staticmethod
    def route_after_plan(state: AgentState) -> str:
        return "reporter" if state.get("llm_error") else "sql_writer"

    @staticmethod
    def route_after_write(state: AgentState) -> str:
        return "reporter" if state.get("llm_error") else "sql_guard"

    @staticmethod
    def route_after_analyst(state: AgentState) -> str:
        return "reporter" if state.get("llm_error") else "charter"

    @staticmethod
    def route_after_guard(state: AgentState) -> str:
        return {"execute": "sql_executor", "approve": "human_approval", "retry": "sql_writer"}.get(
            state.get("next_action", ""), "advance"
        )

    @staticmethod
    def route_after_approval(state: AgentState) -> str:
        return "sql_executor" if state.get("next_action") == "execute" else "advance"

    @staticmethod
    def route_after_execute(state: AgentState) -> str:
        return "sql_writer" if state.get("next_action") == "retry" else "advance"

    @staticmethod
    def route_after_advance(state: AgentState) -> str:
        return "sql_writer" if state.get("current_index", 0) < len(state.get("sub_questions", [])) else "analyst"
