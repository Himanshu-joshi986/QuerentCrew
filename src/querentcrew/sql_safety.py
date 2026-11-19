"""SQL guardrails: validate LLM-written SQL before it touches the database.

Outcomes:
  * ok       -> safe to run automatically
  * review   -> safe, but needs a human approval first (e.g. reads PII columns)
  * blocked  -> never runs (writes, multiple statements, unknown tables, ...)

The guard is intentionally conservative and rule-based (no LLM), so its
behaviour is predictable and unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "replace", "truncate",
    "attach", "detach", "pragma", "vacuum", "reindex", "grant", "revoke", "merge",
    "upsert", "savepoint", "release", "rollback", "commit", "begin", "load_extension",
}

_AGGREGATES = re.compile(r"\b(count|sum|avg|min|max|group_concat|total)\s*\(|\bgroup\s+by\b", re.I)


@dataclass
class GuardResult:
    status: str  # "ok" | "review" | "blocked"
    sql: str  # possibly rewritten (e.g. LIMIT added)
    reasons: list[str] = field(default_factory=list)
    pii_columns: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "sql": self.sql,
            "reasons": self.reasons,
            "pii_columns": self.pii_columns,
            "tables": self.tables,
        }


def strip_sql(sql: str) -> str:
    """Remove markdown fences, comments and trailing semicolons."""
    text = sql.strip()
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.I).strip()
    text = re.sub(r"--[^\n]*", " ", text)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = text.strip().rstrip(";").strip()
    return text


def _without_string_literals(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


class SQLGuard:
    def __init__(
        self,
        table_columns: dict[str, list[tuple[str, str]]],
        pii_columns: list[str] | None = None,
        max_rows: int = 500,
        require_approval_for_pii: bool = True,
        require_approval_for_all: bool = False,
    ):
        self.table_columns = {t.lower(): [c[0].lower() for c in cols] for t, cols in table_columns.items()}
        self.pii_columns = [c.lower() for c in (pii_columns or [])]
        self.max_rows = max_rows
        self.require_approval_for_pii = require_approval_for_pii
        self.require_approval_for_all = require_approval_for_all

    def referenced_tables(self, sql: str) -> list[str]:
        body = _without_string_literals(sql)
        names = re.findall(r'\b(?:from|join)\s+(["`\[]?[A-Za-z_][\w]*["`\]]?)', body, flags=re.I)
        ctes = {m.lower() for m in re.findall(r'\b([A-Za-z_]\w*)\s+as\s*\(', body, flags=re.I)}
        tables = []
        for name in names:
            clean = name.strip('"`[]').lower()
            if clean not in ctes and clean not in tables:
                tables.append(clean)
        return tables

    def pii_in_query(self, sql: str, tables: list[str]) -> list[str]:
        body = _without_string_literals(sql).lower()
        found = [c for c in self.pii_columns if re.search(rf"\b{re.escape(c)}\b", body)]
        if re.search(r"select\s+(?:distinct\s+)?(?:\w+\.)?\*", body):
            for t in tables:
                for col in self.table_columns.get(t, []):
                    if col in self.pii_columns and col not in found:
                        found.append(col)
        return found

    def check(self, raw_sql: str) -> GuardResult:
        sql = strip_sql(raw_sql)
        if not sql:
            return GuardResult("blocked", sql, ["empty query"])

        body = _without_string_literals(sql)
        lowered = body.lower()

        if ";" in body:
            return GuardResult("blocked", sql, ["multiple statements are not allowed"])
        if not re.match(r"^\s*(select|with)\b", lowered):
            return GuardResult("blocked", sql, ["only SELECT / WITH queries are allowed"])
        # Keywords used as statements are blocked; same-named scalar functions such as
        # REPLACE(text, a, b) are allowed (except load_extension, which is always blocked).
        bad = sorted(
            k for k in FORBIDDEN_KEYWORDS
            if re.search(rf"\b{k}\b" if k == "load_extension" else rf"\b{k}\b(?!\s*\()", lowered)
        )
        if bad:
            return GuardResult("blocked", sql, [f"forbidden keyword(s): {', '.join(bad)}"])

        tables = self.referenced_tables(sql)
        unknown = [t for t in tables if t not in self.table_columns]
        if unknown:
            return GuardResult(
                "blocked", sql,
                [f"unknown table(s): {', '.join(unknown)}. Available: {', '.join(sorted(self.table_columns))}"],
                tables=tables,
            )

        reasons: list[str] = []
        # Row cap: add a LIMIT to non-aggregated queries that have none
        if not re.search(r"\blimit\s+\d+", lowered) and not _AGGREGATES.search(body):
            sql = f"{sql}\nLIMIT {self.max_rows}"
            reasons.append(f"added LIMIT {self.max_rows} to a non-aggregated query")

        pii = self.pii_in_query(sql, tables)
        status = "ok"
        if pii and self.require_approval_for_pii:
            status = "review"
            reasons.append(f"reads personal data column(s): {', '.join(pii)}")
        if self.require_approval_for_all:
            status = "review"
            reasons.append("approval is required for every query (REQUIRE_APPROVAL_FOR_ALL)")
        return GuardResult(status, sql, reasons, pii_columns=pii, tables=tables)


def mask_value(value) -> str:
    """Mask a PII value but keep a hint of its shape, e.g. 'ra***@example.com'."""
    if value is None:
        return value
    text = str(value)
    if "@" in text:
        local, _, domain = text.partition("@")
        return f"{local[:2]}***@{domain}"
    if len(text) <= 4:
        return "***"
    return f"{'*' * (len(text) - 4)}{text[-4:]}"
