"""Database access layer.

Queries always run on a READ-ONLY SQLite connection with an authorizer that
rejects anything other than reads, plus a timeout. This is the last line of
defence behind the SQL guard (see sql_safety.py).
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pandas as pd

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


class QueryError(RuntimeError):
    """Raised when a query fails; the message is fed back to the SQL agent for self-correction."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]
    row_count: int
    truncated: bool
    elapsed_ms: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)


class DatabaseManager:
    """Schema introspection, safe query execution and CSV import for a SQLite database."""

    def __init__(self, path: str | Path, timeout_seconds: float = 10.0, max_rows: int = 500):
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows
        if not self.path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.path}. Run `python scripts/seed_database.py` first."
            )

    # ---------- connections ----------
    def _read_only_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)

        def authorizer(action, *_args):
            return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY

        conn.set_authorizer(authorizer)
        return conn

    # ---------- schema ----------
    def table_names(self) -> list[str]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [r[0] for r in rows]

    def table_columns(self) -> dict[str, list[tuple[str, str]]]:
        out: dict[str, list[tuple[str, str]]] = {}
        with sqlite3.connect(self.path) as conn:
            for table in self.table_names():
                cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
                out[table] = [(c[1], c[2] or "TEXT") for c in cols]
        return out

    def row_counts(self) -> dict[str, int]:
        with sqlite3.connect(self.path) as conn:
            return {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in self.table_names()}

    def schema_description(self, sample_rows: int = 2) -> str:
        """Compact schema text for prompts: columns, types, foreign keys, row counts and sample rows."""
        parts = []
        counts = self.row_counts()
        with sqlite3.connect(self.path) as conn:
            for table, cols in self.table_columns().items():
                col_txt = ", ".join(f"{name} {ctype}" for name, ctype in cols)
                fks = conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
                fk_txt = "; ".join(f"{fk[3]} -> {fk[2]}.{fk[4]}" for fk in fks)
                block = f"TABLE {table} ({counts[table]} rows): {col_txt}"
                if fk_txt:
                    block += f"\n  foreign keys: {fk_txt}"
                if sample_rows:
                    sample = conn.execute(f'SELECT * FROM "{table}" LIMIT {int(sample_rows)}').fetchall()
                    for row in sample:
                        block += f"\n  sample: {row}"
                parts.append(block)
        return "\n".join(parts)

    # ---------- execution ----------
    def execute(self, sql: str) -> QueryResult:
        conn = self._read_only_connection()
        deadline = time.monotonic() + self.timeout_seconds
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        start = time.perf_counter()
        try:
            cursor = conn.execute(sql)
            columns = [d[0] for d in cursor.description or []]
            fetched = cursor.fetchmany(self.max_rows + 1)
        except sqlite3.OperationalError as exc:
            msg = str(exc)
            if "interrupted" in msg:
                msg = f"query exceeded the {self.timeout_seconds}s timeout; simplify or aggregate it"
            raise QueryError(msg) from exc
        except sqlite3.DatabaseError as exc:
            raise QueryError(str(exc)) from exc
        finally:
            conn.close()
        truncated = len(fetched) > self.max_rows
        rows = [list(r) for r in fetched[: self.max_rows]]
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    # ---------- data integration ----------
    @staticmethod
    def sanitize_table_name(name: str) -> str:
        clean = re.sub(r"[^0-9a-zA-Z_]", "_", Path(name).stem).strip("_").lower()
        if not clean:
            clean = "uploaded"
        if clean[0].isdigit():
            clean = f"t_{clean}"
        return clean

    def import_csv(self, file: str | Path | IO, table_name: str, replace: bool = True) -> tuple[str, int]:
        """Load a CSV into a new table (Pandas cleans headers and infers types). Returns (table, rows)."""
        df = pd.read_csv(file)
        df.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(c)).strip("_").lower() or f"col_{i}"
                      for i, c in enumerate(df.columns)]
        df = df.dropna(how="all")
        table = self.sanitize_table_name(table_name)
        with sqlite3.connect(self.path) as conn:
            df.to_sql(table, conn, if_exists="replace" if replace else "fail", index=False)
        return table, len(df)
