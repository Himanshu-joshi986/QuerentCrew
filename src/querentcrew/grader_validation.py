"""Checks that the evaluation grader is right, independently of any model.

A grader can be wrong in two ways:
  * it accepts a wrong query (false positive), which inflates accuracy;
  * it rejects a correct query (false negative), which deflates accuracy.

To measure both, every hand-written gold query is transformed automatically:
  * **mutants** change the meaning (wrong status filter, missing discount, SUM -> AVG, wrong region,
    shifted date range, off-by-one LIMIT, ...). A good grader rejects every mutant whose result
    differs from the gold result. This is mutation testing, applied to the grader.
  * **equivalent rewrites** keep the meaning but change the form (rounding removed or changed,
    columns reordered, an extra column added). A good grader accepts all of them.

Run `python evals/validate_grader.py` for the report; the test suite asserts the results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .database import DatabaseManager, QueryError

# (name, pattern, replacement): each applies to the first match only, one mutant per rule and query
MUTATIONS: list[tuple[str, str, str]] = [
    ("completed -> returned", r"status='completed'", "status='returned'"),
    ("drop completed filter", r"(\w+\.)?status='completed'", "1=1"),
    ("returned -> completed", r"status='returned'", "status='completed'"),
    ("cancelled -> returned", r"status='cancelled'", "status='returned'"),
    ("ignore discount", r"\(1-oi\.discount\)", "1"),
    ("SUM -> AVG", r"SUM\(", "AVG("),
    ("SUM -> MAX", r"SUM\(", "MAX("),
    ("COUNT(*) + 1", r"COUNT\(\*\)", "(COUNT(*)+1)"),
    ("COUNT DISTINCT -> COUNT", r"COUNT\(DISTINCT ", "COUNT("),
    ("AVG -> MAX", r"AVG\(", "MAX("),
    ("MIN -> MAX", r"MIN\(", "MAX("),
    ("DESC -> ASC", r"\bDESC\b", "ASC"),
    ("ASC -> DESC", r"\bASC\b", "DESC"),
    ("LIMIT n -> n+1", r"LIMIT (\d+)", lambda m: f"LIMIT {int(m.group(1)) + 1}"),
    ("South -> North", r"'South'", "'North'"),
    ("Electronics -> Furniture", r"'Electronics'", "'Furniture'"),
    ("Software -> Furniture", r"'Software'", "'Furniture'"),
    ("Enterprise -> SMB", r"'Enterprise'", "'SMB'"),
    ("Partner -> Online", r"'Partner'", "'Online'"),
    ("Online -> Partner", r"'Online'", "'Partner'"),
    ("date start + 7 days", r"'(\d{4}-\d{2})-01'", lambda m: f"'{m.group(1)}-08'"),
    ("date end - 7 days", r"'(\d{4}-\d{2})-(3[01])'", lambda m: f"'{m.group(1)}-{int(m.group(2)) - 7}'"),
    ("quarter boundary", r"<= 3\b", "<= 4"),
    ("month -> year", r"'%Y-%m'", "'%Y'"),
    ("percent -> fraction", r"100\.0\*", "1.0*"),
    ("NOT IN -> IN", r"NOT IN", "IN"),
    ("units -> lines", r"SUM\(oi\.quantity\)", "COUNT(oi.quantity)"),
    ("2024 -> 2023", r"'2024-", "'2023-"),
    ("signup window end moved", r"'2024-12-31'", "'2025-06-30'"),
]


def _strip_round(sql: str) -> str:
    """Remove every ROUND(expr, n) wrapper, keeping expr (the unrounded value)."""
    out, i = [], 0
    while True:
        j = sql.upper().find("ROUND(", i)
        if j < 0:
            out.append(sql[i:])
            return "".join(out)
        out.append(sql[i:j])
        depth, k, last_comma = 0, j + len("ROUND("), None
        start = k
        while k < len(sql):
            ch = sql[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            elif ch == "," and depth == 0:
                last_comma = k
            k += 1
        inner = sql[start:last_comma] if last_comma else sql[start:k]
        out.append(_strip_round(inner))
        i = k + 1


def mutants(sql: str) -> list[tuple[str, str]]:
    found = []
    for name, pattern, repl in MUTATIONS:
        new = re.sub(pattern, repl, sql, count=1)
        if new != sql:
            found.append((name, new))
    return found


def equivalent_rewrites(sql: str, columns: list[str]) -> list[tuple[str, str]]:
    quoted = [f'"{c}"' for c in columns]
    rewrites = [
        ("columns reversed", f"SELECT {', '.join(reversed(quoted))} FROM ({sql})"),
        ("extra column added", f"SELECT {', '.join(quoted)}, 'x' AS extra FROM ({sql})"),
    ]
    if "ROUND(" in sql.upper():
        rewrites.append(("rounding removed", _strip_round(sql)))
        rewrites.append(("rounded to 3 places", re.sub(r",\s*([24])\)", ", 3)", sql)))
    return rewrites


@dataclass
class Check:
    id: str
    kind: str        # mutant | rewrite
    name: str
    sql: str
    expected: str    # reject | accept
    verdict: str     # accept | reject | equivalent (mutant with an identical result) | error
    ok: bool


def validate(dataset: list[dict], db: DatabaseManager, grader) -> list[Check]:
    checks: list[Check] = []
    for item in dataset:
        gold = db.execute(item["gold_sql"])
        ordered = item.get("ordered", False)
        checks.append(Check(item["id"], "self", "gold vs itself", item["gold_sql"], "accept",
                            "accept" if grader(gold.columns, gold.rows, gold.columns, gold.rows, ordered)
                            else "reject", False))
        for name, sql in mutants(item["gold_sql"]):
            try:
                res = db.execute(sql)
            except QueryError:
                checks.append(Check(item["id"], "mutant", name, sql, "reject", "error", True))
                continue
            if sorted(map(tuple, res.rows)) == sorted(map(tuple, gold.rows)) and (
                    not ordered or list(map(tuple, res.rows)) == list(map(tuple, gold.rows))):
                # the mutation did not change the result on this data: no grader could reject it
                checks.append(Check(item["id"], "mutant", name, sql, "reject", "equivalent", True))
                continue
            accepted = grader(gold.columns, gold.rows, res.columns, res.rows, ordered)
            checks.append(Check(item["id"], "mutant", name, sql, "reject",
                                "accept" if accepted else "reject", not accepted))
        for name, sql in equivalent_rewrites(item["gold_sql"], gold.columns):
            res = db.execute(sql)
            accepted = grader(gold.columns, gold.rows, res.columns, res.rows, ordered)
            checks.append(Check(item["id"], "rewrite", name, sql, "accept",
                                "accept" if accepted else "reject", accepted))
    for c in checks:
        if c.kind == "self":
            c.ok = c.verdict == "accept"
    return checks


def summarize(checks: list[Check]) -> dict:
    scored_mutants = [c for c in checks if c.kind == "mutant" and c.verdict in {"accept", "reject"}]
    rewrites = [c for c in checks if c.kind == "rewrite"]
    selfs = [c for c in checks if c.kind == "self"]
    return {
        "gold_self_match": f"{sum(c.ok for c in selfs)}/{len(selfs)}",
        "mutants_generated": sum(c.kind == "mutant" for c in checks),
        "mutants_same_result": sum(c.verdict == "equivalent" for c in checks),
        "mutants_sql_error": sum(c.kind == "mutant" and c.verdict == "error" for c in checks),
        "mutants_rejected": f"{sum(c.ok for c in scored_mutants)}/{len(scored_mutants)}",
        "rewrites_accepted": f"{sum(c.ok for c in rewrites)}/{len(rewrites)}",
    }
