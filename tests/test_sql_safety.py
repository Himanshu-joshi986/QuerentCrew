import pytest

from querentcrew.sql_safety import SQLGuard, mask_value, strip_sql


@pytest.fixture
def guard(db):
    return SQLGuard(db.table_columns(), pii_columns=["email", "phone"], max_rows=100)


def test_simple_select_ok(guard):
    res = guard.check("SELECT segment, COUNT(*) FROM customers GROUP BY segment")
    assert res.status == "ok"
    assert res.tables == ["customers"]


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "UPDATE products SET unit_price = 0",
    "DROP TABLE customers",
    "INSERT INTO regions VALUES (9, 'Mars')",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE 'x.db' AS x",
    "SELECT * FROM orders; DROP TABLE orders",
    "SELECT load_extension('evil')",
])
def test_dangerous_sql_blocked(guard, sql):
    assert guard.check(sql).status == "blocked"


def test_unknown_table_blocked(guard):
    res = guard.check("SELECT * FROM salaries")
    assert res.status == "blocked"
    assert "unknown table" in res.reasons[0]


def test_cte_names_are_not_unknown_tables(guard):
    sql = "WITH t AS (SELECT customer_id FROM orders) SELECT COUNT(*) FROM t"
    assert guard.check(sql).status == "ok"


def test_keywords_inside_strings_are_ignored(guard):
    assert guard.check("SELECT COUNT(*) FROM orders WHERE status = 'delete; drop'").status == "ok"


def test_replace_function_is_allowed(guard):
    assert guard.check("SELECT REPLACE(name, ' ', '_') AS n FROM products").status in {"ok"}


def test_limit_added_to_raw_selects(guard):
    res = guard.check("SELECT name FROM products")
    assert res.sql.rstrip().endswith("LIMIT 100")
    assert guard.check("SELECT COUNT(*) FROM products").sql.endswith("products")


def test_pii_requires_review(guard):
    res = guard.check("SELECT name, email FROM customers LIMIT 5")
    assert res.status == "review"
    assert res.pii_columns == ["email"]
    assert guard.check("SELECT * FROM customers LIMIT 5").status == "review"


def test_approval_for_all(db):
    g = SQLGuard(db.table_columns(), require_approval_for_all=True)
    assert g.check("SELECT COUNT(*) FROM orders").status == "review"


def test_strip_sql_removes_fences_and_comments():
    assert strip_sql("```sql\nSELECT 1; -- hi\n```") == "SELECT 1"


def test_mask_value():
    assert mask_value("rohan.das1@example.com") == "ro***@example.com"
    assert mask_value("+91-9123456789").endswith("6789")
    assert "912345" not in mask_value("+91-9123456789")
