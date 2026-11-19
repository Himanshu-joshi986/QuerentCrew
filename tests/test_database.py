import io

import pytest

from querentcrew.database import DatabaseManager, QueryError


def test_schema_description_mentions_tables_and_fks(db):
    text = db.schema_description()
    for table in ["orders", "order_items", "customers", "products", "regions"]:
        assert f"TABLE {table}" in text
    assert "customer_id -> customers.customer_id" in text


def test_execute_returns_rows(db):
    res = db.execute("SELECT COUNT(*) AS n FROM regions")
    assert res.columns == ["n"] and res.rows == [[4]]


def test_read_only_even_if_guard_is_bypassed(db):
    with pytest.raises(QueryError):
        db.execute("DELETE FROM orders")
    assert db.execute("SELECT COUNT(*) FROM orders").rows[0][0] > 0


def test_row_cap_and_truncation(settings):
    small = DatabaseManager(settings.database_path, max_rows=10)
    res = small.execute("SELECT * FROM orders")
    assert res.row_count == 10 and res.truncated


def test_bad_sql_raises_query_error(db):
    with pytest.raises(QueryError, match="no such column"):
        db.execute("SELECT nope FROM orders")


def test_import_csv(fresh_db_path):
    db = DatabaseManager(fresh_db_path)
    csv = io.StringIO("Store Name,Monthly Target\nA,100\nB,200\n")
    table, rows = db.import_csv(csv, "2025 targets.csv")
    assert table == "t_2025_targets" and rows == 2
    assert db.execute(f"SELECT SUM(monthly_target) FROM {table}").rows == [[300]]


def test_missing_database():
    with pytest.raises(FileNotFoundError):
        DatabaseManager("does/not/exist.db")
