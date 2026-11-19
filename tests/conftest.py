import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from querentcrew.config import get_settings  # noqa: E402
from querentcrew.database import DatabaseManager  # noqa: E402
from seed_database import seed  # noqa: E402


@pytest.fixture(scope="session")
def db_path(tmp_path_factory) -> Path:
    return seed(tmp_path_factory.mktemp("data") / "sales.db", 42)


@pytest.fixture
def fresh_db_path(tmp_path) -> Path:
    """A private copy for tests that write (CSV import)."""
    return seed(tmp_path / "sales.db", 42)


@pytest.fixture
def settings(db_path):
    return get_settings(database_path=db_path, glossary_path=ROOT / "data" / "business_glossary.md",
                        max_sql_retries=2, require_approval_for_all=False, require_approval_for_pii=True)


@pytest.fixture
def db(settings) -> DatabaseManager:
    return DatabaseManager(settings.database_path, settings.query_timeout_seconds, settings.max_result_rows)
