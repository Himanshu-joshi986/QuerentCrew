"""Central configuration, loaded from environment variables (and a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> list[str]:
    return [v.strip().lower() for v in os.getenv(name, default).split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    """All runtime settings. Every field can be overridden with an env var."""

    # LLM: any provider supported by langchain's init_chat_model, as "provider:model"
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "groq:openai/gpt-oss-120b"))
    llm_fallback_model: str | None = field(default_factory=lambda: os.getenv("LLM_FALLBACK_MODEL") or None)
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0")))
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2")))

    # Data
    database_path: Path = field(
        default_factory=lambda: Path(os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "data" / "sales.db")))
    )
    glossary_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("GLOSSARY_PATH", str(PROJECT_ROOT / "data" / "business_glossary.md"))
        )
    )
    max_result_rows: int = field(default_factory=lambda: int(os.getenv("MAX_RESULT_ROWS", "500")))
    query_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("QUERY_TIMEOUT_SECONDS", "10")))

    # Agent behaviour
    max_sql_retries: int = field(default_factory=lambda: int(os.getenv("MAX_SQL_RETRIES", "2")))
    max_sub_questions: int = field(default_factory=lambda: int(os.getenv("MAX_SUB_QUESTIONS", "3")))
    history_turns: int = field(default_factory=lambda: int(os.getenv("HISTORY_TURNS", "6")))

    # Governance
    pii_columns: list[str] = field(default_factory=lambda: _list("PII_COLUMNS", "email,phone"))
    require_approval_for_pii: bool = field(default_factory=lambda: _bool("REQUIRE_APPROVAL_FOR_PII", True))
    require_approval_for_all: bool = field(default_factory=lambda: _bool("REQUIRE_APPROVAL_FOR_ALL", False))

    # Prompt versions (see src/querentcrew/prompts/*.yaml)
    prompt_versions: dict[str, str] = field(
        default_factory=lambda: {
            "router": os.getenv("PROMPT_VERSION_ROUTER", "v1"),
            "planner": os.getenv("PROMPT_VERSION_PLANNER", "v1"),
            "sql_writer": os.getenv("PROMPT_VERSION_SQL", "v3"),
            "analyst": os.getenv("PROMPT_VERSION_ANALYST", "v1"),
            "charter": os.getenv("PROMPT_VERSION_CHARTER", "v1"),
            "reporter": os.getenv("PROMPT_VERSION_REPORTER", "v1"),
            "responder": os.getenv("PROMPT_VERSION_RESPONDER", "v1"),
        }
    )

    @property
    def max_sql_attempts(self) -> int:
        return 1 + self.max_sql_retries


def get_settings(**overrides) -> Settings:
    """Build settings from the environment, with optional keyword overrides (useful in tests)."""
    base = Settings()
    if not overrides:
        return base
    values = {f: getattr(base, f) for f in base.__dataclass_fields__}
    values.update(overrides)
    return Settings(**values)
