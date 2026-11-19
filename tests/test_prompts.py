import re

import pytest

from querentcrew.prompts.registry import PromptRegistry

VARS = dict(question="q", history="h", tables="t", schema="s", glossary="g", max_sub_questions=3,
            main_question="m", feedback="f", max_rows=10, results="r", results_overview="o",
            insights="i", notes="n", intent="general")

registry = PromptRegistry()


def test_all_agents_have_prompts():
    assert set(registry.names()) >= {"router", "planner", "sql_writer", "analyst", "charter", "reporter",
                                     "responder"}
    assert registry.versions("sql_writer") == ["v1", "v2", "v3"]


@pytest.mark.parametrize("name", registry.names())
def test_every_version_renders_without_leftover_variables(name):
    for version in registry.versions(name):
        system, user = registry.render(name, version, **VARS)
        leftovers = re.findall(r"\$[a-z_]+", system + user)
        assert not leftovers, f"{name}/{version} has unfilled variables {leftovers}"


def test_sql_braces_survive_rendering():
    system, _ = registry.render("sql_writer", "v3", **VARS)
    assert "strftime('%Y-%m', o.order_date)" in system


def test_unknown_version_raises():
    with pytest.raises(KeyError):
        registry.get("sql_writer", "v99")
