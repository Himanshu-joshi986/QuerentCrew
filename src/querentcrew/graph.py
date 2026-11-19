"""Wires the agents into a LangGraph StateGraph with memory and human-in-the-loop."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import AgentNodes
from .state import AgentState


def _default_checkpointer():
    try:
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError:  # older langgraph
        from langgraph.checkpoint.memory import MemorySaver as InMemorySaver
    return InMemorySaver()


def build_graph(nodes: AgentNodes, checkpointer=None):
    """Build and compile the QuerentCrew workflow.

    The checkpointer stores state per `thread_id`, which gives the assistant
    conversation memory and lets the graph pause at `human_approval` and resume
    later with `Command(resume=...)`.
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", nodes.router)
    graph.add_node("responder", nodes.responder)
    graph.add_node("planner", nodes.planner)
    graph.add_node("sql_writer", nodes.sql_writer)
    graph.add_node("sql_guard", nodes.sql_guard)
    graph.add_node("human_approval", nodes.human_approval)
    graph.add_node("sql_executor", nodes.sql_executor)
    graph.add_node("advance", nodes.advance)
    graph.add_node("analyst", nodes.analyst)
    graph.add_node("charter", nodes.charter)
    graph.add_node("reporter", nodes.reporter)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router", nodes.route_intent,
        {"planner": "planner", "responder": "responder", "reporter": "reporter"},
    )
    graph.add_edge("responder", END)
    graph.add_conditional_edges(
        "planner", nodes.route_after_plan, {"sql_writer": "sql_writer", "reporter": "reporter"}
    )
    graph.add_conditional_edges(
        "sql_writer", nodes.route_after_write, {"sql_guard": "sql_guard", "reporter": "reporter"}
    )
    graph.add_conditional_edges(
        "sql_guard",
        nodes.route_after_guard,
        {"sql_executor": "sql_executor", "human_approval": "human_approval",
         "sql_writer": "sql_writer", "advance": "advance"},
    )
    graph.add_conditional_edges(
        "human_approval", nodes.route_after_approval, {"sql_executor": "sql_executor", "advance": "advance"}
    )
    graph.add_conditional_edges(
        "sql_executor", nodes.route_after_execute, {"sql_writer": "sql_writer", "advance": "advance"}
    )
    graph.add_conditional_edges(
        "advance", nodes.route_after_advance, {"sql_writer": "sql_writer", "analyst": "analyst"}
    )
    graph.add_conditional_edges(
        "analyst", nodes.route_after_analyst, {"charter": "charter", "reporter": "reporter"}
    )
    graph.add_edge("charter", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile(checkpointer=checkpointer if checkpointer is not None else _default_checkpointer())
