"""LangGraph StateGraph runner for webhook investigations.

Supervisor topology — not a chain.
The supervisor routes conditionally between triage, action, execute_action, and comms
based on the current AgentState.
"""

from __future__ import annotations

import os
from inspect import isawaitable
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agent.app.config.settings import get_settings
from agent.app.graph.run_action import run_action
from agent.app.graph.run_comms import run_comms
from agent.app.graph.run_execute_action import run_execute_action
from agent.app.graph.run_supervisor import supervisor_node
from agent.app.graph.run_triage import run_triage
from agent.app.graph.state import AgentState
from agent.app.schemas.drift_alert import DriftAlert
from agent.app.services import investigations
from agent.app.services.manage_checkpoints import create_checkpointer


def _initial_state(drift_alert: DriftAlert) -> AgentState:
    return {
        "investigation_id": str(uuid4()),
        "drift_event_id": drift_alert.event_id,
        "drift_alert": drift_alert,
        "severity": drift_alert.severity,
        "triage_summary": None,
        "recommended_action": None,
        "comms_summary": None,
        "job_id": None,
        "queued": None,
        "queue_name": None,
        "dispatch_error": None,
        "approval_id": None,
        "requires_approval": False,
        "status": "open",
    }


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _set_env_if_value(name: str, value: str | None) -> None:
    if value:
        os.environ[name] = value


def configure_langsmith_environment() -> None:
    """Expose Pydantic-loaded LangSmith settings to LangGraph tracing.

    LangSmith/LangGraph reads tracing configuration from process environment.
    The agent reads `.env` through Pydantic, so we bridge the values here without
    printing or otherwise exposing secrets.
    """

    settings = get_settings()
    if not _truthy(settings.LANGSMITH_TRACING):
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    _set_env_if_value("LANGSMITH_ENDPOINT", settings.LANGSMITH_ENDPOINT)
    _set_env_if_value("LANGCHAIN_ENDPOINT", settings.LANGSMITH_ENDPOINT)
    _set_env_if_value("LANGSMITH_API_KEY", settings.LANGSMITH_API_KEY)
    _set_env_if_value("LANGCHAIN_API_KEY", settings.LANGSMITH_API_KEY)
    _set_env_if_value("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)
    _set_env_if_value("LANGCHAIN_PROJECT", settings.LANGSMITH_PROJECT)
    _set_env_if_value("LANGGRAPH_API_KEY", settings.LANGGRAPH_API_KEY)


def build_agent_graph():
    """Build a LangGraph StateGraph with supervisor topology.

    Flow: START → supervisor → (triage → supervisor → action → supervisor
    → execute_action → supervisor → comms → supervisor → END)

    The supervisor reads AgentState and routes to the next node conditionally.
    Each sub-agent returns to the supervisor after completion.
    """

    configure_langsmith_environment()
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("triage", run_triage)
    graph.add_node("action", run_action)
    graph.add_node("execute_action", run_execute_action)
    graph.add_node("comms", run_comms)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next_node", END),
        {
            "triage": "triage",
            "action": "action",
            "execute_action": "execute_action",
            "comms": "comms",
            END: END,
        },
    )

    graph.add_edge("triage", "supervisor")
    graph.add_edge("action", "supervisor")
    graph.add_edge("execute_action", "supervisor")
    graph.add_edge("comms", "supervisor")

    try:
        checkpointer = create_checkpointer()
    except Exception:
        checkpointer = None

    return graph.compile(checkpointer=checkpointer)


async def run_investigation(drift_alert: DriftAlert) -> AgentState:
    """Run the supervisor topology with durable checkpoints when Postgres is available."""

    try:
        await investigations.ensure_tables()
    except Exception:
        pass

    state = await _load_or_initialize_state(drift_alert)
    node_runners = {
        "triage": run_triage,
        "action": run_action,
        "execute_action": run_execute_action,
        "comms": run_comms,
    }

    while True:
        next_node = supervisor_node(state).get("next_node", END)
        if next_node == END:
            break

        result = node_runners[next_node](state)
        state = await result if isawaitable(result) else result

        try:
            await investigations.save_state(state, last_completed_node=next_node)
        except Exception:
            pass

    return state


async def _load_or_initialize_state(drift_alert: DriftAlert) -> AgentState:
    try:
        saved_state = await investigations.load_state_by_drift_event(drift_alert.event_id)
        if saved_state is not None:
            return saved_state
    except Exception:
        pass

    state = _initial_state(drift_alert)
    try:
        await investigations.save_state(state, last_completed_node=None)
    except Exception:
        pass
    return state
