"""LangGraph wiring.

    parse_query -> plan_retrieval -> retrieve -> [decision] -> assess
                        ^                 |                      |
                        +-- broaden ------+                      v
                                                          verify_citations
                                                                 |
                                                                 v
                                                          decide_verdict -> END

The conditional edge after ``retrieve`` is what makes this a graph rather than a
chain: a weak first pass loops back to ``plan_retrieval`` for one broadened
attempt, and an out-of-scope query skips assessment entirely.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from app.agent import nodes
from app.agent.nodes import AgentDeps
from app.agent.state import AgentState


def build_graph(deps: AgentDeps):
    """Compile the compliance assessment graph."""
    graph = StateGraph(AgentState)

    graph.add_node("parse_query", partial(nodes.parse_query, deps=deps))
    graph.add_node("plan_retrieval", partial(nodes.plan_retrieval, deps=deps))
    graph.add_node("retrieve", partial(nodes.retrieve, deps=deps))
    graph.add_node("assess", partial(nodes.assess, deps=deps))
    graph.add_node("verify_citations", partial(nodes.verify, deps=deps))
    graph.add_node("decide_verdict", partial(nodes.decide_verdict, deps=deps))

    graph.set_entry_point("parse_query")
    graph.add_edge("parse_query", "plan_retrieval")
    graph.add_edge("plan_retrieval", "retrieve")

    graph.add_conditional_edges(
        "retrieve",
        partial(nodes.should_broaden, deps=deps),
        {
            "plan_retrieval": "plan_retrieval",
            "assess": "assess",
            "decide_verdict": "decide_verdict",
        },
    )

    graph.add_edge("assess", "verify_citations")
    graph.add_edge("verify_citations", "decide_verdict")
    graph.add_edge("decide_verdict", END)

    return graph.compile()


def initial_state(query: str, trace_id: str) -> AgentState:
    """Seed state for one assessment."""
    return AgentState(
        query=query,
        trace_id=trace_id,
        retrieval_round=0,
        llm_calls=[],
        node_path=[],
        errors=[],
    )
