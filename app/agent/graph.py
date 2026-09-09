"""LangGraph wiring.

    parse_query -> [in scope?] -> plan_retrieval -> retrieve -> [decision] -> assess
         |                             ^                 |                      |
         | out of scope                +-- broaden ------+                      v
         |                                                            verify_citations
         |                                                                     |
         v                                                                     v
    decide_verdict <-------------------------------------------------------- END

Two conditional edges are what make this a graph rather than a chain.

After ``parse_query``: an out-of-scope query goes straight to the verdict,
skipping planning, embedding, search and reranking entirely.

After ``retrieve``: a weak first pass loops back to ``plan_retrieval`` for one
broadened attempt.
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

    # An out-of-scope query never reaches retrieval. parse_query already knows
    # it is not about a financial product, so embedding, searching and reranking
    # it spends money and sends the text to a second provider to learn nothing.
    graph.add_conditional_edges(
        "parse_query",
        nodes.should_retrieve,
        {
            "plan_retrieval": "plan_retrieval",
            "decide_verdict": "decide_verdict",
        },
    )
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
