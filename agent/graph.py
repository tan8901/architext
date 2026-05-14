"""
agent/graph.py — LangGraph Workflow Definition

Assembles the 4 nodes into a stateful directed graph:

  extract_requirements
         │
         ├─ (needs_clarification=True) → END  [return questions to user]
         │
         └─ (needs_clarification=False)
                 │
         retrieve_context
                 │
         generate_approaches
                 │
         synthesize_report
                 │
               END

Why LangGraph over a plain function pipeline:
- State is checkpointed between nodes → resumable on failure
- Conditional edges let us branch to ask clarifying questions
- Easy to add parallelism later (e.g., evaluate approaches concurrently)
- Thread-safe: multiple users can run concurrent sessions
"""

from __future__ import annotations
import os
import logging
from functools import partial

import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from agent.state import ArchitextState
from agent.nodes.extractor import extract_requirements
from agent.nodes.retriever import retrieve_context
from agent.nodes.generator import generate_approaches
from agent.nodes.synthesizer import synthesize_report

logger = logging.getLogger(__name__)


def _should_clarify(state: ArchitextState) -> str:
    """
    Conditional edge: only block if the extractor found clarifications needed
    AND the user hasn't already been asked once (iteration==1) AND force_proceed
    is not set AND the input is truly sparse (no use_case_summary extracted).

    Design intent: be aggressive about proceeding. A slightly incomplete
    architecture recommendation is more useful than stopping to ask questions.
    The generator handles null constraints gracefully with sensible defaults.
    """
    if state.get("force_proceed"):
        return "continue"

    # Only stop if we have clarifications AND genuinely couldn't extract a summary
    has_clarifications = state.get("needs_clarification") and len(state.get("clarifications_needed", [])) > 0
    truly_empty = not state.get("use_case_summary", "").strip()
    first_pass = state.get("iteration", 0) == 1

    if has_clarifications and truly_empty and first_pass:
        return "needs_clarification"
    return "continue"


def build_graph(
    ollama_base_url: str = "http://localhost:11434",
    model_name: str = "llama3.2",
    chroma_persist_dir: str = "./data/chroma_db",
) -> StateGraph:
    """
    Build and compile the Architext LangGraph workflow.

    Returns a compiled graph ready to invoke.
    """
    logger.info(f"Building graph with model={model_name}, ollama={ollama_base_url}")

    # ── Initialize shared dependencies ──────────────────────────────────────

    llm = ChatOllama(
        model=model_name,
        base_url=ollama_base_url,
        temperature=0.3,           # lower = more consistent architecture decisions
        num_predict=4096,          # enough for full JSON responses
    )

    chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)

    # ── Bind dependencies to nodes via partial application ──────────────────
    # This pattern lets nodes be pure functions (easy to test) while still
    # having access to shared resources.

    _extract = partial(extract_requirements, llm=llm)
    _retrieve = partial(retrieve_context, chroma_client=chroma_client)
    _generate = partial(generate_approaches, llm=llm)
    _synthesize = partial(synthesize_report, llm=llm)

    # ── Define graph ────────────────────────────────────────────────────────

    workflow = StateGraph(ArchitextState)

    workflow.add_node("extract_requirements", _extract)
    workflow.add_node("retrieve_context", _retrieve)
    workflow.add_node("generate_approaches", _generate)
    workflow.add_node("synthesize_report", _synthesize)

    # Entry point
    workflow.set_entry_point("extract_requirements")

    # Conditional edge: clarify or continue
    workflow.add_conditional_edges(
        "extract_requirements",
        _should_clarify,
        {
            "needs_clarification": END,   # return to user with questions
            "continue": "retrieve_context",
        },
    )

    # Linear flow after retrieval
    workflow.add_edge("retrieve_context", "generate_approaches")
    workflow.add_edge("generate_approaches", "synthesize_report")
    workflow.add_edge("synthesize_report", END)

    return workflow.compile()


def run_agent(
    user_input: str,
    clarifications: dict | None = None,
    force_proceed: bool = False,
    ollama_base_url: str = "http://localhost:11434",
    model_name: str = "llama3.2",
    chroma_persist_dir: str = "./data/chroma_db",
) -> ArchitextState:
    """
    Main entrypoint for running the agent end-to-end.

    Args:
        user_input: Raw project description / PRD text
        clarifications: Optional dict of answered clarifying questions
                        (appended to input on second run)
        force_proceed: If True, skip the clarification gate entirely and
                       generate with whatever info is available.
    Returns:
        Final ArchitextState — check state['report'] for the result,
        or state['clarifications_needed'] if more info is required.
    """
    graph = build_graph(
        ollama_base_url=ollama_base_url,
        model_name=model_name,
        chroma_persist_dir=chroma_persist_dir,
    )

    # If user answered clarifications, append them to the input
    full_input = user_input
    if clarifications:
        answers = "\n".join(f"- {q}: {a}" for q, a in clarifications.items())
        full_input += f"\n\nAdditional clarifications:\n{answers}"

    initial_state: ArchitextState = {
        "raw_input": full_input,
        "iteration": 0,
        "needs_clarification": False,
        "force_proceed": force_proceed,
    }

    logger.info("🚀 Starting Architext agent...")
    final_state = graph.invoke(initial_state)
    logger.info("✅ Agent run complete")

    return final_state