"""
agent/nodes/retriever.py — RAG Context Retrieval Node

Queries the local ChromaDB vector store to pull relevant tech documentation,
architecture patterns, and cost data that inform the generation step.

Why RAG here vs. just prompting the LLM:
- LLM training data has outdated pricing (cloud costs change constantly)
- We can inject domain-specific patterns (e.g., HIPAA-compliant architectures)
- Keeps prompts grounded — less hallucination on specifics
- Lets you swap in proprietary docs later (company standards, preferred vendors)

Design: We run 3 parallel queries (use-case, scale, compliance) and
deduplicate results by source before returning the top-k chunks.
"""

from __future__ import annotations
import logging
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from agent.state import ArchitextState

logger = logging.getLogger(__name__)

TOP_K = 8  # chunks to retrieve per query
DEDUP_THRESHOLD = 0.05  # cosine distance threshold for deduplication


def build_retrieval_queries(state: ArchitextState) -> list[str]:
    """Generate targeted search queries from the extracted requirements."""
    queries = []

    summary = state.get("use_case_summary", "")
    constraints = state.get("constraints", {})
    scale = constraints.get("scale_users")
    compliance = constraints.get("compliance", [])
    project_type = state.get("project_type", "greenfield")

    # Primary: what are we building?
    queries.append(f"architecture for {summary}")

    # Scale-aware query
    if scale:
        if scale < 10_000:
            queries.append("small scale startup architecture monolith")
        elif scale < 100_000:
            queries.append("medium scale web application architecture microservices")
        else:
            queries.append("high scale distributed system architecture event-driven")
    else:
        queries.append("web application architecture best practices")

    # Compliance-aware query
    if compliance:
        for c in compliance[:2]:  # top 2 compliance requirements
            queries.append(f"{c} compliant architecture cloud security")

    # Migration-specific
    if project_type == "migration":
        queries.append("cloud migration strategy lift shift re-architecture")

    # NFR-driven queries
    non_functional = state.get("non_functional_reqs", [])
    for nfr in non_functional[:2]:
        queries.append(nfr)

    return queries[:5]  # cap at 5 queries


def retrieve_context(
    state: ArchitextState,
    chroma_client: chromadb.Client,
    collection_name: str = "architext_knowledge",
) -> ArchitextState:
    """
    Run parallel queries against the vector store and return deduplicated,
    ranked context chunks.
    """
    logger.info("🔍 Retrieving relevant context from knowledge base...")

    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        )
    except Exception as e:
        logger.warning(f"Knowledge base not found: {e}. Skipping RAG.")
        return {**state, "retrieved_context": []}

    queries = build_retrieval_queries(state)
    logger.info(f"   Running {len(queries)} queries: {queries[:2]}...")

    seen_ids: set[str] = set()
    results: list[dict[str, Any]] = []

    for query in queries:
        try:
            response = collection.query(
                query_texts=[query],
                n_results=min(TOP_K, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            docs = response["documents"][0]
            metas = response["metadatas"][0]
            dists = response["distances"][0]

            for doc, meta, dist in zip(docs, metas, dists):
                doc_id = meta.get("source", doc[:50])
                # Skip near-duplicates
                if doc_id not in seen_ids and dist < 1.5:
                    seen_ids.add(doc_id)
                    results.append({
                        "text": doc,
                        "source": meta.get("source", "unknown"),
                        "category": meta.get("category", "general"),
                        "score": round(1 - dist, 3),  # convert distance to similarity
                    })
        except Exception as e:
            logger.warning(f"Query failed for '{query}': {e}")

    # Sort by relevance score, take top 10
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:10]

    logger.info(f"✅ Retrieved {len(results)} unique context chunks")

    return {**state, "retrieved_context": results}
