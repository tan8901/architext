"""
agent/state.py — Shared state schema for the LangGraph workflow.

Everything the agent knows lives here. Each node reads from and writes to
this TypedDict. LangGraph checkpoints it between nodes, giving us
persistence + resumability for free.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class Constraint(TypedDict, total=False):
    budget_monthly_usd: Optional[float]
    timeline_weeks: Optional[int]
    team_size: Optional[int]
    scale_users: Optional[int]
    scale_rps: Optional[int]          # requests per second
    compliance: list[str]             # ["HIPAA", "GDPR", ...]
    existing_stack: list[str]         # ["Rails", "PostgreSQL", ...]
    deployment_target: Optional[str]  # "AWS" | "GCP" | "Azure" | "on-prem"


class Component(TypedDict):
    name: str
    type: str          # "database" | "cache" | "api" | "compute" | "queue" | "storage"
    technology: str    # "PostgreSQL" | "Redis" | "FastAPI" ...
    purpose: str
    estimated_monthly_cost_usd: float


class TradeOff(TypedDict):
    category: str   # "scalability" | "cost" | "complexity" | "maintenance"
    pro: str
    con: str


class ArchitectureApproach(TypedDict):
    name: str                        # e.g. "Monolith-First", "Microservices", "Serverless"
    summary: str
    components: list[Component]
    data_flow: str
    trade_offs: list[TradeOff]
    total_monthly_cost_usd: float
    scores: dict[str, int]           # {"scalability": 8, "cost": 6, ...}
    mermaid_diagram: str


class Risk(TypedDict):
    severity: str   # "high" | "medium" | "low"
    area: str
    description: str
    mitigation: str


class ArchitectureReport(TypedDict):
    summary: str
    recommended: ArchitectureApproach
    alternatives: list[ArchitectureApproach]
    comparison_table: str            # markdown
    risk_assessment: list[Risk]
    next_steps: list[str]
    clarifications_asked: list[str]


# ─────────────────────────────────────────────
# The full agent state — passed between nodes
# ─────────────────────────────────────────────

class ArchitextState(TypedDict, total=False):
    # ── Input ──────────────────────────────────
    raw_input: str                            # original user text / PRD

    # ── Extraction node output ─────────────────
    project_type: str                         # "greenfield" | "feature" | "migration"
    use_case_summary: str
    functional_reqs: list[str]
    non_functional_reqs: list[str]
    constraints: Constraint
    clarifications_needed: list[str]          # questions to ask user if unclear

    # ── RAG node output ────────────────────────
    retrieved_context: list[dict[str, Any]]   # [{text, source, score}, ...]

    # ── Generation node output ─────────────────
    approaches: list[ArchitectureApproach]

    # ── Synthesis node output ──────────────────
    report: ArchitectureReport

    # ── Control flow ───────────────────────────
    needs_clarification: bool
    force_proceed: bool       # if True, skip clarification gate entirely
    iteration: int
    error: Optional[str]