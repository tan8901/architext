"""
agent/nodes/synthesizer.py — Synthesis & Report Node

Takes all generated approaches and produces the final ArchitectureReport:
- Selects the recommended approach with reasoning
- Builds a comparison table
- Identifies cross-cutting risks
- Writes actionable next steps

This node does NOT call the LLM for the comparison table or risk matrix —
those are computed deterministically from the structured data.
The LLM is only used for the narrative summary and next steps,
where natural language quality matters.
"""

from __future__ import annotations
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import ArchitextState, ArchitectureReport, Risk

logger = logging.getLogger(__name__)


SCORE_WEIGHTS = {
    "scalability": 0.25,
    "cost_efficiency": 0.20,
    "complexity": 0.15,       # inverted: lower complexity = better
    "time_to_market": 0.20,
    "maintainability": 0.20,
}


def weighted_score(scores: dict[str, int]) -> float:
    """Calculate weighted score for an approach. Higher = better."""
    total = 0.0
    for metric, weight in SCORE_WEIGHTS.items():
        val = scores.get(metric, 5)
        # Invert complexity: a score of 3 complexity → 7 adjusted
        if metric == "complexity":
            val = 10 - val
        total += val * weight
    return round(total, 2)


def build_comparison_table(approaches: list) -> str:
    """Build a markdown comparison table from the scored approaches."""
    if not approaches:
        return ""

    metrics = ["scalability", "cost_efficiency", "complexity", "time_to_market", "maintainability"]
    header = "| Metric | " + " | ".join(a.get("name", "?") for a in approaches) + " |"
    sep = "|--------|" + "--------|" * len(approaches)

    rows = [header, sep]

    for metric in metrics:
        label = metric.replace("_", " ").title()
        vals = []
        for a in approaches:
            v = a.get("scores", {}).get(metric, "?")
            # Add emoji indicators
            if isinstance(v, int):
                if metric == "complexity":
                    emoji = "🟢" if v <= 4 else ("🟡" if v <= 7 else "🔴")
                else:
                    emoji = "🟢" if v >= 7 else ("🟡" if v >= 5 else "🔴")
                vals.append(f"{emoji} {v}/10")
            else:
                vals.append(str(v))
        rows.append(f"| {label} | " + " | ".join(vals) + " |")

    # Cost row
    cost_row = "| Est. Monthly Cost | "
    cost_row += " | ".join(
        f"${a.get('total_monthly_cost_usd', 0):.0f}" for a in approaches
    )
    cost_row += " |"
    rows.append(cost_row)

    # Weighted score row
    ws_row = "| **Overall Score** | "
    ws_row += " | ".join(
        f"**{weighted_score(a.get('scores', {})):.1f}**" for a in approaches
    )
    ws_row += " |"
    rows.append(ws_row)

    return "\n".join(rows)


def infer_risks(state: ArchitextState, recommended) -> list[Risk]:
    """
    Rule-based risk inference — deterministic, no LLM needed.
    Catches common architectural pitfalls based on the constraints.
    """
    risks: list[Risk] = []
    constraints = state.get("constraints", {})
    scale = constraints.get("scale_users", 0) or 0
    compliance = constraints.get("compliance", [])
    team_size = constraints.get("team_size", 0) or 0
    approach_name = recommended.get("name", "").lower()

    # Scale mismatch risks
    if "microservices" in approach_name and team_size < 5:
        risks.append({
            "severity": "high",
            "area": "team capacity",
            "description": "Microservices with a small team creates high operational overhead",
            "mitigation": "Consider a modular monolith first; migrate to microservices when you have 8+ engineers",
        })

    if "monolith" in approach_name and scale > 50_000:
        risks.append({
            "severity": "medium",
            "area": "scalability",
            "description": "Monolithic architecture may hit scaling limits at 50k+ concurrent users",
            "mitigation": "Plan horizontal scaling strategy and identify stateless components early",
        })

    # Compliance risks
    if "HIPAA" in compliance:
        risks.append({
            "severity": "high",
            "area": "compliance",
            "description": "HIPAA requires encryption at rest, audit logging, and BAAs with all vendors",
            "mitigation": "Use HIPAA-eligible AWS/GCP services, implement audit trail from day 1",
        })

    if "GDPR" in compliance:
        risks.append({
            "severity": "medium",
            "area": "compliance",
            "description": "GDPR requires data residency controls and right-to-erasure implementation",
            "mitigation": "Design data deletion workflows early; choose EU regions for user data storage",
        })

    # Cost risks
    budget = constraints.get("budget_monthly_usd")
    estimated_cost = recommended.get("total_monthly_cost_usd", 0)
    if budget and estimated_cost > budget * 0.8:
        risks.append({
            "severity": "high",
            "area": "cost",
            "description": f"Estimated cost (${estimated_cost:.0f}/mo) approaches budget limit (${budget:.0f}/mo)",
            "mitigation": "Use spot/preemptible instances, implement aggressive caching, review right-sizing",
        })

    # No auth mentioned
    functional = " ".join(state.get("functional_reqs", [])).lower()
    if "auth" not in functional and "login" not in functional and "user" in functional:
        risks.append({
            "severity": "medium",
            "area": "security",
            "description": "Authentication/authorization not explicitly specified but system has users",
            "mitigation": "Use managed auth (Auth0, Cognito, Firebase Auth) — don't build from scratch",
        })

    return risks


def generate_summary_and_nextsteps(
    llm: ChatOllama,
    state: ArchitextState,
    recommended,
    report_context: str,
) -> tuple[str, list[str]]:
    """Use LLM for the narrative summary and next steps only."""

    prompt = f"""You are writing the executive summary section of an architecture report.

Project: {state.get('use_case_summary')}
Recommended approach: {recommended.get('name')}
Summary: {recommended.get('summary')}

Constraints: budget=${state.get('constraints', {}).get('budget_monthly_usd')}/mo, 
scale={state.get('constraints', {}).get('scale_users')} users,
team={state.get('constraints', {}).get('team_size')} engineers

{report_context}

Write:
1. A 3-4 sentence executive summary explaining why this architecture was chosen
2. 5-7 concrete next steps (implementation order matters)

Return as JSON:
{{"summary": "...", "next_steps": ["step 1", "step 2", ...]}}

Return ONLY JSON."""

    from langchain_core.messages import SystemMessage, HumanMessage
    import json, re

    messages = [
        SystemMessage(content="You are a senior solutions architect writing a concise technical report. Return only valid JSON."),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    raw = re.sub(r"```(?:json)?|```", "", response.content.strip()).strip()

    try:
        parsed = json.loads(raw)
        return parsed.get("summary", ""), parsed.get("next_steps", [])
    except Exception:
        return recommended.get("summary", ""), [
            "Set up development environment and CI/CD pipeline",
            "Implement core data models and database schema",
            "Build authentication and authorization layer",
            "Implement primary API endpoints",
            "Set up monitoring and observability",
            "Load test against target scale requirements",
        ]


def synthesize_report(state: ArchitextState, llm: ChatOllama) -> ArchitextState:
    """Combine all approaches into a final ranked report."""
    logger.info("📊 Synthesizing final report...")

    approaches = state.get("approaches", [])
    if not approaches:
        return {**state, "error": "No approaches generated to synthesize"}

    # Score and rank approaches
    scored = sorted(
        approaches,
        key=lambda a: weighted_score(a.get("scores", {})),
        reverse=True,
    )

    recommended = scored[0]
    alternatives = scored[1:]

    # Build comparison table (deterministic)
    comparison_table = build_comparison_table(approaches)

    # Infer risks (rule-based)
    risks = infer_risks(state, recommended)

    # LLM for summary and next steps
    report_context = f"Top-scoring approach has weighted score {weighted_score(recommended.get('scores', {})):.1f}/10"
    summary, next_steps = generate_summary_and_nextsteps(
        llm=llm,
        state=state,
        recommended=recommended,
        report_context=report_context,
    )

    report: ArchitectureReport = {
        "summary": summary,
        "recommended": recommended,
        "alternatives": alternatives,
        "comparison_table": comparison_table,
        "risk_assessment": risks,
        "next_steps": next_steps,
        "clarifications_asked": state.get("clarifications_needed", []),
    }

    logger.info(f"✅ Report complete. Recommended: {recommended.get('name')} "
                f"(score: {weighted_score(recommended.get('scores', {})):.1f})")

    return {**state, "report": report}
