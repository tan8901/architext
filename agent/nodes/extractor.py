"""
agent/nodes/extractor.py — Requirement Extraction Node

Parses the raw user input (free-text or PRD doc) into a structured set of
requirements and constraints. This is the hardest NLP task in the pipeline —
getting this right makes every downstream node more accurate.

Design choices:
- We ask the LLM to respond in strict JSON so we can parse it reliably.
- We explicitly prompt for implicit constraints (budget signals, scale hints)
  because users almost never state them directly.
- If critical info is missing, we set `needs_clarification=True` and populate
  `clarifications_needed` so the graph can branch to ask the user.
"""

from __future__ import annotations
import json
import re
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import ArchitextState

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM_PROMPT = """You are a senior solutions architect performing requirement analysis.

Extract structured information from the user's project description.
Return ONLY valid JSON — no markdown fences, no explanation, no preamble.

JSON schema to follow exactly:
{
  "project_type": "greenfield" | "feature_addition" | "migration",
  "use_case_summary": "one sentence describing what this system does",
  "functional_reqs": ["list of what the system must DO"],
  "non_functional_reqs": ["list of quality attributes: performance, security, etc."],
  "constraints": {
    "budget_monthly_usd": null or number,
    "timeline_weeks": null or number,
    "team_size": null or number,
    "scale_users": null or number (estimated concurrent or monthly active users),
    "scale_rps": null or number (requests per second if mentioned),
    "compliance": ["HIPAA", "GDPR", etc. — empty list if none],
    "existing_stack": ["technologies already in use — empty if none"],
    "deployment_target": null or "AWS" | "GCP" | "Azure" | "on-prem" | "any"
  },
  "clarifications_needed": []
}

INFERENCE RULES — always apply these before considering a clarification:
- Infer scale from explicit numbers first; if none, use context: "startup" → 1k users, "small team" → 5k, "enterprise" → 100k+
- Infer budget from scale if not stated: <5k users → $200/mo, 5k-50k → $500/mo, 50k+ → $2000/mo
- Infer deployment from context: if a cloud is named, use it; otherwise default to "any"
- Infer compliance from domain: "healthcare"/"patient"/"medical" → HIPAA; "EU"/"European" → GDPR
- A null value is FINE — the generator handles missing fields with sensible defaults

STRICT CLARIFICATION RULES — clarifications_needed must be EMPTY [] unless ALL of these are true:
  1. The missing information would cause a fundamentally DIFFERENT architecture (not just a parameter tweak)
  2. It cannot be reasonably inferred from ANY context clue in the description
  3. There is no safe default assumption that a senior architect would make

EXAMPLES where clarifications_needed should be [] (empty — just proceed):
- Budget not mentioned → infer from scale, proceed
- Payment gateway not specified → note Stripe as default in functional_reqs, proceed  
- Data volume not specified → infer "moderate", proceed
- Exact uptime SLA not given → default 99.9%, proceed
- Specific auth method not given → default to JWT/OAuth2, proceed

EXAMPLE where a clarification IS justified (rare):
- "Build something for my company" with zero other context — project type is genuinely unknown

When in doubt: PROCEED with reasonable defaults. Do NOT ask clarifying questions."""


def extract_requirements(state: ArchitextState, llm: ChatOllama) -> ArchitextState:
    """
    Parse raw input into structured requirements.
    Sets needs_clarification=True if critical info is missing.
    """
    logger.info("📋 Extracting requirements...")

    messages = [
        SystemMessage(content=EXTRACT_SYSTEM_PROMPT),
        HumanMessage(content=f"Project description:\n\n{state['raw_input']}"),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip any accidental markdown fences the model adds
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed, attempting recovery: {e}")
        # Try to extract JSON object from response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            logger.error("Could not parse LLM output as JSON")
            return {**state, "error": f"Extraction failed: {e}"}

    clarifications = parsed.get("clarifications_needed", [])
    needs_clarification = len(clarifications) > 0

    logger.info(f"✅ Extracted: {parsed.get('project_type')} | "
                f"Scale: {parsed.get('constraints', {}).get('scale_users')} users | "
                f"Clarifications needed: {len(clarifications)}")

    return {
        **state,
        "project_type": parsed.get("project_type", "greenfield"),
        "use_case_summary": parsed.get("use_case_summary", ""),
        "functional_reqs": parsed.get("functional_reqs", []),
        "non_functional_reqs": parsed.get("non_functional_reqs", []),
        "constraints": parsed.get("constraints", {}),
        "clarifications_needed": clarifications,
        "needs_clarification": needs_clarification,
        "iteration": state.get("iteration", 0) + 1,
    }