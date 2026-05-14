"""
agent/nodes/generator.py — Architecture Generation Node

The core reasoning node. Takes structured requirements + RAG context and
generates 2-3 distinct architecture approaches with:
- Component breakdown + cost estimates
- Trade-off analysis
- Mermaid diagram code
- Scoring across 5 dimensions

Design:
- We generate approaches sequentially (not parallel) to avoid overwhelming
  a local Llama model with a huge single prompt.
- Each approach is generated with awareness of the others to ensure diversity
  (we pass previously generated approach names so the LLM contrasts them).
- Mermaid generation is a separate pass — diagram syntax is finicky and
  benefits from a focused prompt.
"""

from __future__ import annotations
import json
import re
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import ArchitextState, ArchitectureApproach

logger = logging.getLogger(__name__)


# ── Prompts ────────────────────────────────────────────────────────────────

ARCHITECT_SYSTEM = """You are a senior solutions architect with 15 years of experience.
You design practical, production-ready systems with clear reasoning.

Return ONLY valid JSON. No markdown fences, no explanation.

JSON schema:
{
  "name": "short name like 'Monolith-First' or 'Event-Driven Microservices'",
  "summary": "2-3 sentence description of the approach and when it makes sense",
  "components": [
    {
      "name": "component display name",
      "type": "database|cache|api|compute|queue|storage|cdn|auth",
      "technology": "specific tech like PostgreSQL, Redis, FastAPI, etc.",
      "purpose": "what this component does in this system",
      "estimated_monthly_cost_usd": 0.0
    }
  ],
  "data_flow": "describe how data moves through the system in 2-3 sentences",
  "trade_offs": [
    {"category": "scalability|cost|complexity|maintenance|time_to_market", "pro": "...", "con": "..."}
  ],
  "total_monthly_cost_usd": 0.0,
  "scores": {
    "scalability": 1-10,
    "cost_efficiency": 1-10,
    "complexity": 1-10,
    "time_to_market": 1-10,
    "maintainability": 1-10
  },
  "mermaid_diagram": "valid Mermaid flowchart code string"
}

Cost estimation guidelines:
- Small compute (1-2 vCPU, 2GB RAM): $10-30/mo
- Medium compute (4 vCPU, 8GB RAM): $50-150/mo
- Managed DB (small): $25-75/mo
- Managed DB (medium): $100-300/mo
- Redis cache (small): $15-50/mo
- CDN: $5-50/mo depending on traffic
- Load balancer: $15-25/mo
- Object storage: $0.023/GB (negligible for most apps)
Scale these by the user count provided.

Mermaid diagram rules — the diagram MUST be valid Mermaid syntax:
- Start with: flowchart TD
- Use: --> for arrows, -- label --> for labeled arrows
- Node IDs: no spaces (use underscores), e.g. API_Gateway
- Wrap node labels in square brackets: API_Gateway[API Gateway]
- Keep it to 8-12 nodes max for readability"""


DIAGRAM_SYSTEM = """You are an expert at writing Mermaid flowchart diagrams.
Return ONLY valid Mermaid code. Nothing else. No explanation, no fences.
Start with: flowchart TD"""


def format_context(retrieved: list[dict]) -> str:
    """Format RAG results into a clean context block for the prompt."""
    if not retrieved:
        return "No specific context retrieved — use general knowledge."
    lines = []
    for i, chunk in enumerate(retrieved[:6], 1):  # top 6 chunks
        lines.append(f"[{i}] ({chunk['category']}) {chunk['text'][:300]}")
    return "\n".join(lines)


def generate_single_approach(
    llm: ChatOllama,
    state: ArchitextState,
    approach_style: str,
    previous_names: list[str],
    context: str,
) -> ArchitectureApproach | None:
    """Generate one architecture approach."""

    constraints = state.get("constraints", {})
    existing = constraints.get("existing_stack", [])
    compliance = constraints.get("compliance", [])

    previous_note = ""
    if previous_names:
        previous_note = f"\nIMPORTANT: Already generated approaches: {previous_names}. This approach MUST be meaningfully different."

    prompt = f"""Project: {state.get('use_case_summary', '')}
Project type: {state.get('project_type', 'greenfield')}

Functional requirements:
{chr(10).join(f'- {r}' for r in state.get('functional_reqs', [])[:8])}

Non-functional requirements:
{chr(10).join(f'- {r}' for r in state.get('non_functional_reqs', [])[:5])}

Constraints:
- Monthly budget: ${constraints.get('budget_monthly_usd', 'unknown')}
- Expected users: {constraints.get('scale_users', 'unknown')}
- Team size: {constraints.get('team_size', 'unknown')}
- Timeline: {constraints.get('timeline_weeks', 'unknown')} weeks
- Compliance: {compliance if compliance else 'none specified'}
- Existing stack: {existing if existing else 'greenfield / no constraints'}
- Deployment target: {constraints.get('deployment_target', 'any cloud')}

Relevant context from knowledge base:
{context}

Generate a "{approach_style}" architecture approach for this project.{previous_note}

Remember: return ONLY valid JSON matching the schema."""

    messages = [
        SystemMessage(content=ARCHITECT_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        approach = json.loads(raw)
        # Validate required fields
        required = ["name", "summary", "components", "trade_offs", "scores"]
        if not all(k in approach for k in required):
            logger.warning(f"Approach missing fields: {[k for k in required if k not in approach]}")
        return approach
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse approach '{approach_style}': {e}")
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return None


def validate_and_repair_mermaid(diagram: str) -> str:
    """Validate and repair common Mermaid syntax errors."""
    if not diagram:
        return "flowchart TD\n    Start[Start] --> End[End]"
    
    lines = diagram.strip().split('\n')
    cleaned_lines = []
    seen_nodes = set()
    
    # Ensure first line is flowchart TD
    if not lines[0].startswith('flowchart'):
        lines.insert(0, 'flowchart TD')
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Fix node definitions: ensure brackets
        # Pattern: node_id[node label] or node_id([node label])
        import re
        
        # Fix common issues:
        # 1. Remove duplicate nodes
        node_match = re.search(r'(\w+)(\[.*?\])', line)
        if node_match:
            node_id = node_match.group(1)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
        
        # 2. Fix arrow syntax: ensure --> not ->
        line = re.sub(r'--->', '-->', line)
        line = re.sub(r'->(?![>-])', '-->', line)
        
        # 3. Fix missing brackets around node labels
        # node_name[Label] is correct
        # node_name Label is wrong
        if '[' not in line and ']' not in line:
            # Try to add brackets
            parts = re.split(r'(\s*-->|\s*-\.->|\s*==>)', line)
            for i, part in enumerate(parts):
                if part and not part.startswith('-') and not part.startswith('='):
                    # This is a node, add brackets if missing
                    if '[' not in part and ']' not in part:
                        # Split into id and label
                        words = part.strip().split()
                        if words:
                            node_id = words[0]
                            label = ' '.join(words[1:]) if len(words) > 1 else node_id
                            if label and not label.startswith('['):
                                parts[i] = f"{node_id}[{label}]"
            line = ''.join(parts)
        
        # 4. Fix labeled arrows: --label--> should be -- label -->
        line = re.sub(r'--([^-\s>]+)-->', r'-- \1 -->', line)
        
        # 5. Remove any markdown code fences
        if line.startswith('```'):
            continue
        
        cleaned_lines.append(line)
    
    # Remove duplicate lines while preserving order
    unique_lines = []
    for line in cleaned_lines:
        if line not in unique_lines:
            unique_lines.append(line)
    
    result = '\n'.join(unique_lines)
    
    # If result is too short, return a simple default
    if len(result.split('\n')) < 3:
        result = """flowchart TD
    Client[Client] --> API[API Gateway]
    API --> App[Application]
    App --> DB[(Database)]"""
    
    return result


def fix_mermaid_diagram(llm: ChatOllama, approach: ArchitectureApproach) -> str:
    """
    Generate and validate Mermaid diagram for an approach.
    """
    logger.info(f"   Generating diagram for {approach.get('name', 'approach')}...")
    
    components_desc = "\n".join(
        f"- {c.get('name', c.get('technology', 'unknown'))} ({c.get('type', 'unknown')}): {c.get('technology', 'unknown')}" 
        for c in approach.get("components", [])
    )
    data_flow = approach.get("data_flow", "")

    prompt = f"""Create a simple Mermaid flowchart for this architecture:

Architecture: {approach.get('name')}
{approach.get('summary', '')}

Components:
{components_desc}

Data flow: {data_flow}

IMPORTANT SYNTAX RULES (MUST FOLLOW):
- Start with: flowchart TD
- Node format: node_id[Node Label]
- Arrow format: NodeA --> NodeB
- Labeled arrow: NodeA -- label text --> NodeB
- Node IDs: use lowercase_with_underscores (no spaces, no hyphens)
- No duplicate nodes
- Keep it simple: 5-8 nodes maximum

Example valid diagram:
flowchart TD
    client[Web Client] --> api[API Gateway]
    api --> auth[Auth Service]
    api --> app[Application Server]
    app --> db[(PostgreSQL Database)]
    app --> cache[Redis Cache]

Return ONLY the Mermaid code, no explanation, no markdown fences."""

    try:
        messages = [
            SystemMessage(content=DIAGRAM_SYSTEM),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        diagram = response.content.strip()
        
        # Clean up any accidental fences
        diagram = re.sub(r"```(?:mermaid)?", "", diagram)
        diagram = re.sub(r"```", "", diagram)
        diagram = diagram.strip()
        
        # Validate and repair
        diagram = validate_and_repair_mermaid(diagram)
        
        # Final check: if still empty, return default
        if not diagram or len(diagram) < 30:
            return """flowchart TD
    Client[Client] --> API[API Gateway]
    API --> Service[Core Service]
    Service --> DB[(Database)]"""
        
        return diagram
        
    except Exception as e:
        logger.warning(f"Diagram generation failed: {e}")
        return """flowchart TD
    Client[Client] --> API[API Gateway]
    API --> Service[Application]
    Service --> DB[(Database)]"""

def generate_approaches(state: ArchitextState, llm: ChatOllama) -> ArchitextState:
    """
    Generate 2-3 architecture approaches with progressively increasing complexity.
    """
    logger.info("🏗️  Generating architecture approaches...")

    context = format_context(state.get("retrieved_context", []))
    scale = (state.get("constraints") or {}).get("scale_users", 0) or 0

    # Pick approach styles based on context
    if state.get("project_type") == "migration":
        styles = ["Lift-and-Shift", "Modernized Cloud-Native", "Strangler Fig Pattern"]
    elif scale > 100_000:
        styles = ["Microservices", "Event-Driven CQRS", "Modular Monolith"]
    elif scale > 10_000:
        styles = ["Modular Monolith", "Microservices", "Serverless"]
    else:
        styles = ["Simple Monolith", "Modular Monolith", "Microservices"]

    approaches: list[ArchitectureApproach] = []
    previous_names: list[str] = []

    for style in styles:
        logger.info(f"   Generating: {style}...")
        approach = generate_single_approach(
            llm=llm,
            state=state,
            approach_style=style,
            previous_names=previous_names,
            context=context,
        )
        if approach:
            # Regenerate diagram in a focused pass
            approach["mermaid_diagram"] = fix_mermaid_diagram(llm, approach)
            approaches.append(approach)
            previous_names.append(approach.get("name", style))

    logger.info(f"✅ Generated {len(approaches)} approaches")
    return {**state, "approaches": approaches}
