"""
tests/test_agent.py — Unit tests for Architext agent nodes

Run with: pytest tests/ -v

Tests are designed to work WITHOUT Ollama running by mocking the LLM.
This is important for CI/CD and for quick iteration during development.

Test philosophy:
- Each node is tested in isolation (unit test, not integration test)
- LLM calls are mocked — we test the logic around LLM calls, not the LLM itself
- We test edge cases: missing fields, bad JSON from LLM, empty inputs
"""

from __future__ import annotations
import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """A mock LLM that returns configurable JSON responses."""
    llm = MagicMock()
    return llm


def make_llm_response(content: str):
    """Wrap content in an AIMessage as LangChain would return it."""
    msg = MagicMock()
    msg.content = content
    return msg


SAMPLE_EXTRACTION_RESPONSE = json.dumps({
    "project_type": "greenfield",
    "use_case_summary": "Real-time chat application for 5,000 concurrent users",
    "functional_reqs": [
        "Users can send and receive messages in real-time",
        "Support group chats up to 100 members",
        "Message history persisted indefinitely",
    ],
    "non_functional_reqs": [
        "< 100ms message delivery latency",
        "99.9% uptime SLA",
        "End-to-end encryption",
    ],
    "constraints": {
        "budget_monthly_usd": 500,
        "timeline_weeks": 12,
        "team_size": 4,
        "scale_users": 5000,
        "scale_rps": None,
        "compliance": [],
        "existing_stack": [],
        "deployment_target": "AWS",
    },
    "clarifications_needed": [],
})


# ── Extractor Tests ──────────────────────────────────────────────────────────

class TestExtractRequirements:
    def test_basic_extraction(self, mock_llm):
        """Happy path: valid JSON response from LLM."""
        from agent.nodes.extractor import extract_requirements

        mock_llm.invoke.return_value = make_llm_response(SAMPLE_EXTRACTION_RESPONSE)

        state = {"raw_input": "Build a chat app for 5000 users, $500/mo budget, AWS"}
        result = extract_requirements(state, llm=mock_llm)

        assert result["project_type"] == "greenfield"
        assert result["constraints"]["scale_users"] == 5000
        assert result["constraints"]["budget_monthly_usd"] == 500
        assert result["needs_clarification"] is False
        assert result["iteration"] == 1

    def test_needs_clarification(self, mock_llm):
        """When LLM flags missing info, needs_clarification should be True."""
        from agent.nodes.extractor import extract_requirements

        response = json.dumps({
            "project_type": "greenfield",
            "use_case_summary": "Some app",
            "functional_reqs": ["Do things"],
            "non_functional_reqs": [],
            "constraints": {
                "budget_monthly_usd": None,
                "scale_users": None,
                "compliance": [],
                "existing_stack": [],
                "deployment_target": None,
            },
            "clarifications_needed": [
                "What is your expected number of users?",
                "What is your monthly infrastructure budget?",
            ],
        })
        mock_llm.invoke.return_value = make_llm_response(response)

        state = {"raw_input": "Build something"}
        result = extract_requirements(state, llm=mock_llm)

        assert result["needs_clarification"] is True
        assert len(result["clarifications_needed"]) == 2

    def test_handles_json_with_fences(self, mock_llm):
        """LLMs sometimes wrap JSON in markdown fences — should handle this."""
        from agent.nodes.extractor import extract_requirements

        fenced = f"```json\n{SAMPLE_EXTRACTION_RESPONSE}\n```"
        mock_llm.invoke.return_value = make_llm_response(fenced)

        state = {"raw_input": "Build a chat app"}
        result = extract_requirements(state, llm=mock_llm)

        assert result["project_type"] == "greenfield"

    def test_handles_malformed_json(self, mock_llm):
        """If LLM returns garbage, should return error state gracefully."""
        from agent.nodes.extractor import extract_requirements

        mock_llm.invoke.return_value = make_llm_response("Sorry, I can't help with that.")

        state = {"raw_input": "Build a chat app"}
        result = extract_requirements(state, llm=mock_llm)

        assert "error" in result


# ── Retriever Tests ──────────────────────────────────────────────────────────

class TestBuildRetrievalQueries:
    def test_generates_queries_from_state(self):
        """Should generate multiple targeted queries from structured state."""
        from agent.nodes.retriever import build_retrieval_queries

        state = {
            "use_case_summary": "Real-time chat application",
            "project_type": "greenfield",
            "constraints": {
                "scale_users": 5000,
                "compliance": ["HIPAA"],
                "deployment_target": "AWS",
            },
            "non_functional_reqs": ["end-to-end encryption", "99.9% uptime"],
        }

        queries = build_retrieval_queries(state)

        assert len(queries) >= 2
        assert any("chat" in q.lower() for q in queries)

    def test_compliance_generates_specific_query(self):
        """HIPAA compliance should generate a compliance-specific query."""
        from agent.nodes.retriever import build_retrieval_queries

        state = {
            "use_case_summary": "Patient management system",
            "project_type": "greenfield",
            "constraints": {
                "scale_users": 1000,
                "compliance": ["HIPAA", "SOC2"],
            },
            "non_functional_reqs": [],
        }

        queries = build_retrieval_queries(state)
        combined = " ".join(queries).lower()
        assert "hipaa" in combined

    def test_migration_project_type(self):
        """Migration projects should get migration-specific queries."""
        from agent.nodes.retriever import build_retrieval_queries

        state = {
            "use_case_summary": "Move CRM to AWS",
            "project_type": "migration",
            "constraints": {"scale_users": 10000, "compliance": []},
            "non_functional_reqs": [],
        }

        queries = build_retrieval_queries(state)
        combined = " ".join(queries).lower()
        assert "migrat" in combined


# ── Synthesizer Tests ────────────────────────────────────────────────────────

class TestWeightedScore:
    def test_higher_scores_rank_higher(self):
        """Approaches with better scores should have higher weighted scores."""
        from agent.nodes.synthesizer import weighted_score

        good = {"scalability": 9, "cost_efficiency": 8, "complexity": 3, "time_to_market": 8, "maintainability": 8}
        bad = {"scalability": 4, "cost_efficiency": 3, "complexity": 9, "time_to_market": 4, "maintainability": 4}

        assert weighted_score(good) > weighted_score(bad)

    def test_complexity_is_inverted(self):
        """Lower complexity score should contribute positively to overall score."""
        from agent.nodes.synthesizer import weighted_score

        low_complexity = {"scalability": 5, "cost_efficiency": 5, "complexity": 2, "time_to_market": 5, "maintainability": 5}
        high_complexity = {"scalability": 5, "cost_efficiency": 5, "complexity": 9, "time_to_market": 5, "maintainability": 5}

        assert weighted_score(low_complexity) > weighted_score(high_complexity)

    def test_score_range(self):
        """Weighted score should be between 0 and 10."""
        from agent.nodes.synthesizer import weighted_score

        perfect = {"scalability": 10, "cost_efficiency": 10, "complexity": 1, "time_to_market": 10, "maintainability": 10}
        worst = {"scalability": 1, "cost_efficiency": 1, "complexity": 10, "time_to_market": 1, "maintainability": 1}

        assert 0 <= weighted_score(worst) <= 10
        assert 0 <= weighted_score(perfect) <= 10


class TestInferRisks:
    def test_hipaa_risk_detected(self):
        """HIPAA compliance should generate a risk entry."""
        from agent.nodes.synthesizer import infer_risks

        state = {
            "constraints": {"compliance": ["HIPAA"], "team_size": 5, "budget_monthly_usd": None, "scale_users": 1000},
            "functional_reqs": ["manage patient records"],
            "non_functional_reqs": [],
        }
        recommended = {"name": "Monolith", "total_monthly_cost_usd": 200}

        risks = infer_risks(state, recommended)
        areas = [r["area"] for r in risks]
        assert "compliance" in areas

    def test_microservices_small_team_risk(self):
        """Microservices with tiny team should flag a capacity risk."""
        from agent.nodes.synthesizer import infer_risks

        state = {
            "constraints": {"compliance": [], "team_size": 2, "budget_monthly_usd": 500, "scale_users": 5000},
            "functional_reqs": [],
            "non_functional_reqs": [],
        }
        recommended = {"name": "Microservices Architecture", "total_monthly_cost_usd": 400}

        risks = infer_risks(state, recommended)
        areas = [r["area"] for r in risks]
        assert "team capacity" in areas

    def test_budget_overrun_risk(self):
        """Cost approaching budget limit should generate a cost risk."""
        from agent.nodes.synthesizer import infer_risks

        state = {
            "constraints": {"compliance": [], "team_size": 5, "budget_monthly_usd": 200, "scale_users": 5000},
            "functional_reqs": [],
            "non_functional_reqs": [],
        }
        recommended = {"name": "Modular Monolith", "total_monthly_cost_usd": 185}

        risks = infer_risks(state, recommended)
        areas = [r["area"] for r in risks]
        assert "cost" in areas


class TestComparisonTable:
    def test_table_has_all_approaches(self):
        """Comparison table should mention each approach name."""
        from agent.nodes.synthesizer import build_comparison_table

        approaches = [
            {"name": "Option A", "scores": {"scalability": 8, "cost_efficiency": 7, "complexity": 4, "time_to_market": 8, "maintainability": 7}, "total_monthly_cost_usd": 150},
            {"name": "Option B", "scores": {"scalability": 5, "cost_efficiency": 9, "complexity": 2, "time_to_market": 9, "maintainability": 8}, "total_monthly_cost_usd": 80},
        ]

        table = build_comparison_table(approaches)

        assert "Option A" in table
        assert "Option B" in table
        assert "Scalability" in table
        assert "$150" in table


# ── Integration smoke test ────────────────────────────────────────────────────

class TestEndToEnd:
    @pytest.mark.skipif(
        not __import__("subprocess").run(
            ["curl", "-s", "http://localhost:11434"],
            capture_output=True, timeout=2
        ).returncode == 0,
        reason="Ollama not running"
    )
    def test_full_pipeline_smoke(self):
        """
        Full pipeline smoke test — only runs if Ollama is available.
        Run manually with: pytest tests/ -v -k smoke
        """
        from agent.graph import run_agent

        state = run_agent(
            user_input="Build a simple todo app for a team of 3 engineers.",
            model_name="llama3.2",
        )

        assert "report" in state or "clarifications_needed" in state
