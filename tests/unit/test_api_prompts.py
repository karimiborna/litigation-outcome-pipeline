"""Tests for the API-side advice and judge prompt builders."""

from __future__ import annotations

from api.prompts import (
    RAG_ADVICE_JUDGE_SYSTEM,
    SIMILARITY_ADVICE_SYSTEM,
    build_rag_advice_judge_prompt,
    build_similarity_advice_prompt,
)


_RETRIEVED = [
    {
        "case_number": "CSM26000001",
        "case_title": "DOE vs SMITH",
        "outcome": "plaintiff_win",
        "similarity_score": 0.81,
        "case_snippet": "Plaintiff sued for unpaid invoices and prevailed.",
    },
    {
        "case_number": "CSM26000002",
        "case_title": "ROE vs JONES",
        "outcome": "defendant_win",
        "similarity_score": 0.74,
        "case_snippet": "Defendant prevailed; plaintiff lacked written contract.",
    },
]

_PERTURBATION_BLOCK = (
    "Top counterfactual changes (sorted by predicted impact on win probability):\n"
    "1. Hire an attorney — currently no; +12.3pp win prob, +$340 expected award. "
    "[actionable]\n"
    "2. Attach photos or physical evidence — not addressed in your claim; "
    "+8.7pp win prob. [actionable]"
)


class TestBuildSimilarityAdvicePrompt:
    def test_message_shape(self):
        messages = build_similarity_advice_prompt(
            case_text="Plaintiff alleges unpaid invoice of $1,500.",
            retrieved_cases=_RETRIEVED,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SIMILARITY_ADVICE_SYSTEM
        assert messages[1]["role"] == "user"

    def test_includes_each_retrieved_case(self):
        messages = build_similarity_advice_prompt(
            case_text="Plaintiff alleges unpaid invoice of $1,500.",
            retrieved_cases=_RETRIEVED,
        )
        user_content = messages[1]["content"]
        assert "CSM26000001" in user_content
        assert "CSM26000002" in user_content
        assert "DOE vs SMITH" in user_content

    def test_perturbation_block_appears_when_provided(self):
        messages = build_similarity_advice_prompt(
            case_text="Plaintiff alleges unpaid invoice of $1,500.",
            retrieved_cases=_RETRIEVED,
            perturbation_summary=_PERTURBATION_BLOCK,
        )
        user_content = messages[1]["content"]
        assert "Top counterfactual changes" in user_content
        assert "Hire an attorney" in user_content
        assert "[actionable]" in user_content

    def test_perturbation_block_omitted_when_none(self):
        messages = build_similarity_advice_prompt(
            case_text="Plaintiff alleges unpaid invoice of $1,500.",
            retrieved_cases=_RETRIEVED,
            perturbation_summary=None,
        )
        user_content = messages[1]["content"]
        assert "Top counterfactual changes" not in user_content

    def test_truncates_long_text(self):
        long_text = "x" * 20000
        messages = build_similarity_advice_prompt(
            case_text=long_text,
            retrieved_cases=_RETRIEVED,
            max_text_length=100,
        )
        assert "[... text truncated ...]" in messages[1]["content"]

    def test_system_prompt_mentions_perturbation_grounding(self):
        # Confirms the system prompt instructs the LLM about the new section.
        assert "perturbation" in SIMILARITY_ADVICE_SYSTEM.lower()
        assert "load-bearing" in SIMILARITY_ADVICE_SYSTEM.lower()


class TestBuildRagAdviceJudgePrompt:
    def test_message_shape(self):
        messages = build_rag_advice_judge_prompt(
            case_text="Plaintiff alleges unpaid invoice of $1,500.",
            retrieved_cases=_RETRIEVED,
            advice="Gather receipts and consider hiring an attorney.",
            comparison_insights="Most similar plaintiff-win cases had documentary evidence.",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == RAG_ADVICE_JUDGE_SYSTEM
        assert messages[1]["role"] == "user"

    def test_perturbation_block_appears_when_provided(self):
        messages = build_rag_advice_judge_prompt(
            case_text="Plaintiff alleges unpaid invoice of $1,500.",
            retrieved_cases=_RETRIEVED,
            advice="Gather receipts and consider hiring an attorney.",
            comparison_insights="Most similar plaintiff-win cases had documentary evidence.",
            perturbation_summary=_PERTURBATION_BLOCK,
        )
        user_content = messages[1]["content"]
        assert "Top counterfactual changes" in user_content
        assert "Hire an attorney" in user_content

    def test_perturbation_block_omitted_when_none(self):
        messages = build_rag_advice_judge_prompt(
            case_text="Plaintiff alleges unpaid invoice of $1,500.",
            retrieved_cases=_RETRIEVED,
            advice="Gather receipts and consider hiring an attorney.",
            comparison_insights="Most similar plaintiff-win cases had documentary evidence.",
            perturbation_summary=None,
        )
        user_content = messages[1]["content"]
        assert "Top counterfactual changes" not in user_content

    def test_system_prompt_mentions_perturbation_factchecking(self):
        assert "perturbation" in RAG_ADVICE_JUDGE_SYSTEM.lower()
