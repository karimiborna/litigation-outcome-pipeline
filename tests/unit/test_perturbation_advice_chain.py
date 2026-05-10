"""Integration test: full chain from FeatureVector → LLM prompt.

Verifies that the perturbation analysis flows through ``select_top_recommendations``,
``format_for_llm``, and into both the advisor and judge prompts as a coherent
block — friendly names instead of raw ``feat_*`` columns, with the expected
header, tags, and verbatim inclusion in both prompts.
"""

from __future__ import annotations

from api.prompts import (
    build_rag_advice_judge_prompt,
    build_similarity_advice_prompt,
)
from counterfactual.analyzer import (
    FEATURE_DISPLAY_NAMES,
    CounterfactualAnalyzer,
    format_for_llm,
    select_top_recommendations,
)
from tests.unit.test_counterfactual import FakeClassifier, FakeRegressor, _weak_case

_RETRIEVED_CASES = [
    {
        "case_number": "CSM26000001",
        "case_title": "DOE vs SMITH",
        "outcome": "plaintiff_win",
        "similarity_score": 0.81,
        "case_snippet": "Plaintiff sued for unpaid invoices and prevailed.",
    },
]


def test_perturbation_chain_feeds_advisor_and_judge_prompts():
    """Weak case → analyzer → top-5 → formatter → advisor + judge prompts.

    Asserts the full data plumbing is intact end-to-end:
    1. The analyzer produces results for a weak case.
    2. ``select_top_recommendations`` narrows them to ≤ 5 helpful flips.
    3. ``format_for_llm`` emits a header, friendly names (no raw ``feat_*``), and tags.
    4. ``build_similarity_advice_prompt`` embeds the summary verbatim.
    5. ``build_rag_advice_judge_prompt`` embeds the same summary verbatim.
    6. The friendly name of the top-ranked perturbation appears in the advisor prompt.
    """
    analyzer = CounterfactualAnalyzer(FakeClassifier(), FakeRegressor())

    all_results = analyzer.analyze(_weak_case())
    assert all_results, "analyzer should produce results for a weak case"

    top = select_top_recommendations(all_results, top_n=5)
    assert 1 <= len(top) <= 5
    assert all(r.direction == "helpful" for r in top), (
        "every top recommendation for a weak case should be helpful"
    )

    summary = format_for_llm(top)
    assert summary.startswith(
        "Top counterfactual changes (sorted by predicted impact on win probability):"
    )
    assert "feat_" not in summary, "friendly names should replace raw feat_* identifiers"
    assert "[actionable]" in summary

    case_text = "Plaintiff alleges unpaid invoice of $1,500 — see attached records."
    advisor_messages = build_similarity_advice_prompt(
        case_text=case_text,
        retrieved_cases=_RETRIEVED_CASES,
        perturbation_summary=summary,
    )
    advisor_user = advisor_messages[1]["content"]
    assert summary in advisor_user, "advisor prompt should embed the summary verbatim"

    top_friendly = FEATURE_DISPLAY_NAMES[top[0].feature_name]
    assert top_friendly in advisor_user, (
        f"top-ranked feature {top[0].feature_name!r} ({top_friendly!r}) "
        "should appear in the advisor prompt"
    )

    judge_messages = build_rag_advice_judge_prompt(
        case_text=case_text,
        retrieved_cases=_RETRIEVED_CASES,
        advice="Hire an attorney and gather written communications.",
        comparison_insights="Plaintiff-win cases all had documentary evidence.",
        perturbation_summary=summary,
    )
    judge_user = judge_messages[1]["content"]
    assert summary in judge_user, "judge prompt should embed the same summary verbatim"
