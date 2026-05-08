"""Prompt templates for LLM-based feature extraction from court case text."""

from __future__ import annotations

FEATURE_EXTRACTION_SYSTEM = (
    "You are a legal analyst AI. Your job is to read small claims court case "
    "documents and extract structured features from them. You must respond ONLY "
    "with valid JSON — no explanation, no markdown, no extra text.\n\n"
    "Be conservative: if information is unclear or missing, use null rather than guessing."
)

FEATURE_EXTRACTION_USER = """Analyze the following small claims court case \
and extract structured features.

Case Number: {case_number}
Case Title: {case_title}
Cause of Action: {cause_of_action}
Filing Date: {filing_date}

--- CASE TEXT ---
{case_text}
--- END CASE TEXT ---

Extract the following features as JSON:

{{
    "evidence_strength": <1-5 integer, 1=very weak, 5=very strong, null if unclear>,
    "evidence_description": <brief description of key evidence presented, null if none>,
    "contract_present": <true/false/null — whether a written contract or agreement is referenced>,
    "contract_type": <string describing contract type if present, null otherwise>,
    "argument_clarity_plaintiff": <1-5 integer, how clear/specific the plaintiff's arguments are>,
    "argument_clarity_defendant": <1-5 integer, how clear/specific the defendant's arguments are>,
    "claim_category": <one of: "property_damage", "unpaid_debt", "service_dispute", \
"security_deposit", "personal_injury", "breach_of_contract", "fraud", "other">,
    "monetary_amount_claimed": <float dollar amount claimed, null if not stated>,
    "prior_attempts_to_resolve": <true/false/null — prior negotiation or demand letters>,
    "witness_count": <integer count of witnesses mentioned, 0 if none>,
    "documentary_evidence": <true/false — whether documents/photos/receipts are referenced>,
    "timeline_clarity": <1-5 integer, how clear the timeline of events is>,
    "legal_representation_plaintiff": <true/false — whether plaintiff has an attorney>,
    "legal_representation_defendant": <true/false — whether defendant has an attorney>,
    "counterclaim_present": <true/false — whether the defendant filed a counterclaim>,
    "default_judgment_likely": <true/false — whether defendant appears absent/non-responsive>
}}

Respond with ONLY the JSON object."""


def build_extraction_prompt(
    case_number: str,
    case_title: str,
    cause_of_action: str | None,
    filing_date: str,
    case_text: str,
    max_text_length: int = 12000,
) -> list[dict[str, str]]:
    """Build the chat messages for feature extraction.

    Truncates case text to max_text_length to manage token usage.
    """
    truncated_text = case_text[:max_text_length]
    if len(case_text) > max_text_length:
        truncated_text += "\n[... text truncated ...]"

    user_content = FEATURE_EXTRACTION_USER.format(
        case_number=case_number,
        case_title=case_title,
        cause_of_action=cause_of_action or "Unknown",
        filing_date=filing_date,
        case_text=truncated_text,
    )

    return [
        {"role": "system", "content": FEATURE_EXTRACTION_SYSTEM},
        {"role": "user", "content": user_content},
    ]


SIMILARITY_ADVICE_SYSTEM = (
    "You are a legal strategy assistant."
    " Review a new small claims case and a set of historical cases "
    "that were selected for similarity."
    " Your answer should compare the new case to the best historical cases, "
    "identify what separates the successful cases from"
    " the current one, and give concise, practical advice."
    "Respond with only JSON, no markdown or extra text."
)


def build_similarity_advice_prompt(
    case_text: str,
    retrieved_cases: list[dict[str, str]],
    max_text_length: int = 3000,
) -> list[dict[str, str]]:
    """Build the chat messages for retrieval-based advice generation."""
    truncated_text = case_text[:max_text_length]
    if len(case_text) > max_text_length:
        truncated_text += "\n[... text truncated ...]"

    case_sections = [
        f"Current case text:\n{truncated_text}",
        "Retrieved historical cases:\n",
    ]
    for idx, case in enumerate(retrieved_cases, start=1):
        snippet = case.get("case_snippet") or "(no snippet available)"
        case_sections.append(
            f"{idx}. Case number: {case['case_number']}\n"
            f"   Title: {case['case_title']}\n"
            f"   Outcome: {case.get('outcome') or 'unknown'}\n"
            f"   Similarity score: {case['similarity_score']:.4f}\n"
            f"   Snippet: {snippet}\n"
        )

    user_content = (
        "A new case is described above, followed by the retrieved cases. "
        "Please answer with JSON only in this format:\n"
        "{\n"
        '  "comparison_insights": '
        "<brief description of how the best historical cases compare to the new case>,\n"
        '  "advice": <concise practical advice for'
        " the plaintiff based on the retrieved examples>\n"
        "}\n"
    )
    user_content += "\n\n" + "\n".join(case_sections)

    return [
        {"role": "system", "content": SIMILARITY_ADVICE_SYSTEM},
        {"role": "user", "content": user_content},
    ]


RAG_ADVICE_JUDGE_SYSTEM = (
    "You are an evaluator for retrieval-grounded legal information."
    " Judge whether the advice is faithful to the current case and retrieved cases,"
    " practical for a small claims litigant, clear, and appropriately cautious."
    " Do not decide the legal merits yourself. Respond with only JSON."
)


def build_rag_advice_judge_prompt(
    case_text: str,
    retrieved_cases: list[dict[str, str]],
    advice: str,
    comparison_insights: str,
    max_text_length: int = 3000,
) -> list[dict[str, str]]:
    """Build messages for LLM-as-a-judge evaluation of RAG advice."""
    truncated_text = case_text[:max_text_length]
    if len(case_text) > max_text_length:
        truncated_text += "\n[... text truncated ...]"

    case_sections = ["Retrieved historical cases:"]
    for idx, case in enumerate(retrieved_cases, start=1):
        snippet = case.get("case_snippet") or "(no snippet available)"
        case_sections.append(
            f"{idx}. Case number: {case['case_number']}\n"
            f"   Title: {case['case_title']}\n"
            f"   Outcome: {case.get('outcome') or 'unknown'}\n"
            f"   Similarity score: {case['similarity_score']:.4f}\n"
            f"   Snippet: {snippet}\n"
        )

    user_content = (
        "Evaluate this retrieval-augmented advice.\n\n"
        f"Current case text:\n{truncated_text}\n\n"
        + "\n".join(case_sections)
        + "\n\n"
        f"Comparison insights:\n{comparison_insights}\n\n"
        f"Advice:\n{advice}\n\n"
        "Return JSON only in this format:\n"
        "{\n"
        '  "score": <integer 1-5, where 5 is excellent>,\n'
        '  "verdict": <"pass", "needs_review", or "fail">,\n'
        '  "rationale": <one short sentence explaining the grade>\n'
        "}\n"
    )

    return [
        {"role": "system", "content": RAG_ADVICE_JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
