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

Extract the following features as JSON. These names must match exactly:

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
    "default_judgment_likely": <true/false — whether defendant appears absent/non-responsive>,

    "user_is_plaintiff": <true/false — whether the input case is from plaintiff perspective>,
    "user_has_attorney": <true/false — whether the filing party has an attorney>,
    "opposing_party_has_attorney": <true/false/null>,
    "opposing_party_filed_response_documents": <true/false/null>,
    "has_photos_or_physical_evidence": <true/false>,
    "has_receipts_or_financial_records": <true/false>,
    "has_written_communications": <true/false>,
    "has_witness_statements": <true/false>,
    "has_signed_contract_attached": <true/false/null>,
    "has_repair_or_replacement_estimate": <true/false>,
    "has_police_report": <true/false>,
    "has_medical_records": <true/false>,
    "has_expert_assessment": <true/false>,
    "has_invoices_or_billing_records": <true/false>,
    "argument_cites_specific_dates": <true/false>,
    "argument_cites_specific_dollar_amounts": <true/false>,
    "argument_cites_contract_or_document": <true/false>,
    "argument_has_chronological_timeline": <true/false>,
    "argument_names_specific_witnesses": <true/false>,
    "argument_quantifies_each_damage_component": <true/false>,
    "argument_cites_statute_or_legal_basis": <true/false>,
    "argument_identifies_specific_location": <true/false>,
    "sent_written_demand_letter": <true/false/null>,
    "sent_certified_mail": <true/false/null>,
    "gave_opportunity_to_cure": <true/false/null>,
    "attempted_mediation": <true/false/null>,
    "contract_is_written": <true/false/null>,
    "contract_is_signed_by_both_parties": <true/false/null>,
    "contract_specifies_deadline_or_term": <true/false/null>,
    "contract_specifies_payment_amount": <true/false/null>,
    "damages_include_out_of_pocket_costs": <true/false/null>,
    "damages_include_lost_wages": <true/false/null>,
    "damages_include_property_value_loss": <true/false/null>,
    "damages_are_ongoing": <true/false/null>,
    "damages_have_third_party_valuation": <true/false/null>,
    "claim_amount_stated_in_dollars": <true/false>,
    "claim_amount_is_within_small_claims_limit": <true/false/null>,
    "user_seeks_interest": <true/false/null>,
    "user_seeks_court_costs": <true/false/null>
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
