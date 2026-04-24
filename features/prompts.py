"""Prompt templates for LLM-based feature extraction from court case text."""

from __future__ import annotations

from typing import Literal

FEATURE_EXTRACTION_SYSTEM = (
    "You are a legal analyst AI. Your job is to read small claims court case "
    "documents and extract structured features based on what appears in the text. "
    "You must respond ONLY with valid JSON — no explanation, no markdown, no extra text.\n\n"
    "Answer every boolean field based on observable evidence in the text, not your "
    "judgment about the merits of the case:\n"
    "  - true: the thing is explicitly mentioned, stated, attached, or named in the text.\n"
    "  - false: the text addresses the topic but the thing is absent (e.g., claim text "
    "describes evidence but no photos are mentioned → has_photos_or_physical_evidence=false).\n"
    "  - null: the text does not address the topic at all, or it is genuinely ambiguous.\n\n"
    "Do NOT infer, guess, or reason about what 'should' be there. Do NOT make subjective "
    "quality judgments. If it's not in the text, it's either false (if addressed) or null "
    "(if unaddressed)."
)

FEATURE_EXTRACTION_USER = """Analyze the following small claims court case \
and extract structured features.

Case Number: {case_number}
Case Title: {case_title}
Cause of Action: {cause_of_action}
Filing Date: {filing_date}
User Side: {user_side}  (the party whose perspective we are modeling — \
"user_*" fields refer to this side, "opposing_party_*" fields refer to the other)

--- CASE TEXT ---
{case_text}
--- END CASE TEXT ---

Extract the following features as JSON. For every boolean, true = explicitly present in \
the text, false = text addresses the topic but the thing is absent, null = topic not \
addressed or unclear.

{{
    "claim_category": <one of: "property_damage", "unpaid_debt", "service_dispute", \
"security_deposit", "personal_injury", "breach_of_contract", "fraud", "other">,
    "monetary_amount_claimed": <float dollar amount claimed, null if not stated>,

    "plaintiff_count": <integer count of distinct plaintiff parties named, null if unclear>,
    "defendant_count": <integer count of distinct defendant parties named, null if unclear>,
    "witness_count": <integer count of witnesses mentioned, 0 if none mentioned, null if unclear>,

    "user_has_attorney": <true if the user side's attorney is named or referenced; false if \
text indicates self-representation; null if unclear>,
    "opposing_party_has_attorney": <true if the opposing party's attorney is named or \
referenced; false if text indicates self-representation; null if unclear>,
    "opposing_party_filed_response_documents": <true if the opposing party has filed any \
response, answer, or counterclaim mentioned in the text; false if text indicates no response; \
null if unclear>,

    "counterclaim_present": <true if a counterclaim is explicitly mentioned; false/null otherwise>,
    "contract_present": <true if a written or verbal contract/agreement is referenced>,

    "has_photos_or_physical_evidence": <true if photos or physical evidence are mentioned/attached>,
    "has_receipts_or_financial_records": <true if receipts or financial records are mentioned/attached>,
    "has_written_communications": <true if emails, texts, or letters are mentioned/attached>,
    "has_witness_statements": <true if witness statements or declarations are mentioned/attached>,
    "has_signed_contract_attached": <true if a signed contract document is attached or explicitly produced>,
    "has_repair_or_replacement_estimate": <true if a repair/replacement estimate is mentioned/attached>,
    "has_police_report": <true if a police report is mentioned/attached>,
    "has_medical_records": <true if medical records or bills are mentioned/attached>,
    "has_expert_assessment": <true if an expert report or professional assessment is mentioned/attached>,
    "has_invoices_or_billing_records": <true if invoices or billing records are mentioned/attached>,

    "argument_cites_specific_dates": <true if specific calendar dates are stated for events>,
    "argument_cites_specific_dollar_amounts": <true if specific dollar amounts are stated>,
    "argument_cites_contract_or_document": <true if the argument references a specific contract or document>,
    "argument_has_chronological_timeline": <true if events are described in chronological order>,
    "argument_names_specific_witnesses": <true if witnesses are named (not just counted)>,
    "argument_quantifies_each_damage_component": <true if damages are itemized into components>,
    "argument_cites_statute_or_legal_basis": <true if a statute, code section, or legal doctrine is cited>,
    "argument_identifies_specific_location": <true if a specific address or location is stated>,

    "sent_written_demand_letter": <true if a written demand letter was sent prior to filing>,
    "sent_certified_mail": <true if certified mail was used for notice>,
    "gave_opportunity_to_cure": <true if opposing party was given a chance to fix the problem before suing>,
    "attempted_mediation": <true if mediation was attempted prior to filing>,

    "contract_is_written": <true if the contract is written (not verbal); null if no contract>,
    "contract_is_signed_by_both_parties": <true if both parties signed; null if no contract>,
    "contract_specifies_deadline_or_term": <true if the contract specifies a deadline or term; null if no contract>,
    "contract_specifies_payment_amount": <true if the contract specifies a payment amount; null if no contract>,

    "damages_include_out_of_pocket_costs": <true if claimed damages include direct out-of-pocket expenses>,
    "damages_include_lost_wages": <true if claimed damages include lost wages or lost income>,
    "damages_include_property_value_loss": <true if claimed damages include loss of property value>,
    "damages_are_ongoing": <true if damages are described as continuing/recurring; false if one-time>,
    "damages_have_third_party_valuation": <true if a third party (mechanic, appraiser, etc.) valued the damages>,

    "claim_amount_stated_in_dollars": <true if a specific dollar amount is claimed; false if vague>,
    "claim_amount_is_within_small_claims_limit": <true if claim <= $10,000 (CA small claims limit); null if amount unclear>,
    "user_seeks_interest": <true if the user is explicitly seeking interest on top of principal>,
    "user_seeks_court_costs": <true if the user is explicitly seeking court costs>
}}

Respond with ONLY the JSON object."""


def build_extraction_prompt(
    case_number: str,
    case_title: str,
    cause_of_action: str | None,
    filing_date: str,
    case_text: str,
    user_side: Literal["plaintiff", "defendant"] = "plaintiff",
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
        user_side=user_side,
        case_text=truncated_text,
    )

    return [
        {"role": "system", "content": FEATURE_EXTRACTION_SYSTEM},
        {"role": "user", "content": user_content},
    ]
