"""Feature schemas — the structured output from LLM extraction and the unified feature vector."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMFeatures(BaseModel):
    """Raw structured features as returned by the LLM.

    All fields are existence-based: the LLM answers "does this appear in the claim text"
    rather than giving a subjective rating. Unilateral perspective — `user_*` vs
    `opposing_party_*` fields depend on the user_side passed into the prompt.
    """

    # Classification + amount
    claim_category: str | None = None
    monetary_amount_claimed: float | None = None

    # Counts
    plaintiff_count: int | None = Field(None, ge=0)
    defendant_count: int | None = Field(None, ge=0)
    witness_count: int | None = Field(None, ge=0)

    # Representation (unilateral)
    user_has_attorney: bool | None = None
    opposing_party_has_attorney: bool | None = None
    opposing_party_filed_response_documents: bool | None = None

    # Counter-filings / contract presence
    counterclaim_present: bool | None = None
    contract_present: bool | None = None

    # Evidence existence
    has_photos_or_physical_evidence: bool | None = None
    has_receipts_or_financial_records: bool | None = None
    has_written_communications: bool | None = None
    has_witness_statements: bool | None = None
    has_signed_contract_attached: bool | None = None
    has_repair_or_replacement_estimate: bool | None = None
    has_police_report: bool | None = None
    has_medical_records: bool | None = None
    has_expert_assessment: bool | None = None
    has_invoices_or_billing_records: bool | None = None

    # Argument-content existence
    argument_cites_specific_dates: bool | None = None
    argument_cites_specific_dollar_amounts: bool | None = None
    argument_cites_contract_or_document: bool | None = None
    argument_has_chronological_timeline: bool | None = None
    argument_names_specific_witnesses: bool | None = None
    argument_quantifies_each_damage_component: bool | None = None
    argument_cites_statute_or_legal_basis: bool | None = None
    argument_identifies_specific_location: bool | None = None

    # Procedural / pre-filing conduct
    sent_written_demand_letter: bool | None = None
    sent_certified_mail: bool | None = None
    gave_opportunity_to_cure: bool | None = None
    attempted_mediation: bool | None = None

    # Contract detail (null if no contract)
    contract_is_written: bool | None = None
    contract_is_signed_by_both_parties: bool | None = None
    contract_specifies_deadline_or_term: bool | None = None
    contract_specifies_payment_amount: bool | None = None

    # Damages breakdown
    damages_include_out_of_pocket_costs: bool | None = None
    damages_include_lost_wages: bool | None = None
    damages_include_property_value_loss: bool | None = None
    damages_are_ongoing: bool | None = None
    damages_have_third_party_valuation: bool | None = None

    # Jurisdictional + claim-scope
    claim_amount_stated_in_dollars: bool | None = None
    claim_amount_is_within_small_claims_limit: bool | None = None
    user_seeks_interest: bool | None = None
    user_seeks_court_costs: bool | None = None

    # v2 dataset features. These mirror dataset.csv so training and API inference
    # can share one preprocessing path.
    user_is_plaintiff: bool | None = None
    user_has_attorney: bool | None = None
    opposing_party_has_attorney: bool | None = None
    opposing_party_filed_response_documents: bool | None = None
    has_photos_or_physical_evidence: bool | None = None
    has_receipts_or_financial_records: bool | None = None
    has_written_communications: bool | None = None
    has_witness_statements: bool | None = None
    has_signed_contract_attached: bool | None = None
    has_repair_or_replacement_estimate: bool | None = None
    has_police_report: bool | None = None
    has_medical_records: bool | None = None
    has_expert_assessment: bool | None = None
    has_invoices_or_billing_records: bool | None = None
    argument_cites_specific_dates: bool | None = None
    argument_cites_specific_dollar_amounts: bool | None = None
    argument_cites_contract_or_document: bool | None = None
    argument_has_chronological_timeline: bool | None = None
    argument_names_specific_witnesses: bool | None = None
    argument_quantifies_each_damage_component: bool | None = None
    argument_cites_statute_or_legal_basis: bool | None = None
    argument_identifies_specific_location: bool | None = None
    sent_written_demand_letter: bool | None = None
    sent_certified_mail: bool | None = None
    gave_opportunity_to_cure: bool | None = None
    attempted_mediation: bool | None = None
    contract_is_written: bool | None = None
    contract_is_signed_by_both_parties: bool | None = None
    contract_specifies_deadline_or_term: bool | None = None
    contract_specifies_payment_amount: bool | None = None
    damages_include_out_of_pocket_costs: bool | None = None
    damages_include_lost_wages: bool | None = None
    damages_include_property_value_loss: bool | None = None
    damages_are_ongoing: bool | None = None
    damages_have_third_party_valuation: bool | None = None
    claim_amount_stated_in_dollars: bool | None = None
    claim_amount_is_within_small_claims_limit: bool | None = None
    user_seeks_interest: bool | None = None
    user_seeks_court_costs: bool | None = None


class FeatureVector(BaseModel):
    """Unified feature vector combining LLM-extracted features and derived counts.

    This is the input to the classification and regression models.
    """

    case_number: str
    feature_version: str = "v2"
    missing_features: bool = False

    # User-side indicator (derived, not LLM-extracted)
    user_is_plaintiff: bool | None = None

    # Classification + amount
    claim_category: str | None = None
    monetary_amount_claimed: float | None = None

    # Representation (unilateral)
    user_has_attorney: bool | None = None
    opposing_party_has_attorney: bool | None = None
    opposing_party_filed_response_documents: bool | None = None

    # Counter-filings / contract presence
    counterclaim_present: bool | None = None
    contract_present: bool | None = None

    # Evidence existence
    has_photos_or_physical_evidence: bool | None = None
    has_receipts_or_financial_records: bool | None = None
    has_written_communications: bool | None = None
    has_witness_statements: bool | None = None
    has_signed_contract_attached: bool | None = None
    has_repair_or_replacement_estimate: bool | None = None
    has_police_report: bool | None = None
    has_medical_records: bool | None = None
    has_expert_assessment: bool | None = None
    has_invoices_or_billing_records: bool | None = None

    # Argument-content existence
    argument_cites_specific_dates: bool | None = None
    argument_cites_specific_dollar_amounts: bool | None = None
    argument_cites_contract_or_document: bool | None = None
    argument_has_chronological_timeline: bool | None = None
    argument_names_specific_witnesses: bool | None = None
    argument_quantifies_each_damage_component: bool | None = None
    argument_cites_statute_or_legal_basis: bool | None = None
    argument_identifies_specific_location: bool | None = None

    # Procedural / pre-filing conduct
    sent_written_demand_letter: bool | None = None
    sent_certified_mail: bool | None = None
    gave_opportunity_to_cure: bool | None = None
    attempted_mediation: bool | None = None

    # Contract detail (null if no contract)
    contract_is_written: bool | None = None
    contract_is_signed_by_both_parties: bool | None = None
    contract_specifies_deadline_or_term: bool | None = None
    contract_specifies_payment_amount: bool | None = None

    # Damages breakdown
    damages_include_out_of_pocket_costs: bool | None = None
    damages_include_lost_wages: bool | None = None
    damages_include_property_value_loss: bool | None = None
    damages_are_ongoing: bool | None = None
    damages_have_third_party_valuation: bool | None = None

    # Jurisdictional + claim-scope
    claim_amount_stated_in_dollars: bool | None = None
    claim_amount_is_within_small_claims_limit: bool | None = None
    user_seeks_interest: bool | None = None
    user_seeks_court_costs: bool | None = None

    # Counts (LLM-extracted where noted)
    plaintiff_count: int | None = None
    defendant_count: int | None = None
    witness_count: int | None = None

    # Derived from source text
    text_length: int = 0
    document_count: int = 0

    def to_model_input(self) -> dict[str, float]:
        """Convert to a flat dict of numeric features for model training/inference.

        Booleans become 0/1, categoricals are excluded (need encoding),
        nulls become -1 sentinel values.
        """
        sentinel = -1.0

        def _bool(v: bool | None) -> float:
            if v is None:
                return sentinel
            return 1.0 if v else 0.0

        def _int(v: int | None) -> float:
            if v is None:
                return sentinel
            return float(v)

        def _float(v: float | None) -> float:
            if v is None:
                return sentinel
            return v

        return {
            # User-side indicator
            "user_is_plaintiff": _bool(self.user_is_plaintiff),
            # Missingness flag
            "missing_features": _bool(self.missing_features),
            # Amount
            "monetary_amount_claimed": _float(self.monetary_amount_claimed),
            # Representation
            "user_has_attorney": _bool(self.user_has_attorney),
            "opposing_party_has_attorney": _bool(self.opposing_party_has_attorney),
            "opposing_party_filed_response_documents": _bool(
                self.opposing_party_filed_response_documents
            ),
            # Counter-filings / contract presence
            "counterclaim_present": _bool(self.counterclaim_present),
            "contract_present": _bool(self.contract_present),
            # Evidence
            "has_photos_or_physical_evidence": _bool(self.has_photos_or_physical_evidence),
            "has_receipts_or_financial_records": _bool(self.has_receipts_or_financial_records),
            "has_written_communications": _bool(self.has_written_communications),
            "has_witness_statements": _bool(self.has_witness_statements),
            "has_signed_contract_attached": _bool(self.has_signed_contract_attached),
            "has_repair_or_replacement_estimate": _bool(self.has_repair_or_replacement_estimate),
            "has_police_report": _bool(self.has_police_report),
            "has_medical_records": _bool(self.has_medical_records),
            "has_expert_assessment": _bool(self.has_expert_assessment),
            "has_invoices_or_billing_records": _bool(self.has_invoices_or_billing_records),
            # Argument content
            "argument_cites_specific_dates": _bool(self.argument_cites_specific_dates),
            "argument_cites_specific_dollar_amounts": _bool(
                self.argument_cites_specific_dollar_amounts
            ),
            "argument_cites_contract_or_document": _bool(self.argument_cites_contract_or_document),
            "argument_has_chronological_timeline": _bool(self.argument_has_chronological_timeline),
            "argument_names_specific_witnesses": _bool(self.argument_names_specific_witnesses),
            "argument_quantifies_each_damage_component": _bool(
                self.argument_quantifies_each_damage_component
            ),
            "argument_cites_statute_or_legal_basis": _bool(
                self.argument_cites_statute_or_legal_basis
            ),
            "argument_identifies_specific_location": _bool(
                self.argument_identifies_specific_location
            ),
            # Procedural
            "sent_written_demand_letter": _bool(self.sent_written_demand_letter),
            "sent_certified_mail": _bool(self.sent_certified_mail),
            "gave_opportunity_to_cure": _bool(self.gave_opportunity_to_cure),
            "attempted_mediation": _bool(self.attempted_mediation),
            # Contract detail
            "contract_is_written": _bool(self.contract_is_written),
            "contract_is_signed_by_both_parties": _bool(self.contract_is_signed_by_both_parties),
            "contract_specifies_deadline_or_term": _bool(self.contract_specifies_deadline_or_term),
            "contract_specifies_payment_amount": _bool(self.contract_specifies_payment_amount),
            # Damages
            "damages_include_out_of_pocket_costs": _bool(self.damages_include_out_of_pocket_costs),
            "damages_include_lost_wages": _bool(self.damages_include_lost_wages),
            "damages_include_property_value_loss": _bool(self.damages_include_property_value_loss),
            "damages_are_ongoing": _bool(self.damages_are_ongoing),
            "damages_have_third_party_valuation": _bool(self.damages_have_third_party_valuation),
            # Jurisdictional
            "claim_amount_stated_in_dollars": _bool(self.claim_amount_stated_in_dollars),
            "claim_amount_is_within_small_claims_limit": _bool(
                self.claim_amount_is_within_small_claims_limit
            ),
            "user_seeks_interest": _bool(self.user_seeks_interest),
            "user_seeks_court_costs": _bool(self.user_seeks_court_costs),
            # Counts
            "plaintiff_count": _int(self.plaintiff_count),
            "defendant_count": _int(self.defendant_count),
            "witness_count": _int(self.witness_count),
            # Derived
            "text_length": float(self.text_length),
            "document_count": float(self.document_count),
        }
