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
            "feat_user_is_plaintiff": _bool(self.user_is_plaintiff),
            "feat_monetary_amount_claimed": _float(self.monetary_amount_claimed),
            "feat_user_has_attorney": _bool(self.user_has_attorney),
            "feat_counterclaim_present": _bool(self.counterclaim_present),
            "feat_has_photos_or_physical_evidence": _bool(self.has_photos_or_physical_evidence),
            "feat_has_receipts_or_financial_records": _bool(self.has_receipts_or_financial_records),
            "feat_has_written_communications": _bool(self.has_written_communications),
            "feat_has_witness_statements": _bool(self.has_witness_statements),
            "feat_has_repair_or_replacement_estimate": _bool(self.has_repair_or_replacement_estimate),
            "feat_has_police_report": _bool(self.has_police_report),
            "feat_has_medical_records": _bool(self.has_medical_records),
            "feat_has_expert_assessment": _bool(self.has_expert_assessment),
            "feat_has_invoices_or_billing_records": _bool(self.has_invoices_or_billing_records),
            "feat_argument_cites_specific_dates": _bool(self.argument_cites_specific_dates),
            "feat_argument_cites_specific_dollar_amounts": _bool(
                self.argument_cites_specific_dollar_amounts
            ),
            "feat_argument_cites_contract_or_document": _bool(self.argument_cites_contract_or_document),
            "feat_argument_has_chronological_timeline": _bool(self.argument_has_chronological_timeline),
            "feat_argument_names_specific_witnesses": _bool(self.argument_names_specific_witnesses),
            "feat_argument_quantifies_each_damage_component": _bool(
                self.argument_quantifies_each_damage_component
            ),
            "feat_argument_cites_statute_or_legal_basis": _bool(
                self.argument_cites_statute_or_legal_basis
            ),
            "feat_argument_identifies_specific_location": _bool(
                self.argument_identifies_specific_location
            ),
            "feat_claim_amount_stated_in_dollars": _bool(self.claim_amount_stated_in_dollars),
            "feat_plaintiff_count": _int(self.plaintiff_count),
            "feat_defendant_count": _int(self.defendant_count),
            "feat_witness_count": _int(self.witness_count),
            "feat_text_length": float(self.text_length),
            "feat_document_count": float(self.document_count),
            "feat_claim_category_breach_of_contract": _bool(self.claim_category=="breach of contract"),
            "feat_claim_category_fraud": _bool(self.claim_category=="fraud"),
            "feat_claim_category_other": _bool(self.claim_category=="other"),
            "feat_claim_category_personal_injury": _bool(self.claim_category=="personal injury"),
            "feat_claim_category_property_damage": _bool(self.claim_category=="property damage"),
            "feat_claim_category_security_deposit": _bool(self.claim_category=="security depost"),
            "feat_claim_category_service_dispute": _bool(self.claim_category=="service dispute"),
            "feat_claim_category_unpaid_debt": _bool(self.claim_category=="unpaid debt")}