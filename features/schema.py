"""Feature schemas — the structured output from LLM extraction and the unified feature vector."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMFeatures(BaseModel):
    """Raw structured features as returned by the LLM."""

    evidence_strength: int | None = Field(None, ge=1, le=5)
    evidence_description: str | None = None
    contract_present: bool | None = None
    contract_type: str | None = None
    argument_clarity_plaintiff: int | None = Field(None, ge=1, le=5)
    argument_clarity_defendant: int | None = Field(None, ge=1, le=5)
    claim_category: str | None = None
    monetary_amount_claimed: float | None = None
    prior_attempts_to_resolve: bool | None = None
    witness_count: int | None = Field(None, ge=0)
    documentary_evidence: bool | None = None
    timeline_clarity: int | None = Field(None, ge=1, le=5)
    legal_representation_plaintiff: bool | None = None
    legal_representation_defendant: bool | None = None
    counterclaim_present: bool | None = None
    default_judgment_likely: bool | None = None

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
    """Unified feature vector combining LLM-extracted features and case metadata.

    This is the input to the classification and regression models.
    """

    case_number: str
    feature_version: str = "v1"

    # LLM-extracted features
    evidence_strength: int | None = None
    contract_present: bool | None = None
    argument_clarity_plaintiff: int | None = None
    argument_clarity_defendant: int | None = None
    claim_category: str | None = None
    monetary_amount_claimed: float | None = None
    prior_attempts_to_resolve: bool | None = None
    witness_count: int | None = None
    documentary_evidence: bool | None = None
    timeline_clarity: int | None = None
    legal_representation_plaintiff: bool | None = None
    legal_representation_defendant: bool | None = None
    counterclaim_present: bool | None = None
    default_judgment_likely: bool | None = None

    # v2 dataset features
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

    # Metadata-derived features
    plaintiff_count: int = 0
    defendant_count: int = 0
    has_attorney_plaintiff: bool = False
    has_attorney_defendant: bool = False
    cause_of_action: str | None = None
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
            "evidence_strength": _int(self.evidence_strength),
            "contract_present": _bool(self.contract_present),
            "argument_clarity_plaintiff": _int(self.argument_clarity_plaintiff),
            "argument_clarity_defendant": _int(self.argument_clarity_defendant),
            "monetary_amount_claimed": _float(self.monetary_amount_claimed),
            "prior_attempts_to_resolve": _bool(self.prior_attempts_to_resolve),
            "witness_count": _int(self.witness_count),
            "documentary_evidence": _bool(self.documentary_evidence),
            "timeline_clarity": _int(self.timeline_clarity),
            "legal_representation_plaintiff": _bool(self.legal_representation_plaintiff),
            "legal_representation_defendant": _bool(self.legal_representation_defendant),
            "counterclaim_present": _bool(self.counterclaim_present),
            "default_judgment_likely": _bool(self.default_judgment_likely),
            "plaintiff_count": float(self.plaintiff_count),
            "defendant_count": float(self.defendant_count),
            "has_attorney_plaintiff": _bool(self.has_attorney_plaintiff),
            "has_attorney_defendant": _bool(self.has_attorney_defendant),
            "text_length": float(self.text_length),
            "document_count": float(self.document_count),
        }
