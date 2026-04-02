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
