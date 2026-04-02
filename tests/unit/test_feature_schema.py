"""Tests for feature extraction schemas."""

from features.schema import FeatureVector, LLMFeatures


class TestLLMFeatures:
    def test_all_null(self):
        f = LLMFeatures()
        assert f.evidence_strength is None
        assert f.contract_present is None

    def test_full_parse(self):
        f = LLMFeatures(
            evidence_strength=4,
            contract_present=True,
            argument_clarity_plaintiff=3,
            argument_clarity_defendant=2,
            claim_category="unpaid_debt",
            monetary_amount_claimed=5000.0,
            prior_attempts_to_resolve=True,
            witness_count=2,
            documentary_evidence=True,
            timeline_clarity=4,
            legal_representation_plaintiff=False,
            legal_representation_defendant=True,
            counterclaim_present=False,
            default_judgment_likely=False,
        )
        assert f.evidence_strength == 4
        assert f.monetary_amount_claimed == 5000.0

    def test_from_dict(self):
        raw = {
            "evidence_strength": 3,
            "contract_present": False,
            "argument_clarity_plaintiff": 4,
            "argument_clarity_defendant": None,
            "claim_category": "service_dispute",
            "monetary_amount_claimed": 1500.0,
            "witness_count": 0,
        }
        f = LLMFeatures.model_validate(raw)
        assert f.evidence_strength == 3
        assert f.witness_count == 0


class TestFeatureVector:
    def test_to_model_input(self):
        v = FeatureVector(
            case_number="SC26001",
            evidence_strength=4,
            contract_present=True,
            argument_clarity_plaintiff=3,
            argument_clarity_defendant=2,
            monetary_amount_claimed=5000.0,
            witness_count=1,
            documentary_evidence=True,
            timeline_clarity=4,
            plaintiff_count=1,
            defendant_count=1,
            text_length=500,
            document_count=2,
        )
        inputs = v.to_model_input()
        assert inputs["evidence_strength"] == 4.0
        assert inputs["contract_present"] == 1.0
        assert inputs["monetary_amount_claimed"] == 5000.0
        assert inputs["plaintiff_count"] == 1.0
        assert inputs["text_length"] == 500.0

    def test_null_features_use_sentinel(self):
        v = FeatureVector(case_number="SC26001")
        inputs = v.to_model_input()
        assert inputs["evidence_strength"] == -1.0
        assert inputs["contract_present"] == -1.0

    def test_round_trip_json(self):
        v = FeatureVector(
            case_number="SC26001",
            evidence_strength=3,
            contract_present=False,
        )
        restored = FeatureVector.model_validate_json(v.model_dump_json())
        assert restored.case_number == v.case_number
        assert restored.evidence_strength == v.evidence_strength
