"""Tests for feature extraction schemas (v2 existence-based schema)."""

from features.schema import FeatureVector, LLMFeatures


class TestLLMFeatures:
    def test_all_null(self):
        f = LLMFeatures()
        assert f.contract_present is None
        assert f.has_photos_or_physical_evidence is None
        assert f.argument_cites_specific_dates is None
        assert f.sent_written_demand_letter is None
        assert f.claim_amount_stated_in_dollars is None

    def test_full_parse(self):
        f = LLMFeatures(
            claim_category="unpaid_debt",
            monetary_amount_claimed=5000.0,
            plaintiff_count=1,
            defendant_count=1,
            witness_count=2,
            user_has_attorney=False,
            opposing_party_has_attorney=True,
            opposing_party_filed_response_documents=False,
            counterclaim_present=False,
            contract_present=True,
            has_photos_or_physical_evidence=True,
            has_receipts_or_financial_records=True,
            argument_cites_specific_dates=True,
            argument_cites_specific_dollar_amounts=True,
            sent_written_demand_letter=True,
            contract_is_written=True,
            damages_include_out_of_pocket_costs=True,
            claim_amount_stated_in_dollars=True,
            claim_amount_is_within_small_claims_limit=True,
        )
        assert f.claim_category == "unpaid_debt"
        assert f.monetary_amount_claimed == 5000.0
        assert f.has_photos_or_physical_evidence is True
        assert f.contract_is_written is True

    def test_from_dict(self):
        raw = {
            "claim_category": "service_dispute",
            "monetary_amount_claimed": 1500.0,
            "witness_count": 0,
            "contract_present": False,
            "has_written_communications": True,
            "argument_cites_specific_dates": True,
            "argument_cites_specific_dollar_amounts": None,
        }
        f = LLMFeatures.model_validate(raw)
        assert f.claim_category == "service_dispute"
        assert f.witness_count == 0
        assert f.has_written_communications is True
        assert f.argument_cites_specific_dollar_amounts is None


class TestFeatureVector:
    def test_to_model_input(self):
        v = FeatureVector(
            case_number="SC26001",
            user_is_plaintiff=True,
            contract_present=True,
            monetary_amount_claimed=5000.0,
            witness_count=1,
            has_photos_or_physical_evidence=True,
            argument_cites_specific_dates=True,
            sent_written_demand_letter=True,
            plaintiff_count=1,
            defendant_count=1,
            text_length=500,
            document_count=2,
        )
        inputs = v.to_model_input()
        assert inputs["user_is_plaintiff"] == 1.0
        assert inputs["contract_present"] == 1.0
        assert inputs["monetary_amount_claimed"] == 5000.0
        assert inputs["has_photos_or_physical_evidence"] == 1.0
        assert inputs["argument_cites_specific_dates"] == 1.0
        assert inputs["sent_written_demand_letter"] == 1.0
        assert inputs["plaintiff_count"] == 1.0
        assert inputs["text_length"] == 500.0

    def test_null_features_use_sentinel(self):
        v = FeatureVector(case_number="SC26001")
        inputs = v.to_model_input()
        assert inputs["contract_present"] == -1.0
        assert inputs["has_photos_or_physical_evidence"] == -1.0
        assert inputs["argument_cites_specific_dates"] == -1.0
        assert inputs["user_is_plaintiff"] == -1.0
        assert inputs["plaintiff_count"] == -1.0
        assert inputs["witness_count"] == -1.0

    def test_round_trip_json(self):
        v = FeatureVector(
            case_number="SC26001",
            user_is_plaintiff=True,
            contract_present=False,
            has_signed_contract_attached=True,
            argument_has_chronological_timeline=True,
            damages_include_lost_wages=False,
        )
        restored = FeatureVector.model_validate_json(v.model_dump_json())
        assert restored.case_number == v.case_number
        assert restored.user_is_plaintiff == v.user_is_plaintiff
        assert restored.contract_present == v.contract_present
        assert restored.has_signed_contract_attached == v.has_signed_contract_attached
        assert restored.argument_has_chronological_timeline == v.argument_has_chronological_timeline
        assert restored.damages_include_lost_wages == v.damages_include_lost_wages
