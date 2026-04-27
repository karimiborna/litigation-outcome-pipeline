"""Integration test for LabelExtractor._parse_response — JSON → CaseLabels."""

from features.labels import FeaturesConfig, LabelExtractor


class TestLabelParsing:
    def setup_method(self):
        config = FeaturesConfig(llm_api_key="fake-key", enable_cache=False)
        self.extractor = LabelExtractor(config=config)

    def test_parses_valid_json(self):
        raw_json = """{
            "outcome": "plaintiff_win",
            "amount_awarded_principal": 1500.00,
            "amount_awarded_costs": 75.00,
            "amount_awarded_interest": 0.0,
            "defendant_appeared": true,
            "judgment_date": "2025-03-15",
            "outcome_summary": "Plaintiff awarded $1500."
        }"""
        result = self.extractor._parse_response("CSM001", raw_json)
        assert result is not None
        assert result.outcome == "plaintiff_win"
        assert result.case_number == "CSM001"
        assert result.total_awarded == 1575.00

    def test_handles_nulls(self):
        raw_json = """{
            "outcome": null,
            "amount_awarded_principal": null,
            "amount_awarded_costs": null,
            "amount_awarded_interest": null,
            "defendant_appeared": null
        }"""
        result = self.extractor._parse_response("CSM002", raw_json)
        assert result is not None
        assert result.outcome is None
        assert result.total_awarded is None

    def test_returns_none_for_invalid_json(self):
        result = self.extractor._parse_response("CSM003", "not json at all")
        assert result is None

    def test_total_awarded_sums_all_parts(self):
        raw_json = """{
            "outcome": "plaintiff_win",
            "amount_awarded_principal": 1000.0,
            "amount_awarded_costs": 200.0,
            "amount_awarded_interest": 50.0
        }"""
        result = self.extractor._parse_response("CSM004", raw_json)
        assert result.total_awarded == 1250.0
