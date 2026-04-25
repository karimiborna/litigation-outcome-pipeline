"""Integration test for LabelExtractor._parse_response — JSON → CaseLabels."""

from features.labels import FeaturesConfig, LabelExtractor


class TestLabelParsing:
    def setup_method(self):
        config = FeaturesConfig(llm_api_key="fake-key", enable_cache=False)
        self.extractor = LabelExtractor(config=config)

    def test_parses_valid_json(self):
        raw_json = """{
            "outcome": "plaintiff_win",
            "awarded_to_plaintiff_principal": 1500.00,
            "awarded_to_plaintiff_costs": 75.00,
            "awarded_to_plaintiff_interest": 0.0,
            "defendant_appeared": true,
            "judgment_date": "2025-03-15",
            "outcome_summary": "Plaintiff awarded $1500."
        }"""
        result = self.extractor._parse_response("CSM001", raw_json)
        assert result is not None
        assert result.outcome == "plaintiff_win"
        assert result.case_number == "CSM001"
        assert result.total_to_plaintiff == 1575.00
        assert result.net_to_plaintiff == 1575.00

    def test_handles_nulls(self):
        raw_json = """{
            "outcome": null,
            "awarded_to_plaintiff_principal": null,
            "awarded_to_plaintiff_costs": null,
            "awarded_to_plaintiff_interest": null,
            "defendant_appeared": null
        }"""
        result = self.extractor._parse_response("CSM002", raw_json)
        assert result is not None
        assert result.outcome is None
        assert result.total_to_plaintiff is None
        assert result.net_to_plaintiff is None

    def test_returns_none_for_invalid_json(self):
        result = self.extractor._parse_response("CSM003", "not json at all")
        assert result is None

    def test_total_to_plaintiff_sums_all_parts(self):
        raw_json = """{
            "outcome": "plaintiff_win",
            "awarded_to_plaintiff_principal": 1000.0,
            "awarded_to_plaintiff_costs": 200.0,
            "awarded_to_plaintiff_interest": 50.0
        }"""
        result = self.extractor._parse_response("CSM004", raw_json)
        assert result.total_to_plaintiff == 1250.0
        assert result.net_to_plaintiff == 1250.0

    def test_mutual_award_parsed(self):
        raw_json = """{
            "outcome": "mutual_award",
            "awarded_to_plaintiff_principal": 1200.0,
            "awarded_to_defendant_principal": 500.0
        }"""
        result = self.extractor._parse_response("CSM005", raw_json)
        assert result.outcome == "mutual_award"
        assert result.total_to_plaintiff == 1200.0
        assert result.total_to_defendant == 500.0
        assert result.net_to_plaintiff == 700.0

    def test_fallback_judgment_when_outcome_null(self, tmp_path):
        (tmp_path / "CSM010_JUDGMENT_ON_PLAINTIFF_S_CLAIM.txt").write_text(
            "Defendant shall pay Plaintiff: $500"
        )
        raw_json = '{"outcome": null}'
        labels = self.extractor._parse_response("CSM010", raw_json)
        result = self.extractor._fallback_outcome(
            "CSM010", tmp_path, labels, "Defendant shall pay Plaintiff: $500"
        )
        assert result.outcome == "plaintiff_win"

    def test_fallback_dismissal_with_prejudice(self, tmp_path):
        (tmp_path / "CSM011_DISMISSAL_OF_ENTIRE_ACTION.txt").write_text(
            "X With prejudice X Entire action"
        )
        raw_json = '{"outcome": null}'
        labels = self.extractor._parse_response("CSM011", raw_json)
        result = self.extractor._fallback_outcome(
            "CSM011", tmp_path, labels, "X With prejudice X Entire action"
        )
        assert result.outcome == "dismissed"
        assert result.dismissal_type == "with_prejudice"

    def test_fallback_dismissal_without_prejudice(self, tmp_path):
        (tmp_path / "CSM012_Small_Claims_Order_of_Dismissal_of_Entir.txt").write_text(
            "The court orders dismissal of the entire action without prejudice."
        )
        raw_json = '{"outcome": null}'
        labels = self.extractor._parse_response("CSM012", raw_json)
        result = self.extractor._fallback_outcome(
            "CSM012",
            tmp_path,
            labels,
            "The court orders dismissal of the entire action without prejudice.",
        )
        assert result.outcome == "dismissed"
        assert result.dismissal_type == "without_prejudice"

    def test_fallback_settled_when_only_stipulation(self, tmp_path):
        (tmp_path / "CSM013_STIPULATION.txt").write_text("Parties stipulate to settle.")
        raw_json = '{"outcome": null}'
        labels = self.extractor._parse_response("CSM013", raw_json)
        result = self.extractor._fallback_outcome(
            "CSM013", tmp_path, labels, "Parties stipulate to settle."
        )
        assert result.outcome == "settled"

    def test_fallback_does_not_override_existing_outcome(self, tmp_path):
        (tmp_path / "CSM014_DISMISSAL_OF_ENTIRE_ACTION.txt").write_text("dismissal text")
        raw_json = '{"outcome": "plaintiff_win"}'
        labels = self.extractor._parse_response("CSM014", raw_json)
        result = self.extractor._fallback_outcome(
            "CSM014", tmp_path, labels, "dismissal text"
        )
        assert result.outcome == "plaintiff_win"

    def test_fallback_judgment_with_only_defendant_amount(self, tmp_path):
        (tmp_path / "CSM015_JUDGMENT_ON_PLAINTIFF_S_CLAIM.txt").write_text("text")
        raw_json = (
            '{"outcome": null, "awarded_to_defendant_principal": 750.0,'
            ' "awarded_to_plaintiff_principal": 0.0}'
        )
        labels = self.extractor._parse_response("CSM015", raw_json)
        result = self.extractor._fallback_outcome("CSM015", tmp_path, labels, "text")
        assert result.outcome == "defendant_win"

    def test_fallback_mutual_award_when_both_amounts_present(self, tmp_path):
        (tmp_path / "CSM016_JUDGMENT_ON_PLAINTIFF_S_CLAIM.txt").write_text("text")
        raw_json = (
            '{"outcome": null, "awarded_to_plaintiff_principal": 1000.0,'
            ' "awarded_to_defendant_principal": 250.0}'
        )
        labels = self.extractor._parse_response("CSM016", raw_json)
        result = self.extractor._fallback_outcome("CSM016", tmp_path, labels, "text")
        assert result.outcome == "mutual_award"
