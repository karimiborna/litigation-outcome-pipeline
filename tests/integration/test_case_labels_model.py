"""Integration test for CaseLabels pydantic model — validation and computed properties."""

from features.labels import CaseLabels


class TestCaseLabelsModel:
    def test_constructs_with_all_fields(self):
        labels = CaseLabels(
            case_number="CSM001",
            outcome="plaintiff_win",
            amount_awarded_principal=1000.0,
            amount_awarded_costs=100.0,
            amount_awarded_interest=25.0,
            defendant_appeared=True,
            judgment_date="2025-01-15",
            outcome_summary="Plaintiff awarded damages.",
        )
        assert labels.case_number == "CSM001"
        assert labels.total_awarded == 1125.0

    def test_defaults_to_none(self):
        labels = CaseLabels(case_number="CSM002")
        assert labels.outcome is None
        assert labels.amount_awarded_principal is None
        assert labels.defendant_appeared is None
        assert labels.total_awarded is None

    def test_total_awarded_treats_none_as_zero(self):
        # If only principal is set, costs/interest (None) are treated as 0
        labels = CaseLabels(case_number="CSM003", amount_awarded_principal=500.0)
        assert labels.total_awarded == 500.0

    def test_model_dump_roundtrip(self):
        labels = CaseLabels(
            case_number="CSM004",
            outcome="dismissed",
            dismissal_type="without_prejudice",
        )
        data = labels.model_dump()
        restored = CaseLabels.model_validate(data)
        assert restored.case_number == labels.case_number
        assert restored.outcome == labels.outcome
        assert restored.dismissal_type == labels.dismissal_type
