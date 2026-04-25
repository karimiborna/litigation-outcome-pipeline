"""Integration test for CaseLabels pydantic model — validation and computed properties."""

from features.labels import CaseLabels


class TestCaseLabelsModel:
    def test_constructs_with_all_fields(self):
        labels = CaseLabels(
            case_number="CSM001",
            outcome="plaintiff_win",
            awarded_to_plaintiff_principal=1000.0,
            awarded_to_plaintiff_costs=100.0,
            awarded_to_plaintiff_interest=25.0,
            defendant_appeared=True,
            judgment_date="2025-01-15",
            outcome_summary="Plaintiff awarded damages.",
        )
        assert labels.case_number == "CSM001"
        assert labels.total_to_plaintiff == 1125.0
        assert labels.total_to_defendant is None
        assert labels.net_to_plaintiff == 1125.0

    def test_defaults_to_none(self):
        labels = CaseLabels(case_number="CSM002")
        assert labels.outcome is None
        assert labels.awarded_to_plaintiff_principal is None
        assert labels.defendant_appeared is None
        assert labels.total_to_plaintiff is None
        assert labels.total_to_defendant is None
        assert labels.net_to_plaintiff is None

    def test_total_treats_none_as_zero(self):
        labels = CaseLabels(case_number="CSM003", awarded_to_plaintiff_principal=500.0)
        assert labels.total_to_plaintiff == 500.0

    def test_mutual_award_net(self):
        labels = CaseLabels(
            case_number="CSM005",
            outcome="mutual_award",
            awarded_to_plaintiff_principal=1000.0,
            awarded_to_defendant_principal=400.0,
        )
        assert labels.total_to_plaintiff == 1000.0
        assert labels.total_to_defendant == 400.0
        assert labels.net_to_plaintiff == 600.0

    def test_defendant_only_award_net_negative(self):
        labels = CaseLabels(
            case_number="CSM006",
            outcome="defendant_win",
            awarded_to_defendant_principal=750.0,
        )
        assert labels.net_to_plaintiff == -750.0

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
