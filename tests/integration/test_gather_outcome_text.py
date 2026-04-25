"""Integration test for gather_outcome_text — reads real files from disk."""

from features.labels import gather_outcome_text


class TestGatherOutcomeText:
    def test_collects_judgment_text(self, tmp_path):
        (tmp_path / "CSM001_JUDGMENT.txt").write_text("Plaintiff wins $500.")
        (tmp_path / "CSM001_CLAIM_OF_PLAINTIFF.txt").write_text("I am owed money.")

        result = gather_outcome_text("CSM001", tmp_path)
        assert "Plaintiff wins $500." in result
        assert "I am owed money." not in result

    def test_combines_multiple_outcome_docs(self, tmp_path):
        (tmp_path / "CSM002_JUDGMENT.txt").write_text("Judgment for plaintiff.")
        (tmp_path / "CSM002_DISMISSAL_OF_ENTIRE_ACTION.txt").write_text(
            "Dismissed without prejudice."
        )

        result = gather_outcome_text("CSM002", tmp_path)
        assert "Judgment for plaintiff." in result
        assert "Dismissed without prejudice." in result

    def test_returns_empty_for_missing_case(self, tmp_path):
        result = gather_outcome_text("CSM999", tmp_path)
        assert result == ""

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "CSM003_JUDGMENT.txt").write_text("")
        (tmp_path / "CSM003_DISMISSAL.txt").write_text("Dismissal text here.")

        result = gather_outcome_text("CSM003", tmp_path)
        assert "Dismissal text here." in result
        assert "[Document: JUDGMENT]" not in result

    def test_excludes_procedural_continue_order(self, tmp_path):
        (tmp_path / "CSM004_JUDGMENT.txt").write_text("Real judgment.")
        (tmp_path / "CSM004_Small_Claims_-_Continue_Order.txt").write_text(
            "Continued for interpreter."
        )

        result = gather_outcome_text("CSM004", tmp_path)
        assert "Real judgment." in result
        assert "Continued for interpreter." not in result

    def test_excludes_off_calendar_order(self, tmp_path):
        (tmp_path / "CSM005_JUDGMENT.txt").write_text("Real judgment.")
        (tmp_path / "CSM005_Off_Calendar_Order.txt").write_text("Off calendar.")

        result = gather_outcome_text("CSM005", tmp_path)
        assert "Real judgment." in result
        assert "Off calendar." not in result

    def test_excludes_motion_to_set_aside(self, tmp_path):
        (tmp_path / "CSM006_DISMISSAL.txt").write_text("Real dismissal.")
        (tmp_path / "CSM006_MOTION_TO_SET_ASIDE_DISMISSAL.txt").write_text(
            "Motion to vacate."
        )

        result = gather_outcome_text("CSM006", tmp_path)
        assert "Real dismissal." in result
        assert "Motion to vacate." not in result

    def test_drops_bare_order_when_specific_outcome_doc_present(self, tmp_path):
        (tmp_path / "CSM007_ORDER.txt").write_text(
            "SC-105A template — every option listed, no checkbox."
        )
        (tmp_path / "CSM007_Small_Claims_Order_of_Dismissal_of_Entir.txt").write_text(
            "The court orders dismissal of the entire action without prejudice."
        )

        result = gather_outcome_text("CSM007", tmp_path)
        assert "dismissal of the entire action" in result
        assert "every option listed" not in result

    def test_keeps_bare_order_when_no_specific_outcome_doc(self, tmp_path):
        (tmp_path / "CSM008_ORDER.txt").write_text("Court orders payment of $500.")

        result = gather_outcome_text("CSM008", tmp_path)
        assert "Court orders payment of $500." in result

    def test_includes_satisfaction_of_judgment(self, tmp_path):
        (tmp_path / "CSM009_SATISFACTION_OF_JUDGMENT.txt").write_text(
            "Judgment satisfied in full."
        )

        result = gather_outcome_text("CSM009", tmp_path)
        assert "Judgment satisfied in full." in result
