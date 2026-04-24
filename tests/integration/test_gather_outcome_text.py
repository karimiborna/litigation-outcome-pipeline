"""Integration test for gather_outcome_text — reads real files from disk."""

from features.labels import gather_outcome_text


class TestGatherOutcomeText:
    def test_collects_judgment_text(self, tmp_path):
        # Create a judgment file — should be included
        (tmp_path / "CSM001_JUDGMENT.txt").write_text("Plaintiff wins $500.")
        # Create a claim file — should NOT be included (not an outcome doc)
        (tmp_path / "CSM001_CLAIM_OF_PLAINTIFF.txt").write_text("I am owed money.")

        result = gather_outcome_text("CSM001", tmp_path)
        assert "Plaintiff wins $500." in result
        assert "I am owed money." not in result

    def test_combines_multiple_outcome_docs(self, tmp_path):
        (tmp_path / "CSM002_JUDGMENT.txt").write_text("Judgment for plaintiff.")
        (tmp_path / "CSM002_ORDER.txt").write_text("Court orders payment.")

        result = gather_outcome_text("CSM002", tmp_path)
        assert "Judgment for plaintiff." in result
        assert "Court orders payment." in result

    def test_returns_empty_for_missing_case(self, tmp_path):
        result = gather_outcome_text("CSM999", tmp_path)
        assert result == ""

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "CSM003_JUDGMENT.txt").write_text("")
        (tmp_path / "CSM003_ORDER.txt").write_text("Order text here.")

        result = gather_outcome_text("CSM003", tmp_path)
        assert "Order text here." in result
        assert "[Document: JUDGMENT]" not in result
