"""Tests for feature extraction prompt building."""

from features.prompts import build_extraction_prompt


class TestBuildExtractionPrompt:
    def test_produces_messages(self):
        messages = build_extraction_prompt(
            case_number="SC26001",
            case_title="DOE vs SMITH",
            cause_of_action="SMALL CLAIMS",
            filing_date="2026-01-15",
            case_text="The plaintiff claims the defendant owes $500.",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "SC26001" in messages[1]["content"]
        assert "DOE vs SMITH" in messages[1]["content"]

    def test_truncates_long_text(self):
        long_text = "x" * 20000
        messages = build_extraction_prompt(
            case_number="SC26001",
            case_title="A vs B",
            cause_of_action=None,
            filing_date="2026-01-01",
            case_text=long_text,
            max_text_length=100,
        )
        assert "[... text truncated ...]" in messages[1]["content"]
        assert len(messages[1]["content"]) < 20000

    def test_handles_none_cause(self):
        messages = build_extraction_prompt(
            case_number="SC26001",
            case_title="A vs B",
            cause_of_action=None,
            filing_date="2026-01-01",
            case_text="Some case text.",
        )
        assert "Unknown" in messages[1]["content"]
