"""Tests for LabelExtractor._smart_truncate — keeps head + tail of long text."""

from features.labels import LabelExtractor


class TestSmartTruncate:
    def test_short_text_returned_unchanged(self):
        text = "Short text."
        assert LabelExtractor._smart_truncate(text, max_chars=8000) == text

    def test_exact_limit_returned_unchanged(self):
        text = "x" * 8000
        assert LabelExtractor._smart_truncate(text, max_chars=8000) == text

    def test_long_text_is_truncated(self):
        text = "x" * 10000
        result = LabelExtractor._smart_truncate(text, max_chars=300)
        assert len(result) < 10000
        assert "[... middle truncated ...]" in result

    def test_keeps_start_and_end(self):
        # Build a string where start and end are identifiable
        text = "START" + ("m" * 10000) + "END"
        result = LabelExtractor._smart_truncate(text, max_chars=300)
        assert result.startswith("START")
        assert result.endswith("END")

    def test_head_is_one_third(self):
        text = "x" * 9000
        result = LabelExtractor._smart_truncate(text, max_chars=300)
        head_part = result.split("\n\n[... middle truncated ...]\n\n")[0]
        # head should be max_chars // 3 = 100 chars
        assert len(head_part) == 100
