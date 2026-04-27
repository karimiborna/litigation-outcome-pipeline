"""Tests for court_api.sanitize_description."""

from scraper.court_api import sanitize_description


class TestSanitizeDescription:
    def test_replaces_spaces_with_underscores(self):
        assert sanitize_description("CLAIM OF PLAINTIFF") == "CLAIM_OF_PLAINTIFF"

    def test_replaces_special_characters(self):
        result = sanitize_description("Order/Judgment (Final)")
        assert "/" not in result
        assert "(" not in result
        assert ")" not in result

    def test_truncates_at_max_len(self):
        long_desc = "A" * 60
        result = sanitize_description(long_desc)
        assert len(result) == 40

    def test_custom_max_len(self):
        assert sanitize_description("abcdef", max_len=3) == "abc"

    def test_already_clean_string(self):
        assert sanitize_description("JUDGMENT") == "JUDGMENT"

    def test_empty_string(self):
        assert sanitize_description("") == ""
