"""Tests for court_api.parse_case_number — extracts case ID from HTML anchor tags."""

from scraper.court_api import parse_case_number


class TestParseCaseNumber:
    def test_extracts_from_typical_html(self):
        html = '<A HREF="?CaseNum=CSM26871146&SessionID=abc123">CSM-26-871146</A>'
        assert parse_case_number(html) == "CSM26871146"

    def test_returns_none_for_no_match(self):
        assert parse_case_number("no case number here") is None

    def test_returns_none_for_empty_string(self):
        assert parse_case_number("") is None

    def test_handles_different_case_prefix(self):
        html = '<A HREF="?CaseNum=ABC12345&SessionID=xyz">ABC-12345</A>'
        assert parse_case_number(html) == "ABC12345"
