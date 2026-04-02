"""Tests for the DataSnap court API client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scraper.config import ScraperConfig
from scraper.court_api import (
    get_cases,
    get_documents,
    parse_case_number,
    sanitize_description,
)
from scraper.session import SessionExpiredError


class TestParseCaseNumber:
    def test_extracts_from_html(self):
        html = (
            '<A HREF="CaseInfo.dll?CaseNum=CSM26871146&SessionID=ABC123">'
            "CSM-26-871146</A>"
        )
        assert parse_case_number(html) == "CSM26871146"

    def test_different_case(self):
        html = '<A HREF="?CaseNum=CSM26001234&SessionID=XYZ">CSM-26-001234</A>'
        assert parse_case_number(html) == "CSM26001234"

    def test_no_match_returns_none(self):
        assert parse_case_number("plain text, no href") is None

    def test_empty_string(self):
        assert parse_case_number("") is None


class TestSanitizeDescription:
    def test_basic(self):
        assert sanitize_description("CLAIM OF PLAINTIFF") == "CLAIM_OF_PLAINTIFF"

    def test_truncation(self):
        result = sanitize_description("A" * 100, max_len=40)
        assert len(result) == 40

    def test_special_chars(self):
        result = sanitize_description("doc (1) / copy & more")
        assert "/" not in result
        assert "&" not in result
        assert "(" not in result


class TestGetCases:
    @patch("scraper.court_api.requests.get")
    def test_returns_cases(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "result": [2, '[{"CASE_NUMBER": "test", "CASETITLE": "A vs B"}]']
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        config = ScraperConfig(session_id="test_sid")
        cases = get_cases("test_sid", "2026-04-01", config)
        assert len(cases) == 1
        assert cases[0]["CASETITLE"] == "A vs B"

    @patch("scraper.court_api.requests.get")
    def test_session_expired(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": [-1, ""]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        config = ScraperConfig(session_id="test_sid")
        with pytest.raises(SessionExpiredError, match="Session expired"):
            get_cases("test_sid", "2026-04-01", config)

    @patch("scraper.court_api.requests.get")
    def test_no_cases(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": [0, ""]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        config = ScraperConfig(session_id="test_sid")
        cases = get_cases("test_sid", "2026-04-01", config)
        assert cases == []


class TestGetDocuments:
    @patch("scraper.court_api.requests.get")
    def test_returns_documents(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "result": [
                1,
                '[{"DESCRIPTION": "CLAIM", "URL": "https://example.com/doc.pdf"}]',
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        config = ScraperConfig(session_id="test_sid")
        docs = get_documents("CSM26871146", "test_sid", config)
        assert len(docs) == 1
        assert docs[0]["DESCRIPTION"] == "CLAIM"

    @patch("scraper.court_api.requests.get")
    def test_session_expired(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": [-1, ""]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        config = ScraperConfig(session_id="test_sid")
        with pytest.raises(SessionExpiredError):
            get_documents("CSM26871146", "test_sid", config)

    @patch("scraper.court_api.requests.get")
    def test_no_documents(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": [0, ""]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        config = ScraperConfig(session_id="test_sid")
        docs = get_documents("CSM26871146", "test_sid", config)
        assert docs == []
