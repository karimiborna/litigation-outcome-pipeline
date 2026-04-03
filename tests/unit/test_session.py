"""Tests for session management."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scraper.session import SessionExpiredError, get_session_id


class TestGetSessionId:
    def test_returns_valid_id_from_env(self):
        config = MagicMock()
        config.session_id = "  ABC123DEF456  "
        sid = get_session_id(config)
        assert sid == "ABC123DEF456"

    @patch("scraper.session._prompt_for_session_id", return_value="PROMPTED_ID")
    def test_prompts_when_empty(self, mock_prompt):
        config = MagicMock()
        config.session_id = ""
        sid = get_session_id(config)
        assert sid == "PROMPTED_ID"
        mock_prompt.assert_called_once()

    @patch("scraper.session._prompt_for_session_id", return_value="PROMPTED_ID")
    def test_prompts_on_placeholder(self, mock_prompt):
        config = MagicMock()
        config.session_id = "your-session-id-here"
        sid = get_session_id(config)
        assert sid == "PROMPTED_ID"
        mock_prompt.assert_called_once()

    def test_prefers_env_over_prompt(self):
        config = MagicMock()
        config.session_id = "REAL_SESSION_HEX_VALUE"
        sid = get_session_id(config)
        assert sid == "REAL_SESSION_HEX_VALUE"


class TestSessionExpiredError:
    def test_is_exception(self):
        exc = SessionExpiredError("test")
        assert isinstance(exc, Exception)
        assert str(exc) == "test"
