"""Tests for PDF text extraction (pymupdf + NVIDIA fallback)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from scraper.extractor import extract_text, extract_with_pymupdf


class TestExtractWithPymupdf:
    @patch("scraper.extractor.fitz")
    def test_returns_text_from_pdf(self, mock_fitz):
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page one text content here."
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_fitz.open.return_value = mock_doc

        result = extract_with_pymupdf(Path("/fake/test.pdf"))
        assert "Page one text content here." in result
        mock_doc.close.assert_called_once()

    @patch("scraper.extractor.fitz")
    def test_empty_when_no_text(self, mock_fitz):
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_fitz.open.return_value = mock_doc

        result = extract_with_pymupdf(Path("/fake/test.pdf"))
        assert result == ""


class TestExtractText:
    @patch("scraper.extractor.extract_with_pymupdf")
    def test_uses_pymupdf_when_enough_text(self, mock_pymupdf):
        mock_pymupdf.return_value = "A" * 200
        result = extract_text(Path("/fake/test.pdf"))
        assert len(result) == 200

    @patch("scraper.extractor.extract_with_nvidia")
    @patch("scraper.extractor.extract_with_pymupdf")
    def test_falls_back_to_nvidia(self, mock_pymupdf, mock_nvidia):
        mock_pymupdf.return_value = ""
        mock_nvidia.return_value = "NVIDIA extracted text"

        result = extract_text(Path("/fake/test.pdf"), nvidia_api_key="nvapi-test")
        assert result == "NVIDIA extracted text"
        mock_nvidia.assert_called_once()

    @patch("scraper.extractor.extract_with_pymupdf")
    def test_empty_when_no_key_and_no_text(self, mock_pymupdf):
        mock_pymupdf.return_value = ""
        result = extract_text(Path("/fake/test.pdf"), nvidia_api_key="")
        assert result == ""
