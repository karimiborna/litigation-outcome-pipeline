"""Tests for data storage utilities."""

from datetime import date
from pathlib import Path

from data.schemas.case import CaseMetadata, ExtractedText
from data.storage import (
    case_pdfs,
    list_scraped_cases,
    load_metadata,
    save_extracted_text,
    save_metadata,
    save_pdf,
    save_raw_html,
)


class TestSaveAndLoadMetadata:
    def test_round_trip(self, tmp_path: Path):
        metadata = CaseMetadata(
            case_number="SC26001",
            case_title="DOE vs SMITH",
            filing_date=date(2026, 1, 15),
        )
        save_metadata(metadata, tmp_path)
        loaded = load_metadata("SC26001", tmp_path)
        assert loaded is not None
        assert loaded.case_number == "SC26001"
        assert loaded.case_title == "DOE vs SMITH"

    def test_load_nonexistent(self, tmp_path: Path):
        result = load_metadata("NONEXISTENT", tmp_path)
        assert result is None


class TestSavePdf:
    def test_save(self, tmp_path: Path):
        content = b"%PDF-1.4 fake content"
        path = save_pdf("SC26001", "doc_000.pdf", content, tmp_path)
        assert path.exists()
        assert path.read_bytes() == content


class TestSaveRawHtml:
    def test_save(self, tmp_path: Path):
        html = "<html><body>test</body></html>"
        path = save_raw_html("SC26001", "detail", html, tmp_path)
        assert path.exists()
        assert path.read_text() == html


class TestSaveExtractedText:
    def test_save(self, tmp_path: Path):
        extracted = ExtractedText(
            case_number="SC26001",
            document_filename="doc_000.pdf",
            pages=["Page 1 text"],
        )
        path = save_extracted_text(extracted, tmp_path)
        assert path.exists()
        assert "Page 1 text" in path.read_text()


class TestListScrapedCases:
    def test_lists_cases(self, tmp_path: Path):
        metadata = CaseMetadata(
            case_number="SC26001",
            case_title="A vs B",
            filing_date=date(2026, 1, 1),
        )
        save_metadata(metadata, tmp_path)
        cases = list_scraped_cases(tmp_path)
        assert "SC26001" in cases

    def test_empty(self, tmp_path: Path):
        assert list_scraped_cases(tmp_path) == []

    def test_nonexistent_dir(self):
        assert list_scraped_cases(Path("/nonexistent")) == []


class TestCasePdfs:
    def test_lists_pdfs(self, tmp_path: Path):
        save_pdf("SC26001", "doc_000.pdf", b"pdf1", tmp_path)
        save_pdf("SC26001", "doc_001.pdf", b"pdf2", tmp_path)
        pdfs = case_pdfs("SC26001", tmp_path)
        assert len(pdfs) == 2

    def test_no_pdfs(self, tmp_path: Path):
        assert case_pdfs("NONEXISTENT", tmp_path) == []
