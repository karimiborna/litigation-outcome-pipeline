"""Integration test for PyMuPDF text extraction — creates real PDFs in memory."""

import fitz  # PyMuPDF

from scraper.extractor import extract_with_pymupdf


class TestPymupdfExtraction:
    def _make_pdf(self, tmp_path, pages_text: list[str]):
        """Helper: create a real PDF with the given page texts."""
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        for text in pages_text:
            page = doc.new_page()
            page.insert_text((72, 72), text)
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path

    def test_extracts_single_page(self, tmp_path):
        pdf = self._make_pdf(tmp_path, ["Hello from page one."])
        result = extract_with_pymupdf(pdf)
        assert "Hello from page one." in result

    def test_extracts_multiple_pages(self, tmp_path):
        pdf = self._make_pdf(tmp_path, ["Page one.", "Page two."])
        result = extract_with_pymupdf(pdf)
        assert "Page one." in result
        assert "Page two." in result

    def test_blank_pdf_returns_empty(self, tmp_path):
        pdf = self._make_pdf(tmp_path, [""])
        result = extract_with_pymupdf(pdf)
        assert result.strip() == ""
