"""Tests for data validation utilities."""

from data.validation import (
    validate_case_metadata,
    validate_extracted_text,
    validate_processed_case,
)


class TestValidateCaseMetadata:
    def test_valid(self):
        result = validate_case_metadata(
            {
                "case_number": "SC26001",
                "case_title": "A vs B",
                "filing_date": "2026-01-15",
            }
        )
        assert result.valid
        assert result.errors == []

    def test_missing_field(self):
        result = validate_case_metadata({"case_number": "SC26001"})
        assert not result.valid
        assert len(result.errors) > 0

    def test_bad_date(self):
        result = validate_case_metadata(
            {
                "case_number": "SC26001",
                "case_title": "A vs B",
                "filing_date": "not-a-date",
            }
        )
        assert not result.valid


class TestValidateExtractedText:
    def test_valid(self):
        result = validate_extracted_text(
            {
                "case_number": "SC26001",
                "document_filename": "doc_000.pdf",
                "pages": ["Some text"],
            }
        )
        assert result.valid

    def test_missing_pages(self):
        result = validate_extracted_text(
            {
                "case_number": "SC26001",
                "document_filename": "doc.pdf",
            }
        )
        assert not result.valid


class TestValidateProcessedCase:
    def test_valid_minimal(self):
        result = validate_processed_case(
            {
                "case_number": "SC26001",
                "case_title": "A vs B",
                "filing_date": "2026-01-01",
            }
        )
        assert result.valid

    def test_invalid_type(self):
        result = validate_processed_case(
            {
                "case_number": 12345,
                "case_title": "A vs B",
                "filing_date": "2026-01-01",
                "plaintiff_count": "not_a_number",
            }
        )
        assert not result.valid
