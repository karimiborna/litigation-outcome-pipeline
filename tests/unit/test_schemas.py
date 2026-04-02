"""Tests for data schemas and validation models."""

from datetime import date

import pytest
from pydantic import ValidationError

from data.schemas.case import (
    Attorney,
    CaseMetadata,
    Document,
    ExtractedText,
    Party,
    PartyType,
    Proceeding,
    ProcessedCase,
)


class TestParty:
    def test_create_plaintiff(self):
        p = Party(name="John Doe", party_type=PartyType.PLAINTIFF)
        assert p.name == "John Doe"
        assert p.party_type == PartyType.PLAINTIFF
        assert p.is_pro_per is False

    def test_pro_per_party(self):
        p = Party(name="Jane Smith", party_type=PartyType.DEFENDANT, is_pro_per=True)
        assert p.is_pro_per is True
        assert p.attorney_name is None


class TestCaseMetadata:
    def test_minimal(self):
        m = CaseMetadata(
            case_number="CIV26001234",
            case_title="DOE vs SMITH",
            filing_date=date(2026, 1, 15),
        )
        assert m.case_number == "CIV26001234"
        assert m.parties == []

    def test_normalizes_case_number(self):
        m = CaseMetadata(
            case_number="CIV-26-001234",
            case_title="DOE vs SMITH",
            filing_date=date(2026, 1, 15),
        )
        assert m.case_number == "CIV26001234"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            CaseMetadata(case_number="X")  # type: ignore[call-arg]

    def test_full_metadata(self):
        m = CaseMetadata(
            case_number="SC26005678",
            case_title="PLAINTIFF vs DEFENDANT",
            cause_of_action="SMALL CLAIMS",
            filing_date=date(2026, 3, 1),
            parties=[
                Party(name="Alice", party_type=PartyType.PLAINTIFF, is_pro_per=True),
                Party(name="Bob", party_type=PartyType.DEFENDANT, attorney_name="Law Firm LLP"),
            ],
            attorneys=[Attorney(name="Law Firm LLP", bar_number="12345")],
            proceedings=[
                Proceeding(date=date(2026, 3, 1), text="Case filed"),
            ],
            documents=[
                Document(date=date(2026, 3, 1), description="Complaint"),
            ],
        )
        assert len(m.parties) == 2
        assert m.parties[0].is_pro_per is True
        assert m.attorneys[0].bar_number == "12345"

    def test_round_trip_json(self):
        m = CaseMetadata(
            case_number="TEST001",
            case_title="A vs B",
            filing_date=date(2026, 2, 1),
        )
        json_str = m.model_dump_json()
        restored = CaseMetadata.model_validate_json(json_str)
        assert restored.case_number == m.case_number
        assert restored.filing_date == m.filing_date


class TestExtractedText:
    def test_create(self):
        et = ExtractedText(
            case_number="SC26001",
            document_filename="doc_000.pdf",
            pages=["Page one text", "Page two text"],
        )
        assert len(et.pages) == 2
        assert et.extraction_method == "nvidia_nemo"


class TestProcessedCase:
    def test_create_minimal(self):
        pc = ProcessedCase(
            case_number="SC-26-001",
            case_title="A vs B",
            filing_date=date(2026, 1, 1),
        )
        assert pc.case_number == "SC26001"
        assert pc.full_text == ""
        assert pc.plaintiff_count == 0
