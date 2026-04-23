"""Pydantic schemas for court case data at every pipeline stage."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PartyType(str, Enum):
    PLAINTIFF = "PLAINTIFF"
    DEFENDANT = "DEFENDANT"
    APPELLANT = "APPELLANT"
    RESPONDENT = "RESPONDENT"
    OTHER = "OTHER"


class Party(BaseModel):
    name: str
    party_type: PartyType
    is_pro_per: bool = False
    attorney_name: str | None = None


class Attorney(BaseModel):
    name: str
    bar_number: str | None = None
    firm: str | None = None
    address: str | None = None
    phone: str | None = None
    parties_represented: list[str] = Field(default_factory=list)


class Proceeding(BaseModel):
    date: date
    text: str
    has_document: bool = False
    fee: str | None = None


class Document(BaseModel):
    date: date
    description: str
    filename: str | None = None
    pdf_path: Path | None = None
    extraction_status: str = "pending"


class CaseMetadata(BaseModel):
    """Raw case metadata as scraped from the court site."""

    case_number: str
    case_title: str
    cause_of_action: str | None = None
    filing_date: date
    scraped_at: datetime = Field(default_factory=datetime.now)

    parties: list[Party] = Field(default_factory=list)
    attorneys: list[Attorney] = Field(default_factory=list)
    proceedings: list[Proceeding] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)

    @field_validator("case_number")
    @classmethod
    def normalize_case_number(cls, v: str) -> str:
        return v.strip().replace("-", "")


class ExtractedText(BaseModel):
    """Text extracted from a single PDF document via OCR."""

    case_number: str
    document_filename: str
    pages: list[str]
    extraction_method: str = "nvidia_nemo"
    extracted_at: datetime = Field(default_factory=datetime.now)


class ProcessedCase(BaseModel):
    """A fully processed case ready for feature extraction.

    Combines metadata with all extracted document text.
    """

    case_number: str
    case_title: str
    cause_of_action: str | None = None
    filing_date: date

    parties: list[Party] = Field(default_factory=list)
    attorneys: list[Attorney] = Field(default_factory=list)
    proceedings: list[Proceeding] = Field(default_factory=list)

    full_text: str = ""
    document_texts: list[ExtractedText] = Field(default_factory=list)

    claim_amount: float | None = None
    has_contract: bool | None = None
    plaintiff_count: int = 0
    defendant_count: int = 0
    has_attorney_plaintiff: bool | None = None
    has_attorney_defendant: bool | None = None
    user_side: Literal["plaintiff", "defendant"] = "plaintiff"

    @field_validator("case_number")
    @classmethod
    def normalize_case_number(cls, v: str) -> str:
        return v.strip().replace("-", "")
