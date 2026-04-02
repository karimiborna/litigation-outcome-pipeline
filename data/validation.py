"""Validation utilities for case data at each pipeline stage."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from data.schemas.case import CaseMetadata, ExtractedText, ProcessedCase

logger = logging.getLogger(__name__)


class ValidationResult:
    def __init__(self, valid: bool, errors: list[str] | None = None):
        self.valid = valid
        self.errors = errors or []

    def __bool__(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        if self.valid:
            return "ValidationResult(valid=True)"
        return f"ValidationResult(valid=False, errors={self.errors})"


def validate_case_metadata(data: dict) -> ValidationResult:
    try:
        CaseMetadata.model_validate(data)
        return ValidationResult(valid=True)
    except ValidationError as e:
        errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        return ValidationResult(valid=False, errors=errors)


def validate_extracted_text(data: dict) -> ValidationResult:
    try:
        ExtractedText.model_validate(data)
        return ValidationResult(valid=True)
    except ValidationError as e:
        errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        return ValidationResult(valid=False, errors=errors)


def validate_processed_case(data: dict) -> ValidationResult:
    try:
        ProcessedCase.model_validate(data)
        return ValidationResult(valid=True)
    except ValidationError as e:
        errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        return ValidationResult(valid=False, errors=errors)


def load_and_validate_metadata(path: Path) -> CaseMetadata | None:
    """Load a metadata JSON file and validate it. Returns None on failure."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return CaseMetadata.model_validate(raw)
    except (json.JSONDecodeError, ValidationError, OSError) as e:
        logger.error("Failed to load metadata from %s: %s", path, e)
        return None


def load_and_validate_extracted(path: Path) -> ExtractedText | None:
    """Load an extracted text JSON file and validate it. Returns None on failure."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ExtractedText.model_validate(raw)
    except (json.JSONDecodeError, ValidationError, OSError) as e:
        logger.error("Failed to load extracted text from %s: %s", path, e)
        return None
