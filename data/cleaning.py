"""Text cleaning utilities for OCR-extracted court document text."""

from __future__ import annotations

import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to their canonical form."""
    return unicodedata.normalize("NFKC", text)


def collapse_whitespace(text: str) -> str:
    """Replace runs of whitespace (excluding newlines) with a single space."""
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_ocr_artifacts(text: str) -> str:
    """Remove common OCR noise patterns from scanned court documents."""
    # Isolated single characters surrounded by spaces (OCR debris)
    text = re.sub(r"(?<= )[^\w\s](?= )", "", text)
    # Repeated punctuation runs (e.g., "....." or "-----")
    text = re.sub(r"([.\-_=])\1{4,}", "", text)
    # Page number artifacts like "Page 1 of 3"
    text = re.sub(r"[Pp]age\s+\d+\s+of\s+\d+", "", text)
    return text


def remove_header_footer_noise(text: str) -> str:
    """Strip common header/footer stamps from scanned court documents."""
    patterns = [
        r"(?i)electronically\s+filed.*",
        r"(?i)superior\s+court\s+of\s+california.*",
        r"(?i)county\s+of\s+san\s+francisco.*",
        r"(?i)clerk\s+of\s+the\s+court.*",
        r"(?i)filed\s+\d{1,2}/\d{1,2}/\d{2,4}.*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text


def clean_extracted_text(text: str, aggressive: bool = False) -> str:
    """Full cleaning pipeline for OCR-extracted text.

    Args:
        text: Raw OCR text from a court document.
        aggressive: If True, also strip header/footer stamps. Use with
            caution since this can remove legitimate content.
    """
    text = normalize_unicode(text)
    text = remove_ocr_artifacts(text)
    if aggressive:
        text = remove_header_footer_noise(text)
    text = collapse_whitespace(text)
    return text


def merge_page_texts(pages: list[str], separator: str = "\n\n") -> str:
    """Join per-page text into a single document string."""
    cleaned = [clean_extracted_text(page) for page in pages]
    return separator.join(page for page in cleaned if page)
