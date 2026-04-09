"""PDF text extraction using PyMuPDF with NVIDIA vision API fallback.

Strategy:
1. Try direct text extraction with PyMuPDF (fast, free, works for text-based PDFs)
2. If no text found (scanned/image PDF), render pages to images and send
   to NVIDIA's vision model for OCR

This is Ernie's approach — much better than nv-ingest-client because it
avoids API calls entirely for PDFs that already have selectable text.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import fitz  # pymupdf
from openai import OpenAI

from scraper.config import NVIDIA_API_BASE, NVIDIA_VISION_MODEL, PDF_IMAGE_DPI

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = (
    "This is a page from a San Francisco small claims court document. "
    "Extract all text exactly as it appears. Include party names, dates, "
    "claim amounts, rulings, and any other information on the page. "
    "Output plain text only."
)

CLAIM_EXTRACTION_PROMPT = (
    "This is a page from a California SC-100 small claims complaint form. "
    "SKIP all boilerplate instructions, bilingual notices, and form guidance. "
    "Extract ONLY the filled-in case-specific information:\n"
    "- Plaintiff name(s), address, phone\n"
    "- Defendant name(s), address, phone\n"
    "- Trial date and department\n"
    "- Dollar amount claimed and basis for the amount\n"
    "- Description of what happened (the plaintiff's narrative)\n"
    "- Any evidence, witnesses, or documents referenced\n"
    "- Whether plaintiff tried to resolve before suing\n\n"
    "If a page is only boilerplate instructions with no filled-in data, "
    "respond with: [NO CASE DATA ON THIS PAGE]\n"
    "Output plain text only."
)

PYMUPDF_MIN_TEXT_LENGTH = 100


def extract_with_pymupdf(pdf_path: Path) -> str:
    """Extract selectable text from a PDF. Returns empty string if none found."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages).strip()


def _page_to_base64(page: fitz.Page, dpi: int = PDF_IMAGE_DPI) -> str:
    """Render a PDF page to JPEG and return as a base64 string."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bytes = pix.tobytes("jpeg")
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_with_nvidia(pdf_path: Path, api_key: str) -> str:
    """Send each page as an image to NVIDIA vision model for OCR."""
    client = OpenAI(base_url=NVIDIA_API_BASE, api_key=api_key)
    doc = fitz.open(str(pdf_path))
    pages_text: list[str] = []

    for page_num, page in enumerate(doc, start=1):
        logger.info("  NVIDIA vision API: page %d/%d ...", page_num, len(doc))
        img_b64 = _page_to_base64(page)

        response = client.chat.completions.create(
            model=NVIDIA_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=2048,
            temperature=0,
        )
        pages_text.append(response.choices[0].message.content or "")

    doc.close()
    return "\n\n--- PAGE BREAK ---\n\n".join(pages_text)


def extract_text(pdf_path: Path, nvidia_api_key: str = "") -> str:
    """Extract text from a PDF.

    Uses PyMuPDF if text is selectable (free, no API quota consumed).
    Falls back to NVIDIA vision API for scanned/image PDFs.
    """
    text = extract_with_pymupdf(pdf_path)

    if len(text) > PYMUPDF_MIN_TEXT_LENGTH:
        logger.info(
            "Extracted %d chars via PyMuPDF (no API quota used): %s",
            len(text),
            pdf_path.name,
        )
        return text

    if not nvidia_api_key:
        logger.warning(
            "No selectable text and no NVIDIA_API_KEY set — cannot extract: %s",
            pdf_path.name,
        )
        return ""

    logger.info("No selectable text — using NVIDIA vision API: %s", pdf_path.name)
    return extract_with_nvidia(pdf_path, nvidia_api_key)
