"""
PDF text extraction using NVIDIA's vision API.

Strategy:
1. Try direct text extraction with pymupdf (fast, free, works for text-based PDFs)
2. If no text found (scanned/image PDF), send pages to NVIDIA vision model

Set NVIDIA_API_KEY env var before use.
"""

import base64
import os
from pathlib import Path

import fitz  # pymupdf
from config import NVIDIA_API_BASE, NVIDIA_VISION_MODEL, PDF_IMAGE_DPI
from openai import OpenAI

EXTRACTION_PROMPT = (
    "This is a page from a San Francisco small claims court document. "
    "Extract all text exactly as it appears. Include party names, dates, "
    "claim amounts, rulings, and any other information on the page. "
    "Output plain text only."
)


def _extract_with_pymupdf(pdf_path: Path) -> str:
    """Extract selectable text from a PDF. Returns empty string if none found."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    text = "\n\n".join(pages).strip()
    return text


def _page_to_base64(page: fitz.Page, dpi: int = PDF_IMAGE_DPI) -> str:
    """Render a PDF page to a JPEG and return as base64 string."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bytes = pix.tobytes("jpeg")
    return base64.b64encode(img_bytes).decode("utf-8")


def _extract_with_nvidia(pdf_path: Path) -> str:
    """Send each PDF page as an image to NVIDIA vision model and collect text."""
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "ERROR: Set NVIDIA_API_KEY env var. Get a free key at https://build.nvidia.com"
        )

    client = OpenAI(base_url=NVIDIA_API_BASE, api_key=api_key)
    doc = fitz.open(str(pdf_path))
    pages_text = []

    for page_num, page in enumerate(doc, start=1):
        print(f"    NVIDIA API: page {page_num}/{len(doc)} ...")
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


def extract_text(pdf_path: Path) -> str:
    """
    Extract text from a PDF.
    Uses pymupdf if text is selectable; falls back to NVIDIA vision API.
    """
    text = _extract_with_pymupdf(pdf_path)

    # Heuristic: if we got meaningful text (>100 chars), trust pymupdf
    if len(text) > 100:
        print(f"    Extracted {len(text)} chars via pymupdf (no API quota used).")
        return text

    print("    No selectable text found — using NVIDIA vision API.")
    return _extract_with_nvidia(pdf_path)
