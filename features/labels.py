"""LLM-based label extraction from court outcome documents.

Reads judgment, order, and dismissal text files for a case and sends them
to GPT-4o-mini to extract structured labels (outcome, amount awarded, etc.).
These labels are what the ML models learn to predict.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from features.config import FeaturesConfig

logger = logging.getLogger(__name__)

LABEL_DOC_KEYWORDS = {
    "JUDGMENT",
    "ORDER",
    "DISMISSAL",
    "Notice_of_Entry_of_Judgment",
    "STIPULATION",
    "COURT_JUDGMENT",
}

LABEL_EXTRACTION_SYSTEM = (
    "You are a legal analyst AI. Your job is to read SF small claims court "
    "outcome documents (judgments, orders, dismissals) and extract the case "
    "outcome as structured JSON. Respond with ONLY valid JSON — no explanation, "
    "no markdown, no extra text.\n\n"
    "Be conservative: if information is unclear or missing, use null."
)

LABEL_EXTRACTION_USER = """\
Analyze the following outcome documents for case {case_number} and extract the case result.

--- OUTCOME DOCUMENTS ---
{outcome_text}
--- END OUTCOME DOCUMENTS ---

Extract the following as JSON:

{{
    "outcome": <one of: "plaintiff_win", "defendant_win", "partial_win", "dismissed", "settled", null>,
    "dismissal_type": <one of: "with_prejudice", "without_prejudice", null>,
    "amount_awarded_principal": <float dollar amount awarded as principal, 0.0 if none, null if unclear>,
    "amount_awarded_costs": <float dollar amount for costs, 0.0 if none, null if unclear>,
    "amount_awarded_interest": <float dollar amount for interest, 0.0 if none, null if unclear>,
    "defendant_appeared": <true/false/null — whether the defendant showed up to court>,
    "has_attorney_plaintiff": <true/false/null — whether an attorney appeared for the plaintiff>,
    "has_attorney_defendant": <true/false/null — whether an attorney appeared for the defendant>,
    "judgment_date": <string date in YYYY-MM-DD format, null if not found>,
    "outcome_summary": <one sentence describing the outcome>
}}

Respond with ONLY the JSON object."""


class CaseLabels(BaseModel):
    """Structured labels extracted from outcome documents."""

    case_number: str
    outcome: str | None = Field(
        None,
        description="plaintiff_win, defendant_win, partial_win, dismissed, or settled",
    )
    dismissal_type: str | None = None
    amount_awarded_principal: float | None = None
    amount_awarded_costs: float | None = None
    amount_awarded_interest: float | None = None
    defendant_appeared: bool | None = None
    has_attorney_plaintiff: bool | None = None
    has_attorney_defendant: bool | None = None
    judgment_date: str | None = None
    outcome_summary: str | None = None

    @property
    def total_awarded(self) -> float | None:
        parts = [
            self.amount_awarded_principal,
            self.amount_awarded_costs,
            self.amount_awarded_interest,
        ]
        if all(p is None for p in parts):
            return None
        return sum(p or 0.0 for p in parts)


def _is_label_doc(filename: str) -> bool:
    """Check if a filename corresponds to an outcome document."""
    name_upper = filename.upper()
    return any(kw.upper() in name_upper for kw in LABEL_DOC_KEYWORDS)


def gather_outcome_text(case_number: str, txt_dir: Path) -> str:
    """Collect all outcome document text for a case into a single string."""
    parts: list[str] = []
    for txt_path in sorted(txt_dir.glob(f"{case_number}_*.txt")):
        if _is_label_doc(txt_path.name):
            doc_type = txt_path.stem.replace(f"{case_number}_", "")
            text = txt_path.read_text(encoding="utf-8").strip()
            if text:
                parts.append(f"[Document: {doc_type}]\n{text}")
    return "\n\n---\n\n".join(parts)


class LabelExtractor:
    """Extracts case outcome labels from court documents using an LLM."""

    def __init__(self, config: FeaturesConfig | None = None):
        self._config = config or FeaturesConfig()
        base_url = self._config.llm_base_url or "https://api.openai.com/v1"
        self._client = OpenAI(
            api_key=self._config.llm_api_key,
            base_url=base_url,
        )
        self._cache_dir = Path(self._config.cache_dir) / "labels"
        if self._config.enable_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_labels(self, case_number: str, txt_dir: Path) -> CaseLabels | None:
        """Extract labels for a single case from its outcome documents."""
        outcome_text = gather_outcome_text(case_number, txt_dir)
        if not outcome_text:
            logger.warning("No outcome documents found for %s", case_number)
            return None

        cached = self._load_cache(case_number, outcome_text)
        if cached is not None:
            logger.debug("Cache hit for labels: %s", case_number)
            return cached

        labels = self._call_llm(case_number, outcome_text)
        if labels:
            self._save_cache(case_number, outcome_text, labels)
        return labels

    def extract_batch(self, case_numbers: list[str], txt_dir: Path) -> dict[str, CaseLabels]:
        """Extract labels for multiple cases. Returns {case_number: CaseLabels}."""
        results: dict[str, CaseLabels] = {}
        for case_number in case_numbers:
            try:
                labels = self.extract_labels(case_number, txt_dir)
                if labels:
                    results[case_number] = labels
            except Exception:
                logger.exception("Label extraction failed for %s", case_number)
        return results

    @staticmethod
    def _smart_truncate(text: str, max_chars: int = 8000) -> str:
        """Keep the start and end of the text, since rulings tend to appear last."""
        if len(text) <= max_chars:
            return text
        head = max_chars // 3
        tail = max_chars - head
        return text[:head] + "\n\n[... middle truncated ...]\n\n" + text[-tail:]

    def _call_llm(self, case_number: str, outcome_text: str) -> CaseLabels | None:
        """Send outcome documents to the LLM and parse the response."""
        user_content = LABEL_EXTRACTION_USER.format(
            case_number=case_number,
            outcome_text=self._smart_truncate(outcome_text),
        )

        try:
            response = self._client.chat.completions.create(
                model=self._config.llm_model,
                messages=[
                    {"role": "system", "content": LABEL_EXTRACTION_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
        except Exception:
            logger.exception("LLM call failed for %s", case_number)
            return None

        content = response.choices[0].message.content or ""
        return self._parse_response(case_number, content)

    def _parse_response(self, case_number: str, content: str) -> CaseLabels | None:
        content = content.strip()
        try:
            raw = json.loads(content)
            raw["case_number"] = case_number
            return CaseLabels.model_validate(raw)
        except Exception:
            logger.exception("Failed to parse LLM response for %s: %s", case_number, content[:200])
            return None

    def _cache_key(self, case_number: str, outcome_text: str) -> str:
        content = f"{case_number}:{outcome_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_cache(self, case_number: str, outcome_text: str) -> CaseLabels | None:
        if not self._config.enable_cache:
            return None
        key = self._cache_key(case_number, outcome_text)
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return CaseLabels.model_validate(raw)
        except Exception:
            return None

    def _save_cache(self, case_number: str, outcome_text: str, labels: CaseLabels) -> None:
        if not self._config.enable_cache:
            return
        key = self._cache_key(case_number, outcome_text)
        path = self._cache_dir / f"{key}.json"
        path.write_text(labels.model_dump_json(indent=2), encoding="utf-8")
