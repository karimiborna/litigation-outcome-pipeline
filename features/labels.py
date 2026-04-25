"""LLM-based label extraction from court outcome documents.

Reads judgment, order, and dismissal text files for a case and sends them
to GPT-4o-mini to extract structured labels (outcome, amount awarded by
direction, etc.). These labels are what the ML models learn to predict.
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
    "DISMISSAL",
    "Notice_of_Entry_of_Judgment",
    "STIPULATION",
    "COURT_JUDGMENT",
    "SATISFACTION_OF_JUDGMENT",
    "ORDER",
}

LABEL_DOC_EXCLUDE_KEYWORDS = {
    "CONTINUE_ORDER",
    "CONTINUATION_OF_HEARING",
    "OFF_CALENDAR",
    "POSTPONE",
    "RESET",
    "SET_ASIDE",
    "CORRECT_OR_CANCEL",
    "NOTICE_OF_TIME_AND_PLACE",
    "NOTICE_OF_APPEAL",
    "SUBPOENA",
    "TRANSFER",
    "TRANSMITTAL",
    "WRIT_OF_EXECUTION",
    "MOTION",
    "REQUEST_TO",
}

SPECIFIC_OUTCOME_KEYWORDS = {
    "JUDGMENT",
    "DISMISSAL",
    "STIPULATION",
    "SATISFACTION_OF_JUDGMENT",
    "NOTICE_OF_ENTRY_OF_JUDGMENT",
}

LABEL_EXTRACTION_SYSTEM = (
    "You are a legal analyst AI. You read SF small claims court outcome documents "
    "(judgments, orders of dismissal, dismissals, notices of entry of judgment, "
    "satisfactions of judgment, stipulations) and extract the case outcome as "
    "structured JSON. Respond with ONLY valid JSON — no explanation, no markdown.\n\n"
    "Decision rules:\n"
    "- A judgment that fills 'Defendant shall pay Plaintiff' with a non-zero amount "
    "is a plaintiff outcome. Put those numbers under awarded_to_plaintiff_*.\n"
    "- A judgment that ALSO fills 'Plaintiff shall pay defendant' with a non-zero "
    "amount means money flows both ways — fill awarded_to_defendant_* too and set "
    "outcome='mutual_award'.\n"
    "- A judgment with $0 to plaintiff and a non-zero amount only to defendant is "
    "outcome='defendant_win'.\n"
    "- A request for dismissal or order of dismissal means outcome='dismissed'. "
    "Set dismissal_type from the 'with prejudice' / 'without prejudice' boxes.\n"
    "- A stipulation without an accompanying judgment indicates the parties settled "
    "— outcome='settled'.\n"
    "- Use outcome='partial_win' only when plaintiff was awarded strictly less than "
    "the amount they claimed AND defendant was awarded nothing.\n"
    "- Boilerplate templates (forms with every option listed and no checkbox visible "
    "in the OCR text) are NOT outcomes — ignore them when a real ruling is present.\n"
    "- Use outcome=null ONLY when no judgment, dismissal, satisfaction, or stipulation "
    "is present, or the text is genuinely unreadable. Whenever any such document is "
    "present, choose the definitive label that fits."
)

LABEL_EXTRACTION_USER = """\
Analyze the following outcome documents for case {case_number} and extract the case result.

--- OUTCOME DOCUMENTS ---
{outcome_text}
--- END OUTCOME DOCUMENTS ---

Return JSON with these fields:

{{
    "outcome": "plaintiff_win" | "defendant_win" | "partial_win" | "mutual_award" | "dismissed" | "settled" | null,
    "dismissal_type": "with_prejudice" | "without_prejudice" | null,
    "awarded_to_plaintiff_principal": <float dollars to plaintiff; 0.0 if explicitly $0; null if not addressed>,
    "awarded_to_plaintiff_costs": <float dollars to plaintiff; 0.0 if explicitly $0; null if not addressed>,
    "awarded_to_plaintiff_interest": <float dollars to plaintiff; 0.0 if explicitly $0; null if not addressed>,
    "awarded_to_defendant_principal": <float dollars to defendant; 0.0 if explicitly $0; null if not addressed>,
    "awarded_to_defendant_costs": <float dollars to defendant; 0.0 if explicitly $0; null if not addressed>,
    "awarded_to_defendant_interest": <float dollars to defendant; 0.0 if explicitly $0; null if not addressed>,
    "defendant_appeared": <true | false | null>,
    "has_attorney_plaintiff": <true | false | null>,
    "has_attorney_defendant": <true | false | null>,
    "judgment_date": <"YYYY-MM-DD" or null>,
    "outcome_summary": <one short sentence describing the outcome>
}}

Respond with ONLY the JSON object."""


class CaseLabels(BaseModel):
    """Structured labels extracted from outcome documents."""

    case_number: str
    outcome: str | None = Field(
        None,
        description=(
            "plaintiff_win | defendant_win | partial_win | mutual_award | "
            "dismissed | settled"
        ),
    )
    dismissal_type: str | None = None

    awarded_to_plaintiff_principal: float | None = None
    awarded_to_plaintiff_costs: float | None = None
    awarded_to_plaintiff_interest: float | None = None
    awarded_to_defendant_principal: float | None = None
    awarded_to_defendant_costs: float | None = None
    awarded_to_defendant_interest: float | None = None

    defendant_appeared: bool | None = None
    has_attorney_plaintiff: bool | None = None
    has_attorney_defendant: bool | None = None
    judgment_date: str | None = None
    outcome_summary: str | None = None

    @staticmethod
    def _sum_or_none(parts: list[float | None]) -> float | None:
        if all(p is None for p in parts):
            return None
        return sum(p or 0.0 for p in parts)

    @property
    def total_to_plaintiff(self) -> float | None:
        return self._sum_or_none(
            [
                self.awarded_to_plaintiff_principal,
                self.awarded_to_plaintiff_costs,
                self.awarded_to_plaintiff_interest,
            ]
        )

    @property
    def total_to_defendant(self) -> float | None:
        return self._sum_or_none(
            [
                self.awarded_to_defendant_principal,
                self.awarded_to_defendant_costs,
                self.awarded_to_defendant_interest,
            ]
        )

    @property
    def net_to_plaintiff(self) -> float | None:
        p = self.total_to_plaintiff
        d = self.total_to_defendant
        if p is None and d is None:
            return None
        return (p or 0.0) - (d or 0.0)


def _is_label_doc(filename: str) -> bool:
    name_upper = filename.upper()
    if any(kw in name_upper for kw in LABEL_DOC_EXCLUDE_KEYWORDS):
        return False
    return any(kw.upper() in name_upper for kw in LABEL_DOC_KEYWORDS)


def _has_specific_outcome_doc(filenames_upper: list[str]) -> bool:
    return any(
        any(kw in name for kw in SPECIFIC_OUTCOME_KEYWORDS) for name in filenames_upper
    )


def gather_outcome_text(case_number: str, txt_dir: Path) -> str:
    """Collect all outcome document text for a case into a single string.

    Skips procedural orders and drops the bare ``ORDER.txt`` template form when
    a more specific outcome document (judgment, dismissal, stipulation,
    satisfaction, notice of entry) is available for the same case.
    """
    candidates = [p for p in sorted(txt_dir.glob(f"{case_number}_*.txt")) if _is_label_doc(p.name)]
    if not candidates:
        return ""

    upper_names = [p.name.upper() for p in candidates]
    has_specific = _has_specific_outcome_doc(upper_names)
    bare_order_name = f"{case_number}_ORDER.txt"

    parts: list[str] = []
    for txt_path in candidates:
        if has_specific and txt_path.name == bare_order_name:
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        doc_type = txt_path.stem.replace(f"{case_number}_", "")
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
        self._cache_dir = Path(self._config.cache_dir)
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
        if labels is None:
            labels = CaseLabels(case_number=case_number)
        labels = self._fallback_outcome(case_number, txt_dir, labels, outcome_text)
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

    @staticmethod
    def _fallback_outcome(
        case_number: str,
        txt_dir: Path,
        labels: CaseLabels,
        outcome_text: str,
    ) -> CaseLabels:
        """Derive outcome from filename evidence when the LLM returned null.

        Catches the long tail where messy OCR makes the model defer to null even
        though the doc-type evidence is unambiguous.
        """
        if labels.outcome is not None:
            return labels

        files_upper = [f.name.upper() for f in txt_dir.glob(f"{case_number}_*.txt")]
        if not files_upper:
            return labels

        def _present(positive: str, *negatives: str) -> bool:
            return any(positive in n and not any(neg in n for neg in negatives) for n in files_upper)

        has_judgment = _present("JUDGMENT", "MOTION", "REQUEST_TO") or any(
            "NOTICE_OF_ENTRY_OF_JUDGMENT" in n for n in files_upper
        )
        has_satisfaction = any("SATISFACTION_OF_JUDGMENT" in n for n in files_upper)
        has_dismissal = _present("DISMISSAL", "MOTION_TO_SET_ASIDE")
        has_stipulation = any("STIPULATION" in n for n in files_upper)

        p_total = labels.total_to_plaintiff or 0.0
        d_total = labels.total_to_defendant or 0.0

        if has_judgment or has_satisfaction:
            if p_total > 0 and d_total > 0:
                labels.outcome = "mutual_award"
            elif p_total > 0:
                labels.outcome = "plaintiff_win"
            elif d_total > 0:
                labels.outcome = "defendant_win"
            else:
                labels.outcome = "plaintiff_win"
        elif has_dismissal:
            labels.outcome = "dismissed"
            if labels.dismissal_type is None:
                text_lower = outcome_text.lower()
                if "without prejudice" in text_lower:
                    labels.dismissal_type = "without_prejudice"
                elif "with prejudice" in text_lower:
                    labels.dismissal_type = "with_prejudice"
        elif has_stipulation:
            labels.outcome = "settled"

        return labels

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
