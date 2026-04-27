"""LLM-based feature extraction pipeline.

Calls an LLM to extract structured signals from case text and merges
them with metadata to produce a unified feature vector.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from data.schemas.case import ProcessedCase
from features.config import FeaturesConfig
from features.prompts import build_extraction_prompt
from features.schema import FeatureVector, LLMFeatures

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts ML-ready features from processed case data using an LLM."""

    def __init__(self, config: FeaturesConfig | None = None):
        self._config = config or FeaturesConfig()
        self._cache_dir = Path(self._config.cache_dir)
        if self._config.enable_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def extract(self, case: ProcessedCase) -> FeatureVector:
        """Extract features for a single case. Uses cache if available."""
        cache_key = self._cache_key(case)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.debug("Cache hit for %s", case.case_number)
            return cached

        llm_features = await self._call_llm(case)
        vector = self._build_feature_vector(case, llm_features)

        self._save_cache(cache_key, vector)
        return vector

    async def extract_batch(self, cases: list[ProcessedCase]) -> list[FeatureVector]:
        """Extract features for multiple cases."""
        results: list[FeatureVector] = []
        for case in cases:
            try:
                vector = await self.extract(case)
                results.append(vector)
            except Exception:
                logger.exception("Feature extraction failed for %s", case.case_number)
        return results

    async def _call_llm(self, case: ProcessedCase) -> LLMFeatures:
        """Call the LLM to extract structured features from case text."""
        messages = build_extraction_prompt(
            case_number=case.case_number,
            case_title=case.case_title,
            cause_of_action=case.cause_of_action,
            filing_date=case.filing_date.isoformat(),
            case_text=case.full_text,
        )

        body: dict[str, Any] = {
            "model": self._config.llm_model,
            "messages": messages,
            "temperature": self._config.llm_temperature,
            "max_tokens": self._config.llm_max_tokens,
        }

        base_url = self._config.llm_base_url or "https://api.openai.com/v1"
        headers = {
            "Authorization": f"Bearer {self._config.llm_api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self._config.llm_timeout) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        return self._parse_llm_response(content)

    def _parse_llm_response(self, content: str) -> LLMFeatures:
        """Parse the LLM's JSON response into structured features."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        raw = json.loads(content)
        return LLMFeatures.model_validate(raw)

    def _build_feature_vector(self, case: ProcessedCase, llm: LLMFeatures) -> FeatureVector:
        """Merge LLM-extracted features with case metadata."""
        return FeatureVector(
            case_number=case.case_number,
            feature_version=self._config.feature_version,
            evidence_strength=llm.evidence_strength,
            contract_present=llm.contract_present,
            argument_clarity_plaintiff=llm.argument_clarity_plaintiff,
            argument_clarity_defendant=llm.argument_clarity_defendant,
            claim_category=llm.claim_category,
            monetary_amount_claimed=llm.monetary_amount_claimed or case.claim_amount,
            prior_attempts_to_resolve=llm.prior_attempts_to_resolve,
            witness_count=llm.witness_count,
            documentary_evidence=llm.documentary_evidence,
            timeline_clarity=llm.timeline_clarity,
            legal_representation_plaintiff=llm.legal_representation_plaintiff,
            legal_representation_defendant=llm.legal_representation_defendant,
            counterclaim_present=llm.counterclaim_present,
            default_judgment_likely=llm.default_judgment_likely,
            user_is_plaintiff=llm.user_is_plaintiff,
            user_has_attorney=(
                llm.user_has_attorney
                if llm.user_has_attorney is not None
                else llm.legal_representation_plaintiff
            ),
            opposing_party_has_attorney=(
                llm.opposing_party_has_attorney
                if llm.opposing_party_has_attorney is not None
                else llm.legal_representation_defendant
            ),
            opposing_party_filed_response_documents=(
                llm.opposing_party_filed_response_documents
            ),
            has_photos_or_physical_evidence=llm.has_photos_or_physical_evidence,
            has_receipts_or_financial_records=llm.has_receipts_or_financial_records,
            has_written_communications=llm.has_written_communications,
            has_witness_statements=llm.has_witness_statements,
            has_signed_contract_attached=llm.has_signed_contract_attached,
            has_repair_or_replacement_estimate=llm.has_repair_or_replacement_estimate,
            has_police_report=llm.has_police_report,
            has_medical_records=llm.has_medical_records,
            has_expert_assessment=llm.has_expert_assessment,
            has_invoices_or_billing_records=llm.has_invoices_or_billing_records,
            argument_cites_specific_dates=llm.argument_cites_specific_dates,
            argument_cites_specific_dollar_amounts=llm.argument_cites_specific_dollar_amounts,
            argument_cites_contract_or_document=llm.argument_cites_contract_or_document,
            argument_has_chronological_timeline=llm.argument_has_chronological_timeline,
            argument_names_specific_witnesses=llm.argument_names_specific_witnesses,
            argument_quantifies_each_damage_component=(
                llm.argument_quantifies_each_damage_component
            ),
            argument_cites_statute_or_legal_basis=(
                llm.argument_cites_statute_or_legal_basis
            ),
            argument_identifies_specific_location=llm.argument_identifies_specific_location,
            sent_written_demand_letter=llm.sent_written_demand_letter,
            sent_certified_mail=llm.sent_certified_mail,
            gave_opportunity_to_cure=llm.gave_opportunity_to_cure,
            attempted_mediation=llm.attempted_mediation,
            contract_is_written=llm.contract_is_written,
            contract_is_signed_by_both_parties=llm.contract_is_signed_by_both_parties,
            contract_specifies_deadline_or_term=llm.contract_specifies_deadline_or_term,
            contract_specifies_payment_amount=llm.contract_specifies_payment_amount,
            damages_include_out_of_pocket_costs=llm.damages_include_out_of_pocket_costs,
            damages_include_lost_wages=llm.damages_include_lost_wages,
            damages_include_property_value_loss=llm.damages_include_property_value_loss,
            damages_are_ongoing=llm.damages_are_ongoing,
            damages_have_third_party_valuation=llm.damages_have_third_party_valuation,
            claim_amount_stated_in_dollars=llm.claim_amount_stated_in_dollars,
            claim_amount_is_within_small_claims_limit=(
                llm.claim_amount_is_within_small_claims_limit
            ),
            user_seeks_interest=llm.user_seeks_interest,
            user_seeks_court_costs=llm.user_seeks_court_costs,
            plaintiff_count=case.plaintiff_count,
            defendant_count=case.defendant_count,
            has_attorney_plaintiff=case.has_attorney_plaintiff,
            has_attorney_defendant=case.has_attorney_defendant,
            cause_of_action=case.cause_of_action,
            text_length=len(case.full_text),
            document_count=len(case.document_texts),
        )

    def _cache_key(self, case: ProcessedCase) -> str:
        """Deterministic cache key based on case content and feature version."""
        content = f"{case.case_number}:{case.full_text}:{self._config.feature_version}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_cache(self, key: str) -> FeatureVector | None:
        if not self._config.enable_cache:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return FeatureVector.model_validate(raw)
        except Exception:
            return None

    def _save_cache(self, key: str, vector: FeatureVector) -> None:
        if not self._config.enable_cache:
            return
        path = self._cache_dir / f"{key}.json"
        path.write_text(vector.model_dump_json(indent=2), encoding="utf-8")
