# Features Module

LLM-based feature extraction — converts raw case text into structured ML signals.

## File Map

| File | Purpose |
|---|---|
| `extraction.py` | FeatureExtractor class — extract(), extract_batch(), LLM calls, caching |
| `prompts.py` | LLM prompt templates — FEATURE_EXTRACTION_SYSTEM, build_extraction_prompt() |
| `schema.py` | LLMFeatures, FeatureVector, to_model_input() |
| `config.py` | FeaturesConfig (pydantic-settings) — LLM provider, model, caching |

## What Gets Extracted

The prompt (`prompts.py`) asks the LLM for two tiers of features:

**Core (v1) fields** — always present, used in `FeatureVector.to_model_input()`:
`evidence_strength`, `contract_present`, `argument_clarity_plaintiff/defendant`, `claim_category`, `monetary_amount_claimed`, `prior_attempts_to_resolve`, `witness_count`, `documentary_evidence`, `timeline_clarity`, `legal_representation_plaintiff/defendant`, `counterclaim_present`, `default_judgment_likely`

**Granular (v2) fields** — used by the real model (`models/dataset.py` v2 feature set):
- Evidence type: `has_photos_or_physical_evidence`, `has_receipts_or_financial_records`, `has_written_communications`, `has_witness_statements`, `has_signed_contract_attached`, `has_repair_or_replacement_estimate`, `has_police_report`, `has_medical_records`, `has_expert_assessment`, `has_invoices_or_billing_records`
- Argument quality: `argument_cites_specific_dates`, `argument_cites_specific_dollar_amounts`, `argument_cites_contract_or_document`, `argument_has_chronological_timeline`, `argument_names_specific_witnesses`, `argument_quantifies_each_damage_component`, `argument_cites_statute_or_legal_basis`, `argument_identifies_specific_location`
- Procedural: `sent_written_demand_letter`, `sent_certified_mail`, `gave_opportunity_to_cure`, `attempted_mediation`
- Contract: `contract_is_written`, `contract_is_signed_by_both_parties`, `contract_specifies_deadline_or_term`, `contract_specifies_payment_amount`
- Damages: `damages_include_out_of_pocket_costs`, `damages_include_lost_wages`, `damages_include_property_value_loss`, `damages_are_ongoing`, `damages_have_third_party_valuation`
- Claim: `claim_amount_stated_in_dollars`, `claim_amount_is_within_small_claims_limit`, `user_seeks_interest`, `user_seeks_court_costs`
- Role: `user_is_plaintiff`, `user_has_attorney`, `opposing_party_has_attorney`, `opposing_party_filed_response_documents`

All fields are optional (`null` if unclear) — LLM is instructed to be conservative.

## Key rules

- LLM is used strictly for feature extraction, NOT for prediction
- Prompts should produce consistent, parseable structured output (JSON)
- Feature extraction is idempotent — same input always produces same features (cache is SHA-256 keyed)
- Cost and latency of LLM calls are managed via the file-based cache in `features_cache/`
- Feature definitions are versioned (`feature_version` field on `FeatureVector`)

## Related modules

- **`labels.py`** — separate LLM pipeline for **supervision**: structured labels (outcome, amounts, attorney flags) from judgment / order text, not the same as training-feature extraction in `extraction.py`.
