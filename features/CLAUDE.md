# Features Module

LLM-based feature extraction and preprocessing pipeline that converts raw case text into structured ML-ready features.

## Schema version: v2 (existence-based)

The v2 schema retired all subjective 1–5 ratings (`evidence_strength`, `argument_clarity_*`, `timeline_clarity`) in favor of ~40 existence-based booleans the LLM can observe in the claim text. The design rule is that the LLM answers **"does X appear in the text"** rather than making quality judgments.

Every boolean follows this contract (reinforced in the system prompt):
- `true` — thing is explicitly mentioned/stated/attached/named in the text
- `false` — text addresses the topic but the thing is absent
- `null` — topic is not addressed, or genuinely ambiguous

## Unilateral perspective

Features are written as `user_*` / `opposing_party_*`, not symmetric `plaintiff_*` / `defendant_*`. The `user_side` (plaintiff | defendant) is threaded through `build_extraction_prompt` so the LLM knows which party to treat as "user" when populating those fields. The derived `user_is_plaintiff` boolean is stored alongside LLM output so one model can serve both perspectives.

At training time (Colab notebook), `user_side` is auto-detected per case: `"defendant"` if a `DEFENDANT_S_CLAIM` txt exists for the case, else `"plaintiff"`.

## Feature sections

Grouped roughly by what the LLM is looking for:
- Classification + amount: `claim_category`, `monetary_amount_claimed`
- Representation: `user_has_attorney`, `opposing_party_has_attorney`, `opposing_party_filed_response_documents`
- Counter-filings / contract presence: `counterclaim_present`, `contract_present`
- Evidence existence (10): `has_*` booleans for photos, receipts, written comms, witness statements, signed contracts, repair estimates, police reports, medical records, expert assessments, invoices
- Argument content (8): `argument_cites_*` / `argument_has_*` / `argument_names_*` / `argument_quantifies_*` / `argument_identifies_*`
- Procedural (4): `sent_written_demand_letter`, `sent_certified_mail`, `gave_opportunity_to_cure`, `attempted_mediation`
- Contract detail (4, null if no contract): `contract_is_written`, `contract_is_signed_by_both_parties`, `contract_specifies_deadline_or_term`, `contract_specifies_payment_amount`
- Damages breakdown (5): `damages_include_*` booleans + `damages_are_ongoing` + `damages_have_third_party_valuation`
- Jurisdictional (4): `claim_amount_stated_in_dollars`, `claim_amount_is_within_small_claims_limit`, `user_seeks_interest`, `user_seeks_court_costs`
- Counts + derived: `plaintiff_count`, `defendant_count`, `witness_count`, `text_length`, `document_count`, `user_is_plaintiff`

Authoritative list: `features/schema.py::FeatureVector`.

## Leakage firewall

`FeatureExtractor` must **never** see outcome documents. The notebook builder and any future local aggregator filter txts using `features.labels.LABEL_DOC_KEYWORDS` / `_is_label_doc` — same list the label pipeline uses as its inclusion filter. Feature and label pipelines read disjoint subsets of a case's txts.

## Responsibilities

- Parse the LLM's JSON response into `LLMFeatures` (see `extraction.py::_parse_llm_response`)
- Merge `LLMFeatures` + derived counts into `FeatureVector` (`_build_feature_vector`)
- Cache per-case `FeatureVector` JSONs keyed by `sha256(case_number + full_text + feature_version)[:16]` — idempotent across reruns and teammates
- Serialize to `to_model_input()` dict of numeric features, with nulls mapping to a `-1.0` sentinel (bool/int/float all use the same sentinel; categoricals like `claim_category` are excluded and must be encoded downstream)

## Key Considerations

- LLM is used strictly for feature extraction, NOT for prediction.
- `feature_version` is pinned on `FeaturesConfig` (default `"v2"`) and included in the cache key — bumping the version invalidates cached outputs automatically.
- `extract_batch` logs + continues on per-case failures, so it's safe to pass hundreds of cases.
- Cost/latency management: `gpt-4o-mini` via OpenAI-compatible endpoint, `temperature=0.0`, max_tokens tunable on `FeaturesConfig`. At ~500 cases with the v2 prompt (~3000 input / ~600 output tokens), total cost is ~$0.30.
- Case text is truncated to `max_text_length=12000` chars by `build_extraction_prompt`; the truncation is marked inline so the LLM knows.

## Related modules

- **`labels.py`** — separate LLM pipeline for **supervision**: structured outcome labels (win/loss, amounts awarded, attorney presence, dates) from judgment / order / dismissal text. Shared `LABEL_DOC_KEYWORDS` serves as both the label-side inclusion filter and the feature-side exclusion filter.
- **`features/prompts.py`** — `build_extraction_prompt(..., user_side=...)` — all prompt engineering lives here.
- **`features/schema.py`** — `LLMFeatures` (raw LLM output), `FeatureVector` (training-ready), `FeatureVector.to_model_input()` (numeric dict for sklearn).
