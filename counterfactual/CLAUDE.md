# Counterfactual Analysis Module

Simulates changes in key features to show how they would affect predicted outcomes.

## Responsibilities

- Accept a case's feature vector and a set of feature perturbations
- Run the trained model(s) on the modified feature vectors
- Compare original vs. counterfactual predictions
- Present results as "if X were different, the outcome would change by Y"
- Support both classification (win probability shift) and regression (monetary change) counterfactuals

## Key Considerations

- Perturbations must respect feature constraints (e.g., claim amount can't be negative)
- Only perturb features that are meaningful and **actionable by the user**. The v2 perturbable set focuses on things a litigant can plausibly change about their own case (gather evidence, hire an attorney, rewrite their argument, take pre-filing steps). Features describing the other party's choices, the structural facts of the case, or the nature of the harm are excluded.
- Depends on trained models from `models/` — loads from MLflow registry
- Output should be human-readable and useful for non-technical users
- Feature interaction effects may be important — changing one feature may affect interpretation of others
- Bool perturbations flip 0↔1; numeric perturbations step by `+1` (witness_count, capped at 5) or follow declared min/max in `FEATURE_CONSTRAINTS`

## v2 perturbable feature set (28 total: 27 booleans + 1 numeric)

The curated v2 set, defined in `analyzer.py::PERTURBABLE_FEATURES`. All `feat_*` prefixed:

**Representation (1)**
- `feat_user_has_attorney`

**Evidence existence (10)**
- `feat_has_photos_or_physical_evidence`, `feat_has_receipts_or_financial_records`, `feat_has_written_communications`, `feat_has_witness_statements`, `feat_has_signed_contract_attached`, `feat_has_repair_or_replacement_estimate`, `feat_has_police_report`, `feat_has_medical_records`, `feat_has_expert_assessment`, `feat_has_invoices_or_billing_records`

**Argument content (8)**
- `feat_argument_cites_specific_dates`, `feat_argument_cites_specific_dollar_amounts`, `feat_argument_cites_contract_or_document`, `feat_argument_has_chronological_timeline`, `feat_argument_names_specific_witnesses`, `feat_argument_quantifies_each_damage_component`, `feat_argument_cites_statute_or_legal_basis`, `feat_argument_identifies_specific_location`

**Procedural / pre-filing conduct (4)**
- `feat_sent_written_demand_letter`, `feat_sent_certified_mail`, `feat_gave_opportunity_to_cure`, `feat_attempted_mediation`

**Claim framing (2)**
- `feat_user_seeks_interest`, `feat_user_seeks_court_costs`

**Contract presence (1)**
- `feat_contract_present` — kept perturbable as a "did you forget there was a written/oral agreement?" nudge

**Damages valuation (1)**
- `feat_damages_have_third_party_valuation` — actionable: get an appraisal/estimate

**Numeric (1)**
- `feat_witness_count` — `+1` step, capped at 5, with early-stop (see below)

### Explicitly excluded (and why)

- `feat_user_is_plaintiff` — role is fixed by who sued whom
- `feat_text_length`, `feat_document_count`, `feat_plaintiff_count`, `feat_defendant_count` — derived metadata / case structure
- `feat_claim_category_*` (8 one-hots) — nature of the dispute is fixed
- `feat_opposing_party_has_attorney`, `feat_opposing_party_filed_response_documents`, `feat_counterclaim_present` — other side's choices, not user-actionable
- `feat_claim_amount_is_within_small_claims_limit` — jurisdictional rule, capped by statute
- `feat_monetary_amount_claimed`, `feat_claim_amount_stated_in_dollars` — mechanical/trivial; the regressor is anchored to monetary_amount_claimed so perturbing it produces uninformative deltas
- `feat_damages_include_out_of_pocket_costs`, `feat_damages_include_lost_wages`, `feat_damages_include_property_value_loss`, `feat_damages_are_ongoing` — describe the nature of the harm, not knobs the user can turn

## Implementation plan

### 1. Real v2 `FEATURE_CONSTRAINTS`

Replace the v1-only constraint dict with v2 entries keyed by `feat_*` column names. Booleans get `{"min": 0, "max": 1, "type": "bool"}`; `feat_witness_count` gets `{"min": 0, "max": 5, "type": "int", "step": 1}`. `_clamp` then actually does something on v2.

### 2. Curated perturbation generator

`_auto_perturbations_v2` rewritten to iterate `PERTURBABLE_FEATURES` only. For each feature in the case's `base_input`:
- Boolean: flip whatever the current value is (`1.0 - current`). Treat the `-1.0` null sentinel as "feature not addressed in text" and skip.
- `feat_witness_count`: see witness-count handling below.

### 3. Batched predicts

Build a single `pd.DataFrame` with one row per perturbation (each row = base row with one column changed). Call `predict_proba` and `predict` once each on the full batch. Map results back to feature names by row order. Collapses 2N sklearn calls into 2.

### 4. Witness-count stepping with early-stop

`feat_witness_count` is the only multi-step perturbation. Add up to four rows to the batch (`current+1`, `current+2`, ..., capped at 5). After predictions, walk the steps in order and drop any step where `|win_prob_delta - prior_step_delta| < 0.005`. Keeps the first step always (so the user sees at least one witness-count signal), prunes diminishing-returns steps.

### 5. Helpful/harmful classification + top-5 selection

Each result is tagged:
- `helpful` if the flip moves toward a user-reachable better state: `false → true` for action features (`has_*`, `argument_*`, `sent_*`, `gave_*`, `attempted_*`, `user_has_attorney`, `user_seeks_*`, `contract_present`, `damages_have_third_party_valuation`); `+1` for `witness_count`.
- `harmful` otherwise (`true → false` on those same features). Still computed and returned, but not advice.

For surfacing top-5: sort all results by `|win_prob_delta|`. Take helpful ones from the top first; admit harmful ones only if they appear in the overall top-N by magnitude (so high-impact "this is load-bearing" findings still get shown). Default top-N for the magnitude threshold: 5.

### 6. Output shape

`CounterfactualResult.to_dict()` adds `"direction": "helpful" | "harmful"`. The analyzer's public method returns the full sorted list (callers can paginate); a separate helper picks the top-5 with the helpful-first / magnitude-exception rule.

### Latency budget

GBM predict on a single row is ~2–4ms; batching 27+ rows costs roughly the same as 1–2 single-row calls. Whole perturbation loop should land in ~10–15ms after batching, vs ~80ms today. The dominant `/counterfactual` cost remains the upstream LLM feature extraction.

## LLM integration (shipped)

The perturbation analysis is surfaced two ways: the existing similarity-advice LLM call grounds its prose in the top-5 (instructed to reference each by name and reconcile with retrieved cases), and the same top-5 ships as a structured `top_recommendations: list[CounterfactualItem]` on `LexRatioAnalysisResponse`. The judge prompt receives the same block so it can fact-check quoted deltas.

### Components

The build below was followed bottom-up. All steps are now live; the section is kept as a record of what fits where.

1. `counterfactual/analyzer.py` — add `FEATURE_DISPLAY_NAMES` (28-entry dict) + `format_for_llm(results)` helper. Pure data-in/string-out, fully unit-testable. Output sample:

   ```
   Top counterfactual changes (sorted by predicted impact on win probability):
   1. Hire an attorney — you don't have one; if you hired one: +12.3pp win prob, +$340 expected award. [actionable]
   2. Attach photos or physical evidence — not addressed in your claim; if you had them: +8.7pp win prob. [actionable]
   3. Send a written demand letter before filing — you didn't; if you had: +6.1pp. [actionable]
   4. Add witnesses — you have 2; with 3 witnesses: +3.1pp. [actionable]
   5. Attach receipts or financial records — you have these; if removed: −9.5pp win prob. [load-bearing — keep this]
   ```

   Tag rule: `[actionable]` for `direction == "helpful"`, `[load-bearing — keep this]` for `direction == "harmful"`. State phrase derives from `original_value` (`null/NaN` → "not addressed in your claim", `0.0` → "you don't have this / you stated this is absent", `1.0` → "you have this"). For `feat_witness_count`, render the actual integer counts.

2. `api/prompts.py` — extend `build_similarity_advice_prompt` and `build_rag_advice_judge_prompt` with an optional `perturbation_summary: str` argument. Inject as a new section in the user content. Update `SIMILARITY_ADVICE_SYSTEM` to add four bullets: (a) reference top-5 by name, (b) explain each, (c) when perturbations and historical patterns diverge, surface the divergence ("our model estimates +X% from doing Y, though in case Z the same change didn't help because…"), (d) treat load-bearing perturbations as warnings, not actions. Keep JSON-only output rule.

3. `api/schemas.py` — add `top_recommendations: list[CounterfactualItem] = []` to `LexRatioAnalysisResponse`.

4. `api/app.py` — in `_build_rag_context` (or thread `feature_vector` through to it), call `app_state.counterfactual_analyzer.analyze` → `select_top_recommendations(top_n=5)` → `format_for_llm`. Pass the formatted string into both `_build_similarity_advice` and `_evaluate_rag_advice`. Populate `top_recommendations` in `_build_frontend_analysis_response` with the same 5 results converted to `CounterfactualItem`s.

5. Tests — formatter unit tests (state phrase per original-value, witness rendering, tag selection); prompt-builder tests confirming the perturbation block appears in the messages. Skip live API tests (env lacks `faiss`/`fitz`).

### Friendly-name map (approved)

```
feat_user_has_attorney                              → Hire an attorney
feat_has_photos_or_physical_evidence                → Attach photos or physical evidence
feat_has_receipts_or_financial_records              → Attach receipts or financial records
feat_has_written_communications                     → Attach written communications (emails, texts, letters)
feat_has_witness_statements                         → Obtain witness statements
feat_has_signed_contract_attached                   → Attach a signed contract
feat_has_repair_or_replacement_estimate             → Obtain a repair or replacement estimate
feat_has_police_report                              → Obtain a police report
feat_has_medical_records                            → Attach medical records
feat_has_expert_assessment                          → Obtain an expert assessment
feat_has_invoices_or_billing_records                → Attach invoices or billing records
feat_argument_cites_specific_dates                  → Cite specific dates in your argument
feat_argument_cites_specific_dollar_amounts         → Cite specific dollar amounts
feat_argument_cites_contract_or_document            → Cite the contract or document
feat_argument_has_chronological_timeline            → Present events as a chronological timeline
feat_argument_names_specific_witnesses              → Name specific witnesses in your argument
feat_argument_quantifies_each_damage_component      → Quantify each component of damages
feat_argument_cites_statute_or_legal_basis          → Cite the statute or legal basis
feat_argument_identifies_specific_location          → Identify the specific location of events
feat_sent_written_demand_letter                     → Send a written demand letter before filing
feat_sent_certified_mail                            → Use certified mail for your demand letter
feat_gave_opportunity_to_cure                       → Give the other party a chance to fix the issue before filing
feat_attempted_mediation                            → Attempt mediation before filing
feat_user_seeks_interest                            → Request interest in your claim
feat_user_seeks_court_costs                         → Request court costs in your claim
feat_contract_present                               → Identify a written or oral agreement governing the dispute
feat_damages_have_third_party_valuation             → Obtain a third-party valuation of damages
feat_witness_count                                  → Add witnesses
```

### Decisions locked in

- One LLM call (the existing similarity-advice call), not a separate perturbation-explanation call.
- Judge prompt also receives the same top-5 string so it can fact-check the advice's references.
- Response carries both the LLM `advice` (prose, references perturbations) and a structured `top_recommendations` list (raw deltas + direction) so a frontend panel is possible later.
- Latency is ~10ms for the analyzer call — noise next to the LLM calls already in the path. No caching.
- `app_state.counterfactual_analyzer` is already loaded by `dependencies.AppState.load_models`; no init wiring needed.
- The advice prompt may need iteration after we see real outputs; that's a later concern.
