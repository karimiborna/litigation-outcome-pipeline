"""Tests for counterfactual analysis (v2 feature space)."""

from __future__ import annotations

import numpy as np
import pytest

from counterfactual.analyzer import (
    FEATURE_DISPLAY_NAMES,
    PERTURBABLE_FEATURES,
    WITNESS_COUNT_FEATURE,
    CounterfactualAnalyzer,
    CounterfactualResult,
    _is_null,
    format_for_llm,
    select_top_recommendations,
)
from features.schema import FeatureVector
from models.dataset import MODEL_FEATURE_COLUMNS


# A small set of features whose absence/presence we expect to move the fake
# classifier's predicted win probability.
_EVIDENCE_BOOLS = (
    "feat_has_photos_or_physical_evidence",
    "feat_has_receipts_or_financial_records",
    "feat_has_written_communications",
    "feat_has_witness_statements",
    "feat_has_signed_contract_attached",
    "feat_has_repair_or_replacement_estimate",
    "feat_has_invoices_or_billing_records",
)


class FakeClassifier:
    """Fake classifier whose win prob is monotone-increasing in evidence count + witnesses.

    Exposes ``feature_names_in_`` so the analyzer takes its v2 path and aligns
    columns the same way the production model does.
    """

    feature_names_in_ = np.array(MODEL_FEATURE_COLUMNS)

    def predict_proba(self, x):
        evidence_score = sum(
            (x[c].clip(lower=0) for c in _EVIDENCE_BOOLS),
            start=x[WITNESS_COUNT_FEATURE].clip(lower=0) * 0.5,
        )
        prob = (evidence_score / 8.0).clip(lower=0.05, upper=0.95)
        return np.column_stack([1 - prob, prob])


class FakeRegressor:
    """Fake regressor proportional to the claim amount."""

    feature_names_in_ = np.array(MODEL_FEATURE_COLUMNS)

    def predict(self, x):
        amount = x["feat_monetary_amount_claimed"].clip(lower=0)
        return (amount * 0.6).to_numpy()


@pytest.fixture
def analyzer():
    return CounterfactualAnalyzer(FakeClassifier(), FakeRegressor())


def _weak_case() -> FeatureVector:
    """A case missing most of the helpful evidence — gives lots of helpful flips."""
    return FeatureVector(
        case_number="WEAK-1",
        claim_category="unpaid_debt",
        monetary_amount_claimed=2500.0,
        user_is_plaintiff=True,
        user_has_attorney=False,
        opposing_party_has_attorney=False,
        opposing_party_filed_response_documents=False,
        counterclaim_present=False,
        contract_present=False,
        has_photos_or_physical_evidence=False,
        has_receipts_or_financial_records=False,
        has_written_communications=False,
        has_witness_statements=False,
        has_signed_contract_attached=False,
        has_repair_or_replacement_estimate=False,
        has_police_report=False,
        has_medical_records=False,
        has_expert_assessment=False,
        has_invoices_or_billing_records=False,
        argument_cites_specific_dates=False,
        argument_cites_specific_dollar_amounts=False,
        argument_cites_contract_or_document=False,
        argument_has_chronological_timeline=False,
        argument_names_specific_witnesses=False,
        argument_quantifies_each_damage_component=False,
        argument_cites_statute_or_legal_basis=False,
        argument_identifies_specific_location=False,
        sent_written_demand_letter=False,
        sent_certified_mail=False,
        gave_opportunity_to_cure=False,
        attempted_mediation=False,
        damages_include_out_of_pocket_costs=True,
        damages_include_lost_wages=False,
        damages_include_property_value_loss=False,
        damages_are_ongoing=False,
        damages_have_third_party_valuation=False,
        claim_amount_stated_in_dollars=True,
        claim_amount_is_within_small_claims_limit=True,
        user_seeks_interest=False,
        user_seeks_court_costs=False,
        plaintiff_count=1,
        defendant_count=1,
        witness_count=0,
        text_length=500,
        document_count=2,
    )


def _strong_case() -> FeatureVector:
    """A case that already has the helpful features — flips will be harmful."""
    return FeatureVector(
        case_number="STRONG-1",
        claim_category="unpaid_debt",
        monetary_amount_claimed=2500.0,
        user_is_plaintiff=True,
        user_has_attorney=True,
        opposing_party_has_attorney=False,
        opposing_party_filed_response_documents=True,
        counterclaim_present=False,
        contract_present=True,
        has_photos_or_physical_evidence=True,
        has_receipts_or_financial_records=True,
        has_written_communications=True,
        has_witness_statements=True,
        has_signed_contract_attached=True,
        has_repair_or_replacement_estimate=True,
        has_police_report=False,
        has_medical_records=False,
        has_expert_assessment=False,
        has_invoices_or_billing_records=True,
        argument_cites_specific_dates=True,
        argument_cites_specific_dollar_amounts=True,
        argument_cites_contract_or_document=True,
        argument_has_chronological_timeline=True,
        argument_names_specific_witnesses=True,
        argument_quantifies_each_damage_component=True,
        argument_cites_statute_or_legal_basis=False,
        argument_identifies_specific_location=True,
        sent_written_demand_letter=True,
        sent_certified_mail=True,
        gave_opportunity_to_cure=True,
        attempted_mediation=False,
        damages_include_out_of_pocket_costs=True,
        damages_include_lost_wages=False,
        damages_include_property_value_loss=False,
        damages_are_ongoing=False,
        damages_have_third_party_valuation=True,
        claim_amount_stated_in_dollars=True,
        claim_amount_is_within_small_claims_limit=True,
        user_seeks_interest=True,
        user_seeks_court_costs=True,
        plaintiff_count=1,
        defendant_count=1,
        witness_count=2,
        text_length=900,
        document_count=4,
    )


def test_perturbable_features_are_v2_columns():
    """Every perturbable feature must be a real v2 model column."""
    model_columns = set(MODEL_FEATURE_COLUMNS)
    for feat in PERTURBABLE_FEATURES:
        assert feat in model_columns, f"{feat} is not in MODEL_FEATURE_COLUMNS"


def test_perturbable_set_size():
    """Curated set should be 27 booleans + 1 numeric = 28."""
    assert len(PERTURBABLE_FEATURES) == 28


def test_excluded_features_never_perturbed(analyzer):
    """Non-actionable features must not appear in auto perturbations."""
    forbidden = {
        "feat_user_is_plaintiff",
        "feat_text_length",
        "feat_document_count",
        "feat_plaintiff_count",
        "feat_defendant_count",
        "feat_opposing_party_has_attorney",
        "feat_opposing_party_filed_response_documents",
        "feat_counterclaim_present",
        "feat_claim_amount_is_within_small_claims_limit",
        "feat_monetary_amount_claimed",
        "feat_claim_amount_stated_in_dollars",
        "feat_damages_include_out_of_pocket_costs",
        "feat_damages_include_lost_wages",
        "feat_damages_include_property_value_loss",
        "feat_damages_are_ongoing",
    }
    results = analyzer.analyze(_weak_case())
    perturbed = {r.feature_name for r in results}
    assert perturbed.isdisjoint(forbidden), (
        f"Non-actionable features were perturbed: {perturbed & forbidden}"
    )


def test_weak_case_yields_only_helpful_flips(analyzer):
    """A case missing every helpful feature only produces helpful perturbations."""
    results = analyzer.analyze(_weak_case())
    assert results, "expected non-empty perturbations for a weak case"
    assert all(r.direction == "helpful" for r in results)


def test_strong_case_yields_harmful_flips(analyzer):
    """A case that already has helpful features produces harmful direction flips."""
    results = analyzer.analyze(_strong_case())
    assert results
    assert any(r.direction == "harmful" for r in results)


def test_results_sorted_by_absolute_delta(analyzer):
    """Results must be sorted by |win_prob_delta| descending."""
    results = analyzer.analyze(_weak_case())
    deltas = [abs(r.win_prob_delta) for r in results]
    assert deltas == sorted(deltas, reverse=True)


def test_witness_count_caps_at_max(analyzer):
    """Witness-count perturbations never propose values above WITNESS_COUNT_MAX."""
    results = analyzer.analyze(_weak_case())
    witness_results = [r for r in results if r.feature_name == WITNESS_COUNT_FEATURE]
    assert witness_results
    for r in witness_results:
        assert r.new_value <= 5.0


def test_witness_count_skips_when_already_at_max(analyzer):
    """No witness perturbations when the case already has the max witnesses."""
    case = _weak_case()
    case.witness_count = 5
    results = analyzer.analyze(case)
    assert all(r.feature_name != WITNESS_COUNT_FEATURE for r in results)


def test_custom_perturbations_clamp_to_constraints(analyzer):
    """Custom values are clamped — e.g. 99 witnesses gets capped at 5."""
    results = analyzer.analyze(
        _weak_case(),
        perturbations={WITNESS_COUNT_FEATURE: 99.0},
    )
    witness_results = [r for r in results if r.feature_name == WITNESS_COUNT_FEATURE]
    assert witness_results
    assert witness_results[0].new_value == 5.0


def test_custom_perturbations_drop_unknown_features(analyzer, caplog):
    """Unknown feature names are logged and ignored, not raised."""
    with caplog.at_level("WARNING"):
        results = analyzer.analyze(
            _weak_case(),
            perturbations={"feat_does_not_exist": 1.0},
        )
    assert results == []
    assert any("Unknown feature" in record.message for record in caplog.records)


def test_null_sentinel_treated_as_missing_helpful(analyzer):
    """A feature the LLM reported as null becomes a helpful "what if you had this" flip."""
    case = _weak_case()
    case.has_photos_or_physical_evidence = None  # -> NULL_SENTINEL
    results = analyzer.analyze(case)
    photo_results = [
        r for r in results if r.feature_name == "feat_has_photos_or_physical_evidence"
    ]
    assert len(photo_results) == 1
    assert _is_null(photo_results[0].original_value)
    assert photo_results[0].new_value == 1.0
    assert photo_results[0].direction == "helpful"


def _make_result(feature: str, delta: float, direction: str) -> CounterfactualResult:
    return CounterfactualResult(
        feature_name=feature,
        original_value=0.0,
        new_value=1.0,
        original_win_prob=0.5,
        new_win_prob=0.5 + delta,
        original_monetary=1000.0,
        new_monetary=1000.0,
        direction=direction,
    )


def test_select_top_recommendations_helpful_only_by_default():
    """When the top-N is all helpful, only helpful ones are returned."""
    results = [
        _make_result("a", 0.20, "helpful"),
        _make_result("b", 0.15, "helpful"),
        _make_result("c", 0.10, "helpful"),
        _make_result("d", 0.05, "helpful"),
        _make_result("e", 0.04, "helpful"),
        _make_result("f", 0.03, "helpful"),
        _make_result("g", 0.001, "harmful"),
    ]
    top = select_top_recommendations(results, top_n=5)
    assert len(top) == 5
    assert all(r.direction == "helpful" for r in top)


def test_select_top_recommendations_admits_harmful_when_high_magnitude():
    """If a harmful flip ranks in the overall top-N, it's surfaced."""
    results = [
        _make_result("loadbearing", 0.30, "harmful"),
        _make_result("a", 0.20, "helpful"),
        _make_result("b", 0.15, "helpful"),
        _make_result("c", 0.10, "helpful"),
        _make_result("d", 0.05, "helpful"),
        _make_result("e", 0.04, "helpful"),
    ]
    top = select_top_recommendations(results, top_n=5)
    assert any(r.feature_name == "loadbearing" for r in top)
    assert top[0].feature_name == "loadbearing"


def test_select_top_recommendations_empty_input():
    assert select_top_recommendations([]) == []


def test_helpful_then_harmful_are_distinct_per_state():
    """The same feature flipped from a different starting state has the opposite direction."""
    weak_results = analyzer_factory().analyze(_weak_case())
    strong_results = analyzer_factory().analyze(_strong_case())
    weak_attorney = next(
        r for r in weak_results if r.feature_name == "feat_user_has_attorney"
    )
    strong_attorney = next(
        r for r in strong_results if r.feature_name == "feat_user_has_attorney"
    )
    assert weak_attorney.direction == "helpful"
    assert strong_attorney.direction == "harmful"


def analyzer_factory() -> CounterfactualAnalyzer:
    return CounterfactualAnalyzer(FakeClassifier(), FakeRegressor())


def test_display_names_cover_every_perturbable_feature():
    """Every perturbable column needs an advice-friendly label."""
    missing = [f for f in PERTURBABLE_FEATURES if f not in FEATURE_DISPLAY_NAMES]
    assert missing == [], f"display name missing for: {missing}"


def test_format_for_llm_empty_results():
    assert format_for_llm([]) == "(no actionable perturbations identified for this case)"


def test_format_for_llm_helpful_boolean_with_monetary():
    """A helpful flip with a non-trivial monetary delta renders both deltas + tag."""
    r = CounterfactualResult(
        feature_name="feat_user_has_attorney",
        original_value=0.0,
        new_value=1.0,
        original_win_prob=0.40,
        new_win_prob=0.523,
        original_monetary=1200.0,
        new_monetary=1540.0,
        direction="helpful",
    )
    out = format_for_llm([r])
    assert "1. Hire an attorney" in out
    assert "currently no" in out
    assert "+12.3pp win prob" in out
    assert "+$340 expected award" in out
    assert "[actionable]" in out


def test_format_for_llm_helpful_null_state():
    """A null/NaN original is rendered as 'not addressed in your claim'."""
    r = CounterfactualResult(
        feature_name="feat_has_photos_or_physical_evidence",
        original_value=float("nan"),
        new_value=1.0,
        original_win_prob=0.40,
        new_win_prob=0.487,
        original_monetary=1000.0,
        new_monetary=1000.0,
        direction="helpful",
    )
    out = format_for_llm([r])
    assert "not addressed in your claim" in out
    assert "+8.7pp win prob" in out
    # No monetary delta below the $1 threshold.
    assert "expected award" not in out


def test_format_for_llm_harmful_boolean():
    """A harmful flip renders the negative sign and the load-bearing tag."""
    r = CounterfactualResult(
        feature_name="feat_has_receipts_or_financial_records",
        original_value=1.0,
        new_value=0.0,
        original_win_prob=0.70,
        new_win_prob=0.605,
        original_monetary=1000.0,
        new_monetary=1000.0,
        direction="harmful",
    )
    out = format_for_llm([r])
    assert "currently yes" in out
    assert "-9.5pp win prob" in out
    assert "[load-bearing — keep this]" in out


def test_format_for_llm_witness_count():
    """Witness-count rendering shows the integer counts and the helpful tag."""
    r = CounterfactualResult(
        feature_name=WITNESS_COUNT_FEATURE,
        original_value=2.0,
        new_value=3.0,
        original_win_prob=0.50,
        new_win_prob=0.531,
        original_monetary=1500.0,
        new_monetary=1500.0,
        direction="helpful",
    )
    out = format_for_llm([r])
    assert "Add witnesses" in out
    assert "currently 2 (would be 3)" in out
    assert "+3.1pp win prob" in out
    assert "[actionable]" in out


def test_format_for_llm_numbers_and_orders_results():
    """Multiple results are numbered 1..N in the order they were passed."""
    rs = [
        _make_result("feat_user_has_attorney", 0.20, "helpful"),
        _make_result("feat_has_photos_or_physical_evidence", 0.10, "helpful"),
        _make_result("feat_sent_written_demand_letter", 0.05, "helpful"),
    ]
    out = format_for_llm(rs)
    lines = out.strip().split("\n")
    # First line is the header, then 3 numbered entries.
    assert len(lines) == 4
    assert lines[1].startswith("1. Hire an attorney")
    assert lines[2].startswith("2. Attach photos or physical evidence")
    assert lines[3].startswith("3. Send a written demand letter before filing")


def test_format_for_llm_unknown_feature_falls_back_to_raw_name():
    """Defensive: an unmapped feature name still renders without crashing."""
    r = CounterfactualResult(
        feature_name="feat_some_future_addition",
        original_value=0.0,
        new_value=1.0,
        original_win_prob=0.5,
        new_win_prob=0.55,
        original_monetary=0.0,
        new_monetary=0.0,
        direction="helpful",
    )
    out = format_for_llm([r])
    assert "feat_some_future_addition" in out
