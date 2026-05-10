"""Counterfactual analysis — simulates feature changes and shows predicted outcome shifts."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from features.schema import FeatureVector
from models.dataset import MODEL_FEATURE_COLUMNS, feature_vector_to_model_frame

logger = logging.getLogger(__name__)

# Curated v2 perturbable booleans — only features the user can plausibly act on.
# See counterfactual/CLAUDE.md for rationale. Excluded groups: case-structure
# metadata, opposing-party choices, claim-category, jurisdictional facts,
# damages composition, and the monetary amount itself.
_BOOLEAN_FEATURES: tuple[str, ...] = (
    # Representation (1)
    "feat_user_has_attorney",
    # Evidence existence (10)
    "feat_has_photos_or_physical_evidence",
    "feat_has_receipts_or_financial_records",
    "feat_has_written_communications",
    "feat_has_witness_statements",
    "feat_has_signed_contract_attached",
    "feat_has_repair_or_replacement_estimate",
    "feat_has_police_report",
    "feat_has_medical_records",
    "feat_has_expert_assessment",
    "feat_has_invoices_or_billing_records",
    # Argument content (8)
    "feat_argument_cites_specific_dates",
    "feat_argument_cites_specific_dollar_amounts",
    "feat_argument_cites_contract_or_document",
    "feat_argument_has_chronological_timeline",
    "feat_argument_names_specific_witnesses",
    "feat_argument_quantifies_each_damage_component",
    "feat_argument_cites_statute_or_legal_basis",
    "feat_argument_identifies_specific_location",
    # Procedural / pre-filing (4)
    "feat_sent_written_demand_letter",
    "feat_sent_certified_mail",
    "feat_gave_opportunity_to_cure",
    "feat_attempted_mediation",
    # Claim framing (2)
    "feat_user_seeks_interest",
    "feat_user_seeks_court_costs",
    # Contract presence (1) — kept as a "did you forget there's an agreement?" nudge
    "feat_contract_present",
    # Damages valuation (1) — actionable: get an appraisal
    "feat_damages_have_third_party_valuation",
)

WITNESS_COUNT_FEATURE = "feat_witness_count"
WITNESS_COUNT_MAX = 5
WITNESS_COUNT_STEP_EPSILON = 0.005

NULL_SENTINEL = -1.0

FEATURE_CONSTRAINTS: dict[str, dict[str, Any]] = {
    **{f: {"min": 0.0, "max": 1.0, "type": "bool"} for f in _BOOLEAN_FEATURES},
    WITNESS_COUNT_FEATURE: {
        "min": 0.0,
        "max": float(WITNESS_COUNT_MAX),
        "type": "int",
    },
}

PERTURBABLE_FEATURES: list[str] = list(FEATURE_CONSTRAINTS.keys())

# Advice-friendly labels for each perturbable feature. The phrasing assumes the
# helpful direction (false → true for booleans, +1 for witnesses) so the LLM
# can read them as imperative actions.
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "feat_user_has_attorney": "Hire an attorney",
    "feat_has_photos_or_physical_evidence": "Attach photos or physical evidence",
    "feat_has_receipts_or_financial_records": "Attach receipts or financial records",
    "feat_has_written_communications": "Attach written communications (emails, texts, letters)",
    "feat_has_witness_statements": "Obtain witness statements",
    "feat_has_signed_contract_attached": "Attach a signed contract",
    "feat_has_repair_or_replacement_estimate": "Obtain a repair or replacement estimate",
    "feat_has_police_report": "Obtain a police report",
    "feat_has_medical_records": "Attach medical records",
    "feat_has_expert_assessment": "Obtain an expert assessment",
    "feat_has_invoices_or_billing_records": "Attach invoices or billing records",
    "feat_argument_cites_specific_dates": "Cite specific dates in your argument",
    "feat_argument_cites_specific_dollar_amounts": "Cite specific dollar amounts",
    "feat_argument_cites_contract_or_document": "Cite the contract or document",
    "feat_argument_has_chronological_timeline": "Present events as a chronological timeline",
    "feat_argument_names_specific_witnesses": "Name specific witnesses in your argument",
    "feat_argument_quantifies_each_damage_component": "Quantify each component of damages",
    "feat_argument_cites_statute_or_legal_basis": "Cite the statute or legal basis",
    "feat_argument_identifies_specific_location": "Identify the specific location of events",
    "feat_sent_written_demand_letter": "Send a written demand letter before filing",
    "feat_sent_certified_mail": "Use certified mail for your demand letter",
    "feat_gave_opportunity_to_cure": "Give the other party a chance to fix the issue before filing",
    "feat_attempted_mediation": "Attempt mediation before filing",
    "feat_user_seeks_interest": "Request interest in your claim",
    "feat_user_seeks_court_costs": "Request court costs in your claim",
    "feat_contract_present": "Identify a written or oral agreement governing the dispute",
    "feat_damages_have_third_party_valuation": "Obtain a third-party valuation of damages",
    "feat_witness_count": "Add witnesses",
}


def _is_null(value: float) -> bool:
    """``True`` for both pandas NaN and the ``-1.0`` sentinel used by ``to_model_input``."""
    return value != value or value == NULL_SENTINEL


def _state_phrase(feature_name: str, original_value: float, new_value: float) -> str:
    """Render the current-state phrase used inside :func:`format_for_llm`."""
    if feature_name == WITNESS_COUNT_FEATURE:
        orig_n = 0 if _is_null(original_value) else int(original_value)
        new_n = int(new_value)
        return f"currently {orig_n} (would be {new_n})"
    if _is_null(original_value):
        return "not addressed in your claim"
    if original_value == 1.0:
        return "currently yes"
    return "currently no"


def _delta_phrase(win_prob_delta: float, monetary_delta: float) -> str:
    """Render the ``+/- X.Xpp win prob[, +/-$Y expected award]`` segment."""
    win_pp = win_prob_delta * 100
    sign_win = "+" if win_pp >= 0 else "-"
    out = f"{sign_win}{abs(win_pp):.1f}pp win prob"
    if abs(monetary_delta) >= 1:
        sign_money = "+" if monetary_delta >= 0 else "-"
        out += f", {sign_money}${abs(monetary_delta):,.0f} expected award"
    return out


def format_for_llm(results: list[CounterfactualResult]) -> str:
    """Render top counterfactual results as advice-ready prose for an LLM prompt.

    Each line carries the friendly feature name, the user's current state,
    the predicted deltas, and a tag (``[actionable]`` for helpful flips,
    ``[load-bearing — keep this]`` for harmful ones). The output is
    deterministic given the inputs and intended for direct concatenation
    into a chat-completion user message.
    """
    if not results:
        return "(no actionable perturbations identified for this case)"

    lines = [
        "Top counterfactual changes (sorted by predicted impact on win probability):"
    ]
    for idx, r in enumerate(results, start=1):
        name = FEATURE_DISPLAY_NAMES.get(r.feature_name, r.feature_name)
        state = _state_phrase(r.feature_name, r.original_value, r.new_value)
        deltas = _delta_phrase(r.win_prob_delta, r.monetary_delta)
        tag = "actionable" if r.direction == "helpful" else "load-bearing — keep this"
        lines.append(f"{idx}. {name} — {state}; {deltas}. [{tag}]")
    return "\n".join(lines)


class CounterfactualResult:
    """Result of a single counterfactual perturbation."""

    def __init__(
        self,
        feature_name: str,
        original_value: float,
        new_value: float,
        original_win_prob: float,
        new_win_prob: float,
        original_monetary: float,
        new_monetary: float,
        direction: str = "helpful",
    ):
        self.feature_name = feature_name
        self.original_value = original_value
        self.new_value = new_value
        self.original_win_prob = original_win_prob
        self.new_win_prob = new_win_prob
        self.original_monetary = original_monetary
        self.new_monetary = new_monetary
        self.direction = direction

    @property
    def win_prob_delta(self) -> float:
        return self.new_win_prob - self.original_win_prob

    @property
    def monetary_delta(self) -> float:
        return self.new_monetary - self.original_monetary

    def to_dict(self) -> dict:
        return {
            "feature": self.feature_name,
            "original_value": self.original_value,
            "new_value": self.new_value,
            "direction": self.direction,
            "win_probability": {
                "original": round(self.original_win_prob, 4),
                "new": round(self.new_win_prob, 4),
                "delta": round(self.win_prob_delta, 4),
            },
            "monetary_outcome": {
                "original": round(self.original_monetary, 2),
                "new": round(self.new_monetary, 2),
                "delta": round(self.monetary_delta, 2),
            },
            "description": self.describe(),
        }

    def describe(self) -> str:
        direction_word = "increase" if self.win_prob_delta > 0 else "decrease"
        pct = abs(self.win_prob_delta) * 100
        return (
            f"If {self.feature_name} changed from {self.original_value} to "
            f"{self.new_value}, win probability would {direction_word} by {pct:.1f}%"
        )

    # Back-compat: existing api/app.py mapping calls _describe().
    _describe = describe


def select_top_recommendations(
    results: list[CounterfactualResult],
    *,
    top_n: int = 5,
) -> list[CounterfactualResult]:
    """Pick the top recommendations to surface to the user.

    Default: top-N helpful perturbations (actionable advice). If any of the
    overall top-N by ``|win_prob_delta|`` is harmful, return the overall
    top-N as-is so high-magnitude "this is load-bearing for your case"
    findings are not hidden.
    """
    if not results:
        return []

    by_magnitude = sorted(results, key=lambda r: abs(r.win_prob_delta), reverse=True)
    overall_top = by_magnitude[:top_n]

    if any(r.direction == "harmful" for r in overall_top):
        return overall_top

    helpful = [r for r in by_magnitude if r.direction == "helpful"]
    return helpful[:top_n]


class CounterfactualAnalyzer:
    """Analyzes how feature changes would affect predicted outcomes."""

    def __init__(self, classifier: Any, regressor: Any):
        self._classifier = classifier
        self._regressor = regressor

    def analyze(
        self,
        feature_vector: FeatureVector,
        perturbations: dict[str, float] | None = None,
    ) -> list[CounterfactualResult]:
        """Run counterfactual analysis on a single case.

        With ``perturbations=None`` (default), generates the curated v2
        perturbation set (28 features). Otherwise applies the provided
        ``feature → new_value`` map verbatim, clamped to the declared
        constraints. Returned results are sorted by ``|win_prob_delta|``
        descending. Use :func:`select_top_recommendations` to pick what
        to surface.
        """
        base_df, expected_columns = self._build_base_df(feature_vector)
        base_input = base_df.iloc[0].to_dict()

        base_win_prob = float(self._classifier.predict_proba(base_df)[0, 1])
        base_monetary = float(self._regressor.predict(base_df)[0])

        candidates = self._build_candidates(base_input, perturbations)
        if not candidates:
            return []

        rows = []
        for cand in candidates:
            row = base_input.copy()
            row[cand["feature"]] = cand["new_value"]
            rows.append(row)
        batch_df = pd.DataFrame(rows)
        if expected_columns is not None:
            batch_df = batch_df.reindex(columns=expected_columns)

        new_win_probs = self._classifier.predict_proba(batch_df)[:, 1]
        new_monetary = self._regressor.predict(batch_df)

        results: list[CounterfactualResult] = []
        for idx, cand in enumerate(candidates):
            feat = cand["feature"]
            orig_val = float(base_input[feat])
            new_val = float(cand["new_value"])
            results.append(
                CounterfactualResult(
                    feature_name=feat,
                    original_value=orig_val,
                    new_value=new_val,
                    original_win_prob=base_win_prob,
                    new_win_prob=float(new_win_probs[idx]),
                    original_monetary=base_monetary,
                    new_monetary=float(new_monetary[idx]),
                    direction=self._classify_direction(feat, orig_val, new_val),
                )
            )

        results = self._prune_witness_count_steps(results)
        results.sort(key=lambda r: abs(r.win_prob_delta), reverse=True)
        return results

    def _build_base_df(
        self, feature_vector: FeatureVector
    ) -> tuple[pd.DataFrame, list[str] | None]:
        """Build the base feature row and the column order to align all batches against."""
        expected = getattr(self._classifier, "feature_names_in_", None)
        if expected is not None and set(expected).issubset(set(MODEL_FEATURE_COLUMNS)):
            df = feature_vector_to_model_frame(feature_vector)
            expected_cols = list(expected)
            return df.reindex(columns=expected_cols), expected_cols
        # Fallback for tests / classifiers without feature_names_in_.
        return pd.DataFrame([feature_vector.to_model_input()]), None

    def _build_candidates(
        self,
        base_input: dict[str, float],
        perturbations: dict[str, float] | None,
    ) -> list[dict[str, Any]]:
        """Return a list of ``{"feature", "new_value"}`` perturbation entries."""
        if perturbations is None:
            return self._auto_perturbations_v2(base_input)

        out: list[dict[str, Any]] = []
        for feat, new_val in perturbations.items():
            if feat not in base_input:
                logger.warning("Unknown feature: %s", feat)
                continue
            clamped = self._clamp(feat, float(new_val))
            if abs(clamped - float(base_input[feat])) < 1e-6:
                continue
            out.append({"feature": feat, "new_value": clamped})
        return out

    def _auto_perturbations_v2(
        self, base_input: dict[str, float]
    ) -> list[dict[str, Any]]:
        """Generate the curated v2 perturbation set."""
        candidates: list[dict[str, Any]] = []

        for feat in _BOOLEAN_FEATURES:
            if feat not in base_input:
                continue
            current = float(base_input[feat])
            if _is_null(current):
                # Topic not addressed in text — model the "what if you had this" lift.
                new_val = 1.0
            elif current == 0.0:
                new_val = 1.0
            elif current == 1.0:
                new_val = 0.0
            else:
                continue
            candidates.append({"feature": feat, "new_value": new_val})

        if WITNESS_COUNT_FEATURE in base_input:
            current = float(base_input[WITNESS_COUNT_FEATURE])
            if _is_null(current):
                current = 0.0
            current = max(0.0, current)
            for witness in range(int(current) + 1, WITNESS_COUNT_MAX + 1):
                candidates.append(
                    {"feature": WITNESS_COUNT_FEATURE, "new_value": float(witness)}
                )

        return candidates

    def _prune_witness_count_steps(
        self, results: list[CounterfactualResult]
    ) -> list[CounterfactualResult]:
        """Drop diminishing-returns witness_count steps after batched scoring.

        Walks the witness_count steps in ascending order. Always keeps the
        first step. Stops as soon as a step's ``win_prob_delta`` differs from
        the previous kept step by less than :data:`WITNESS_COUNT_STEP_EPSILON`.
        """
        witness_results = [r for r in results if r.feature_name == WITNESS_COUNT_FEATURE]
        if len(witness_results) <= 1:
            return results

        witness_results.sort(key=lambda r: r.new_value)
        kept: list[CounterfactualResult] = [witness_results[0]]
        prev_delta = witness_results[0].win_prob_delta
        for r in witness_results[1:]:
            if abs(r.win_prob_delta - prev_delta) < WITNESS_COUNT_STEP_EPSILON:
                break
            kept.append(r)
            prev_delta = r.win_prob_delta

        kept_ids = {id(r) for r in kept}
        return [
            r
            for r in results
            if r.feature_name != WITNESS_COUNT_FEATURE or id(r) in kept_ids
        ]

    def _classify_direction(
        self, feature_name: str, original_value: float, new_value: float
    ) -> str:
        """Tag a perturbation as ``helpful`` (toward better) or ``harmful``."""
        if _is_null(original_value):
            # Treat null/NaN as "absent" — moving to 1 (or witnesses up) is helpful.
            if feature_name == WITNESS_COUNT_FEATURE:
                return "helpful" if new_value > 0 else "harmful"
            return "helpful" if new_value == 1.0 else "harmful"
        if feature_name == WITNESS_COUNT_FEATURE:
            return "helpful" if new_value > original_value else "harmful"
        # All curated booleans treat 1.0 as the user-reachable better state.
        if new_value > original_value:
            return "helpful"
        return "harmful"

    def _clamp(self, feature_name: str, value: float) -> float:
        """Clamp a value to respect feature constraints (no-op for unknown features)."""
        if feature_name not in FEATURE_CONSTRAINTS:
            return value
        constraints = FEATURE_CONSTRAINTS[feature_name]
        min_val = constraints.get("min")
        max_val = constraints.get("max")
        if min_val is not None:
            value = max(value, min_val)
        if max_val is not None:
            value = min(value, max_val)
        return value
