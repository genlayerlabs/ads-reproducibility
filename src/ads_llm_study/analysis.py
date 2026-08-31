from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ads_llm_study.certificate import (
    certification_boundaries,
    clopper_pearson,
    familywise_operational_certificate,
    fixed_share_attack_probability,
    interval_certifies,
    mass_controlled_certificate,
    quota,
    targeted_attack_probability,
)
from ads_llm_study.dataset import declared_item_weights, load_and_validate_dataset
from ads_llm_study.io_utils import (
    git_commit,
    git_dirty,
    iter_jsonl,
    read_json,
    sha256_file,
    write_json,
)
from ads_llm_study.paths import DATA_DIR, REPO_ROOT, RESULTS_DIR, from_root
from ads_llm_study.validation import validate_run

plt.rcParams.update(
    {
        "figure.dpi": 170,
        "font.family": "serif",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.24,
    }
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _binary_entropy(probability: float) -> float:
    if probability <= 0 or probability >= 1:
        return 0.0
    return float(
        -(probability * math.log2(probability) + (1 - probability) * math.log2(1 - probability))
    )


def _float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _route_field(record: dict[str, Any], field: str) -> Any:
    routing = record.get("routing", {})
    if field in routing:
        return routing[field]
    decision = routing.get("decision_trace", {}) if isinstance(routing, dict) else {}
    return decision.get(field) if isinstance(decision, dict) else None


def _per_item_rows(
    manifest: dict[str, Any],
    items: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], bool]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["item_id"]].append(record)
    item_weights = declared_item_weights(items)
    interval_alpha = manifest["analysis"]["eta_e"] * manifest["analysis"]["xi_e"]
    threshold = manifest["analysis"]["threshold"]
    delta = manifest["analysis"]["delta"]
    panel_sizes = manifest["analysis"]["candidate_panel_sizes"]
    certifications: dict[tuple[str, int], bool] = {}
    output: list[dict[str, Any]] = []
    for item in items:
        item_records = grouped[item["id"]]
        trials = len(item_records)
        accepts = sum(record["global_vote"] for record in item_records)
        p_hat = accepts / trials
        lower, upper = clopper_pearson(accepts, trials, interval_alpha)
        certified_sizes: list[int] = []
        for panel_size in panel_sizes:
            certified = interval_certifies(lower, upper, panel_size, threshold, delta)
            certifications[(item["id"], panel_size)] = certified
            if certified:
                certified_sizes.append(panel_size)
        population_label = "accept" if p_hat >= threshold else "reject"
        confidence_values = [
            record["parsed_confidence"]
            for record in item_records
            if record["parsed_confidence"] is not None
        ]
        output.append(
            {
                "id": item["id"],
                "family": item["family"],
                "clarity_stratum": item["clarity_stratum"],
                "gold_label": item["gold_label"],
                "ambiguity_axis": item["ambiguity_axis"],
                "declared_item_weight": item_weights[item["id"]],
                "row_count": trials,
                "accept_count": accepts,
                "acceptance_rate": p_hat,
                "interval_lower": lower,
                "interval_upper": upper,
                "clarity_hat": abs(p_hat - threshold),
                "population_label_hat": population_label,
                "population_label_correct": (
                    population_label == item["gold_label"]
                    if item["gold_label"] in {"accept", "reject"}
                    else None
                ),
                "binary_entropy": _binary_entropy(p_hat),
                "mean_reported_confidence": (
                    float(np.mean(confidence_values)) if confidence_values else None
                ),
                "non_success_cells": sum(
                    record["status"] != "success" for record in item_records
                ),
                "certified_panel_sizes": certified_sizes,
            }
        )
    return output, certifications


def _coverage_rows(
    manifest: dict[str, Any],
    per_item: list[dict[str, Any]],
    certifications: dict[tuple[str, int], bool],
) -> list[dict[str, Any]]:
    analysis = manifest["analysis"]
    evaluator_row_counts = {int(row["row_count"]) for row in per_item}
    if len(evaluator_row_counts) != 1:
        raise ValueError("primary certificate requires the same evaluator-row count per item")
    certificate = mass_controlled_certificate(
        column_counts=[int(row["accept_count"]) for row in per_item],
        evaluator_sample_size=evaluator_row_counts.pop(),
        panel_sizes=analysis["candidate_panel_sizes"],
        threshold=analysis["threshold"],
        delta=analysis["delta"],
        eta_e=analysis["eta_e"],
        eta_g=analysis["eta_g"],
        xi_e=analysis["xi_e"],
    )
    bounds_by_size = {bound.panel_size: bound for bound in certificate.panel_bounds}
    raw_weights = np.asarray([row["declared_item_weight"] for row in per_item], dtype=float)
    normalized_weights = raw_weights / raw_weights.sum()
    rows: list[dict[str, Any]] = []
    for panel_size in analysis["candidate_panel_sizes"]:
        flags = np.asarray(
            [certifications[(row["id"], panel_size)] for row in per_item], dtype=float
        )
        uniform = float(flags.mean())
        weighted = float(np.dot(normalized_weights, flags))
        bound = bounds_by_size[panel_size]
        if int(flags.sum()) != bound.certified_columns:
            raise AssertionError("per-item and reference certificate implementations diverged")
        rows.append(
            {
                "panel_size": panel_size,
                "delta": analysis["delta"],
                "uniform_certified_fraction": uniform,
                "declared_weight_certified_fraction": weighted,
                "generator_correction": certificate.generator_correction,
                "evaluator_mass_charge": certificate.evaluator_mass_correction,
                "exact_mass_controlled_lower_bound": bound.exact_lower_coverage,
                "hoeffding_mass_controlled_lower_bound": bound.hoeffding_lower_coverage,
                # Backwards-compatible name; the exact bound is now primary.
                "mass_controlled_lower_bound": bound.exact_lower_coverage,
                "target_coverage": 1 - analysis["beta"],
                "passes_target": bound.exact_lower_coverage >= 1 - analysis["beta"],
                "hoeffding_passes_target": (
                    bound.hoeffding_lower_coverage >= 1 - analysis["beta"]
                ),
                "claim_scope": (
                    "confirmatory_iid"
                    if manifest["phase"] == "confirmatory"
                    and manifest["dataset"]["population_mode"] == "iid_workload_generator"
                    else "pilot_diagnostic_only"
                ),
            }
        )
    return rows


def _finite_catalogue_rows(
    manifest: dict[str, Any],
    per_item: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Certify the fully enumerated frozen catalogue without outer sampling."""
    analysis = manifest["analysis"]
    row_counts = {int(row["row_count"]) for row in per_item}
    if len(row_counts) != 1:
        raise ValueError("finite-catalogue analysis requires a common evaluator count")
    counts = [int(row["accept_count"]) for row in per_item]
    evaluator_rows = row_counts.pop()
    familywise = familywise_operational_certificate(
        counts,
        evaluator_rows,
        analysis["candidate_panel_sizes"],
        analysis["threshold"],
        analysis["delta"],
        analysis["eta_e"],
        analysis["eta_g"],
    )
    mass = mass_controlled_certificate(
        counts,
        evaluator_rows,
        analysis["candidate_panel_sizes"],
        analysis["threshold"],
        analysis["delta"],
        analysis["eta_e"],
        analysis["eta_g"],
        analysis["xi_e"],
    )
    raw_weights = np.asarray(
        [float(row["declared_item_weight"]) for row in per_item], dtype=float
    )
    weights = raw_weights / raw_weights.sum()
    output: list[dict[str, Any]] = []
    for panel_size in analysis["candidate_panel_sizes"]:
        familywise_flags = np.asarray(
            [
                interval_certifies(
                    lower,
                    upper,
                    panel_size,
                    analysis["threshold"],
                    analysis["delta"],
                )
                for lower, upper in familywise.evaluator_intervals
            ],
            dtype=float,
        )
        mass_flags = np.asarray(
            [
                interval_certifies(
                    lower,
                    upper,
                    panel_size,
                    analysis["threshold"],
                    analysis["delta"],
                )
                for lower, upper in mass.evaluator_intervals
            ],
            dtype=float,
        )
        output.append(
            {
                "panel_size": panel_size,
                "delta": analysis["delta"],
                "catalogue_size": len(per_item),
                "familywise_certified_items": int(familywise_flags.sum()),
                "familywise_uniform_lower_coverage": float(familywise_flags.mean()),
                "familywise_declared_weight_lower_coverage": float(
                    np.dot(weights, familywise_flags)
                ),
                "familywise_confidence": 1 - analysis["eta_e"],
                "mass_certified_items": int(mass_flags.sum()),
                "mass_uniform_certified_fraction": float(mass_flags.mean()),
                "mass_declared_weight_certified_fraction": float(np.dot(weights, mass_flags)),
                "mass_uniform_lower_coverage": max(
                    0.0, float(mass_flags.mean()) - analysis["xi_e"]
                ),
                "mass_declared_weight_lower_coverage": max(
                    0.0,
                    float(np.dot(weights, mass_flags)) - analysis["xi_e"],
                ),
                "mass_confidence": 1 - analysis["eta_e"],
                "claim_scope": "fully_enumerated_frozen_catalogue",
            }
        )
    return output


def _finite_census_error(
    evaluator_rows: int,
    accept_count: int,
    panel_size: int,
    threshold: float,
    corruption_budget: int = 0,
) -> float:
    """Exact worst-case disagreement with the honest frozen census decision."""
    census_accepts = accept_count >= quota(evaluator_rows, threshold)
    if census_accepts:
        corrupted_count = max(0, accept_count - corruption_budget)
    else:
        corrupted_count = min(evaluator_rows, accept_count + corruption_budget)
    panel_quota = quota(panel_size, threshold)
    if census_accepts:
        return float(
            stats.hypergeom.cdf(
                panel_quota - 1,
                evaluator_rows,
                corrupted_count,
                panel_size,
            )
        )
    return float(
        stats.hypergeom.sf(
            panel_quota - 1,
            evaluator_rows,
            corrupted_count,
            panel_size,
        )
    )


def _finite_census_rows(
    manifest: dict[str, Any],
    per_item: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return per-item, aggregate, and corruption results for the frozen census."""
    analysis = manifest["analysis"]
    threshold = float(analysis["threshold"])
    delta = float(analysis["delta"])
    row_counts = {int(row["row_count"]) for row in per_item}
    if len(row_counts) != 1:
        raise ValueError("finite-census analysis requires a common evaluator count")
    evaluator_rows = row_counts.pop()
    raw_weights = np.asarray(
        [float(row["declared_item_weight"]) for row in per_item], dtype=float
    )
    weights = raw_weights / raw_weights.sum()

    errors_by_item: dict[str, list[float]] = {}
    per_item_output: list[dict[str, Any]] = []
    for row in per_item:
        accept_count = int(row["accept_count"])
        errors = [
            _finite_census_error(
                evaluator_rows,
                accept_count,
                panel_size,
                threshold,
            )
            for panel_size in range(1, evaluator_rows + 1)
        ]
        errors_by_item[str(row["id"])] = errors
        certified = [error <= delta for error in errors]
        minimum = next(index + 1 for index, flag in enumerate(certified) if flag)
        stable = evaluator_rows
        suffix_all = True
        for index in range(evaluator_rows - 1, -1, -1):
            suffix_all = suffix_all and certified[index]
            if suffix_all:
                stable = index + 1
        selected_errors = {
            str(panel_size): errors[panel_size - 1]
            for panel_size in (5, 7, 11, 13, 23, 25, 39, 40)
            if panel_size <= evaluator_rows
        }
        per_item_output.append(
            {
                "id": row["id"],
                "evaluator_census_size": evaluator_rows,
                "accept_count": accept_count,
                "census_decision": (
                    "accept" if accept_count >= quota(evaluator_rows, threshold) else "reject"
                ),
                "minimum_certifying_panel_size": minimum,
                "stable_certifying_panel_size": stable,
                "selected_panel_errors": selected_errors,
                "delta": delta,
                "claim_scope": "exact_frozen_census_no_replacement",
            }
        )

    aggregate_output: list[dict[str, Any]] = []
    for panel_size in range(1, evaluator_rows + 1):
        flags = np.asarray(
            [errors_by_item[str(row["id"])][panel_size - 1] <= delta for row in per_item],
            dtype=float,
        )
        aggregate_output.append(
            {
                "panel_size": panel_size,
                "certified_items": int(flags.sum()),
                "uniform_certified_fraction": float(flags.mean()),
                "declared_weight_certified_fraction": float(np.dot(weights, flags)),
                "delta": delta,
                "claim_scope": "exact_frozen_census_no_replacement",
            }
        )

    corruption_output: list[dict[str, Any]] = []
    panel_sizes = [size for size in (5, 7, 11, 13, 23, 25, 39, 40) if size <= evaluator_rows]
    if not panel_sizes:
        panel_sizes = list(range(1, evaluator_rows + 1))
    budgets = sorted(
        {
            0,
            1,
            evaluator_rows // 20,
            evaluator_rows // 10,
            3 * evaluator_rows // 20,
        }
    )
    for budget in budgets:
        for panel_size in panel_sizes:
            errors = np.asarray(
                [
                    _finite_census_error(
                        evaluator_rows,
                        int(row["accept_count"]),
                        panel_size,
                        threshold,
                        corruption_budget=budget,
                    )
                    for row in per_item
                ]
            )
            safe = (errors <= delta).astype(float)
            corruption_output.append(
                {
                    "corruption_budget": budget,
                    "corruption_fraction": budget / evaluator_rows,
                    "panel_size": panel_size,
                    "safe_items": int(safe.sum()),
                    "uniform_safe_fraction": float(safe.mean()),
                    "declared_weight_safe_fraction": float(np.dot(weights, safe)),
                    "delta": delta,
                    "attack_model": "exact_nonadaptive_targeted_census_contamination",
                    "claim_scope": "frozen_census_sensitivity",
                }
            )
    return per_item_output, aggregate_output, corruption_output


def _population_sensitivity_rows(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Descriptive decision sensitivity to realized configuration weighting."""
    threshold = float(manifest["analysis"]["threshold"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["item_id"])].append(record)
    output: list[dict[str, Any]] = []
    for item_id, item_records in sorted(grouped.items()):
        by_configuration: dict[str, list[int]] = defaultdict(list)
        for record in item_records:
            by_configuration[str(record["configuration_id"])].append(int(record["global_vote"]))
        baseline_rate = float(np.mean([int(row["global_vote"]) for row in item_records]))
        configuration_rate = float(
            np.mean([np.mean(votes) for votes in by_configuration.values()])
        )
        baseline_decision = baseline_rate >= threshold
        leave_one_changes: list[str] = []
        for configuration in sorted(by_configuration):
            retained = [
                int(row["global_vote"])
                for row in item_records
                if str(row["configuration_id"]) != configuration
            ]
            if (float(np.mean(retained)) >= threshold) != baseline_decision:
                leave_one_changes.append(configuration)
        output.append(
            {
                "id": item_id,
                "realized_row_acceptance_rate": baseline_rate,
                "equal_realized_configuration_acceptance_rate": configuration_rate,
                "realized_row_decision": "accept" if baseline_decision else "reject",
                "equal_realized_configuration_decision": (
                    "accept" if configuration_rate >= threshold else "reject"
                ),
                "equal_configuration_decision_changed": (
                    (configuration_rate >= threshold) != baseline_decision
                ),
                "leave_one_configuration_decision_changed": bool(leave_one_changes),
                "configurations_causing_leave_one_change": leave_one_changes,
                "realized_configurations": len(by_configuration),
                "estimation_status": "descriptive_reweighting_not_certified",
            }
        )
    return output


def _failure_policy_rows(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    panel_size: int,
) -> list[dict[str, Any]]:
    """Descriptive sensitivity to three explicit non-success policies."""
    analysis = manifest["analysis"]
    threshold = float(analysis["threshold"])
    interval_alpha = float(analysis["eta_e"] * analysis["xi_e"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["item_id"])].append(record)
    output: list[dict[str, Any]] = []
    for item_id, item_records in sorted(grouped.items()):
        policy_samples = {
            "declared_map_to_reject": [int(row["global_vote"]) for row in item_records],
            "all_non_success_to_accept": [
                1 if row["status"] != "success" else int(row["global_vote"])
                for row in item_records
            ],
            "successes_only": [
                int(row["global_vote"]) for row in item_records if row["status"] == "success"
            ],
        }
        evaluated: dict[str, tuple[float, bool, bool]] = {}
        for policy, votes in policy_samples.items():
            accepts = int(sum(votes))
            trials = len(votes)
            lower, upper = clopper_pearson(accepts, trials, interval_alpha)
            rate = accepts / trials
            decision = rate >= threshold
            certified = interval_certifies(
                lower,
                upper,
                panel_size,
                threshold,
                analysis["delta"],
            )
            evaluated[policy] = (rate, decision, certified)
            output.append(
                {
                    "id": item_id,
                    "policy": policy,
                    "trials": trials,
                    "accept_count": accepts,
                    "acceptance_rate": rate,
                    "interval_lower": lower,
                    "interval_upper": upper,
                    "population_decision": "accept" if decision else "reject",
                    "certified_at_panel_size": certified,
                    "panel_size": panel_size,
                    "delta": analysis["delta"],
                    "estimation_status": "descriptive_failure_policy_sensitivity",
                }
            )
        baseline = evaluated["declared_map_to_reject"]
        for row in output[-3:]:
            policy_result = evaluated[str(row["policy"])]
            row["decision_changed_from_declared"] = policy_result[1] != baseline[1]
            row["certification_changed_from_declared"] = policy_result[2] != baseline[2]
    return output


def _boundary_rows(
    manifest: dict[str, Any],
    per_item: list[dict[str, Any]],
    certifications: dict[tuple[str, int], bool],
) -> list[dict[str, Any]]:
    analysis = manifest["analysis"]
    threshold = float(analysis["threshold"])
    raw_weights = np.asarray([row["declared_item_weight"] for row in per_item], dtype=float)
    weights = raw_weights / raw_weights.sum()
    output: list[dict[str, Any]] = []
    for panel_size in analysis["candidate_panel_sizes"]:
        reject_boundary, accept_boundary = certification_boundaries(
            panel_size, threshold, analysis["delta"]
        )
        threshold_crossing = np.asarray(
            [row["interval_lower"] <= threshold <= row["interval_upper"] for row in per_item],
            dtype=float,
        )
        boundary_crossing = np.asarray(
            [
                row["interval_lower"] <= reject_boundary <= row["interval_upper"]
                or row["interval_lower"] <= accept_boundary <= row["interval_upper"]
                for row in per_item
            ],
            dtype=float,
        )
        uncertified = np.asarray(
            [not certifications[(row["id"], panel_size)] for row in per_item], dtype=float
        )
        distances = np.asarray(
            [
                min(
                    abs(row["acceptance_rate"] - reject_boundary),
                    abs(row["acceptance_rate"] - accept_boundary),
                )
                for row in per_item
            ],
            dtype=float,
        )
        output.append(
            {
                "panel_size": panel_size,
                "delta": analysis["delta"],
                "reject_probability_boundary": reject_boundary,
                "accept_probability_boundary": accept_boundary,
                "uncertainty_band_width": accept_boundary - reject_boundary,
                "uniform_threshold_interval_mass": float(threshold_crossing.mean()),
                "declared_weight_threshold_interval_mass": float(
                    np.dot(weights, threshold_crossing)
                ),
                "uniform_certification_boundary_interval_mass": float(boundary_crossing.mean()),
                "declared_weight_certification_boundary_interval_mass": float(
                    np.dot(weights, boundary_crossing)
                ),
                "uniform_uncertified_fraction": float(uncertified.mean()),
                "declared_weight_uncertified_fraction": float(np.dot(weights, uncertified)),
                "median_plugin_distance_to_nearest_boundary": float(np.median(distances)),
                "p10_plugin_distance_to_nearest_boundary": float(np.quantile(distances, 0.10)),
                "estimation_status": "pilot_diagnostic_interval_geometry",
            }
        )
    return output


def _security_rows(
    manifest: dict[str, Any],
    per_item: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_weights = np.asarray([row["declared_item_weight"] for row in per_item], dtype=float)
    weights = raw_weights / raw_weights.sum()
    clarity = np.asarray([row["clarity_hat"] for row in per_item], dtype=float)
    delta = manifest["analysis"]["delta"]
    output: list[dict[str, Any]] = []
    for panel_size in manifest["analysis"]["candidate_panel_sizes"]:
        for alpha in manifest["analysis"]["adversarial_fractions"]:
            for attack_model, function in (
                ("fixed_byzantine_share", fixed_share_attack_probability),
                ("targeted_census_contamination", targeted_attack_probability),
            ):
                probabilities = np.asarray(function(panel_size, alpha, clarity), dtype=float)
                safe = probabilities <= delta
                output.append(
                    {
                        "panel_size": panel_size,
                        "adversarial_fraction": alpha,
                        "attack_model": attack_model,
                        "uniform_safe_workload_fraction": float(np.mean(safe)),
                        "declared_weight_safe_workload_fraction": float(np.dot(weights, safe)),
                        "uniform_mean_attack_probability": float(np.mean(probabilities)),
                        "declared_weight_mean_attack_probability": float(
                            np.dot(weights, probabilities)
                        ),
                        "delta": delta,
                        "estimation_status": "plug_in_pilot_not_certified",
                    }
                )
    return output


def _provider_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["configuration_id"],
            str(_route_field(record, "provider") or "unknown"),
            str(_route_field(record, "served_model_id") or "unknown"),
        )
        groups[key].append(record)
    output: list[dict[str, Any]] = []
    for (configuration, provider, served_model), group in sorted(groups.items()):
        costs = [
            value
            for record in group
            if (value := _float(_route_field(record, "cost_usd"))) is not None
        ]
        output.append(
            {
                "configuration_id": configuration,
                "provider": provider,
                "served_model_id": served_model,
                "calls": len(group),
                "successes": sum(record["status"] == "success" for record in group),
                "policy_fingerprints": sorted(
                    {
                        str(value)
                        for record in group
                        if (value := _route_field(record, "policy_fingerprint")) is not None
                    }
                ),
                "mean_latency_ms": float(np.mean([record["latency_ms"] for record in group])),
                "total_cost_usd": float(math.fsum(costs)) if costs else None,
            }
        )
    return output


def _public_vote_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the minimal sanitized cell ledger needed to reproduce the analysis."""
    return [
        {
            "cell_id": record["cell_id"],
            "item_id": record["item_id"],
            "row_index": record["row_index"],
            "configuration_id": record["configuration_id"],
            "global_vote": record["global_vote"],
            "status": record["status"],
        }
        for record in sorted(records, key=lambda row: (row["item_id"], row["row_index"]))
    ]


def _summary(
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    per_item: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    clear_items = [row for row in per_item if row["gold_label"] in {"accept", "reject"}]
    gold_accept = [row for row in clear_items if row["gold_label"] == "accept"]
    gold_reject = [row for row in clear_items if row["gold_label"] == "reject"]
    unresolvable = [row for row in per_item if row["gold_label"] == "unresolvable"]
    clear_ids = {row["id"]: row["gold_label"] for row in clear_items}
    correct_cells = [
        record["global_vote"] == int(clear_ids[record["item_id"]] == "accept")
        for record in records
        if record["item_id"] in clear_ids
    ]
    adversarial_clear = [
        row for row in clear_items if row["clarity_stratum"] == "ambiguous_adversarial"
    ]
    adversarial_clear_ids = {row["id"]: row["gold_label"] for row in adversarial_clear}
    adversarial_correct_cells = [
        record["global_vote"] == int(adversarial_clear_ids[record["item_id"]] == "accept")
        for record in records
        if record["item_id"] in adversarial_clear_ids
    ]
    accept_ids = {row["id"] for row in gold_accept}
    reject_ids = {row["id"] for row in gold_reject}
    false_reject_cells = [
        record["global_vote"] == 0 for record in records if record["item_id"] in accept_ids
    ]
    false_accept_cells = [
        record["global_vote"] == 1 for record in records if record["item_id"] in reject_ids
    ]
    exact_passing = [row["panel_size"] for row in coverage if row["passes_target"]]
    hoeffding_passing = [
        row["panel_size"] for row in coverage if row["hoeffding_passes_target"]
    ]
    # Preserve the panel reported by the frozen pilot analysis.  New
    # confirmatory runs use the exact exterior inversion as the primary rule.
    passing = hoeffding_passing if manifest["phase"] == "pilot" else exact_passing
    selected_panel_size = min(passing) if passing else None
    selected_coverage = next(
        (row for row in coverage if row["panel_size"] == selected_panel_size), None
    )
    costs = [
        value
        for record in records
        if (value := _float(_route_field(record, "cost_usd"))) is not None
    ]
    policy_fingerprints = sorted(
        {
            str(value)
            for record in records
            if (value := _route_field(record, "policy_fingerprint")) is not None
        }
    )
    providers = sorted(
        {
            str(value)
            for record in records
            if (value := _route_field(record, "provider")) is not None
        }
    )
    served_models = sorted(
        {
            str(value)
            for record in records
            if (value := _route_field(record, "served_model_id")) is not None
        }
    )
    status_counts = dict(sorted(Counter(record["status"] for record in records).items()))
    return {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "phase": manifest["phase"],
        "claim_scope": coverage[0]["claim_scope"],
        "items": len(per_item),
        "evaluator_rows": len(metadata["selected_row_indices"]),
        "cells": len(records),
        "status_counts": status_counts,
        "successful_response_fraction": status_counts.get("success", 0) / len(records),
        "declared_evaluator_configurations": len(
            manifest["evaluator_population"]["configurations"]
        ),
        "realized_evaluator_configurations": len(
            {record["configuration_id"] for record in records}
        ),
        "policy_fingerprints_observed": policy_fingerprints,
        "providers_observed": providers,
        "served_models_observed": served_models,
        "selected_panel_size": selected_panel_size,
        "exact_selected_panel_size": min(exact_passing) if exact_passing else None,
        "hoeffding_selected_panel_size": (
            min(hoeffding_passing) if hoeffding_passing else None
        ),
        "target_certified": bool(passing),
        "selected_uniform_certified_fraction": (
            selected_coverage["uniform_certified_fraction"] if selected_coverage else None
        ),
        "selected_mass_controlled_lower_bound": (
            selected_coverage["mass_controlled_lower_bound"] if selected_coverage else None
        ),
        "selected_hoeffding_mass_controlled_lower_bound": (
            selected_coverage["hoeffding_mass_controlled_lower_bound"]
            if selected_coverage
            else None
        ),
        "target_coverage": coverage[0]["target_coverage"],
        "construction_resolvable_items": len(clear_items),
        "construction_resolvable_population_matches": sum(
            row["population_label_correct"] for row in clear_items
        ),
        "unresolvable_items": len(unresolvable),
        "unresolvable_certified_at_selected_panel": (
            sum(selected_panel_size in row["certified_panel_sizes"] for row in unresolvable)
            if selected_panel_size is not None
            else None
        ),
        "population_label_accuracy_on_clear_items": (
            float(np.mean([row["population_label_correct"] for row in clear_items]))
            if clear_items
            else None
        ),
        "cell_accuracy_on_clear_items": float(np.mean(correct_cells))
        if correct_cells
        else None,
        "population_false_rejection_rate": (
            float(np.mean([row["population_label_hat"] == "reject" for row in gold_accept]))
            if gold_accept
            else None
        ),
        "population_false_acceptance_rate": (
            float(np.mean([row["population_label_hat"] == "accept" for row in gold_reject]))
            if gold_reject
            else None
        ),
        "cell_false_rejection_rate": (
            float(np.mean(false_reject_cells)) if false_reject_cells else None
        ),
        "cell_false_acceptance_rate": (
            float(np.mean(false_accept_cells)) if false_accept_cells else None
        ),
        "population_accuracy_on_adversarial_clear_items": (
            float(np.mean([row["population_label_correct"] for row in adversarial_clear]))
            if adversarial_clear
            else None
        ),
        "cell_accuracy_on_adversarial_clear_items": (
            float(np.mean(adversarial_correct_cells)) if adversarial_correct_cells else None
        ),
        "mean_entropy_on_clear_items": (
            float(np.mean([row["binary_entropy"] for row in clear_items]))
            if clear_items
            else None
        ),
        "mean_entropy_on_unresolvable_items": (
            float(np.mean([row["binary_entropy"] for row in unresolvable]))
            if unresolvable
            else None
        ),
        "unresolvable_minus_clear_mean_entropy": (
            float(
                np.mean([row["binary_entropy"] for row in unresolvable])
                - np.mean([row["binary_entropy"] for row in clear_items])
            )
            if unresolvable and clear_items
            else None
        ),
        "total_cost_usd_reported": float(math.fsum(costs)) if costs else None,
        "mean_latency_ms": float(np.mean([record["latency_ms"] for record in records])),
        "interpretation": (
            "Pilot diagnostics only; construction labels and stratified seed catalogue do not "
            "support a confirmatory population claim."
            if manifest["phase"] == "pilot"
            else "Interpret under the frozen confirmatory manifest and acceptance gates."
        ),
    }


def _plot_coverage(path: Path, coverage: list[dict[str, Any]]) -> None:
    x = np.asarray([row["panel_size"] for row in coverage])
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(
        x,
        [row["uniform_certified_fraction"] for row in coverage],
        marker="o",
        label="Certified sample fraction",
    )
    ax.plot(
        x,
        [row["declared_weight_certified_fraction"] for row in coverage],
        marker="s",
        label="Declared-weight fraction",
    )
    ax.plot(
        x,
        [row["mass_controlled_lower_bound"] for row in coverage],
        color="#2166AC",
        linewidth=2.2,
        label="Exact outer diagnostic",
    )
    ax.plot(
        x,
        [row["hoeffding_mass_controlled_lower_bound"] for row in coverage],
        color="#E07A1F",
        linestyle="-.",
        linewidth=1.8,
        label="Hoeffding outer diagnostic",
    )
    ax.axhline(
        coverage[0]["target_coverage"],
        color="#B2182B",
        linestyle="--",
        label="Target $1-\\beta$",
    )
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Deployment panel size $K$")
    ax.set_ylabel("Certified fraction / diagnostic lower bound")
    ax.set_title("Stratified pilot catalogue (no workload-population claim)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_clarity(path: Path, per_item: list[dict[str, Any]]) -> None:
    order = ["clear_accept", "clear_reject", "ambiguous_benign", "ambiguous_adversarial"]
    labels = [
        "clear accept",
        "clear reject",
        "ambiguous benign",
        "input-presentation stress",
    ]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for index, stratum in enumerate(order):
        values = [row["clarity_hat"] for row in per_item if row["clarity_stratum"] == stratum]
        offsets = np.linspace(-0.14, 0.14, len(values)) if values else np.asarray([])
        ax.scatter(index + offsets, values, alpha=0.8, s=34)
        if values:
            ax.hlines(np.median(values), index - 0.25, index + 0.25, color="black", linewidth=2)
    ax.set_xticks(range(len(order)), labels, rotation=12)
    ax.set_ylim(-0.01, 0.51)
    ax.set_ylabel(r"Plug-in clarity $|\widehat p-1/2|$")
    ax.set_title("Observed clarity by preregistered pilot stratum")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_pilot_overview(
    path: Path,
    per_item: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
) -> None:
    order = ["clear_accept", "clear_reject", "ambiguous_benign", "ambiguous_adversarial"]
    labels = [
        "clear\naccept",
        "clear\nreject",
        "ambiguous\nbenign",
        "input-presentation\nstress",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))

    clarity_axis, coverage_axis = axes
    for index, stratum in enumerate(order):
        values = [row["clarity_hat"] for row in per_item if row["clarity_stratum"] == stratum]
        offsets = np.linspace(-0.14, 0.14, len(values)) if values else np.asarray([])
        clarity_axis.scatter(index + offsets, values, alpha=0.8, s=30)
        if values:
            clarity_axis.hlines(
                np.median(values), index - 0.25, index + 0.25, color="black", linewidth=2
            )
    clarity_axis.set_xticks(range(len(order)), labels)
    clarity_axis.set_ylim(-0.01, 0.51)
    clarity_axis.set_ylabel(r"Plug-in clarity $|\widehat p-1/2|$")
    clarity_axis.set_title("(a) Observed clarity by pilot stratum")

    x = np.asarray([row["panel_size"] for row in coverage])
    coverage_axis.plot(
        x,
        [row["uniform_certified_fraction"] for row in coverage],
        marker="o",
        label="Certified sample fraction",
    )
    coverage_axis.plot(
        x,
        [row["mass_controlled_lower_bound"] for row in coverage],
        color="#2166AC",
        linewidth=2.2,
        label="Exact outer diagnostic",
    )
    coverage_axis.plot(
        x,
        [row["hoeffding_mass_controlled_lower_bound"] for row in coverage],
        color="#E07A1F",
        linestyle="-.",
        linewidth=1.6,
        label="Hoeffding diagnostic",
    )
    coverage_axis.axhline(
        coverage[0]["target_coverage"],
        color="#B2182B",
        linestyle="--",
        label="Pilot target $1-\\beta$",
    )
    coverage_axis.set_xscale("log")
    coverage_axis.set_ylim(-0.02, 1.02)
    coverage_axis.set_xlabel("Deployment panel size $K$")
    coverage_axis.set_ylabel("Certified fraction / lower bound")
    coverage_axis.set_title("(b) Catalogue and outer-sampling diagnostics")
    coverage_axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_security(path: Path, security: list[dict[str, Any]]) -> None:
    panel_size = max(row["panel_size"] for row in security)
    subset = [row for row in security if row["panel_size"] == panel_size]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    styles = {
        "fixed_byzantine_share": ("#2166AC", "Fixed Byzantine share"),
        "targeted_census_contamination": ("#B2182B", "Targeted contamination"),
    }
    for model, (color, label) in styles.items():
        rows = [row for row in subset if row["attack_model"] == model]
        ax.plot(
            [row["adversarial_fraction"] for row in rows],
            [row["declared_weight_safe_workload_fraction"] for row in rows],
            marker="o",
            color=color,
            label=label,
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, 0.4)
    ax.set_xlabel(r"Adversarial fraction $\alpha$")
    ax.set_ylabel("Plug-in safe catalogue fraction")
    ax.set_title(f"Pilot catalogue security profiles, $K={panel_size}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _derived_diagnostic_summary(
    manifest: dict[str, Any],
    per_item: list[dict[str, Any]],
    catalogue: list[dict[str, Any]],
    finite_census_per_item: list[dict[str, Any]],
    population_sensitivity: list[dict[str, Any]],
    failure_sensitivity: list[dict[str, Any]],
    selected_panel_size: int,
) -> dict[str, Any]:
    catalogue_row = next(
        row for row in catalogue if int(row["panel_size"]) == selected_panel_size
    )
    unstable = [
        row for row in finite_census_per_item if int(row["stable_certifying_panel_size"]) > 7
    ]
    threshold = float(manifest["analysis"]["threshold"])
    threshold_crossers = [
        str(row["id"])
        for row in per_item
        if float(row["interval_lower"]) <= threshold <= float(row["interval_upper"])
    ]
    return {
        "finite_catalogue_panel_size": selected_panel_size,
        "finite_catalogue_familywise_certified_items": catalogue_row[
            "familywise_certified_items"
        ],
        "finite_catalogue_familywise_uniform_lower_coverage": catalogue_row[
            "familywise_uniform_lower_coverage"
        ],
        "finite_catalogue_familywise_declared_weight_lower_coverage": catalogue_row[
            "familywise_declared_weight_lower_coverage"
        ],
        "finite_catalogue_familywise_confidence": catalogue_row["familywise_confidence"],
        "finite_catalogue_mass_declared_weight_lower_coverage": catalogue_row[
            "mass_declared_weight_lower_coverage"
        ],
        "finite_census_items_stable_by_k7": len(finite_census_per_item) - len(unstable),
        "finite_census_items_total": len(finite_census_per_item),
        "finite_census_late_stable_items": [
            {
                "id": row["id"],
                "stable_panel_size": row["stable_certifying_panel_size"],
            }
            for row in unstable
        ],
        "mass_interval_threshold_crossing_items": threshold_crossers,
        "equal_configuration_decision_changes": sorted(
            str(row["id"])
            for row in population_sensitivity
            if bool(row["equal_configuration_decision_changed"])
        ),
        "leave_one_configuration_decision_changes": sorted(
            str(row["id"])
            for row in population_sensitivity
            if bool(row["leave_one_configuration_decision_changed"])
        ),
        "failure_policy_decision_changes": sorted(
            {
                str(row["id"])
                for row in failure_sensitivity
                if bool(row["decision_changed_from_declared"])
            }
        ),
        "failure_policy_certification_changes": sorted(
            {
                str(row["id"])
                for row in failure_sensitivity
                if bool(row["certification_changed_from_declared"])
            }
        ),
        "sensitivity_interpretation": (
            "Configuration reweighting and failure-policy variants are descriptive; "
            "the released matrix was not sampled to certify alternative evaluator laws."
        ),
    }


def _write_tex_table(path: Path, summary: dict[str, Any]) -> None:
    def render(value: Any) -> str:
        if value is None:
            return "---"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def percentage(value: float | None, digits: int = 1) -> str:
        return "---" if value is None else f"{100 * value:.{digits}f}\\%"

    success_count = summary["status_counts"].get("success", 0)
    family_count = (
        f"{summary['realized_evaluator_configurations']}/"
        f"{summary['declared_evaluator_configurations']}"
    )
    resolvable_match = (
        f"{summary['construction_resolvable_population_matches']}/"
        f"{summary['construction_resolvable_items']}"
    )
    unresolvable_certified = (
        f"{summary['unresolvable_certified_at_selected_panel']}/{summary['unresolvable_items']}"
    )
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Pilot result \\",
        r"\midrule",
        f"Frozen workload units & {summary['items']} \\\\",
        f"Evaluator rows / cells & {summary['evaluator_rows']} / {summary['cells']} \\\\",
        f"Realized / declared model families & {family_count} \\\\",
        f"Valid structured responses & {success_count}/{summary['cells']} "
        f"({percentage(summary['successful_response_fraction'], 2)}) \\\\",
        f"Aggregate matches on construction-resolvable units & {resolvable_match} \\\\",
        f"Selected $K$ & {render(summary['selected_panel_size'])} \\\\",
        f"Fixed-catalogue familywise certificate at selected $K$ & "
        f"{summary['finite_catalogue_familywise_certified_items']}/"
        f"{summary['items']} "
        f"({percentage(summary['finite_catalogue_familywise_uniform_lower_coverage'])}) \\\\",
        f"Fixed-catalogue declared-weight lower coverage & "
        f"{render(summary['finite_catalogue_familywise_declared_weight_lower_coverage'])} \\\\",
        f"Diagnostic exact outer-sampling lower bound & "
        f"{render(summary['selected_mass_controlled_lower_bound'])} \\\\",
        f"Diagnostic Hoeffding outer-sampling lower bound & "
        f"{render(summary['selected_hoeffding_mass_controlled_lower_bound'])} \\\\",
        f"Frozen-census units stable by $K=7$ & "
        f"{summary['finite_census_items_stable_by_k7']}/"
        f"{summary['finite_census_items_total']} \\\\",
        f"Construction-unresolvable units certified at selected $K$ & "
        f"{unresolvable_certified} \\\\",
        f"Mean entropy: resolvable / unresolvable & "
        f"{render(summary['mean_entropy_on_clear_items'])} / "
        f"{render(summary['mean_entropy_on_unresolvable_items'])} \\\\",
        f"Reported cost (USD) & {render(summary['total_cost_usd_reported'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_run(
    manifest_path: Path,
    manifest: dict[str, Any],
    run_id: str,
) -> Path:
    run_errors = validate_run(run_id, manifest)
    if run_errors:
        raise ValueError("run validation failed:\n" + "\n".join(run_errors))
    run_dir = DATA_DIR / "runs" / run_id
    metadata = read_json(run_dir / "run.json")
    records = list(iter_jsonl(run_dir / "responses.jsonl"))
    all_items = load_and_validate_dataset(
        from_root(manifest["dataset"]["path"]), manifest["dataset"]["item_count"]
    )
    selected = set(metadata["selected_item_ids"])
    items = [item for item in all_items if item["id"] in selected]
    per_item, certifications = _per_item_rows(manifest, items, records)
    coverage = _coverage_rows(manifest, per_item, certifications)
    catalogue = _finite_catalogue_rows(manifest, per_item)
    finite_census, finite_census_coverage, finite_census_security = _finite_census_rows(
        manifest, per_item
    )
    boundaries = _boundary_rows(manifest, per_item, certifications)
    security = _security_rows(manifest, per_item)
    providers = _provider_rows(records)
    public_votes = _public_vote_rows(records)
    population_sensitivity = _population_sensitivity_rows(manifest, public_votes)
    selection_field = (
        "hoeffding_passes_target" if manifest["phase"] == "pilot" else "passes_target"
    )
    passing_panel_sizes = [row["panel_size"] for row in coverage if row[selection_field]]
    sensitivity_panel_size = (
        min(passing_panel_sizes)
        if passing_panel_sizes
        else max(manifest["analysis"]["candidate_panel_sizes"])
    )
    failure_sensitivity = _failure_policy_rows(manifest, public_votes, sensitivity_panel_size)
    evaluator_rows = list(iter_jsonl(run_dir / "evaluator-rows.jsonl"))
    summary = _summary(manifest, metadata, per_item, coverage, records)
    summary.update(
        _derived_diagnostic_summary(
            manifest,
            per_item,
            catalogue,
            finite_census,
            population_sensitivity,
            failure_sensitivity,
            sensitivity_panel_size,
        )
    )

    output_dir = RESULTS_DIR / manifest["phase"] / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_item.csv", per_item)
    _write_csv(output_dir / "operational_coverage.csv", coverage)
    _write_csv(output_dir / "finite_catalogue_coverage.csv", catalogue)
    _write_csv(output_dir / "finite_census_per_item.csv", finite_census)
    _write_csv(output_dir / "finite_census_coverage.csv", finite_census_coverage)
    _write_csv(output_dir / "finite_census_security.csv", finite_census_security)
    _write_csv(output_dir / "population_sensitivity.csv", population_sensitivity)
    _write_csv(output_dir / "failure_policy_sensitivity.csv", failure_sensitivity)
    _write_csv(output_dir / "boundary_diagnostics.csv", boundaries)
    _write_csv(output_dir / "security_profiles.csv", security)
    _write_csv(output_dir / "provider_summary.csv", providers)
    _write_csv(output_dir / "global_verdicts.csv", public_votes)
    _write_csv(output_dir / "evaluator_rows.csv", evaluator_rows)
    write_json(output_dir / "summary.json", summary)
    _plot_coverage(output_dir / "operational_coverage.png", coverage)
    _plot_clarity(output_dir / "clarity_by_stratum.png", per_item)
    _plot_pilot_overview(output_dir / "pilot_overview.png", per_item, coverage)
    _plot_security(output_dir / "security_profiles.png", security)
    _write_tex_table(output_dir / "operational_coverage_table.tex", summary)
    artifact_names = [
        "per_item.csv",
        "operational_coverage.csv",
        "finite_catalogue_coverage.csv",
        "finite_census_per_item.csv",
        "finite_census_coverage.csv",
        "finite_census_security.csv",
        "population_sensitivity.csv",
        "failure_policy_sensitivity.csv",
        "boundary_diagnostics.csv",
        "security_profiles.csv",
        "provider_summary.csv",
        "global_verdicts.csv",
        "evaluator_rows.csv",
        "summary.json",
        "operational_coverage.png",
        "clarity_by_stratum.png",
        "pilot_overview.png",
        "security_profiles.png",
        "operational_coverage_table.tex",
    ]
    write_json(
        output_dir / "provenance.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "dataset_sha256": sha256_file(from_root(manifest["dataset"]["path"])),
            "responses_sha256": sha256_file(run_dir / "responses.jsonl"),
            "request_ledger_sha256": sha256_file(run_dir / "request-ledger.jsonl"),
            "global_verdicts_sha256": sha256_file(run_dir / "global-verdicts.jsonl"),
            "evaluator_rows_sha256": sha256_file(run_dir / "evaluator-rows.jsonl"),
            "run_metadata_sha256": sha256_file(run_dir / "run.json"),
            "execution_code_commit": metadata.get("code_commit"),
            "execution_code_dirty": metadata.get("code_dirty"),
            "analysis_code_commit": git_commit(REPO_ROOT),
            "analysis_code_dirty": git_dirty(REPO_ROOT),
            "dependency_lock_sha256": metadata["provenance"]["dependency_lock_sha256"],
            "system_prompt_sha256": metadata["provenance"]["system_prompt_sha256"],
            "user_template_sha256": metadata["provenance"]["user_template_sha256"],
            "claim_scope": summary["claim_scope"],
            "derived_artifact_sha256": {
                name: sha256_file(output_dir / name) for name in artifact_names
            },
        },
    )
    return output_dir


def _read_public_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def refresh_released_artifacts(manifest: dict[str, Any]) -> Path:
    """Recompute public diagnostics from the released sanitized ledgers.

    This path deliberately performs no provider calls and does not require the
    private raw-response archive.  It is sufficient for every binary-vote
    analysis added after the original pilot release.
    """
    target = RESULTS_DIR / "paper"
    per_item = _read_public_csv(target / "per_item.csv")
    integer_fields = ("row_count", "accept_count", "non_success_cells")
    float_fields = (
        "declared_item_weight",
        "acceptance_rate",
        "interval_lower",
        "interval_upper",
        "clarity_hat",
        "binary_entropy",
        "mean_reported_confidence",
    )
    for row in per_item:
        for field in integer_fields:
            row[field] = int(row[field])  # type: ignore[assignment]
        for field in float_fields:
            row[field] = float(row[field])  # type: ignore[assignment]
        row["certified_panel_sizes"] = json.loads(row["certified_panel_sizes"])

    public_votes = _read_public_csv(target / "global_verdicts.csv")
    for row in public_votes:
        row["row_index"] = int(row["row_index"])  # type: ignore[assignment]
        row["global_vote"] = int(row["global_vote"])  # type: ignore[assignment]

    ledger_counts: dict[str, tuple[int, int]] = {}
    for item_id, group in _group_records_by_item(public_votes).items():
        ledger_counts[item_id] = (
            len(group),
            sum(int(record["global_vote"]) for record in group),
        )
    for row in per_item:
        expected = (int(row["row_count"]), int(row["accept_count"]))
        if ledger_counts[str(row["id"])] != expected:
            raise ValueError(f"released ledger diverges from per_item.csv for {row['id']}")

    certifications: dict[tuple[str, int], bool] = {}
    for row in per_item:
        for panel_size in manifest["analysis"]["candidate_panel_sizes"]:
            certifications[(str(row["id"]), panel_size)] = interval_certifies(
                float(row["interval_lower"]),
                float(row["interval_upper"]),
                panel_size,
                manifest["analysis"]["threshold"],
                manifest["analysis"]["delta"],
            )

    coverage = _coverage_rows(manifest, per_item, certifications)
    catalogue = _finite_catalogue_rows(manifest, per_item)
    finite_census, finite_census_coverage, finite_census_security = _finite_census_rows(
        manifest, per_item
    )
    boundaries = _boundary_rows(manifest, per_item, certifications)
    security = _security_rows(manifest, per_item)
    population_sensitivity = _population_sensitivity_rows(manifest, public_votes)
    selection_field = (
        "hoeffding_passes_target" if manifest["phase"] == "pilot" else "passes_target"
    )
    passing = [row["panel_size"] for row in coverage if row[selection_field]]
    selected_panel_size = (
        min(passing) if passing else max(manifest["analysis"]["candidate_panel_sizes"])
    )
    failure_sensitivity = _failure_policy_rows(manifest, public_votes, selected_panel_size)

    selected_coverage = next(
        row for row in coverage if int(row["panel_size"]) == selected_panel_size
    )
    summary = read_json(target / "summary.json")
    exact_passing = [row["panel_size"] for row in coverage if row["passes_target"]]
    hoeffding_passing = [
        row["panel_size"] for row in coverage if row["hoeffding_passes_target"]
    ]
    summary.update(
        {
            "selected_panel_size": selected_panel_size,
            "exact_selected_panel_size": min(exact_passing) if exact_passing else None,
            "hoeffding_selected_panel_size": (
                min(hoeffding_passing) if hoeffding_passing else None
            ),
            "target_certified": bool(passing),
            "selected_uniform_certified_fraction": selected_coverage[
                "uniform_certified_fraction"
            ],
            "selected_mass_controlled_lower_bound": selected_coverage[
                "mass_controlled_lower_bound"
            ],
            "selected_hoeffding_mass_controlled_lower_bound": selected_coverage[
                "hoeffding_mass_controlled_lower_bound"
            ],
        }
    )
    summary.update(
        _derived_diagnostic_summary(
            manifest,
            per_item,
            catalogue,
            finite_census,
            population_sensitivity,
            failure_sensitivity,
            selected_panel_size,
        )
    )

    _write_csv(target / "operational_coverage.csv", coverage)
    _write_csv(target / "finite_catalogue_coverage.csv", catalogue)
    _write_csv(target / "finite_census_per_item.csv", finite_census)
    _write_csv(target / "finite_census_coverage.csv", finite_census_coverage)
    _write_csv(target / "finite_census_security.csv", finite_census_security)
    _write_csv(target / "population_sensitivity.csv", population_sensitivity)
    _write_csv(target / "failure_policy_sensitivity.csv", failure_sensitivity)
    _write_csv(target / "boundary_diagnostics.csv", boundaries)
    _write_csv(target / "security_profiles.csv", security)
    write_json(target / "summary.json", summary)
    _plot_coverage(target / "operational_coverage.png", coverage)
    _plot_clarity(target / "clarity_by_stratum.png", per_item)
    _plot_pilot_overview(target / "pilot_overview.png", per_item, coverage)
    _plot_security(target / "security_profiles.png", security)
    _write_tex_table(target / "operational_coverage_table.tex", summary)

    artifact_names = [
        "per_item.csv",
        "operational_coverage.csv",
        "finite_catalogue_coverage.csv",
        "finite_census_per_item.csv",
        "finite_census_coverage.csv",
        "finite_census_security.csv",
        "population_sensitivity.csv",
        "failure_policy_sensitivity.csv",
        "boundary_diagnostics.csv",
        "security_profiles.csv",
        "provider_summary.csv",
        "global_verdicts.csv",
        "evaluator_rows.csv",
        "summary.json",
        "operational_coverage.png",
        "clarity_by_stratum.png",
        "pilot_overview.png",
        "security_profiles.png",
        "operational_coverage_table.tex",
    ]
    provenance = read_json(target / "provenance.json")
    provenance.update(
        {
            "analysis_code_commit": git_commit(REPO_ROOT),
            "analysis_code_dirty": git_dirty(REPO_ROOT),
            "release_refresh_source": "sanitized_binary_ledger_and_published_per_item_counts",
            "derived_artifact_sha256": {
                name: sha256_file(target / name) for name in artifact_names
            },
        }
    )
    write_json(target / "provenance.json", provenance)
    return target


def _group_records_by_item(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["item_id"])].append(record)
    return grouped


def publish_paper_artifacts(
    manifest_path: Path,
    manifest: dict[str, Any],
    run_id: str,
) -> Path:
    source_dir = analyze_run(manifest_path, manifest, run_id)
    target = RESULTS_DIR / "paper"
    target.mkdir(parents=True, exist_ok=True)
    names = [
        "per_item.csv",
        "operational_coverage.csv",
        "finite_catalogue_coverage.csv",
        "finite_census_per_item.csv",
        "finite_census_coverage.csv",
        "finite_census_security.csv",
        "population_sensitivity.csv",
        "failure_policy_sensitivity.csv",
        "boundary_diagnostics.csv",
        "operational_coverage.png",
        "clarity_by_stratum.png",
        "security_profiles.csv",
        "security_profiles.png",
        "provider_summary.csv",
        "global_verdicts.csv",
        "evaluator_rows.csv",
        "operational_coverage_table.tex",
        "pilot_overview.png",
        "summary.json",
        "provenance.json",
    ]
    for name in names:
        shutil.copy2(source_dir / name, target / name)
    return target
