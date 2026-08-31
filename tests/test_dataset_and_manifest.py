from __future__ import annotations

import math
from collections import Counter

from ads_llm_study.config import load_manifest
from ads_llm_study.dataset import declared_item_weights, load_and_validate_dataset
from ads_llm_study.paths import from_root
from ads_llm_study.validation import validate_repository


def test_repository_offline_validation() -> None:
    result = validate_repository("preregistration/pilot.yaml")
    assert result["ok"] is True
    assert result["items"] == 50
    assert result["families"] == 13
    assert result["evaluator_rows"] == 40
    assert result["expected_cells"] == 2000


def test_seed_dataset_composition_and_weights() -> None:
    _, manifest = load_manifest("preregistration/pilot.yaml")
    items = load_and_validate_dataset(
        from_root(manifest["dataset"]["path"]), manifest["dataset"]["item_count"]
    )
    assert Counter(item["gold_label"] for item in items) == {
        "accept": 15,
        "reject": 22,
        "unresolvable": 13,
    }
    assert Counter(item["clarity_stratum"] for item in items) == {
        "clear_accept": 13,
        "clear_reject": 13,
        "ambiguous_benign": 12,
        "ambiguous_adversarial": 12,
    }
    weights = declared_item_weights(items)
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)
