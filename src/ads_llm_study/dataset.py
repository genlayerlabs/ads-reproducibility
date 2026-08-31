from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from ads_llm_study.io_utils import iter_jsonl, read_json
from ads_llm_study.paths import SCHEMAS_DIR


class DatasetError(ValueError):
    pass


def load_and_validate_dataset(
    path: Path, expected_count: int | None = None
) -> list[dict[str, Any]]:
    items = list(iter_jsonl(path))
    schema = read_json(SCHEMAS_DIR / "dataset-item.schema.json")
    validator = Draft202012Validator(schema)
    failures: list[str] = []
    for line_number, item in enumerate(items, start=1):
        for error in validator.iter_errors(item):
            failures.append(f"line {line_number} {error.json_path}: {error.message}")
    if failures:
        raise DatasetError("dataset schema validation failed:\n" + "\n".join(failures))
    if expected_count is not None and len(items) != expected_count:
        raise DatasetError(f"expected {expected_count} items, found {len(items)}")

    ids = [item["id"] for item in items]
    duplicates = [item_id for item_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise DatasetError(f"duplicate item ids: {duplicates}")

    family_weights: dict[str, set[float]] = defaultdict(set)
    for item in items:
        family_weights[item["family"]].add(float(item["family_weight"]))
        if item["clarity_stratum"] == "clear_accept" and item["gold_label"] != "accept":
            raise DatasetError(f"{item['id']}: clear_accept must have accept gold label")
        if item["clarity_stratum"] == "clear_reject" and item["gold_label"] != "reject":
            raise DatasetError(f"{item['id']}: clear_reject must have reject gold label")
        if item["gold_label"] == "unresolvable":
            if set(item["defensible_labels"]) != {"accept", "reject"}:
                raise DatasetError(
                    f"{item['id']}: unresolvable item needs both defensible labels"
                )
        elif item["defensible_labels"] != [item["gold_label"]]:
            raise DatasetError(f"{item['id']}: clear item defensible_labels must equal gold")
    inconsistent = {
        family: values for family, values in family_weights.items() if len(values) != 1
    }
    if inconsistent:
        raise DatasetError(f"inconsistent weights within family: {inconsistent}")
    total = math.fsum(next(iter(values)) for values in family_weights.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise DatasetError(f"unique family weights sum to {total}, expected 1")
    return items


def declared_item_weights(items: list[dict[str, Any]]) -> dict[str, float]:
    family_counts = Counter(item["family"] for item in items)
    return {
        item["id"]: float(item["family_weight"]) / family_counts[item["family"]]
        for item in items
    }
