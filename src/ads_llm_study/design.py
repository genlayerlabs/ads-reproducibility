from __future__ import annotations

import hashlib
from typing import Any

from ads_llm_study.io_utils import sha256_object


def _digest_int(value: str, bits: int = 64) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[: bits // 8], "big")


def _unit_interval(value: str) -> float:
    return _digest_int(value, 64) / 2**64


def generate_evaluator_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    population = manifest["evaluator_population"]
    base_seed = int(population["row_seed"])
    configurations = population["configurations"]
    cumulative: list[tuple[float, dict[str, Any]]] = []
    total = 0.0
    for configuration in configurations:
        total += float(configuration["weight"])
        cumulative.append((total, configuration))

    rows: list[dict[str, Any]] = []
    for row_index in range(int(population["row_count"])):
        draw = _unit_interval(f"choice|{base_seed}|{row_index}")
        configuration = next(config for ceiling, config in cumulative if draw < ceiling + 1e-15)
        row = {
            "row_index": row_index,
            "configuration_id": configuration["id"],
            "model": configuration["model"],
            "row_seed": _digest_int(f"row|{base_seed}|{row_index}", 32) & 0x7FFFFFFF,
        }
        row["design_hash"] = sha256_object(row)
        rows.append(row)
    return rows


def generate_configuration_smoke_rows(
    manifest: dict[str, Any], configurations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build deterministic diagnostic rows outside the primary IID sample."""
    population = manifest["evaluator_population"]
    base_seed = int(population["row_seed"])
    first_diagnostic_index = int(population["row_count"])
    declared_positions = {
        configuration["id"]: position
        for position, configuration in enumerate(population["configurations"])
    }
    rows: list[dict[str, Any]] = []
    for configuration in configurations:
        configuration_id = configuration["id"]
        position = declared_positions[configuration_id]
        row = {
            "row_index": first_diagnostic_index + position,
            "configuration_id": configuration_id,
            "model": configuration["model"],
            "row_seed": _digest_int(f"configuration-smoke|{base_seed}|{configuration_id}", 32)
            & 0x7FFFFFFF,
        }
        row["design_hash"] = sha256_object(row)
        rows.append(row)
    return rows


def derive_cell_seed(row_seed: int, item_id: str) -> int:
    return _digest_int(f"cell|{row_seed}|{item_id}", 32) & 0x7FFFFFFF
