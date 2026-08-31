from __future__ import annotations

from ads_llm_study.config import load_manifest
from ads_llm_study.design import (
    derive_cell_seed,
    generate_configuration_smoke_rows,
    generate_evaluator_rows,
)


def test_evaluator_design_is_deterministic_and_unique() -> None:
    _, manifest = load_manifest("preregistration/pilot.yaml")
    first = generate_evaluator_rows(manifest)
    second = generate_evaluator_rows(manifest)
    assert first == second
    assert len(first) == 40
    assert len({row["design_hash"] for row in first}) == 40
    assert all(row["model"].startswith("family:") for row in first)


def test_cell_seed_is_stable_and_item_specific() -> None:
    assert derive_cell_seed(123, "sla-01") == derive_cell_seed(123, "sla-01")
    assert derive_cell_seed(123, "sla-01") != derive_cell_seed(123, "sla-02")


def test_configuration_smoke_rows_cover_declared_population_outside_iid_sample() -> None:
    _, manifest = load_manifest("preregistration/pilot.yaml")
    configurations = manifest["evaluator_population"]["configurations"]

    selected = generate_configuration_smoke_rows(manifest, configurations)

    assert len(selected) == len(configurations) == 21
    assert [row["configuration_id"] for row in selected] == [
        configuration["id"] for configuration in configurations
    ]
    assert min(row["row_index"] for row in selected) == 40

    subset = generate_configuration_smoke_rows(manifest, configurations[2:5])
    assert [row["configuration_id"] for row in subset] == [
        configuration["id"] for configuration in configurations[2:5]
    ]
