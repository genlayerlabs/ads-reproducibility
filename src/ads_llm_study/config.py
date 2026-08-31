from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from ads_llm_study.io_utils import read_json, sha256_file
from ads_llm_study.paths import REPO_ROOT, SCHEMAS_DIR, from_root


class ConfigurationError(ValueError):
    pass


def load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = from_root(path).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ConfigurationError(f"{manifest_path}: manifest must be a mapping")
    schema = read_json(SCHEMAS_DIR / "run-manifest.schema.json")
    errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=str)
    if errors:
        rendered = "\n".join(f"- {error.json_path}: {error.message}" for error in errors)
        raise ConfigurationError(f"manifest schema validation failed:\n{rendered}")
    validate_manifest_semantics(manifest_path, manifest)
    return manifest_path, manifest


def validate_manifest_semantics(path: Path, manifest: dict[str, Any]) -> None:
    if manifest["phase"] == "confirmatory" and manifest["status"] != "frozen":
        raise ConfigurationError("a confirmatory manifest must have status=frozen")
    if manifest["phase"] == "confirmatory" and "REPLACE_ME" in path.read_text():
        raise ConfigurationError("confirmatory manifest still contains REPLACE_ME")
    if manifest["phase"] == "confirmatory":
        if manifest["dataset"]["population_mode"] != "iid_workload_generator":
            raise ConfigurationError(
                "confirmatory mass-controlled claims require population_mode="
                "iid_workload_generator"
            )
        if manifest["dataset"]["human_relabel_required_for_confirmatory"]:
            raise ConfigurationError(
                "confirmatory data must complete independent human relabelling first"
            )
        if not manifest["execution"]["preregistration_hash_required"]:
            raise ConfigurationError(
                "confirmatory execution must require a preregistration hash"
            )

    dataset_path = from_root(manifest["dataset"]["path"])
    if not dataset_path.is_file():
        raise ConfigurationError(f"dataset does not exist: {dataset_path}")
    actual_hash = sha256_file(dataset_path)
    if actual_hash != manifest["dataset"]["sha256"]:
        expected_hash = manifest["dataset"]["sha256"]
        raise ConfigurationError(
            f"dataset hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

    prompt = manifest["prompt"]
    for key in ("system_path", "template_path"):
        prompt_path = from_root(prompt[key])
        if not prompt_path.is_file():
            raise ConfigurationError(f"prompt file does not exist: {prompt_path}")
    exposed = set(prompt["exposed_item_fields"])
    forbidden = set(prompt["forbidden_item_fields"])
    overlap = exposed & forbidden
    if overlap:
        raise ConfigurationError(
            f"prompt fields are both exposed and forbidden: {sorted(overlap)}"
        )

    population = manifest["evaluator_population"]
    configurations = population["configurations"]
    ids = [entry["id"] for entry in configurations]
    if len(ids) != len(set(ids)):
        raise ConfigurationError("evaluator configuration ids must be unique")
    total_weight = math.fsum(float(entry["weight"]) for entry in configurations)
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ConfigurationError(f"evaluator weights sum to {total_weight}, expected 1")

    request = manifest["request"]
    if request.get("max_concurrency_per_configuration", 1) > request["concurrency"]:
        raise ConfigurationError(
            "max_concurrency_per_configuration cannot exceed global concurrency"
        )
    if request["inter_call_delay_seconds"] > 0 and request["concurrency"] != 1:
        raise ConfigurationError(
            "inter_call_delay_seconds requires concurrency=1 for deterministic pacing"
        )

    router = manifest["router"]
    mandatory_trace_fields = {
        "provider",
        "model_family",
        "served_model_id",
        "policy_fingerprint",
        "decision_trace",
    }
    missing_trace_fields = mandatory_trace_fields - set(router["required_trace_fields"])
    if missing_trace_fields:
        raise ConfigurationError(
            f"required_trace_fields omits provenance fields: {sorted(missing_trace_fields)}"
        )
    if manifest["phase"] == "confirmatory":
        if not router["expected_policy_fingerprint"]:
            raise ConfigurationError(
                "confirmatory execution requires expected_policy_fingerprint"
            )
        if not router["allowed_providers"]:
            raise ConfigurationError("confirmatory execution requires allowed_providers")
        if router["provider_drift_policy"] != "abort_on_mismatch":
            raise ConfigurationError(
                "confirmatory execution must abort on router policy/provider drift"
            )

    analysis = manifest["analysis"]
    if analysis["xi_e"] >= analysis["beta"]:
        raise ConfigurationError("xi_e must be strictly smaller than beta")
    sizes = analysis["candidate_panel_sizes"]
    if sizes != sorted(sizes):
        raise ConfigurationError("candidate_panel_sizes must be increasing")
    if any(size % 2 == 0 for size in sizes):
        raise ConfigurationError("strict-majority pilot panel sizes must be odd")

    expected_calls = manifest["dataset"]["item_count"] * population["row_count"]
    declared_calls = manifest["execution"]["pilot_calls_expected"]
    if manifest["phase"] == "pilot" and expected_calls != declared_calls:
        raise ConfigurationError(
            f"pilot_calls_expected={declared_calls}, but design implies {expected_calls}"
        )


def manifest_hash(path: str | Path) -> str:
    return sha256_file(from_root(path))


def resolved_provenance(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    prompt = manifest["prompt"]
    return {
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_sha256": sha256_file(from_root(manifest["dataset"]["path"])),
        "system_prompt_sha256": sha256_file(from_root(prompt["system_path"])),
        "user_template_sha256": sha256_file(from_root(prompt["template_path"])),
        "dependency_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
        "source_manifest_sha256": sha256_file(REPO_ROOT / "data" / "source-manifest.json"),
    }
