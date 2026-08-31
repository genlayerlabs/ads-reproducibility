from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from ads_llm_study.config import load_manifest
from ads_llm_study.dataset import load_and_validate_dataset
from ads_llm_study.design import (
    generate_configuration_smoke_rows,
    generate_evaluator_rows,
)
from ads_llm_study.io_utils import iter_jsonl, read_json, sha256_file, sha256_object
from ads_llm_study.paths import DATA_DIR, SCHEMAS_DIR, from_root
from ads_llm_study.prompts import build_messages, template_fields

_SECRET_PATTERN = re.compile(
    r"(?:llmr_[A-Za-z0-9_.-]{12,}|sk-[A-Za-z0-9_-]{12,}|Authorization\s*:\s*Bearer)"
)


def _schema_errors(values: list[dict[str, Any]], schema_name: str) -> list[str]:
    schema = read_json(SCHEMAS_DIR / schema_name)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors: list[str] = []
    for index, value in enumerate(values):
        for error in validator.iter_errors(value):
            errors.append(f"record {index + 1} {error.json_path}: {error.message}")
    return errors


def _validate_source_manifest() -> list[str]:
    errors: list[str] = []
    source_manifest = read_json(DATA_DIR / "source-manifest.json")
    for entry in source_manifest["files"]:
        path = from_root(entry["path"])
        if not path.is_file():
            errors.append(f"missing imported source file: {entry['path']}")
        elif sha256_file(path) != entry["sha256"]:
            errors.append(f"imported source hash mismatch: {entry['path']}")
    return errors


def _validate_prompt_contract(
    manifest: dict[str, Any], items: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    prompt = manifest["prompt"]
    template_path = from_root(prompt["template_path"])
    template = template_path.read_text(encoding="utf-8")
    fields = template_fields(template)
    exposed = set(prompt["exposed_item_fields"])
    forbidden = set(prompt["forbidden_item_fields"])
    if fields != exposed:
        errors.append(
            "template placeholders "
            f"{sorted(fields)} do not exactly match exposed fields {sorted(exposed)}"
        )
    if fields & forbidden:
        errors.append(f"template exposes forbidden fields: {sorted(fields & forbidden)}")
    messages = build_messages(
        items[0],
        from_root(prompt["system_path"]),
        template_path,
        prompt["exposed_item_fields"],
    )
    serialized = json.dumps(messages, ensure_ascii=False)
    for field in forbidden:
        if f'"{field}"' in serialized:
            errors.append(f"prompt serialization contains forbidden field name {field}")
    return errors


def validate_run(run_id: str, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    run_dir = DATA_DIR / "runs" / run_id
    if not run_dir.is_dir():
        return [f"run does not exist: {run_id}"]
    required = {
        "run.json",
        "resolved-manifest.json",
        "evaluator-rows.jsonl",
        "request-ledger.jsonl",
        "responses.jsonl",
        "global-verdicts.jsonl",
    }
    missing = sorted(name for name in required if not (run_dir / name).is_file())
    if missing:
        return [f"run is missing required artifact: {name}" for name in missing]

    metadata = read_json(run_dir / "run.json")
    resolved_manifest = read_json(run_dir / "resolved-manifest.json")
    responses = list(iter_jsonl(run_dir / "responses.jsonl"))
    rows = list(iter_jsonl(run_dir / "evaluator-rows.jsonl"))
    ledgers = list(iter_jsonl(run_dir / "request-ledger.jsonl"))
    votes = list(iter_jsonl(run_dir / "global-verdicts.jsonl"))
    schema_errors = [
        *_schema_errors([metadata], "run-metadata.schema.json"),
        *_schema_errors(responses, "response-record.schema.json"),
        *_schema_errors(rows, "evaluator-row.schema.json"),
        *_schema_errors(ledgers, "request-ledger.schema.json"),
        *_schema_errors(votes, "global-verdict.schema.json"),
    ]
    errors.extend(schema_errors)
    if schema_errors:
        return errors

    if resolved_manifest != manifest:
        errors.append("resolved run manifest differs from the analysis manifest")
    if metadata.get("run_id") != run_id:
        errors.append("run metadata identifier does not match its directory")
    if metadata.get("status") != "complete":
        errors.append(f"run status is {metadata.get('status')!r}, expected 'complete'")

    row_indices = [row["row_index"] for row in rows]
    if len(row_indices) != len(set(row_indices)):
        errors.append("evaluator design contains duplicate row indices")
    if set(row_indices) != set(metadata.get("selected_row_indices", [])):
        errors.append("evaluator rows do not match selected_row_indices")
    if metadata.get("design_mode") == "configuration_coverage_smoke":
        selected_configuration_ids = metadata.get("selected_configuration_ids", [])
        declared_configurations = manifest["evaluator_population"]["configurations"]
        configurations_by_id = {
            configuration["id"]: configuration for configuration in declared_configurations
        }
        unknown = sorted(set(selected_configuration_ids) - configurations_by_id.keys())
        if unknown:
            errors.append(
                "diagnostic design names unknown configurations: " + ", ".join(unknown)
            )
        selected_configurations = [
            configurations_by_id[configuration_id]
            for configuration_id in selected_configuration_ids
            if configuration_id in configurations_by_id
        ]
        expected_design = generate_configuration_smoke_rows(manifest, selected_configurations)
    else:
        expected_design = generate_evaluator_rows(manifest)
    expected_rows = {row["row_index"]: row for row in expected_design}
    if metadata.get("design_mode") == "configuration_coverage_smoke" and set(
        row_indices
    ) != set(expected_rows):
        errors.append("configuration-coverage rows do not match selected_configuration_ids")
    for row in rows:
        if expected_rows.get(row["row_index"]) != row:
            errors.append(f"row {row['row_index']}: evaluator design differs from manifest")

    cell_ids = [record["cell_id"] for record in responses]
    if len(cell_ids) != len(set(cell_ids)):
        errors.append("response matrix contains duplicate cell ids")
    expected = {
        f"{item_id}::row-{row_index:04d}"
        for item_id in metadata["selected_item_ids"]
        for row_index in metadata["selected_row_indices"]
    }
    observed = set(cell_ids)
    if observed != expected:
        errors.append(
            f"matrix mismatch: {len(expected - observed)} missing, "
            f"{len(observed - expected)} unexpected"
        )
    if metadata.get("expected_cells") != len(expected):
        errors.append("run metadata expected_cells is inconsistent with its selected design")
    if metadata.get("completed_cells") != len(responses):
        errors.append("run metadata completed_cells is inconsistent with responses")
    if responses != sorted(responses, key=lambda value: (value["item_id"], value["row_index"])):
        errors.append("response matrix is not in canonical item/row order")

    records_by_cell = {record["cell_id"]: record for record in responses}
    for record in responses:
        canonical_cell = f"{record['item_id']}::row-{record['row_index']:04d}"
        if record["cell_id"] != canonical_cell:
            errors.append(f"{record['cell_id']}: non-canonical cell identifier")
        if record["run_id"] != run_id:
            errors.append(f"{record['cell_id']}: response has the wrong run id")
        expected_vote = int(
            record["status"] == "success" and record["parsed_verdict"] == "ACCEPT"
        )
        if record["global_vote"] != expected_vote:
            errors.append(f"{record['cell_id']}: nondeterministic global vote")
    if metadata.get("responses_sha256") != sha256_object(responses):
        errors.append("run response hash does not match canonical response matrix")

    ledger_ids = [entry.get("cell_id") for entry in ledgers]
    if len(ledger_ids) != len(set(ledger_ids)):
        errors.append("request ledger contains duplicate cell ids")
    if set(ledger_ids) != expected:
        errors.append("request ledger does not match the selected matrix")
    for entry in ledgers:
        cell_id = entry.get("cell_id")
        payload = entry.get("payload")
        response = records_by_cell.get(cell_id)
        if not isinstance(payload, dict) or response is None:
            errors.append(f"{cell_id}: malformed or unmatched request ledger entry")
            continue
        payload_hash = sha256_object(payload)
        if entry.get("request_sha256") != payload_hash:
            errors.append(f"{cell_id}: ledger payload hash mismatch")
        if response["request_sha256"] != payload_hash:
            errors.append(f"{cell_id}: response does not match frozen request payload")

    vote_ids = [entry.get("cell_id") for entry in votes]
    if len(vote_ids) != len(set(vote_ids)):
        errors.append("global verdict table contains duplicate cell ids")
    if set(vote_ids) != expected:
        errors.append("global verdict table does not match the selected matrix")
    for vote in votes:
        cell_id = vote.get("cell_id")
        response = records_by_cell.get(cell_id)
        if response is None:
            continue
        if vote.get("global_vote") != response["global_vote"]:
            errors.append(f"{cell_id}: global verdict differs from response aggregation")
        if vote.get("source_response_sha256") != sha256_object(response):
            errors.append(f"{cell_id}: global verdict source hash mismatch")

    for artifact in run_dir.iterdir():
        if (
            artifact.is_file()
            and artifact.suffix in {".json", ".jsonl", ".txt"}
            and _SECRET_PATTERN.search(artifact.read_text(encoding="utf-8", errors="replace"))
        ):
            errors.append(f"secret-like value found in run artifact: {artifact.name}")
    return errors


def validate_repository(manifest_path: str | Path, run_id: str | None = None) -> dict[str, Any]:
    _, manifest = load_manifest(manifest_path)
    items = load_and_validate_dataset(
        from_root(manifest["dataset"]["path"]), manifest["dataset"]["item_count"]
    )
    rows = generate_evaluator_rows(manifest)
    errors = _validate_source_manifest()
    errors.extend(_validate_prompt_contract(manifest, items))
    errors.extend(_schema_errors(rows, "evaluator-row.schema.json"))
    configuration_counts: dict[str, int] = {}
    for row in rows:
        configuration_counts[row["configuration_id"]] = (
            configuration_counts.get(row["configuration_id"], 0) + 1
        )
    if run_id:
        errors.extend(validate_run(run_id, manifest))
        noncomplete_runs: list[str] = []
    elif (DATA_DIR / "runs").is_dir():
        noncomplete_runs = []
        for child in sorted((DATA_DIR / "runs").iterdir()):
            if child.is_dir() and (child / "run.json").is_file():
                metadata = read_json(child / "run.json")
                if metadata.get("status") != "complete":
                    noncomplete_runs.append(child.name)
                    continue
                resolved_manifest_path = child / "resolved-manifest.json"
                if not resolved_manifest_path.is_file():
                    errors.append(
                        f"{child.name}: run is missing required artifact: "
                        "resolved-manifest.json"
                    )
                    continue
                run_manifest = read_json(resolved_manifest_path)
                manifest_schema_errors = _schema_errors(
                    [run_manifest], "run-manifest.schema.json"
                )
                errors.extend(
                    f"{child.name}: resolved manifest {error}"
                    for error in manifest_schema_errors
                )
                if manifest_schema_errors:
                    continue
                errors.extend(
                    f"{child.name}: {error}" for error in validate_run(child.name, run_manifest)
                )
    else:
        noncomplete_runs = []
    if errors:
        raise ValueError("validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    return {
        "ok": True,
        "items": len(items),
        "families": len({item["family"] for item in items}),
        "evaluator_rows": len(rows),
        "expected_cells": len(items) * len(rows),
        "configuration_row_counts": dict(sorted(configuration_counts.items())),
        "noncomplete_runs_skipped": noncomplete_runs,
    }
