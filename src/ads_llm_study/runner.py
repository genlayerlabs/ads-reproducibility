from __future__ import annotations

import os
import re
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ads_llm_study.config import resolved_provenance
from ads_llm_study.credentials import resolve_api_key, resolve_base_url
from ads_llm_study.dataset import load_and_validate_dataset
from ads_llm_study.design import (
    derive_cell_seed,
    generate_configuration_smoke_rows,
    generate_evaluator_rows,
)
from ads_llm_study.io_utils import (
    append_jsonl,
    git_commit,
    git_dirty,
    iter_jsonl,
    read_json,
    sha256_object,
    write_json,
    write_jsonl,
)
from ads_llm_study.paths import DATA_DIR, REPO_ROOT, from_root
from ads_llm_study.prompts import build_messages, parse_verdict
from ads_llm_study.router import RouterClient, RouterError, extract_content, sanitize_metadata

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _request_payload(
    manifest: dict[str, Any],
    item: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    prompt = manifest["prompt"]
    messages = build_messages(
        item,
        from_root(prompt["system_path"]),
        from_root(prompt["template_path"]),
        prompt["exposed_item_fields"],
    )
    request = manifest["request"]
    return {
        "model": row["model"],
        "messages": messages,
        "temperature": request["temperature"],
        "top_p": request["top_p"],
        "max_tokens": request["max_tokens"],
        "seed": derive_cell_seed(row["row_seed"], item["id"]),
        "response_format": {"type": request["response_format"]},
    }


def _route_summary(routing: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "provider",
        "served_model_id",
        "served_by",
        "cost_usd",
        "policy_fingerprint",
    }
    return {key: value for key, value in routing.items() if key in keep}


def _validate_routing_contract(
    manifest: dict[str, Any],
    row: dict[str, Any],
    response: dict[str, Any],
) -> None:
    routing = response.get("x_router")
    if not isinstance(routing, dict) or not routing:
        if manifest["router"]["provider_drift_policy"] == "abort_on_mismatch":
            raise RouterError("required x_router trace is absent", status_code=422)
        return

    missing = [
        field
        for field in manifest["router"]["required_trace_fields"]
        if routing.get(field) in (None, "", {})
    ]
    if missing:
        raise RouterError(
            "x_router trace is missing required fields: " + ", ".join(sorted(missing)),
            status_code=422,
        )

    observed_family = str(routing["model_family"])
    if observed_family != row["configuration_id"]:
        raise RouterError(
            f"router served family {observed_family!r} for declared row "
            f"{row['configuration_id']!r}",
            status_code=422,
        )

    expected_fingerprint = manifest["router"]["expected_policy_fingerprint"]
    observed_fingerprint = str(routing["policy_fingerprint"])
    if expected_fingerprint and observed_fingerprint != expected_fingerprint:
        raise RouterError(
            "router policy fingerprint differs from the preregistration",
            status_code=422,
        )

    allowed_providers = manifest["router"]["allowed_providers"]
    observed_provider = str(routing["provider"])
    if allowed_providers and observed_provider not in allowed_providers:
        raise RouterError(
            f"router provider {observed_provider!r} is outside the preregistered population",
            status_code=422,
        )


def _record_from_response(
    *,
    manifest: dict[str, Any],
    run_id: str,
    item: dict[str, Any],
    row: dict[str, Any],
    payload: dict[str, Any],
    response: dict[str, Any] | None,
    attempts: int,
    started_at: str,
    elapsed_ms: float,
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_text = extract_content(response) if response is not None else None
    choices = response.get("choices") if response is not None else None
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    finish_reason = (
        sanitize_metadata(first_choice.get("finish_reason"))
        if isinstance(first_choice, dict)
        else None
    )
    verdict, confidence, reason = parse_verdict(raw_text)
    routing = sanitize_metadata(response.get("x_router", {})) if response else {}
    usage = sanitize_metadata(response.get("usage", {})) if response else {}

    if response is None:
        status = "transport_error_mapped_reject"
        verdict = None
        confidence = None
        reason = None
    elif manifest["router"]["require_routing_trace"] and not routing:
        status = "routing_trace_missing_mapped_reject"
        verdict = None
    elif raw_text is None or not raw_text.strip():
        status = "missing_mapped_reject"
        verdict = None
    elif verdict is None:
        status = "malformed_mapped_reject"
    else:
        status = "success"

    global_vote = 1 if status == "success" and verdict == "ACCEPT" else 0
    cell_id = f"{item['id']}::row-{row['row_index']:04d}"
    return {
        "schema_version": 1,
        "run_id": run_id,
        "cell_id": cell_id,
        "item_id": item["id"],
        "row_index": row["row_index"],
        "configuration_id": row["configuration_id"],
        "requested_model": row["model"],
        "row_seed": row["row_seed"],
        "cell_seed": payload["seed"],
        "request_sha256": sha256_object(payload),
        "status": status,
        "attempts": attempts,
        "started_at": started_at,
        "finished_at": utc_now(),
        "latency_ms": round(elapsed_ms, 3),
        "finish_reason": finish_reason,
        "raw_text": raw_text,
        "parsed_verdict": verdict,
        "parsed_confidence": confidence,
        "parsed_reason": reason,
        "global_vote": global_vote,
        "routing": routing
        if manifest["router"]["store_decision_trace"]
        else _route_summary(routing),
        "usage": usage,
        "error": sanitize_metadata(error),
    }


def _call_cell(
    client: RouterClient,
    manifest: dict[str, Any],
    run_id: str,
    item: dict[str, Any],
    row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _request_payload(manifest, item, row)
    started_at = utc_now()
    start = time.monotonic()
    request_config = manifest["request"]
    response: dict[str, Any] | None = None
    final_error: dict[str, Any] | None = None
    attempts = 0
    for attempts in range(1, int(request_config["max_transport_attempts"]) + 1):
        try:
            response = client.chat_completion(payload)
            _validate_routing_contract(manifest, row, response)
            final_error = None
            break
        except RouterError as exc:
            final_error = {
                "type": type(exc).__name__,
                "status_code": exc.status_code,
                "message": str(exc),
            }
            retryable = (
                exc.status_code is None or exc.status_code in request_config["retry_statuses"]
            )
            if exc.status_code is not None and 400 <= exc.status_code < 500 and not retryable:
                raise RouterError(
                    "fatal router request configuration/authentication failure: " + str(exc),
                    status_code=exc.status_code,
                ) from exc
            if not retryable or attempts >= int(request_config["max_transport_attempts"]):
                break
            time.sleep(float(request_config["retry_backoff_seconds"]))
    elapsed_ms = (time.monotonic() - start) * 1000
    record = _record_from_response(
        manifest=manifest,
        run_id=run_id,
        item=item,
        row=row,
        payload=payload,
        response=response,
        attempts=attempts,
        started_at=started_at,
        elapsed_ms=elapsed_ms,
        error=final_error,
    )
    ledger = {
        "run_id": run_id,
        "cell_id": record["cell_id"],
        "item_id": item["id"],
        "row_index": row["row_index"],
        "configuration_id": row["configuration_id"],
        "request_sha256": record["request_sha256"],
        "payload": payload,
    }
    return ledger, record


def run_experiment(
    manifest_path: Path,
    manifest: dict[str, Any],
    run_id: str,
    *,
    confirm_spend: bool,
    limit_items: int | None = None,
    limit_rows: int | None = None,
    one_row_per_configuration: bool = False,
    configuration_ids: list[str] | None = None,
) -> dict[str, Any]:
    if not confirm_spend:
        raise ValueError("live execution requires --confirm-spend")
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, numbers, dot, underscore, and dash")
    if one_row_per_configuration and limit_rows is not None:
        raise ValueError("--one-row-per-configuration cannot be combined with --limit-rows")
    if configuration_ids and not one_row_per_configuration:
        raise ValueError("--configuration-id requires --one-row-per-configuration")
    if configuration_ids and len(configuration_ids) != len(set(configuration_ids)):
        raise ValueError("--configuration-id values must be unique")
    full_manifest_design = (
        limit_items is None
        and limit_rows is None
        and not one_row_per_configuration
        and not configuration_ids
    )
    if (
        manifest["phase"] == "pilot"
        and full_manifest_design
        and os.environ.get("CONFIRM_LIVE_PILOT") != "YES"
    ):
        raise ValueError(
            "full pilot requires CONFIRM_LIVE_PILOT=YES in addition to --confirm-spend"
        )
    if (
        manifest["phase"] == "confirmatory"
        and os.environ.get("CONFIRM_LIVE_CONFIRMATORY") != "YES"
    ):
        raise ValueError(
            "confirmatory execution requires CONFIRM_LIVE_CONFIRMATORY=YES in addition "
            "to --confirm-spend"
        )

    items = load_and_validate_dataset(
        from_root(manifest["dataset"]["path"]), manifest["dataset"]["item_count"]
    )
    rows = generate_evaluator_rows(manifest)
    selected_configurations: list[dict[str, Any]] | None = None
    if one_row_per_configuration:
        configurations = manifest["evaluator_population"]["configurations"]
        if configuration_ids:
            known_configuration_ids = {configuration["id"] for configuration in configurations}
            unknown = sorted(set(configuration_ids) - known_configuration_ids)
            if unknown:
                raise ValueError("unknown --configuration-id values: " + ", ".join(unknown))
            requested_configuration_ids = set(configuration_ids)
            configurations = [
                configuration
                for configuration in configurations
                if configuration["id"] in requested_configuration_ids
            ]
        selected_configurations = configurations
        rows = generate_configuration_smoke_rows(
            manifest,
            selected_configurations,
        )
    if limit_items is not None:
        if limit_items < 1:
            raise ValueError("limit_items must be positive")
        items = items[:limit_items]
    if limit_rows is not None:
        if limit_rows < 1:
            raise ValueError("limit_rows must be positive")
        rows = rows[:limit_rows]

    run_dir = DATA_DIR / "runs" / run_id
    response_path = run_dir / "responses.jsonl"
    ledger_path = run_dir / "request-ledger.jsonl"
    vote_path = run_dir / "global-verdicts.jsonl"
    metadata_path = run_dir / "run.json"
    expected_cell_ids = {
        f"{item['id']}::row-{row['row_index']:04d}" for item in items for row in rows
    }

    base_url = resolve_base_url(manifest)
    api_key, credential_source = resolve_api_key(manifest)
    provenance = resolved_provenance(manifest_path, manifest)
    code_commit = git_commit(REPO_ROOT)
    code_dirty = git_dirty(REPO_ROOT)
    if manifest["phase"] == "confirmatory" and (not code_commit or code_dirty is not False):
        raise ValueError("confirmatory execution requires a clean committed Git worktree")
    existing_metadata = read_json(metadata_path) if metadata_path.exists() else None
    if (
        existing_metadata
        and existing_metadata["provenance"]["manifest_sha256"] != provenance["manifest_sha256"]
    ):
        raise ValueError("cannot resume run with a different manifest hash")

    completed_records = list(iter_jsonl(response_path)) if response_path.exists() else []
    completed = {record["cell_id"]: record for record in completed_records}
    unexpected = set(completed) - expected_cell_ids
    if unexpected:
        raise ValueError(
            f"run contains cells outside the selected design: {sorted(unexpected)[:5]}"
        )
    items_by_id = {item["id"]: item for item in items}
    rows_by_index = {row["row_index"]: row for row in rows}
    for record in completed.values():
        expected_payload = _request_payload(
            manifest,
            items_by_id[record["item_id"]],
            rows_by_index[record["row_index"]],
        )
        if record.get("request_sha256") != sha256_object(expected_payload):
            raise ValueError(
                f"cannot resume {record['cell_id']}: frozen request hash does not match"
            )

    configured_models = [
        entry["model"] for entry in manifest["evaluator_population"]["configurations"]
    ]
    timeout = float(
        os.environ.get("ADS_LLM_TIMEOUT_SECONDS", manifest["request"]["timeout_seconds"])
    )
    with RouterClient(base_url, api_key, timeout) as client:
        router_probe = client.probe(configured_models)
    unavailable = sorted(
        family
        for family, available in router_probe["configured_family_availability"].items()
        if not available
    )
    if unavailable:
        raise ValueError(
            "router catalog is missing preregistered evaluator families: "
            + ", ".join(unavailable)
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": manifest["phase"],
        "status": "running",
        "created_at": existing_metadata.get("created_at", utc_now())
        if existing_metadata
        else utc_now(),
        "updated_at": utc_now(),
        "base_url": base_url,
        "credential_source": credential_source,
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "provenance": provenance,
        "selected_item_ids": [item["id"] for item in items],
        "selected_row_indices": [row["row_index"] for row in rows],
        "design_mode": (
            "configuration_coverage_smoke"
            if one_row_per_configuration
            else "iid_full"
            if full_manifest_design
            else "iid_subset"
        ),
        "selected_configuration_ids": (
            [configuration["id"] for configuration in selected_configurations]
            if selected_configurations is not None
            else []
        ),
        "expected_cells": len(expected_cell_ids),
        "full_manifest_design": full_manifest_design,
        "router_probe": router_probe,
    }
    write_json(metadata_path, metadata)
    write_jsonl(run_dir / "evaluator-rows.jsonl", rows)
    write_json(run_dir / "resolved-manifest.json", manifest)

    pending = [
        (item, row)
        for item in items
        for row in rows
        if f"{item['id']}::row-{row['row_index']:04d}" not in completed
    ]
    print(
        f"Run {run_id}: {len(expected_cell_ids)} cells; "
        f"{len(completed)} complete; {len(pending)} live calls pending."
    )
    if pending:
        concurrency = min(int(manifest["request"]["concurrency"]), len(pending))
        configuration_concurrency = min(
            int(manifest["request"].get("max_concurrency_per_configuration", 1)),
            concurrency,
        )
        inter_call_delay = float(manifest["request"]["inter_call_delay_seconds"])
        with RouterClient(base_url, api_key, timeout) as client:
            executor = ThreadPoolExecutor(max_workers=concurrency)
            pending_queue = deque(pending)
            active_configurations: Counter[str] = Counter()
            futures: dict[
                Future[tuple[dict[str, Any], dict[str, Any]]], tuple[str, int, str]
            ] = {}

            def submit_available() -> bool:
                for _ in range(len(pending_queue)):
                    item, row = pending_queue.popleft()
                    configuration_id = row["configuration_id"]
                    if active_configurations[configuration_id] >= configuration_concurrency:
                        pending_queue.append((item, row))
                        continue
                    future = executor.submit(_call_cell, client, manifest, run_id, item, row)
                    futures[future] = (
                        item["id"],
                        row["row_index"],
                        configuration_id,
                    )
                    active_configurations[configuration_id] += 1
                    return True
                return False

            def fill_available_slots() -> None:
                while len(futures) < concurrency and submit_available():
                    pass

            fill_available_slots()
            try:
                done_count = 0
                while futures:
                    future = next(as_completed(tuple(futures)))
                    _, _, configuration_id = futures.pop(future)
                    active_configurations[configuration_id] -= 1
                    if active_configurations[configuration_id] == 0:
                        del active_configurations[configuration_id]
                    ledger, record = future.result()
                    append_jsonl(ledger_path, ledger)
                    append_jsonl(response_path, record)
                    append_jsonl(
                        vote_path,
                        {
                            "run_id": run_id,
                            "cell_id": record["cell_id"],
                            "item_id": record["item_id"],
                            "row_index": record["row_index"],
                            "global_vote": record["global_vote"],
                            "source_response_sha256": sha256_object(record),
                        },
                    )
                    completed[record["cell_id"]] = record
                    done_count += 1
                    if done_count == len(pending) or done_count % 25 == 0:
                        print(f"Completed {len(completed)}/{len(expected_cell_ids)} cells")
                    if inter_call_delay > 0 and concurrency == 1 and done_count < len(pending):
                        print(f"Pacing next evaluator call by {inter_call_delay:g}s")
                        time.sleep(inter_call_delay)
                    fill_available_slots()
            except BaseException as exc:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                partial_records = sorted(
                    completed.values(),
                    key=lambda value: (value["item_id"], value["row_index"]),
                )
                metadata.update(
                    {
                        "status": "aborted",
                        "updated_at": utc_now(),
                        "completed_cells": len(partial_records),
                        "status_counts": dict(
                            sorted(Counter(row["status"] for row in partial_records).items())
                        ),
                        "responses_sha256": sha256_object(partial_records),
                        "terminal_error": sanitize_metadata(
                            {"type": type(exc).__name__, "message": str(exc)}
                        ),
                    }
                )
                write_json(metadata_path, metadata)
                raise
            else:
                executor.shutdown(wait=True)

    records = sorted(
        completed.values(), key=lambda value: (value["item_id"], value["row_index"])
    )
    write_jsonl(response_path, records)
    ledgers = []
    for record in records:
        payload = _request_payload(
            manifest,
            items_by_id[record["item_id"]],
            rows_by_index[record["row_index"]],
        )
        request_sha256 = sha256_object(payload)
        if request_sha256 != record["request_sha256"]:
            raise ValueError(f"{record['cell_id']}: response/request ledger hash mismatch")
        ledgers.append(
            {
                "run_id": run_id,
                "cell_id": record["cell_id"],
                "item_id": record["item_id"],
                "row_index": record["row_index"],
                "configuration_id": record["configuration_id"],
                "request_sha256": request_sha256,
                "payload": payload,
            }
        )
    write_jsonl(ledger_path, ledgers)
    votes = [
        {
            "run_id": run_id,
            "cell_id": record["cell_id"],
            "item_id": record["item_id"],
            "row_index": record["row_index"],
            "global_vote": record["global_vote"],
            "source_response_sha256": sha256_object(record),
        }
        for record in records
    ]
    write_jsonl(vote_path, votes)

    statuses = Counter(record["status"] for record in records)
    metadata.update(
        {
            "status": "complete" if len(records) == len(expected_cell_ids) else "incomplete",
            "updated_at": utc_now(),
            "completed_cells": len(records),
            "status_counts": dict(sorted(statuses.items())),
            "responses_sha256": sha256_object(records),
        }
    )
    write_json(metadata_path, metadata)
    return metadata
