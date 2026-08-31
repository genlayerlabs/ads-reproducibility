from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from ads_llm_study.analysis import (
    analyze_run,
    publish_paper_artifacts,
    refresh_released_artifacts,
)
from ads_llm_study.config import load_manifest, manifest_hash
from ads_llm_study.credentials import resolve_api_key, resolve_base_url
from ads_llm_study.router import RouterClient
from ads_llm_study.runner import run_experiment
from ads_llm_study.validation import validate_repository

DEFAULT_MANIFEST = "preregistration/pilot.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ads-study",
        description="Run and analyze the ADS real-LLM resolvability study.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate offline study artifacts")
    validate.add_argument("--manifest", default=DEFAULT_MANIFEST)
    validate.add_argument("--run-id")

    digest = subparsers.add_parser("hash-manifest", help="print exact manifest SHA-256")
    digest.add_argument("--manifest", default=DEFAULT_MANIFEST)

    probe = subparsers.add_parser(
        "probe", help="probe router auth and catalog without inference"
    )
    probe.add_argument("--manifest", default=DEFAULT_MANIFEST)

    run = subparsers.add_parser("run", help="execute a guarded, resumable live matrix")
    run.add_argument("--manifest", default=DEFAULT_MANIFEST)
    run.add_argument("--run-id", required=True)
    run.add_argument("--limit-items", type=int)
    run.add_argument("--limit-rows", type=int)
    run.add_argument("--one-row-per-configuration", action="store_true")
    run.add_argument("--configuration-id", action="append", dest="configuration_ids")
    run.add_argument("--confirm-spend", action="store_true")

    analyze = subparsers.add_parser("analyze", help="analyze one complete run matrix")
    analyze.add_argument("--manifest", default=DEFAULT_MANIFEST)
    analyze.add_argument("--run-id", required=True)

    artifacts = subparsers.add_parser(
        "paper-artifacts", help="regenerate and stage publication artifacts"
    )
    artifacts.add_argument("--manifest", default=DEFAULT_MANIFEST)
    artifacts.add_argument("--run-id", required=True)

    refresh = subparsers.add_parser(
        "refresh-release",
        help="recompute binary-vote diagnostics from the sanitized release",
    )
    refresh.add_argument("--manifest", default=DEFAULT_MANIFEST)
    return parser


def _timeout(manifest: dict[str, Any]) -> float:
    override = os.environ.get("ADS_LLM_TIMEOUT_SECONDS")
    return float(override) if override else float(manifest["request"]["timeout_seconds"])


def _main(args: argparse.Namespace) -> None:
    if args.command == "validate":
        result = validate_repository(args.manifest, args.run_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "hash-manifest":
        print(manifest_hash(args.manifest))
        return

    manifest_path, manifest = load_manifest(args.manifest)
    if args.command == "probe":
        base_url = resolve_base_url(manifest)
        api_key, source = resolve_api_key(manifest)
        models = [
            entry["model"] for entry in manifest["evaluator_population"]["configurations"]
        ]
        with RouterClient(base_url, api_key, _timeout(manifest)) as client:
            result = client.probe(models)
        result["credential_source"] = source
        result["credential_present"] = True
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "run":
        result = run_experiment(
            manifest_path,
            manifest,
            args.run_id,
            confirm_spend=args.confirm_spend,
            limit_items=args.limit_items,
            limit_rows=args.limit_rows,
            one_row_per_configuration=args.one_row_per_configuration,
            configuration_ids=args.configuration_ids,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "analyze":
        output = analyze_run(manifest_path, manifest, args.run_id)
        print(output)
        return
    if args.command == "paper-artifacts":
        output = publish_paper_artifacts(manifest_path, manifest, args.run_id)
        print(output)
        return
    if args.command == "refresh-release":
        output = refresh_released_artifacts(manifest)
        print(output)
        return
    raise AssertionError(f"unhandled command: {args.command}")


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        _main(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
