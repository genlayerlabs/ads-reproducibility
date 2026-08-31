from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import ads_llm_study.analysis as analysis_module
import ads_llm_study.runner as runner_module
import ads_llm_study.validation as validation_module
from ads_llm_study.config import load_manifest
from ads_llm_study.io_utils import iter_jsonl


class FakeRouterClient:
    chat_calls = 0
    probe_calls = 0

    def __init__(self, base_url: str, api_key: str, timeout_seconds: float):
        assert base_url == "https://router.ygr.ai/v1"
        assert api_key == "unit-test-consumer-key"
        assert timeout_seconds > 0

    def __enter__(self) -> FakeRouterClient:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def probe(self, configured_models: list[str]) -> dict[str, Any]:
        type(self).probe_calls += 1
        return {
            "base_url": "https://router.ygr.ai/v1",
            "health": {"ok": True},
            "catalog_model_count": len(configured_models),
            "configured_family_availability": {
                model.removeprefix("family:"): True for model in configured_models
            },
            "catalog_sample": configured_models,
            "policy_templates_available": True,
        }

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        type(self).chat_calls += 1
        verdict = "ACCEPT" if payload["seed"] % 2 == 0 else "REJECT"
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {"verdict": verdict, "confidence": 0.8, "reason": "fixture"}
                        )
                    },
                }
            ],
            "x_router": {
                "provider": "fixture-provider",
                "model_family": payload["model"].removeprefix("family:"),
                "served_model_id": payload["model"],
                "cost_usd": 0.0001,
                "policy_fingerprint": "fixture-policy-v1",
                "decision_trace": {
                    "route": "fixture",
                    "policy_fingerprint": "fixture-policy-v1",
                },
            },
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }


class MissingFamilyRouterClient(FakeRouterClient):
    def probe(self, configured_models: list[str]) -> dict[str, Any]:
        result = super().probe(configured_models)
        first_family = configured_models[0].removeprefix("family:")
        result["configured_family_availability"][first_family] = False
        return result


class ConcurrencyTrackingRouterClient(FakeRouterClient):
    lock = threading.Lock()
    active_total = 0
    max_active_total = 0
    active_by_family: Counter[str] = Counter()
    max_active_by_family: Counter[str] = Counter()

    @classmethod
    def reset(cls) -> None:
        with cls.lock:
            cls.active_total = 0
            cls.max_active_total = 0
            cls.active_by_family = Counter()
            cls.max_active_by_family = Counter()

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        family = payload["model"].removeprefix("family:")
        with type(self).lock:
            type(self).active_total += 1
            type(self).active_by_family[family] += 1
            type(self).max_active_total = max(
                type(self).max_active_total, type(self).active_total
            )
            type(self).max_active_by_family[family] = max(
                type(self).max_active_by_family[family],
                type(self).active_by_family[family],
            )
        try:
            time.sleep(0.01)
            return super().chat_completion(payload)
        finally:
            with type(self).lock:
                type(self).active_total -= 1
                type(self).active_by_family[family] -= 1


def _patch_runtime_dirs(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    data_dir = root / "data"
    results_dir = root / "results"
    monkeypatch.setattr(runner_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(validation_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(analysis_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(analysis_module, "RESULTS_DIR", results_dir)


def test_mocked_run_resume_validation_and_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, manifest = load_manifest("preregistration/pilot.yaml")
    _patch_runtime_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "unit-test-consumer-key")
    monkeypatch.setattr(runner_module, "RouterClient", FakeRouterClient)
    pacing_sleeps: list[float] = []
    monkeypatch.setattr(runner_module.time, "sleep", pacing_sleeps.append)
    FakeRouterClient.chat_calls = 0
    FakeRouterClient.probe_calls = 0

    metadata = runner_module.run_experiment(
        manifest_path,
        manifest,
        "mock-smoke",
        confirm_spend=True,
        limit_items=3,
        limit_rows=4,
    )
    assert metadata["status"] == "complete"
    assert metadata["expected_cells"] == 12
    assert metadata["completed_cells"] == 12
    assert FakeRouterClient.chat_calls == 12
    assert pacing_sleeps == []
    assert validation_module.validate_run("mock-smoke", manifest) == []

    output_dir = analysis_module.analyze_run(manifest_path, manifest, "mock-smoke")
    expected_outputs = {
        "boundary_diagnostics.csv",
        "clarity_by_stratum.png",
        "evaluator_rows.csv",
        "global_verdicts.csv",
        "operational_coverage.csv",
        "operational_coverage.png",
        "operational_coverage_table.tex",
        "per_item.csv",
        "pilot_overview.png",
        "provenance.json",
        "provider_summary.csv",
        "security_profiles.csv",
        "security_profiles.png",
        "summary.json",
    }
    assert expected_outputs <= {path.name for path in output_dir.iterdir()}

    resumed = runner_module.run_experiment(
        manifest_path,
        manifest,
        "mock-smoke",
        confirm_spend=True,
        limit_items=3,
        limit_rows=4,
    )
    assert resumed["completed_cells"] == 12
    assert FakeRouterClient.chat_calls == 12
    assert FakeRouterClient.probe_calls == 2


def test_live_spend_guards_fire_before_credentials_are_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = load_manifest("preregistration/pilot.yaml")
    monkeypatch.delenv("CONFIRM_LIVE_PILOT", raising=False)

    with pytest.raises(ValueError, match="--confirm-spend"):
        runner_module.run_experiment(
            manifest_path, manifest, "blocked", confirm_spend=False, limit_items=1
        )
    with pytest.raises(ValueError, match="CONFIRM_LIVE_PILOT=YES"):
        runner_module.run_experiment(
            manifest_path, manifest, "blocked-full", confirm_spend=True
        )


def test_targeted_family_smoke_selects_one_row_per_requested_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, manifest = load_manifest("preregistration/pilot.yaml")
    _patch_runtime_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "unit-test-consumer-key")
    monkeypatch.setattr(runner_module, "RouterClient", FakeRouterClient)
    monkeypatch.setattr(runner_module.time, "sleep", lambda _: None)

    requested = ["claude-sonnet-4-6", "gemini-3.1-pro-preview"]
    metadata = runner_module.run_experiment(
        manifest_path,
        manifest,
        "mock-family-retest",
        confirm_spend=True,
        limit_items=1,
        one_row_per_configuration=True,
        configuration_ids=requested,
    )

    assert metadata["expected_cells"] == 2
    rows = list(iter_jsonl(tmp_path / "data/runs/mock-family-retest/evaluator-rows.jsonl"))
    assert [row["configuration_id"] for row in rows] == requested
    responses = list(iter_jsonl(tmp_path / "data/runs/mock-family-retest/responses.jsonl"))
    assert {response["finish_reason"] for response in responses} == {"stop"}


def test_catalog_preflight_blocks_calls_when_a_family_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, manifest = load_manifest("preregistration/pilot.yaml")
    _patch_runtime_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "unit-test-consumer-key")
    monkeypatch.setattr(runner_module, "RouterClient", MissingFamilyRouterClient)
    MissingFamilyRouterClient.chat_calls = 0

    with pytest.raises(ValueError, match="missing preregistered evaluator families"):
        runner_module.run_experiment(
            manifest_path,
            manifest,
            "missing-family",
            confirm_spend=True,
            limit_items=1,
            limit_rows=1,
        )
    assert MissingFamilyRouterClient.chat_calls == 0
    assert not (tmp_path / "data" / "runs" / "missing-family").exists()


def test_scheduler_serializes_calls_within_each_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, manifest = load_manifest("preregistration/pilot.yaml")
    _patch_runtime_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "unit-test-consumer-key")
    monkeypatch.setattr(runner_module, "RouterClient", ConcurrencyTrackingRouterClient)
    ConcurrencyTrackingRouterClient.reset()

    metadata = runner_module.run_experiment(
        manifest_path,
        manifest,
        "mock-configuration-aware-scheduler",
        confirm_spend=True,
        limit_items=1,
    )

    assert metadata["completed_cells"] == 40
    assert ConcurrencyTrackingRouterClient.max_active_total == 3
    assert set(ConcurrencyTrackingRouterClient.max_active_by_family.values()) == {1}
