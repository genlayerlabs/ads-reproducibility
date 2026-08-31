from __future__ import annotations

from pathlib import Path

import pytest

import ads_llm_study.credentials as credentials_module


def _manifest() -> dict:
    return {
        "router": {
            "api_key_env": "LLM_API_KEY",
            "api_key_env_aliases": ["UNHARDCODED_API_KEY"],
        }
    }


def test_repository_env_alias_is_loaded_without_exposing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(credentials_module, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("UNHARDCODED_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "UNHARDCODED_API_KEY=unit-test-consumer-key\n", encoding="utf-8"
    )

    key, source = credentials_module.resolve_api_key(_manifest())
    assert key == "unit-test-consumer-key"
    assert source == ".env:UNHARDCODED_API_KEY"


def test_process_environment_precedes_local_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(credentials_module, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "environment-key")
    (tmp_path / ".env.local").write_text("LLM_API_KEY=local-key\n", encoding="utf-8")

    key, source = credentials_module.resolve_api_key(_manifest())
    assert key == "environment-key"
    assert source == "environment:LLM_API_KEY"
