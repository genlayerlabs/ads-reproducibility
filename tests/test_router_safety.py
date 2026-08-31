from __future__ import annotations

from ads_llm_study.router import extract_content, sanitize_metadata


def test_metadata_sanitizer_redacts_credentials_but_not_token_counts() -> None:
    fake_consumer = "llmr_" + "unit_test_not_a_key"
    fake_provider = "sk-" + "unit_test_not_a_key"
    value = sanitize_metadata(
        {
            "api_key": fake_consumer,
            "message": f"Bearer {fake_consumer}",
            "tokens_in": 12,
            "nested": {"access_token": fake_provider},
        }
    )
    assert value["api_key"] == "[REDACTED]"
    assert "llmr_" not in value["message"]
    assert value["tokens_in"] == 12
    assert value["nested"]["access_token"] == "[REDACTED]"


def test_extract_content_supports_string_and_parts() -> None:
    assert extract_content({"choices": [{"message": {"content": "ok"}}]}) == "ok"
    assert (
        extract_content(
            {"choices": [{"message": {"content": [{"type": "text", "text": "ok"}]}}]}
        )
        == "ok"
    )
