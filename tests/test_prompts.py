from __future__ import annotations

import json

from ads_llm_study.config import load_manifest
from ads_llm_study.dataset import load_and_validate_dataset
from ads_llm_study.paths import from_root
from ads_llm_study.prompts import build_messages, parse_verdict


def test_prompt_exposes_only_predeclared_fields() -> None:
    _, manifest = load_manifest("preregistration/pilot.yaml")
    items = load_and_validate_dataset(
        from_root(manifest["dataset"]["path"]), manifest["dataset"]["item_count"]
    )
    prompt = manifest["prompt"]
    messages = build_messages(
        items[0],
        from_root(prompt["system_path"]),
        from_root(prompt["template_path"]),
        prompt["exposed_item_fields"],
    )
    rendered = json.dumps(messages)
    for field in prompt["forbidden_item_fields"]:
        assert f'"{field}"' not in rendered
    assert items[0]["notes"] not in messages[1]["content"]


def test_parse_exact_and_fenced_json() -> None:
    assert parse_verdict('{"verdict":"ACCEPT","confidence":0.8,"reason":"rule"}') == (
        "ACCEPT",
        0.8,
        "rule",
    )
    assert parse_verdict('```json\n{"verdict":"REJECT","confidence":1,"reason":"x"}\n```') == (
        "REJECT",
        1.0,
        "x",
    )


def test_malformed_output_has_no_vote() -> None:
    assert parse_verdict("I think accept") == (None, None, None)
    assert parse_verdict('{"verdict":"ABSTAIN"}') == (None, None, None)
