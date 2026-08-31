from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any


class PromptError(ValueError):
    pass


def template_fields(template: str) -> set[str]:
    return {
        field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name
    }


def build_messages(
    item: dict[str, Any],
    system_path: Path,
    template_path: Path,
    exposed_fields: list[str],
) -> list[dict[str, str]]:
    system = system_path.read_text(encoding="utf-8").strip()
    template = template_path.read_text(encoding="utf-8")
    fields = template_fields(template)
    exposed = set(exposed_fields)
    if not fields <= exposed:
        raise PromptError(f"template references non-exposed fields: {sorted(fields - exposed)}")
    values = {field: str(item[field]) for field in fields}
    user = template.format(**values).strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_verdict(raw_text: str | None) -> tuple[str | None, float | None, str | None]:
    if raw_text is None or not raw_text.strip():
        return None, None, None
    text = raw_text.strip()
    candidates = [text]
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]))
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            candidates.append(json.dumps(value))
            break
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(value, dict):
            continue
        verdict = str(value.get("verdict", "")).upper()
        if verdict not in {"ACCEPT", "REJECT"}:
            continue
        confidence = value.get("confidence")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= float(confidence) <= 1
        ):
            confidence = None
        else:
            confidence = float(confidence)
        reason = value.get("reason")
        if reason is not None and not isinstance(reason, str):
            reason = str(reason)
        return verdict, confidence, reason
    return None, None, None
