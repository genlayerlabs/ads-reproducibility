from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ads_llm_study.paths import REPO_ROOT


class CredentialError(RuntimeError):
    pass


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        values[name] = value.strip().strip('"').strip("'")
    return values


def resolve_base_url(manifest: dict[str, Any]) -> str:
    value = os.environ.get("LLM_BASE_URL") or manifest["router"]["base_url"]
    normalized = value.rstrip("/")
    if not normalized.endswith("/v1"):
        raise CredentialError("LLM_BASE_URL must end in /v1")
    return normalized


def resolve_api_key(manifest: dict[str, Any]) -> tuple[str, str]:
    router = manifest["router"]
    env_names = [router["api_key_env"], *router.get("api_key_env_aliases", [])]
    for env_name in env_names:
        if value := os.environ.get(env_name):
            return value, f"environment:{env_name}"

    for filename in (".env.local", ".env"):
        local_env = _read_env_file(REPO_ROOT / filename)
        for env_name in env_names:
            if value := local_env.get(env_name):
                return value, f"{filename}:{env_name}"

    fallback = manifest["router"].get("local_development_fallback")
    if fallback:
        path = (REPO_ROOT / fallback["path"]).resolve()
        if value := _read_env_file(path).get(fallback["variable"]):
            return value, f"local-fallback:{path.name}:{fallback['variable']}"

    raise CredentialError(
        "No router consumer key found. Export one of "
        + ", ".join(env_names)
        + " or configure the local fallback."
    )
