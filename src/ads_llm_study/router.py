from __future__ import annotations

import re
from typing import Any

import httpx


class RouterError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


_SECRET_FIELD_NAMES = {
    "api_key",
    "authorization",
    "bearer",
    "access_token",
    "refresh_token",
    "provider_key",
    "secret",
}
_SECRET_VALUE = re.compile(r"(?:llmr_|sk-[A-Za-z0-9_-]*|Bearer\s+)[A-Za-z0-9_.-]+")


def sanitize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in _SECRET_FIELD_NAMES else sanitize_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_metadata(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED]", value)
    return value


def extract_content(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(parts) or None
    return None


class RouterClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: float):
        self.base_url = base_url.rstrip("/")
        self.root_url = self.base_url[: -len("/v1")]
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Authorization": f"Bearer {api_key}"},
            follow_redirects=True,
        )

    def __enter__(self) -> RouterClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _json_response(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            detail = sanitize_metadata(response.text[:1000])
            raise RouterError(
                f"router returned HTTP {response.status_code}: {detail}",
                status_code=response.status_code,
            )
        try:
            value = response.json()
        except ValueError as exc:
            raise RouterError(
                "router returned a non-JSON response", response.status_code
            ) from exc
        if not isinstance(value, dict):
            raise RouterError("router JSON response is not an object", response.status_code)
        return value

    def probe(self, configured_models: list[str]) -> dict[str, Any]:
        try:
            health_response = httpx.get(
                f"{self.root_url}/healthz", timeout=10, follow_redirects=True
            )
            health = self._json_response(health_response)
            models = self._json_response(self._client.get(f"{self.base_url}/models"))
        except RouterError:
            raise
        except httpx.HTTPError as exc:
            raise RouterError(
                f"router probe transport failure: {type(exc).__name__}: {exc}"
            ) from exc
        data = models.get("data", [])
        model_ids = sorted(
            str(item.get("id"))
            for item in data
            if isinstance(item, dict) and item.get("id") is not None
        )
        configured_families = [model.removeprefix("family:") for model in configured_models]
        availability = {
            family: any(
                model_id == family
                or model_id == f"family:{family}"
                or model_id.endswith(f"/{family}")
                for model_id in model_ids
            )
            for family in configured_families
        }
        templates: dict[str, Any] | None = None
        try:
            templates = self._json_response(
                self._client.get(f"{self.root_url}/x/policy/templates")
            )
        except (RouterError, httpx.HTTPError):
            templates = None
        return sanitize_metadata(
            {
                "base_url": self.base_url,
                "health": health,
                "catalog_model_count": len(model_ids),
                "configured_family_availability": availability,
                "catalog_sample": model_ids[:25],
                "policy_templates_available": templates is not None,
            }
        )

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise RouterError(f"router transport failure: {type(exc).__name__}: {exc}") from exc
        return self._json_response(response)
