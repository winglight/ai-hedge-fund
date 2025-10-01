

from typing import Any


def get_api_key_from_state(state: dict, api_key_name: str) -> str | None:
    """Get an API key from the state object."""
    if state and state.get("metadata", {}).get("request"):
        request = state["metadata"]["request"]
        if hasattr(request, "api_keys") and request.api_keys:
            return request.api_keys.get(api_key_name)
    return None


def get_provider_settings_from_state(state: dict) -> tuple[str | None, dict[str, Any]]:
    request = state.get("metadata", {}).get("request") if state else None
    if not request:
        return None, {}
    provider = getattr(request, "data_provider", None)
    options = dict(getattr(request, "data_provider_options", {}) or {})
    return provider, options