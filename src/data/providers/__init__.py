"""Data provider implementations and resolver utilities."""

from __future__ import annotations

from typing import Mapping

from .base import (
    AssetDataProvider,
    FundamentalDataProvider,
    MarketDataProvider,
    ProviderCredentials,
    get_provider,
    has_provider,
    get_provider_context,
    provider_context,
    register_provider,
    set_provider_context,
)

DEFAULT_PROVIDER_NAME = "financial_datasets"


def resolve_provider(
    provider_name: str | None,
    *,
    credentials: Mapping[str, object] | None = None,
) -> MarketDataProvider:
    name = provider_name or DEFAULT_PROVIDER_NAME
    return get_provider(name, credentials=credentials)


__all__ = [
    "AssetDataProvider",
    "FundamentalDataProvider",
    "MarketDataProvider",
    "ProviderCredentials",
    "DEFAULT_PROVIDER_NAME",
    "get_provider",
    "has_provider",
    "register_provider",
    "resolve_provider",
    "set_provider_context",
    "get_provider_context",
    "provider_context",
]

# Ensure built-in providers are registered on import
from . import financial_datasets as _financial_datasets  # noqa: F401
from . import ibbot as _ibbot  # noqa: F401


