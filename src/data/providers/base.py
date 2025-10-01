"""Core abstractions and registry for market data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)


ProviderCredentials = Mapping[str, Any]


class AssetDataProvider(ABC):
    """Interface for providers that offer asset/market data."""

    name: str

    def __init__(self, *, credentials: ProviderCredentials | None = None):
        self.credentials: Dict[str, Any] = dict(credentials or {})

    @abstractmethod
    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Price]:
        raise NotImplementedError

    @abstractmethod
    def get_company_news(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str,
        limit: int,
    ) -> list[CompanyNews]:
        raise NotImplementedError

    @abstractmethod
    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        raise NotImplementedError


class FundamentalDataProvider(ABC):
    """Interface for providers that serve fundamentals and reference data."""

    name: str

    def __init__(self, *, credentials: ProviderCredentials | None = None):
        self.credentials: Dict[str, Any] = dict(credentials or {})

    @abstractmethod
    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str,
        limit: int,
    ) -> list[FinancialMetrics]:
        raise NotImplementedError

    @abstractmethod
    def search_line_items(
        self,
        ticker: str,
        line_items: list[str],
        end_date: str,
        period: str,
        limit: int,
    ) -> list[LineItem]:
        raise NotImplementedError

    @abstractmethod
    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None,
        limit: int,
    ) -> list[InsiderTrade]:
        raise NotImplementedError


class MarketDataProvider(AssetDataProvider, FundamentalDataProvider, ABC):
    """Concrete providers implement both asset and fundamental interfaces."""


_PROVIDER_REGISTRY: Dict[str, Callable[[ProviderCredentials | None], MarketDataProvider]] = {}


def register_provider(name: str, factory: Callable[[ProviderCredentials | None], MarketDataProvider]) -> None:
    """Register a data provider factory under the given name."""
    normalized = name.lower()
    _PROVIDER_REGISTRY[normalized] = factory


def get_provider(name: str, *, credentials: ProviderCredentials | None = None) -> MarketDataProvider:
    """Instantiate the provider registered under ``name``."""
    normalized = name.lower()
    if normalized not in _PROVIDER_REGISTRY:
        raise ValueError(f"Unknown data provider: {name}")
    factory = _PROVIDER_REGISTRY[normalized]
    return factory(credentials)


def has_provider(name: str) -> bool:
    return name.lower() in _PROVIDER_REGISTRY


_provider_context: MutableMapping[str, Any] = {}


def set_provider_context(provider_name: str | None, credentials: ProviderCredentials | None) -> None:
    if provider_name:
        _provider_context["provider"] = provider_name
    else:
        _provider_context.pop("provider", None)

    if credentials:
        _provider_context["credentials"] = dict(credentials)
    else:
        _provider_context.pop("credentials", None)


def get_provider_context() -> tuple[str | None, ProviderCredentials | None]:
    provider = _provider_context.get("provider")
    credentials = _provider_context.get("credentials")
    return provider, credentials


@contextmanager
def provider_context(provider_name: str | None, credentials: ProviderCredentials | None):
    previous = dict(_provider_context)
    try:
        set_provider_context(provider_name, credentials)
        yield
    finally:
        _provider_context.clear()
        _provider_context.update(previous)

