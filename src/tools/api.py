"""Convenience façade for market data provider access."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import time
import requests

import pandas as pd

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)
from src.data.providers import (
    DEFAULT_PROVIDER_NAME,
    get_provider_context,
    resolve_provider,
)

_cache = get_cache()


def _merge_credentials(
    provider_name: str,
    *,
    api_key: str | None = None,
    credentials: Dict[str, Any] | None = None,
) -> tuple[str, Dict[str, Any]]:
    context_provider, context_credentials = get_provider_context()
    active_provider = provider_name or context_provider or DEFAULT_PROVIDER_NAME
    merged: Dict[str, Any] = {}
    if context_credentials:
        merged.update(context_credentials)
    if credentials:
        merged.update(credentials)

    if api_key and "api_key" not in merged:
        merged["api_key"] = api_key

    return active_provider, merged


def _get_provider(
    *,
    provider: str | None = None,
    api_key: str | None = None,
    credentials: Dict[str, Any] | None = None,
):
    provider_name, merged_credentials = _merge_credentials(
        provider or DEFAULT_PROVIDER_NAME,
        api_key=api_key,
        credentials=credentials,
    )
    return resolve_provider(provider_name, credentials=merged_credentials), provider_name


def _make_api_request(
    url: str,
    headers: dict[str, str],
    method: str = "GET",
    json_data: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> requests.Response:
    """Backward-compatible helper for integration tests."""
    for attempt in range(max_retries + 1):
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)

        if response.status_code == 429 and attempt < max_retries:
            delay = 60 + (30 * attempt)
            time.sleep(delay)
            continue

        return response

    return response


def get_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    api_key: str | None = None,
    provider: str | None = None,
    credentials: Dict[str, Any] | None = None,
) -> list[Price]:
    provider_instance, provider_name = _get_provider(
        provider=provider,
        api_key=api_key,
        credentials=credentials,
    )
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached := _cache.get_prices(cache_key, source=provider_name):
        return [Price(**row) for row in cached]

    prices = provider_instance.get_prices(ticker, start_date, end_date)
    if prices:
        _cache.set_prices(cache_key, [p.model_dump() for p in prices], source=provider_name)
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    *,
    period: str = "ttm",
    limit: int = 10,
    api_key: str | None = None,
    provider: str | None = None,
    credentials: Dict[str, Any] | None = None,
) -> list[FinancialMetrics]:
    provider_instance, provider_name = _get_provider(
        provider=provider,
        api_key=api_key,
        credentials=credentials,
    )
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    if cached := _cache.get_financial_metrics(cache_key, source=provider_name):
        return [FinancialMetrics(**item) for item in cached]

    metrics = provider_instance.get_financial_metrics(ticker, end_date, period, limit)
    if metrics:
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics], source=provider_name)
    return metrics


def search_line_items(
    ticker: str,
    line_items: Iterable[str],
    *,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str | None = None,
    provider: str | None = None,
    credentials: Dict[str, Any] | None = None,
) -> list[LineItem]:
    provider_instance, provider_name = _get_provider(
        provider=provider,
        api_key=api_key,
        credentials=credentials,
    )
    cache_key = f"{ticker}_{period}_{end_date}_{'_'.join(sorted(line_items))}_{limit}"
    if cached := _cache.get_line_items(cache_key, source=provider_name):
        return [LineItem(**item) for item in cached]

    results = provider_instance.search_line_items(
        ticker,
        list(line_items),
        end_date,
        period,
        limit,
    )
    if results:
        _cache.set_line_items(cache_key, [item.model_dump() for item in results], source=provider_name)
    return results


def get_insider_trades(
    ticker: str,
    end_date: str,
    *,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str | None = None,
    provider: str | None = None,
    credentials: Dict[str, Any] | None = None,
) -> list[InsiderTrade]:
    provider_instance, provider_name = _get_provider(
        provider=provider,
        api_key=api_key,
        credentials=credentials,
    )
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached := _cache.get_insider_trades(cache_key, source=provider_name):
        return [InsiderTrade(**item) for item in cached]

    trades = provider_instance.get_insider_trades(ticker, end_date, start_date, limit)
    if trades:
        _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in trades], source=provider_name)
    return trades


def get_company_news(
    ticker: str,
    end_date: str,
    *,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str | None = None,
    provider: str | None = None,
    credentials: Dict[str, Any] | None = None,
) -> list[CompanyNews]:
    provider_instance, provider_name = _get_provider(
        provider=provider,
        api_key=api_key,
        credentials=credentials,
    )
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached := _cache.get_company_news(cache_key, source=provider_name):
        return [CompanyNews(**item) for item in cached]

    news = provider_instance.get_company_news(ticker, start_date, end_date, limit)
    if news:
        _cache.set_company_news(cache_key, [item.model_dump() for item in news], source=provider_name)
    return news


def get_market_cap(
    ticker: str,
    end_date: str,
    *,
    api_key: str | None = None,
    provider: str | None = None,
    credentials: Dict[str, Any] | None = None,
) -> float | None:
    provider_instance, _ = _get_provider(
        provider=provider,
        api_key=api_key,
        credentials=credentials,
    )
    return provider_instance.get_market_cap(ticker, end_date)


def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    api_key: str | None = None,
    provider: str | None = None,
    credentials: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    prices = get_prices(
        ticker,
        start_date,
        end_date,
        api_key=api_key,
        provider=provider,
        credentials=credentials,
    )
    return prices_to_df(prices)


def prices_to_df(prices: Iterable[Price]) -> pd.DataFrame:
    df = pd.DataFrame([p.model_dump() for p in prices])
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    for column in ["open", "close", "high", "low", "volume"]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df.sort_index(inplace=True)
    return df

