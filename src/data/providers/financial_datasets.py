"""Financial Datasets API implementation of the market data provider."""

from __future__ import annotations

import datetime
import os
import time
from typing import Any

import requests

from src.data.cache import get_cache
from src.data.models import (
    CompanyFactsResponse,
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    InsiderTrade,
    InsiderTradeResponse,
    LineItem,
    LineItemResponse,
    Price,
    PriceResponse,
)

from . import register_provider
from .base import MarketDataProvider, ProviderCredentials


class FinancialDatasetsProvider(MarketDataProvider):
    name = "financial_datasets"

    def __init__(self, *, credentials: ProviderCredentials | None = None):
        super().__init__(credentials=credentials)
        self._cache = get_cache()
        self._api_key = self.credentials.get("api_key") or os.environ.get("FINANCIAL_DATASETS_API_KEY")

    @property
    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["X-API-KEY"] = str(self._api_key)
        return headers

    def _make_request(
        self,
        url: str,
        *,
        method: str = "GET",
        json_data: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> requests.Response:
        for attempt in range(max_retries + 1):
            if method.upper() == "POST":
                response = requests.post(url, headers=self._headers, json=json_data)
            else:
                response = requests.get(url, headers=self._headers)

            if response.status_code == 429 and attempt < max_retries:
                delay = 60 + (30 * attempt)
                time.sleep(delay)
                continue

            return response

        return response

    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Price]:
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cached := self._cache.get_prices(cache_key, source=self.name):
            return [Price(**row) for row in cached]

        url = (
            "https://api.financialdatasets.ai/prices/"
            f"?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
        )
        response = self._make_request(url)
        if response.status_code != 200:
            raise Exception(f"Error fetching prices for {ticker}: {response.status_code} - {response.text}")

        price_response = PriceResponse(**response.json())
        prices = price_response.prices
        if prices:
            self._cache.set_prices(cache_key, [p.model_dump() for p in prices], source=self.name)
        return prices

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str,
        limit: int,
    ) -> list[FinancialMetrics]:
        cache_key = f"{ticker}_{period}_{end_date}_{limit}"
        if cached := self._cache.get_financial_metrics(cache_key, source=self.name):
            return [FinancialMetrics(**metric) for metric in cached]

        url = (
            "https://api.financialdatasets.ai/financial-metrics/"
            f"?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
        )
        response = self._make_request(url)
        if response.status_code != 200:
            raise Exception(f"Error fetching financial metrics for {ticker}: {response.status_code} - {response.text}")

        metrics_response = FinancialMetricsResponse(**response.json())
        metrics = metrics_response.financial_metrics
        if metrics:
            self._cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics], source=self.name)
        return metrics

    def search_line_items(
        self,
        ticker: str,
        line_items: list[str],
        end_date: str,
        period: str,
        limit: int,
    ) -> list[LineItem]:
        cache_key = f"{ticker}_{period}_{end_date}_{'_'.join(sorted(line_items))}_{limit}"
        if cached := self._cache.get_line_items(cache_key, source=self.name):
            return [LineItem(**item) for item in cached]

        url = "https://api.financialdatasets.ai/financials/search/line-items"
        body = {
            "tickers": [ticker],
            "line_items": line_items,
            "end_date": end_date,
            "period": period,
            "limit": limit,
        }
        response = self._make_request(url, method="POST", json_data=body)
        if response.status_code != 200:
            raise Exception(f"Error searching line items for {ticker}: {response.status_code} - {response.text}")

        response_model = LineItemResponse(**response.json())
        results = response_model.search_results
        if results:
            self._cache.set_line_items(cache_key, [r.model_dump() for r in results], source=self.name)
        return results[:limit]

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None,
        limit: int,
    ) -> list[InsiderTrade]:
        cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
        if cached := self._cache.get_insider_trades(cache_key, source=self.name):
            return [InsiderTrade(**trade) for trade in cached]

        all_trades: list[InsiderTrade] = []
        current_end = end_date
        while True:
            url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end}&limit={limit}"
            if start_date:
                url += f"&filing_date_gte={start_date}"

            response = self._make_request(url)
            if response.status_code != 200:
                raise Exception(f"Error fetching insider trades for {ticker}: {response.status_code} - {response.text}")

            response_model = InsiderTradeResponse(**response.json())
            trades = response_model.insider_trades
            if not trades:
                break

            all_trades.extend(trades)

            if not start_date or len(trades) < limit:
                break

            current_end = min(t.filing_date.split("T")[0] for t in trades)
            if current_end <= start_date:
                break

        if all_trades:
            self._cache.set_insider_trades(cache_key, [t.model_dump() for t in all_trades], source=self.name)
        return all_trades

    def get_company_news(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str,
        limit: int,
    ) -> list[CompanyNews]:
        cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
        if cached := self._cache.get_company_news(cache_key, source=self.name):
            return [CompanyNews(**item) for item in cached]

        all_news: list[CompanyNews] = []
        current_end = end_date
        while True:
            url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end}&limit={limit}"
            if start_date:
                url += f"&start_date={start_date}"

            response = self._make_request(url)
            if response.status_code != 200:
                raise Exception(f"Error fetching news for {ticker}: {response.status_code} - {response.text}")

            response_model = CompanyNewsResponse(**response.json())
            news_batch = response_model.news
            if not news_batch:
                break

            all_news.extend(news_batch)

            if not start_date or len(news_batch) < limit:
                break

            current_end = min(item.date.split("T")[0] for item in news_batch)
            if current_end <= start_date:
                break

        if all_news:
            self._cache.set_company_news(cache_key, [n.model_dump() for n in all_news], source=self.name)
        return all_news

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
            response = self._make_request(f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}")
            if response.status_code != 200:
                return None
            facts = CompanyFactsResponse(**response.json())
            return facts.company_facts.market_cap

        metrics = self.get_financial_metrics(ticker, end_date, period="ttm", limit=1)
        if not metrics:
            return None
        return metrics[0].market_cap


def _factory(credentials: ProviderCredentials | None = None) -> FinancialDatasetsProvider:
    return FinancialDatasetsProvider(credentials=credentials)


register_provider(FinancialDatasetsProvider.name, _factory)

