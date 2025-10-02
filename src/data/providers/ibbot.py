"""Ibbot market data provider implementation."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable

import requests

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    InsiderTrade,
    InsiderTradeResponse,
    LineItem,
    LineItemResponse,
    Price,
)

from . import register_provider
from .base import MarketDataProvider, ProviderCredentials


class IbbotDataProvider(MarketDataProvider):
    name = "ibbot"

    def __init__(self, *, credentials: ProviderCredentials | None = None):
        super().__init__(credentials=credentials)
        self._cache = get_cache()
        self.host = self._resolve_credential("host", env="IBBOT_HOST")
        self.account = self._resolve_credential("account", env="IBBOT_ACCOUNT")
        self.access_token = self._resolve_credential("access_token", env="IBBOT_ACCESS_TOKEN")
        self.refresh_token = self._resolve_credential("refresh_token", env="IBBOT_REFRESH_TOKEN", required=False)
        self._contracts: Dict[str, str] = {}

        if not self.host or not self.account or not self.access_token:
            raise ValueError("Ibbot provider requires host, account, and access token credentials")

    def _resolve_credential(self, key: str, *, env: str, required: bool = True) -> str | None:
        if key in self.credentials and self.credentials[key]:
            return str(self.credentials[key])
        env_value = os.environ.get(env)
        if env_value:
            return env_value
        if required:
            raise ValueError(f"Missing required Ibbot credential: {key}")
        return None

    @property
    def _base_url(self) -> str:
        host = str(self.host).strip().rstrip("/")
        if not host:
            raise ValueError("Ibbot host is empty")

        if "://" not in host:
            normalized_host = host.lstrip("/")
            if normalized_host.startswith(("localhost", "127.0.0.1", "0.0.0.0")):
                host = f"http://{normalized_host}"
            else:
                host = f"https://{normalized_host}"

        return host

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "X-Ibbot-Account": self.account,
            "Content-Type": "application/json",
        }

    def _refresh_access_token(self) -> None:
        if not self.refresh_token:
            raise Exception("Ibbot access token expired and no refresh token provided")
        url = f"{self._base_url}/auth/token"
        response = requests.post(
            url,
            json={"account": self.account, "refresh_token": self.refresh_token},
            timeout=30,
        )
        if response.status_code != 200:
            raise Exception(f"Failed to refresh Ibbot token: {response.status_code} - {response.text}")
        payload = response.json()
        new_token = payload.get("access_token") or payload.get("token")
        if not new_token:
            raise Exception("Ibbot token refresh response missing access_token")
        self.access_token = new_token

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        headers = self._auth_headers()
        response = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_body,
            timeout=30,
        )

        if response.status_code == 401 and self.refresh_token:
            self._refresh_access_token()
            headers = self._auth_headers()
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=30,
            )

        if response.status_code >= 400:
            raise Exception(f"Ibbot request failed ({response.status_code}): {response.text}")

        if not response.content:
            return {}

        return response.json()

    def _ensure_contract(self, ticker: str) -> str:
        normalized = ticker.upper()
        if normalized in self._contracts:
            return self._contracts[normalized]

        payload = {"symbol": normalized, "account": self.account}
        data = self._request("POST", "/v1/contracts/search", json_body=payload)
        contracts: Iterable[dict[str, Any]] = data.get("contracts") or data.get("data") or []
        for contract in contracts:
            symbol = contract.get("symbol") or contract.get("ticker")
            if symbol and symbol.upper() == normalized:
                contract_id = str(contract.get("id") or contract.get("contract_id") or contract.get("conid"))
                if contract_id:
                    self._contracts[normalized] = contract_id
                    return contract_id

        raise Exception(f"Unable to resolve Ibbot contract for ticker {ticker}")

    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Price]:
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cached := self._cache.get_prices(cache_key, source=self.name):
            return [Price(**row) for row in cached]

        contract_id = self._ensure_contract(ticker)
        params = {"start": start_date, "end": end_date, "interval": "1d"}
        data = self._request("GET", f"/v1/market-data/{contract_id}/prices", params=params)
        raw_prices: Iterable[dict[str, Any]] = data.get("prices") or data.get("data") or []
        prices = [self._normalize_price(item, ticker) for item in raw_prices]
        if prices:
            self._cache.set_prices(cache_key, [p.model_dump() for p in prices], source=self.name)
        return prices

    def get_intraday_prices(
        self,
        ticker: str,
        start: str,
        end: str,
        *,
        interval: str = "minute",
        interval_multiplier: int = 5,
    ) -> list[Price]:
        cache_key = f"intraday_{ticker}_{interval}_{interval_multiplier}_{start}_{end}"
        if cached := self._cache.get_prices(cache_key, source=self.name):
            return [Price(**row) for row in cached]

        contract_id = self._ensure_contract(ticker)
        normalized_interval = interval.lower()
        if normalized_interval.startswith("min"):
            suffix = "m"
        elif normalized_interval.startswith("hour") or normalized_interval.startswith("hr"):
            suffix = "h"
        else:
            suffix = normalized_interval[:1]
        interval_code = f"{interval_multiplier}{suffix}"

        params = {"start": start, "end": end, "interval": interval_code}
        data = self._request("GET", f"/v1/market-data/{contract_id}/prices", params=params)
        raw_prices: Iterable[dict[str, Any]] = data.get("prices") or data.get("data") or []
        prices = [self._normalize_price(item, ticker) for item in raw_prices]
        if prices:
            self._cache.set_prices(cache_key, [p.model_dump() for p in prices], source=self.name)
        return prices

    def _normalize_price(self, payload: dict[str, Any], ticker: str) -> Price:
        return Price(
            ticker=ticker,
            time=str(payload.get("time") or payload.get("timestamp") or payload.get("date")),
            open=float(payload.get("open", 0)),
            high=float(payload.get("high", 0)),
            low=float(payload.get("low", 0)),
            close=float(payload.get("close", payload.get("price", 0))),
            volume=float(payload.get("volume", payload.get("vol", 0) or 0)),
        )

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

        contract_id = self._ensure_contract(ticker)
        params = {"end": end_date, "period": period, "limit": limit}
        data = self._request("GET", f"/v1/fundamentals/{contract_id}/metrics", params=params)
        raw_metrics: Iterable[dict[str, Any]] = data.get("financial_metrics") or data.get("data") or []
        response = FinancialMetricsResponse(financial_metrics=[self._normalize_metric(m, ticker) for m in raw_metrics])
        metrics = response.financial_metrics
        if metrics:
            self._cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics], source=self.name)
        return metrics

    def _normalize_metric(self, payload: dict[str, Any], ticker: str) -> dict[str, Any]:
        normalized = dict(payload)
        normalized.setdefault("ticker", ticker)
        return normalized

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

        contract_id = self._ensure_contract(ticker)
        payload = {
            "line_items": line_items,
            "end": end_date,
            "period": period,
            "limit": limit,
        }
        data = self._request("POST", f"/v1/fundamentals/{contract_id}/line-items:search", json_body=payload)
        results = data.get("results") or data.get("data") or []
        response = LineItemResponse(search_results=[LineItem(**item) for item in results])
        items = response.search_results
        if items:
            self._cache.set_line_items(cache_key, [i.model_dump() for i in items], source=self.name)
        return items[:limit]

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

        contract_id = self._ensure_contract(ticker)
        params = {"end": end_date, "limit": limit}
        if start_date:
            params["start"] = start_date

        data = self._request("GET", f"/v1/fundamentals/{contract_id}/insider-trades", params=params)
        trades = data.get("insider_trades") or data.get("data") or []
        response = InsiderTradeResponse(insider_trades=[InsiderTrade(**trade) for trade in trades])
        results = response.insider_trades
        if results:
            self._cache.set_insider_trades(cache_key, [t.model_dump() for t in results], source=self.name)
        return results

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

        contract_id = self._ensure_contract(ticker)
        params = {"end": end_date, "limit": limit}
        if start_date:
            params["start"] = start_date

        data = self._request("GET", f"/v1/news/{contract_id}", params=params)
        news_payload = data.get("news") or data.get("data") or []
        response = CompanyNewsResponse(news=[CompanyNews(**item) for item in news_payload])
        news_items = response.news
        if news_items:
            self._cache.set_company_news(cache_key, [n.model_dump() for n in news_items], source=self.name)
        return news_items

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        metrics = self.get_financial_metrics(ticker, end_date, period="ttm", limit=1)
        if not metrics:
            return None
        return metrics[0].market_cap


def _factory(credentials: ProviderCredentials | None = None) -> IbbotDataProvider:
    return IbbotDataProvider(credentials=credentials)


register_provider(IbbotDataProvider.name, _factory)

