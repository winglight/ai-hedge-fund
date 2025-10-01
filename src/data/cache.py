from typing import Any


class Cache:
    """In-memory cache for API responses."""

    def __init__(self):
        self._prices_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._financial_metrics_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._line_items_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._insider_trades_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._company_news_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def _get_namespace(self, cache: dict[str, dict[str, list[dict[str, Any]]]], source: str | None) -> dict[str, list[dict[str, Any]]]:
        """Return (and lazily create) the cache namespace for a provider source."""
        namespace = source or "__default__"
        if namespace not in cache:
            cache[namespace] = {}
        return cache[namespace]

    def _merge_data(self, existing: list[dict] | None, new_data: list[dict], key_field: str) -> list[dict]:
        """Merge existing and new data, avoiding duplicates based on a key field."""
        if not existing:
            return new_data

        # Create a set of existing keys for O(1) lookup
        existing_keys = {item[key_field] for item in existing}

        # Only add items that don't exist yet
        merged = existing.copy()
        merged.extend([item for item in new_data if item[key_field] not in existing_keys])
        return merged

    def get_prices(self, cache_key: str, *, source: str | None = None) -> list[dict[str, Any]] | None:
        """Get cached price data if available for a provider namespace."""
        namespace = self._get_namespace(self._prices_cache, source)
        return namespace.get(cache_key)

    def set_prices(self, cache_key: str, data: list[dict[str, Any]], *, source: str | None = None):
        """Append new price data to cache for a provider namespace."""
        namespace = self._get_namespace(self._prices_cache, source)
        namespace[cache_key] = self._merge_data(namespace.get(cache_key), data, key_field="time")

    def get_financial_metrics(self, cache_key: str, *, source: str | None = None) -> list[dict[str, Any]] | None:
        """Get cached financial metrics if available for a provider namespace."""
        namespace = self._get_namespace(self._financial_metrics_cache, source)
        return namespace.get(cache_key)

    def set_financial_metrics(self, cache_key: str, data: list[dict[str, Any]], *, source: str | None = None):
        """Append new financial metrics to cache for a provider namespace."""
        namespace = self._get_namespace(self._financial_metrics_cache, source)
        namespace[cache_key] = self._merge_data(namespace.get(cache_key), data, key_field="report_period")

    def get_line_items(self, cache_key: str, *, source: str | None = None) -> list[dict[str, Any]] | None:
        """Get cached line items if available for a provider namespace."""
        namespace = self._get_namespace(self._line_items_cache, source)
        return namespace.get(cache_key)

    def set_line_items(self, cache_key: str, data: list[dict[str, Any]], *, source: str | None = None):
        """Append new line items to cache for a provider namespace."""
        namespace = self._get_namespace(self._line_items_cache, source)
        namespace[cache_key] = self._merge_data(namespace.get(cache_key), data, key_field="report_period")

    def get_insider_trades(self, cache_key: str, *, source: str | None = None) -> list[dict[str, Any]] | None:
        """Get cached insider trades if available for a provider namespace."""
        namespace = self._get_namespace(self._insider_trades_cache, source)
        return namespace.get(cache_key)

    def set_insider_trades(self, cache_key: str, data: list[dict[str, Any]], *, source: str | None = None):
        """Append new insider trades to cache for a provider namespace."""
        namespace = self._get_namespace(self._insider_trades_cache, source)
        namespace[cache_key] = self._merge_data(namespace.get(cache_key), data, key_field="filing_date")  # Could also use transaction_date if preferred

    def get_company_news(self, cache_key: str, *, source: str | None = None) -> list[dict[str, Any]] | None:
        """Get cached company news if available for a provider namespace."""
        namespace = self._get_namespace(self._company_news_cache, source)
        return namespace.get(cache_key)

    def set_company_news(self, cache_key: str, data: list[dict[str, Any]], *, source: str | None = None):
        """Append new company news to cache for a provider namespace."""
        namespace = self._get_namespace(self._company_news_cache, source)
        namespace[cache_key] = self._merge_data(namespace.get(cache_key), data, key_field="date")


# Global cache instance
_cache = Cache()


def get_cache() -> Cache:
    """Get the global cache instance."""
    return _cache
