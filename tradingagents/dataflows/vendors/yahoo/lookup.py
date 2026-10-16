"""Yahoo quote search and read-child cache setup for the service boundary."""

from pathlib import Path

import yfinance as yf

from tradingagents.dataflows.errors import VendorUnavailableError


def set_cache_location(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(path))
    except Exception as exc:
        raise VendorUnavailableError("Yahoo cache setup failed") from exc


def search_quotes(query: str) -> list[dict]:
    try:
        quotes = yf.Search(
            query,
            max_results=8,
            news_count=0,
            lists_count=0,
            recommended=0,
            include_cb=False,
            include_nav_links=False,
            include_research=False,
            include_cultural_assets=False,
            enable_fuzzy_query=False,
            timeout=10,
            raise_errors=True,
        ).quotes
        if not isinstance(quotes, list):
            raise VendorUnavailableError("Yahoo returned malformed quotes")
        return quotes
    except VendorUnavailableError:
        raise
    except Exception as exc:
        raise VendorUnavailableError("Yahoo search failed") from exc
