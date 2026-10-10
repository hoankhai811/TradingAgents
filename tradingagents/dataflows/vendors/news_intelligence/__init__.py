"""News Intelligence vendor package for TradingAgents dataflows.

Exposes news and global news fetching adapters conforming to the
canonical MarketNewsEvent contract from Repo 1 (trading-news-intelligence).
"""

from tradingagents.dataflows.vendors.news_intelligence.client import (
    NewsIntelligenceClient,
    filter_and_sanitize_event,
    format_article,
    get_global_news_news_intelligence,
    get_news_news_intelligence,
    map_ticker_for_news_intelligence,
    parse_utc_timestamp,
    to_iso_utc,
)

# Canonical aliases
get_news = get_news_news_intelligence
get_global_news = get_global_news_news_intelligence

__all__ = [
    "NewsIntelligenceClient",
    "filter_and_sanitize_event",
    "format_article",
    "get_global_news",
    "get_global_news_news_intelligence",
    "get_news",
    "get_news_news_intelligence",
    "map_ticker_for_news_intelligence",
    "parse_utc_timestamp",
    "to_iso_utc",
]
