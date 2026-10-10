"""News Intelligence vendor client and data-fetching adapters.

Integrates Repo 1 (trading-news-intelligence) into TradingAgents dataflows
through the router extension point, conforming to the canonical MarketNewsEvent
contract with strict bi-temporal point-in-time protections and ticker mapping.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import (
    VendorNotConfiguredError,
    VendorUnavailableError,
)
from tradingagents.dataflows.symbols import _CRYPTO_BASES, crypto_base, normalize_symbol

logger = logging.getLogger(__name__)


def map_ticker_for_news_intelligence(ticker: str) -> tuple[str, str]:
    """Map a user-supplied ticker/instrument to canonical query symbol and display resolution.

    Handles:
    - Crypto pairs (BTC-USD, BTCUSD, BTCUSDT -> BTC)
    - Bare crypto tickers (BTC, ETH, SOL -> BTC, ETH, SOL)
    - Stock symbols (AAPL, MSFT, NVDA -> AAPL, MSFT, NVDA)
    - Foreign listings / exchange suffixes (700.HK -> 0700.HK)

    Returns:
        tuple[str, str]: (query_symbol, resolved_annotation)
    """
    if not isinstance(ticker, str):
        return "", ""
    raw = ticker.strip()
    compact = raw.upper().rstrip("+").replace("-", "")

    # Bare crypto base directly
    if compact in _CRYPTO_BASES:
        query_symbol = compact
    else:
        # USD/USDT/USDC quoted crypto
        base = crypto_base(raw)
        query_symbol = base or normalize_symbol(raw)

    resolved = "" if query_symbol == raw else f" (resolved to {query_symbol})"
    return query_symbol, resolved


def to_iso_utc(date_str: str, end_of_day: bool = False) -> str:
    """Format a date or date-time string into canonical UTC ISO 8601 representation.

    Plain YYYY-MM-DD dates are expanded:
    - start of day (end_of_day=False): YYYY-MM-DDT00:00:00Z
    - end of day (end_of_day=True): YYYY-MM-DDT23:59:59Z
    """
    if not date_str or not isinstance(date_str, str):
        return ""
    clean = date_str.strip()
    if "T" in clean:
        if clean.endswith("Z") or "+" in clean[-6:] or "-" in clean[-6:]:
            return clean
        return f"{clean}Z"
    if end_of_day:
        return f"{clean}T23:59:59Z"
    return f"{clean}T00:00:00Z"


def parse_utc_timestamp(ts: str | None) -> datetime | None:
    """Parse an ISO 8601 UTC timestamp string with up to microsecond precision."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        clean = ts.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def filter_and_sanitize_event(
    event: dict[str, Any],
    cutoff_dt: datetime,
    min_pub_dt: datetime | None = None,
    max_pub_dt: datetime | None = None,
) -> dict[str, Any] | None:
    """Validate event and enforce bi-temporal point-in-time protections.

    Rules:
    1. Mandatory fields: 'publishedAt' and 'receivedAt' must be valid UTC timestamps.
    2. Article Availability:
       - publishedAt <= cutoff AND receivedAt <= cutoff.
       - If publishedAt > cutoff: REJECT (future publication).
       - If receivedAt > cutoff: REJECT (future ingestion / late receipt lookahead).
    3. Publication Window:
       - If min_pub_dt is provided: publishedAt >= min_pub_dt.
       - If max_pub_dt is provided: publishedAt <= max_pub_dt.
    4. Derived Fact Availability:
       - enrichedAt <= cutoff.
       - If enrichedAt is None or enrichedAt > cutoff:
         WITHHOLD all derived intelligence facts (sentiment, computed summary,
         clusterId, enrichedAt) to eliminate future-enrichment leakage.

    Returns:
        Sanitized event dict, or None if rejected.
    """
    if not isinstance(event, dict):
        return None

    pub_at_str = event.get("publishedAt")
    rec_at_str = event.get("receivedAt")

    pub_dt = parse_utc_timestamp(pub_at_str)
    rec_dt = parse_utc_timestamp(rec_at_str)

    if pub_dt is None or rec_dt is None:
        logger.warning("Rejecting event with invalid timestamps: %r", event.get("id"))
        return None

    # Bi-temporal availability check: both publication and receipt must be on or before cutoff
    if pub_dt > cutoff_dt:
        logger.debug("Withholding future publication: %s > cutoff %s", pub_dt, cutoff_dt)
        return None

    if rec_dt > cutoff_dt:
        logger.debug("Withholding late-received article: %s > cutoff %s", rec_dt, cutoff_dt)
        return None

    # Publication window check
    if min_pub_dt is not None and pub_dt < min_pub_dt:
        return None
    if max_pub_dt is not None and pub_dt > max_pub_dt:
        return None

    sanitized = dict(event)

    # Derived fact availability check:
    enr_at_str = event.get("enrichedAt")
    enr_dt = parse_utc_timestamp(enr_at_str)

    if enr_dt is None or enr_dt > cutoff_dt:
        # Withhold derived intelligence facts
        sanitized["sentiment"] = None
        sanitized["clusterId"] = None
        sanitized["summary"] = None
        sanitized["enrichedAt"] = None

    return sanitized


def format_article(event: dict[str, Any]) -> str:
    """Format a sanitized MarketNewsEvent into markdown for News Analyst consumption."""
    title = event.get("title") or "No title"
    source = event.get("source") or "News Intelligence"
    out = f"### {title} (source: {source})\n"

    sentiment = event.get("sentiment")
    if sentiment and isinstance(sentiment, dict):
        label = sentiment.get("label")
        score = sentiment.get("score")
        if label and score is not None:
            out += f"Sentiment: {label} (score: {score:+.2f})\n"
        elif label:
            out += f"Sentiment: {label}\n"
        elif score is not None:
            out += f"Sentiment score: {score:+.2f}\n"

    # Prefer summary when available (and permitted by cutoff), else content
    body = event.get("summary") or event.get("content") or ""
    if body:
        out += f"{body.strip()}\n"

    url = event.get("url")
    if url:
        out += f"Link: {url}\n"

    return out


class NewsIntelligenceClient:
    """Client for querying the News Intelligence platform REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
        session: requests.Session | None = None,
    ):
        config = get_config()
        raw_url = base_url or config.get("news_intelligence_url") or "http://localhost:3001"
        clean_url = raw_url.strip().rstrip("/")
        if clean_url.endswith("/api/v1"):
            clean_url = clean_url[:-len("/api/v1")]
        elif clean_url.endswith("/v1"):
            clean_url = clean_url[:-len("/v1")]

        if not clean_url:
            raise VendorNotConfiguredError("News Intelligence API URL is not configured")

        self.base_url = clean_url
        self.token = token if token is not None else config.get("news_intelligence_token")
        self.timeout = float(timeout if timeout is not None else config.get("news_intelligence_timeout", 10.0))
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "TradingAgents/NewsIntelligenceVendor",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _make_request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            resp = self.session.get(url, params=params, headers=self._headers(), timeout=self.timeout)
        except requests.exceptions.Timeout as exc:
            raise VendorUnavailableError(f"News Intelligence API timed out after {self.timeout}s: {exc}") from exc
        except (requests.exceptions.ConnectionError, requests.exceptions.RequestException) as exc:
            raise VendorUnavailableError(f"News Intelligence API connection error: {exc}") from exc

        if resp.status_code == 404:
            return {"events": [], "count": 0}

        if resp.status_code >= 500:
            raise VendorUnavailableError(f"News Intelligence API server error ({resp.status_code}): {resp.text}")

        if not resp.ok:
            raise VendorUnavailableError(f"News Intelligence API error ({resp.status_code}): {resp.text}")

        try:
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")
            return data
        except Exception as exc:
            raise VendorUnavailableError(f"News Intelligence API invalid JSON: {exc}") from exc

    def get_news(self, ticker: str, start_date: str, end_date: str) -> str:
        """Fetch and format asset-specific news with bi-temporal point-in-time protections."""
        query_symbol, resolved = map_ticker_for_news_intelligence(ticker)
        if not query_symbol:
            return f"No news found for {ticker} between {start_date} and {end_date}"

        config = get_config()
        limit = config.get("news_article_limit", 20)

        as_of = to_iso_utc(end_date, end_of_day=True)
        min_pub = to_iso_utc(start_date, end_of_day=False)
        max_pub = to_iso_utc(end_date, end_of_day=True)

        cutoff_dt = parse_utc_timestamp(as_of)
        min_pub_dt = parse_utc_timestamp(min_pub)
        max_pub_dt = parse_utc_timestamp(max_pub)

        params = {
            "symbol": query_symbol,
            "asOf": as_of,
            "from": min_pub,
            "to": max_pub,
            "limit": limit,
        }

        data = self._make_request("/api/v1/news", params)
        raw_events = data.get("events") or []

        sanitized_events = []
        for raw_event in raw_events:
            ev = filter_and_sanitize_event(
                raw_event,
                cutoff_dt=cutoff_dt,
                min_pub_dt=min_pub_dt,
                max_pub_dt=max_pub_dt,
            )
            if ev is not None:
                sanitized_events.append(ev)

        if sanitized_events:
            formatted = "\n".join(format_article(e) for e in sanitized_events[:limit])
            return f"## {ticker}{resolved} News, from {start_date} to {end_date}:\n\n{formatted}"

        return f"No news found for {ticker}{resolved} between {start_date} and {end_date}"

    def get_global_news(
        self,
        as_of_date: str,
        look_back_days: int | None = None,
        limit: int | None = None,
    ) -> str:
        """Fetch and format global macro news with bi-temporal point-in-time protections."""
        config = get_config()
        if look_back_days is None:
            look_back_days = config.get("global_news_lookback_days", 7)
        if limit is None:
            limit = config.get("global_news_article_limit", 10)

        curr_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
        start_dt = curr_dt - timedelta(days=look_back_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        as_of = to_iso_utc(as_of_date, end_of_day=True)
        min_pub = to_iso_utc(start_date, end_of_day=False)
        max_pub = to_iso_utc(as_of_date, end_of_day=True)

        cutoff_dt = parse_utc_timestamp(as_of)
        min_pub_dt = parse_utc_timestamp(min_pub)
        max_pub_dt = parse_utc_timestamp(max_pub)

        params = {
            "asOf": as_of,
            "from": min_pub,
            "to": max_pub,
            "limit": limit,
        }

        data = self._make_request("/api/v1/news/global", params)
        raw_events = data.get("events") or []

        sanitized_events = []
        for raw_event in raw_events:
            ev = filter_and_sanitize_event(
                raw_event,
                cutoff_dt=cutoff_dt,
                min_pub_dt=min_pub_dt,
                max_pub_dt=max_pub_dt,
            )
            if ev is not None:
                sanitized_events.append(ev)

        if sanitized_events:
            formatted = "\n".join(format_article(e) for e in sanitized_events[:limit])
            return f"## Global Market News, from {start_date} to {as_of_date}:\n\n{formatted}"

        return f"No global news found between {start_date} and {as_of_date}"


def get_news_news_intelligence(ticker: str, start_date: str, end_date: str) -> str:
    """Router implementation entry point for asset-specific news."""
    client = NewsIntelligenceClient()
    return client.get_news(ticker, start_date, end_date)


def get_global_news_news_intelligence(
    as_of_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Router implementation entry point for global macro news."""
    client = NewsIntelligenceClient()
    return client.get_global_news(as_of_date, look_back_days, limit)
