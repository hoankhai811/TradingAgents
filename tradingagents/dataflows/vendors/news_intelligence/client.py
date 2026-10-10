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
        tuple[str, str]: (query_symbol, display_resolution_string)
    """
    if not ticker or not isinstance(ticker, str):
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
    - end of day (end_of_day=True): YYYY-MM-DDT23:59:59.999999Z (inclusive microsecond end)
    """
    if not date_str or not isinstance(date_str, str):
        return ""
    clean = date_str.strip()
    if "T" in clean:
        if clean.endswith("Z") or "+" in clean[-6:] or "-" in clean[-6:]:
            return clean
        return f"{clean}Z"
    if end_of_day:
        return f"{clean}T23:59:59.999999Z"
    return f"{clean}T00:00:00Z"


def parse_utc_timestamp(ts: str | None) -> datetime | None:
    """Parse an ISO 8601 UTC timestamp string with up to microsecond precision.

    Requires an explicit timezone indicator ('Z' or '+/-HH:MM'). Timezone-free timestamps are rejected.
    """
    if not ts or not isinstance(ts, str):
        return None
    try:
        clean = ts.strip()
        # Verify explicit timezone before parsing; reject timezone-free strings
        if not (clean.endswith("Z") or ("+" in clean[-6:] or "-" in clean[-6:])):
            return None
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            return None
        return dt.astimezone(UTC)
    except Exception:
        return None


def validate_canonical_event(event: dict[str, Any]) -> bool:
    """Validate that an event conforms to the canonical MarketNewsEvent contract."""
    if not isinstance(event, dict):
        return False

    # Mandatory identity/title/source fields
    event_id = event.get("id")
    if not isinstance(event_id, str) or not event_id.strip():
        return False

    source = event.get("source")
    if not isinstance(source, str) or not source.strip():
        return False

    title = event.get("title")
    if not isinstance(title, str) or not title.strip():
        return False

    symbols = event.get("symbols")
    if not isinstance(symbols, list) or not all(isinstance(s, str) for s in symbols):
        return False

    # Required publication and receipt timestamps with explicit timezone
    pub_dt = parse_utc_timestamp(event.get("publishedAt"))
    rec_dt = parse_utc_timestamp(event.get("receivedAt"))
    if pub_dt is None or rec_dt is None:
        return False

    # EnrichedAt format if provided
    enr_at_val = event.get("enrichedAt")
    if enr_at_val is not None and (not isinstance(enr_at_val, str) or parse_utc_timestamp(enr_at_val) is None):
        return False

    # Sentiment validation if provided
    sentiment = event.get("sentiment")
    if sentiment is not None:
        if not isinstance(sentiment, dict):
            return False
        score = sentiment.get("score")
        if score is None or not isinstance(score, (int, float)) or not (-1.0 <= score <= 1.0):
            return False
        label = sentiment.get("label")
        if label not in ("positive", "neutral", "negative"):
            return False
        confidence = sentiment.get("confidence")
        if confidence is not None and (not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0)):
            return False

    # Optional string fields
    for field in ("summary", "clusterId", "url", "content"):
        val = event.get(field)
        if val is not None and not isinstance(val, str):
            return False

    return True


def filter_and_sanitize_event(
    event: dict[str, Any],
    cutoff_dt: datetime,
    min_pub_dt: datetime | None = None,
    max_pub_dt: datetime | None = None,
) -> dict[str, Any] | None:
    """Validate event and enforce bi-temporal point-in-time protections.

    Rules:
    1. Mandatory fields: 'id', 'source', 'title', 'symbols', 'publishedAt', 'receivedAt'.
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
         clusterId, enrichedAt, symbols -> []) to eliminate future-enrichment leakage.

    Returns:
        Sanitized event dict, or None if rejected.
    """
    if not validate_canonical_event(event):
        return None

    pub_dt = parse_utc_timestamp(event["publishedAt"])
    rec_dt = parse_utc_timestamp(event["receivedAt"])

    assert pub_dt is not None and rec_dt is not None

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
    enr_dt = parse_utc_timestamp(enr_at_str) if enr_at_str else None

    if enr_dt is None or enr_dt > cutoff_dt:
        # Withhold derived intelligence facts per canonical schema
        sanitized["sentiment"] = None
        sanitized["clusterId"] = None
        sanitized["summary"] = None
        sanitized["enrichedAt"] = None
        sanitized["symbols"] = []

    return sanitized


def format_article(event: dict[str, Any]) -> str:
    """Format a sanitized MarketNewsEvent into markdown for News Analyst consumption."""
    title = event.get("title") or "No title"
    source = event.get("source") or "News Intelligence"
    out = f"### {title} (source: {source})\n"

    sentiment = event.get("sentiment")
    if sentiment and isinstance(sentiment, dict):
        score = sentiment.get("score")
        label = sentiment.get("label")
        if score is not None and label is not None:
            try:
                f_score = float(score)
                out += f"- Sentiment: {label} (score: {f_score:+.2f})\n"
            except (ValueError, TypeError):
                pass

    pub_at = event.get("publishedAt")
    if pub_at:
        out += f"- Published: {pub_at}\n"

    cluster_id = event.get("clusterId")
    if cluster_id:
        out += f"- Story Cluster: {cluster_id}\n"

    url = event.get("url")
    if url:
        out += f"- URL: {url}\n"

    summary = event.get("summary")
    if summary:
        out += f"- Summary: {summary}\n"
    elif event.get("content"):
        # Fall back to content snippet if summary is not available
        content_snippet = str(event["content"])[:300].replace("\n", " ").strip()
        out += f"- Content: {content_snippet}...\n"

    return out.strip()


class NewsIntelligenceClient:
    """HTTP client for Repo 1 (trading-news-intelligence) REST endpoints.

    Integrates:
    - GET /api/v1/news?symbol={symbol}&asOf={cutoff}&from={start}&to={end}&limit={limit}
    - GET /api/v1/news/global?asOf={cutoff}&from={start}&to={end}&limit={limit}
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        config = get_config()
        raw_url = base_url if base_url is not None else config.get("news_intelligence_url")
        if not raw_url:
            raise VendorNotConfiguredError("News Intelligence API URL is not configured")

        # Strip trailing slash and trailing /api/v1 or /v1 path if supplied in base URL
        clean_url = str(raw_url).rstrip("/")
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

        if not resp.ok:
            raise VendorUnavailableError(f"News Intelligence API error ({resp.status_code}): {resp.text}")

        try:
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")
        except Exception as exc:
            raise VendorUnavailableError(f"News Intelligence API invalid JSON: {exc}") from exc

        if "events" not in data or not isinstance(data["events"], list):
            raise VendorUnavailableError("News Intelligence API response missing or invalid 'events' list")

        return data

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
            if ev is not None and query_symbol in ev.get("symbols", []):
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
