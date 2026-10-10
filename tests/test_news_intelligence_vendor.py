"""Unit tests for the News Intelligence vendor integration.

Verifies:
1. Ticker and instrument symbol mapping (equities, crypto pairs, bare crypto, HKEX).
2. Date windowing and ISO timestamp formatting (including microsecond end-of-day bounds).
3. Strict bi-temporal point-in-time protections (publishedAt <= cutoff, receivedAt <= cutoff,
   enrichedAt <= cutoff for derived intelligence facts, subsecond precision).
4. Canonical MarketNewsEvent contract enforcement (mandatory fields, timezone presence, sentiment bounds).
5. News formatting for asset-specific and global macro news.
6. Vendor routing, configuration precedence, disabled-by-default behavior, and fallback chains.
"""

from __future__ import annotations

import copy
from datetime import UTC
from unittest import mock

import pytest
import requests

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors.news_intelligence import (
    NewsIntelligenceClient,
    filter_and_sanitize_event,
    format_article,
    get_global_news_news_intelligence,
    get_news_news_intelligence,
    map_ticker_for_news_intelligence,
    parse_utc_timestamp,
    to_iso_utc,
    validate_canonical_event,
)


def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture(autouse=True)
def clean_config():
    _reset_config()
    yield
    _reset_config()


def canonical_event(**overrides):
    """Helper creating a 100% valid canonical MarketNewsEvent fixture."""
    base = {
        "id": "review-event",
        "source": "review-feed",
        "title": "REVIEW HEADLINE",
        "publishedAt": "2026-08-05T12:00:00Z",
        "receivedAt": "2026-08-05T12:01:00Z",
        "enrichedAt": "2026-08-05T12:02:00Z",
        "symbols": ["AAPL"],
        "content": "Original body",
        "summary": "Derived summary",
        "clusterId": "cluster-1",
        "sentiment": {"score": 0.25, "label": "positive"},
    }
    return {**base, **overrides}


# =====================================================================
# 1. Ticker and Instrument Symbol Mapping
# =====================================================================


@pytest.mark.unit
class TestTickerMapping:
    @pytest.mark.parametrize(
        "raw_ticker, expected_query, expected_annotation",
        [
            # Equities
            ("AAPL", "AAPL", ""),
            ("NVDA", "NVDA", ""),
            ("MSFT", "MSFT", ""),
            ("GOOGL", "GOOGL", ""),
            # Crypto pairs -> mapped to base symbol
            ("BTC-USD", "BTC", " (resolved to BTC)"),
            ("BTCUSD", "BTC", " (resolved to BTC)"),
            ("BTCUSDT", "BTC", " (resolved to BTC)"),
            ("BTCUSDC", "BTC", " (resolved to BTC)"),
            ("ETH-USD", "ETH", " (resolved to ETH)"),
            ("ETHUSD", "ETH", " (resolved to ETH)"),
            ("ETHUSDT", "ETH", " (resolved to ETH)"),
            ("SOLUSDT", "SOL", " (resolved to SOL)"),
            ("ADA-USD", "ADA", " (resolved to ADA)"),
            ("DOGEUSD", "DOGE", " (resolved to DOGE)"),
            # Bare crypto tickers
            ("BTC", "BTC", ""),
            ("ETH", "ETH", ""),
            ("SOL", "SOL", ""),
            # Foreign exchange suffix padding
            ("700.HK", "0700.HK", " (resolved to 0700.HK)"),
            ("9992.HK", "9992.HK", ""),
            ("600519.SH", "600519.SS", " (resolved to 600519.SS)"),
            # Whitespace and lowercase
            ("  aapl  ", "AAPL", " (resolved to AAPL)"),
            ("btc-usd", "BTC", " (resolved to BTC)"),
        ],
    )
    def test_map_ticker_for_news_intelligence(self, raw_ticker, expected_query, expected_annotation):
        query_symbol, resolved = map_ticker_for_news_intelligence(raw_ticker)
        assert query_symbol == expected_query
        assert resolved == expected_annotation

    def test_invalid_ticker_returns_empty(self):
        query_symbol, resolved = map_ticker_for_news_intelligence("")
        assert query_symbol == ""
        assert resolved == ""


# =====================================================================
# 2. Date and Timestamp Helpers
# =====================================================================


@pytest.mark.unit
class TestDateHelpers:
    def test_to_iso_utc_start_of_day(self):
        assert to_iso_utc("2026-08-01", end_of_day=False) == "2026-08-01T00:00:00Z"

    def test_to_iso_utc_end_of_day(self):
        # R4: end-of-day date expansion includes microsecond fractional bound
        assert to_iso_utc("2026-08-07", end_of_day=True) == "2026-08-07T23:59:59.999999Z"

    def test_to_iso_utc_preserves_existing_iso(self):
        assert to_iso_utc("2026-08-01T15:30:00Z") == "2026-08-01T15:30:00Z"
        assert to_iso_utc("2026-08-01T15:30:00+00:00") == "2026-08-01T15:30:00+00:00"

    def test_parse_utc_timestamp_precision(self):
        ts1 = parse_utc_timestamp("2026-08-01T10:00:00.123456Z")
        assert ts1 is not None
        assert ts1.year == 2026
        assert ts1.month == 8
        assert ts1.day == 1
        assert ts1.hour == 10
        assert ts1.microsecond == 123456
        assert ts1.tzinfo == UTC

        ts2 = parse_utc_timestamp("2026-08-01T10:00:00.123000Z")
        assert ts1 > ts2

    def test_parse_utc_timestamp_invalid(self):
        assert parse_utc_timestamp("") is None
        assert parse_utc_timestamp(None) is None
        assert parse_utc_timestamp("invalid-date") is None

    def test_parse_utc_timestamp_rejects_timezone_free_strings(self):
        # R3: explicit timezone indicator is mandatory
        assert parse_utc_timestamp("2026-08-05T12:01:00") is None
        assert parse_utc_timestamp("2026-08-05 12:01:00") is None


# =====================================================================
# 3. Canonical Contract & Schema Validation (R3)
# =====================================================================


@pytest.mark.unit
class TestCanonicalSchemaValidation:
    def test_valid_canonical_event_passes(self):
        ev = canonical_event()
        assert validate_canonical_event(ev) is True

    def test_missing_mandatory_fields_rejected(self):
        # Mandatory: id, source, title, symbols, publishedAt, receivedAt
        assert validate_canonical_event(canonical_event(id="")) is False
        assert validate_canonical_event(canonical_event(source="")) is False
        assert validate_canonical_event(canonical_event(title="")) is False
        assert validate_canonical_event(canonical_event(publishedAt=None)) is False
        assert validate_canonical_event(canonical_event(receivedAt=None)) is False
        assert validate_canonical_event(canonical_event(symbols=None)) is False
        assert validate_canonical_event(canonical_event(symbols="not-a-list")) is False

    def test_sentiment_score_bounds_enforced(self):
        # Valid range [-1.0, 1.0]
        assert validate_canonical_event(canonical_event(sentiment={"score": -1.0, "label": "negative"})) is True
        assert validate_canonical_event(canonical_event(sentiment={"score": 1.0, "label": "positive"})) is True
        assert validate_canonical_event(canonical_event(sentiment={"score": 0.0, "label": "neutral"})) is True
        # R3: Score 500 must be rejected
        assert validate_canonical_event(canonical_event(sentiment={"score": 500, "label": "positive"})) is False
        assert validate_canonical_event(canonical_event(sentiment={"score": -1.05, "label": "negative"})) is False
        assert validate_canonical_event(canonical_event(sentiment={"score": 1.05, "label": "positive"})) is False
        # Invalid label
        assert validate_canonical_event(canonical_event(sentiment={"score": 0.5, "label": "unknown"})) is False

    def test_out_of_range_sentiment_filtered_out(self):
        # Reviewer proof test: out of range sentiment rejected by filter_and_sanitize_event
        ev = canonical_event(sentiment={"score": 500, "label": "positive"})
        assert filter_and_sanitize_event(ev, parse_utc_timestamp("2026-08-05T23:59:59Z")) is None

    def test_timezone_free_received_at_filtered_out(self):
        # Reviewer proof test: timezone-free timestamp rejected
        ev = canonical_event(receivedAt="2026-08-05T12:01:00")
        assert filter_and_sanitize_event(ev, parse_utc_timestamp("2026-08-05T23:59:59Z")) is None


# =====================================================================
# 4. Bi-Temporal Point-in-Time Protections (R1, R4)
# =====================================================================


@pytest.mark.unit
class TestBiTemporalProtections:
    def test_scenario_1_late_receipt_excluded(self):
        event = canonical_event(
            publishedAt="2026-08-01T10:00:00Z",
            receivedAt="2026-08-03T12:00:00Z",
            symbols=["AAPL"],
        )
        cutoff_dt = parse_utc_timestamp("2026-08-02T23:59:59Z")
        res = filter_and_sanitize_event(event, cutoff_dt=cutoff_dt)
        assert res is None

    def test_scenario_2_receipt_advancement_accepted(self):
        event = canonical_event(
            id="ev-1",
            publishedAt="2026-08-01T10:00:00Z",
            receivedAt="2026-08-03T12:00:00Z",
            symbols=["AAPL"],
        )
        cutoff_dt = parse_utc_timestamp("2026-08-04T00:00:00Z")
        res = filter_and_sanitize_event(event, cutoff_dt=cutoff_dt)
        assert res is not None
        assert res["id"] == "ev-1"

    def test_scenario_3_late_enrichment_withheld(self):
        # R1: when enrichment is late, derived facts AND symbols are withheld
        event = canonical_event(
            publishedAt="2026-08-01T10:00:00Z",
            receivedAt="2026-08-01T11:00:00Z",
            enrichedAt="2026-08-03T09:00:00Z",
            title="NVIDIA Earnings Surprise",
            content="Raw body of earnings report...",
            symbols=["NVDA"],
        )
        cutoff_dt = parse_utc_timestamp("2026-08-02T23:59:59Z")
        res = filter_and_sanitize_event(event, cutoff_dt=cutoff_dt)

        assert res is not None
        assert res["title"] == "NVIDIA Earnings Surprise"
        assert res["content"] == "Raw body of earnings report..."
        # Derived facts must be stripped / withheld
        assert res["summary"] is None
        assert res["sentiment"] is None
        assert res["clusterId"] is None
        assert res["enrichedAt"] is None
        # R1: symbols cleared to []
        assert res["symbols"] == []

    @pytest.mark.parametrize("enriched_at", [None, "2026-08-06T12:00:00Z"])
    def test_future_or_null_symbols_are_withheld(self, enriched_at):
        # Reviewer proof test: future or null enrichment withholds symbols to []
        row = filter_and_sanitize_event(
            canonical_event(enrichedAt=enriched_at),
            parse_utc_timestamp("2026-08-05T23:59:59Z"),
        )
        assert row is not None
        assert row["symbols"] == []
        assert row["sentiment"] is None
        assert row["summary"] is None

    def test_scenario_4_enrichment_advancement_accepted(self):
        event = canonical_event(
            publishedAt="2026-08-01T10:00:00Z",
            receivedAt="2026-08-01T11:00:00Z",
            enrichedAt="2026-08-03T09:00:00Z",
            summary="Derived LLM summary generated in future",
            clusterId="cluster-999",
            sentiment={"score": 0.85, "label": "positive"},
            symbols=["NVDA"],
        )
        cutoff_dt = parse_utc_timestamp("2026-08-04T00:00:00Z")
        res = filter_and_sanitize_event(event, cutoff_dt=cutoff_dt)

        assert res is not None
        assert res["summary"] == "Derived LLM summary generated in future"
        assert res["sentiment"] == {"score": 0.85, "label": "positive"}
        assert res["clusterId"] == "cluster-999"
        assert res["enrichedAt"] == "2026-08-03T09:00:00Z"
        assert res["symbols"] == ["NVDA"]

    def test_future_publication_excluded(self):
        event = canonical_event(
            publishedAt="2026-08-05T10:00:00Z",
            receivedAt="2026-08-05T10:01:00Z",
        )
        cutoff_dt = parse_utc_timestamp("2026-08-02T23:59:59Z")
        assert filter_and_sanitize_event(event, cutoff_dt=cutoff_dt) is None

    def test_publication_window_bounds(self):
        event_early = canonical_event(
            publishedAt="2026-07-31T23:59:59Z",
            receivedAt="2026-07-31T23:59:59Z",
        )
        event_in_window = canonical_event(
            publishedAt="2026-08-02T10:00:00Z",
            receivedAt="2026-08-02T10:05:00Z",
        )
        event_late = canonical_event(
            publishedAt="2026-08-08T00:00:01Z",
            receivedAt="2026-08-08T00:00:01Z",
        )
        min_pub_dt = parse_utc_timestamp("2026-08-01T00:00:00Z")
        max_pub_dt = parse_utc_timestamp("2026-08-07T23:59:59.999999Z")
        cutoff_dt = parse_utc_timestamp("2026-08-10T00:00:00Z")

        assert filter_and_sanitize_event(event_early, cutoff_dt, min_pub_dt, max_pub_dt) is None
        assert filter_and_sanitize_event(event_late, cutoff_dt, min_pub_dt, max_pub_dt) is None
        assert filter_and_sanitize_event(event_in_window, cutoff_dt, min_pub_dt, max_pub_dt) is not None

    def test_microsecond_precision_cutoff(self):
        event = canonical_event(
            publishedAt="2026-08-01T10:00:00.123456Z",
            receivedAt="2026-08-01T10:00:00.123456Z",
        )
        cutoff_before = parse_utc_timestamp("2026-08-01T10:00:00.123455Z")
        assert filter_and_sanitize_event(event, cutoff_before) is None

        cutoff_exact = parse_utc_timestamp("2026-08-01T10:00:00.123456Z")
        assert filter_and_sanitize_event(event, cutoff_exact) is not None

        cutoff_after = parse_utc_timestamp("2026-08-01T10:00:00.123457Z")
        assert filter_and_sanitize_event(event, cutoff_after) is not None

    @pytest.mark.parametrize("global_news", [False, True])
    def test_date_window_includes_last_fractional_second(self, global_news):
        # Reviewer proof test: R4 inclusion of final fractional second
        row = canonical_event(
            publishedAt="2026-08-05T23:59:59.500000Z",
            receivedAt="2026-08-05T23:59:59.600000Z",
            enrichedAt="2026-08-05T23:59:59.700000Z",
            symbols=["AAPL"],
        )
        if global_news:
            row.update(enrichedAt=None, symbols=[], summary=None, sentiment=None, clusterId=None)

        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.return_value = {"events": [row], "count": 1}

        session = mock.Mock()
        session.get.return_value = mock_resp
        vendor = NewsIntelligenceClient(base_url="http://test-server:3001", session=session)

        result = (
            vendor.get_global_news("2026-08-05")
            if global_news
            else vendor.get_news("AAPL", "2026-08-01", "2026-08-05")
        )
        assert "REVIEW HEADLINE" in result


# =====================================================================
# 5. Article Formatting
# =====================================================================


@pytest.mark.unit
class TestArticleFormatting:
    def test_format_article_with_sentiment_and_summary(self):
        event = canonical_event(
            title="Major Tech Acquisition",
            source="bloomberg",
            sentiment={"label": "positive", "score": 0.75},
            summary="Tech giant acquires startup for $2B.",
            url="https://example.com/story",
        )
        formatted = format_article(event)
        assert "### Major Tech Acquisition (source: bloomberg)" in formatted
        assert "Sentiment: positive (score: +0.75)" in formatted
        assert "Tech giant acquires startup for $2B." in formatted
        assert "URL: https://example.com/story" in formatted

    def test_format_article_without_sentiment(self):
        event = canonical_event(
            title="Raw Ingestion Story",
            source="rss-reuters",
            sentiment=None,
            content="Raw story body text.",
            url="https://example.com/raw",
            summary=None,
            enrichedAt=None,
            symbols=[],
        )
        formatted = format_article(event)
        assert "### Raw Ingestion Story (source: rss-reuters)" in formatted
        assert "Sentiment:" not in formatted
        assert "Raw story body text." in formatted
        assert "URL: https://example.com/raw" in formatted


# =====================================================================
# 6. NewsIntelligenceClient & API Calls (R2)
# =====================================================================


@pytest.mark.unit
class TestNewsIntelligenceClient:
    def test_get_news_success_with_crypto_mapping(self):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {
            "symbol": "BTC",
            "count": 1,
            "events": [
                canonical_event(
                    id="btc-1",
                    source="binance",
                    publishedAt="2026-08-02T10:00:00Z",
                    receivedAt="2026-08-02T10:05:00Z",
                    enrichedAt="2026-08-02T10:06:00Z",
                    title="Bitcoin Surges Past Resistance",
                    summary="BTC shows strong buying volume.",
                    sentiment={"label": "positive", "score": 0.60},
                    symbols=["BTC"],
                    url="https://binance.com/news/1",
                )
            ],
        }

        mock_session = mock.Mock()
        mock_session.get.return_value = mock_response

        client = NewsIntelligenceClient(base_url="http://test-server:3001", session=mock_session)
        result = client.get_news("BTC-USD", "2026-08-01", "2026-08-05")

        mock_session.get.assert_called_once()
        call_url, call_kwargs = mock_session.get.call_args
        assert call_url[0] == "http://test-server:3001/api/v1/news"
        assert call_kwargs["params"]["symbol"] == "BTC"
        assert call_kwargs["params"]["asOf"] == "2026-08-05T23:59:59.999999Z"
        assert "## BTC-USD (resolved to BTC) News" in result

    def test_get_global_news_success(self):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {
            "count": 1,
            "events": [
                canonical_event(
                    id="macro-1",
                    source="reuters",
                    publishedAt="2026-08-05T12:00:00Z",
                    receivedAt="2026-08-05T12:02:00Z",
                    enrichedAt="2026-08-05T12:05:00Z",
                    title="Fed Signals Pause in Rate Cuts",
                    summary="Federal Reserve indicates policy caution.",
                    sentiment={"label": "neutral", "score": 0.0},
                    symbols=[],
                    url="https://reuters.com/macro",
                )
            ],
        }

        mock_session = mock.Mock()
        mock_session.get.return_value = mock_response

        client = NewsIntelligenceClient(base_url="http://test-server:3001", session=mock_session)
        result = client.get_global_news("2026-08-07", look_back_days=7, limit=5)

        mock_session.get.assert_called_once()
        call_url, call_kwargs = mock_session.get.call_args
        assert call_url[0] == "http://test-server:3001/api/v1/news/global"
        assert call_kwargs["params"]["asOf"] == "2026-08-07T23:59:59.999999Z"
        assert "## Global Market News, from 2026-07-31 to 2026-08-07:" in result
        assert "### Fed Signals Pause in Rate Cuts (source: reuters)" in result

    def test_a_missing_list_route_is_not_an_empty_feed(self):
        # R2: HTTP 404 raises VendorUnavailableError, never treated as empty feed
        mock_response = mock.Mock(status_code=404, ok=False, text="Route Not Found")
        mock_session = mock.Mock()
        mock_session.get.return_value = mock_response

        client = NewsIntelligenceClient(base_url="http://test-server:3001", session=mock_session)
        with pytest.raises(VendorUnavailableError):
            client.get_news("AAPL", "2026-08-01", "2026-08-05")

    @pytest.mark.parametrize("payload", [{"status": "ok"}, {"events": {}, "count": 1}])
    def test_wrong_response_shape_is_not_an_empty_feed(self, payload):
        # R2: Missing or non-array 'events' field raises VendorUnavailableError
        mock_response = mock.Mock(status_code=200, ok=True)
        mock_response.json.return_value = payload
        mock_session = mock.Mock()
        mock_session.get.return_value = mock_response

        client = NewsIntelligenceClient(base_url="http://test-server:3001", session=mock_session)
        with pytest.raises(VendorUnavailableError):
            client.get_news("AAPL", "2026-08-01", "2026-08-05")

    def test_empty_news_returns_clear_message_on_valid_200(self):
        # Valid HTTP 200 with empty array returns standard empty message
        mock_response = mock.Mock(status_code=200, ok=True)
        mock_response.json.return_value = {"symbol": "AAPL", "count": 0, "events": []}
        mock_session = mock.Mock()
        mock_session.get.return_value = mock_response

        client = NewsIntelligenceClient(base_url="http://test-server:3001", session=mock_session)
        result = client.get_news("AAPL", "2026-08-01", "2026-08-05")
        assert result == "No news found for AAPL between 2026-08-01 and 2026-08-05"

    def test_network_connection_error_raises_vendor_unavailable(self):
        mock_session = mock.Mock()
        mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = NewsIntelligenceClient(base_url="http://test-server:3001", session=mock_session)
        with pytest.raises(VendorUnavailableError) as exc_info:
            client.get_news("AAPL", "2026-08-01", "2026-08-05")
        assert "connection error" in str(exc_info.value).lower()

    def test_timeout_raises_vendor_unavailable(self):
        mock_session = mock.Mock()
        mock_session.get.side_effect = requests.exceptions.Timeout("Read timeout")

        client = NewsIntelligenceClient(base_url="http://test-server:3001", session=mock_session)
        with pytest.raises(VendorUnavailableError) as exc_info:
            client.get_news("AAPL", "2026-08-01", "2026-08-05")
        assert "timed out" in str(exc_info.value).lower()

    def test_server_500_error_raises_vendor_unavailable(self):
        mock_response = mock.Mock(status_code=500, ok=False, text="Internal Server Error")
        mock_session = mock.Mock()
        mock_session.get.return_value = mock_response

        client = NewsIntelligenceClient(base_url="http://test-server:3001", session=mock_session)
        with pytest.raises(VendorUnavailableError) as exc_info:
            client.get_news("AAPL", "2026-08-01", "2026-08-05")
        assert "500" in str(exc_info.value)

    def test_bearer_token_authorization_header(self):
        mock_response = mock.Mock(status_code=200, ok=True)
        mock_response.json.return_value = {"events": [], "count": 0}
        mock_session = mock.Mock()
        mock_session.get.return_value = mock_response

        client = NewsIntelligenceClient(
            base_url="http://test-server:3001",
            token="test-jwt",
            session=mock_session,
        )
        client.get_news("AAPL", "2026-08-01", "2026-08-05")

        call_url, call_kwargs = mock_session.get.call_args
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-jwt"

    def test_base_url_normalization(self):
        client1 = NewsIntelligenceClient(base_url="http://test-server:3001/api/v1/")
        assert client1.base_url == "http://test-server:3001"

        client2 = NewsIntelligenceClient(base_url="http://test-server:3001/v1")
        assert client2.base_url == "http://test-server:3001"

        client3 = NewsIntelligenceClient(base_url="http://test-server:3001")
        assert client3.base_url == "http://test-server:3001"


# =====================================================================
# 7. Vendor Router & Fallback Chain Integration (R2)
# =====================================================================


@pytest.mark.unit
class TestVendorRoutingIntegration:
    def test_disabled_by_default(self):
        cfg = default_config.DEFAULT_CONFIG
        assert cfg["data_vendors"]["news_data"] == "yfinance"

    def test_news_intelligence_is_registered_in_vendor_methods(self):
        assert "news_intelligence" in router.VENDOR_METHODS["get_news"]
        assert "news_intelligence" in router.VENDOR_METHODS["get_global_news"]

    def test_single_news_intelligence_routing(self):
        set_config({"data_vendors": {"news_data": "news_intelligence"}})
        mock_get_news = mock.Mock(return_value="NEWS_INTEL_REPORT")
        with mock.patch.dict(
            router.VENDOR_METHODS["get_news"],
            {"news_intelligence": mock_get_news},
            clear=False,
        ):
            res = router.route_to_vendor("get_news", "NVDA", "2026-08-01", "2026-08-07")
            assert res == "NEWS_INTEL_REPORT"
            mock_get_news.assert_called_once_with("NVDA", "2026-08-01", "2026-08-07")

    def test_fallback_chain_from_news_intelligence_to_yfinance(self):
        set_config({"data_vendors": {"news_data": "news_intelligence,yfinance"}})
        mock_intel = mock.Mock(side_effect=VendorUnavailableError("Connection refused to news-intelligence"))
        mock_yf = mock.Mock(return_value="YFINANCE_FALLBACK_REPORT")
        with mock.patch.dict(
            router.VENDOR_METHODS["get_news"],
            {"news_intelligence": mock_intel, "yfinance": mock_yf},
            clear=False,
        ):
            res = router.route_to_vendor("get_news", "AAPL", "2026-08-01", "2026-08-07")
            assert res == "YFINANCE_FALLBACK_REPORT"
            mock_intel.assert_called_once_with("AAPL", "2026-08-01", "2026-08-07")
            mock_yf.assert_called_once_with("AAPL", "2026-08-01", "2026-08-07")

    def test_fallback_chain_triggers_on_404_and_malformed_envelope(self):
        # R2: Verify that HTTP 404 or bad envelope properly falls back to yfinance
        set_config({"data_vendors": {"news_data": "news_intelligence,yfinance"}})
        mock_intel_404 = mock.Mock(side_effect=VendorUnavailableError("API error (404)"))
        mock_yf = mock.Mock(return_value="YFINANCE_FALLBACK_REPORT")
        with mock.patch.dict(
            router.VENDOR_METHODS["get_news"],
            {"news_intelligence": mock_intel_404, "yfinance": mock_yf},
            clear=False,
        ):
            res = router.route_to_vendor("get_news", "AAPL", "2026-08-01", "2026-08-07")
            assert res == "YFINANCE_FALLBACK_REPORT"

    def test_get_insider_transactions_routes_to_yfinance_when_chain_configured(self):
        set_config({"data_vendors": {"news_data": "news_intelligence,yfinance"}})
        mock_yf_insider = mock.Mock(return_value="INSIDER_REPORT")
        with mock.patch.dict(
            router.VENDOR_METHODS["get_insider_transactions"],
            {"yfinance": mock_yf_insider},
            clear=False,
        ):
            res = router.route_to_vendor("get_insider_transactions", "AAPL", "2026-08-07")
            assert res == "INSIDER_REPORT"
            mock_yf_insider.assert_called_once_with("AAPL", "2026-08-07")

    def test_single_vendor_unavailable_returns_data_unavailable_sentinel(self):
        set_config({"data_vendors": {"news_data": "news_intelligence"}})
        mock_intel = mock.Mock(side_effect=VendorUnavailableError("Endpoint unreachable"))
        with mock.patch.dict(
            router.VENDOR_METHODS["get_news"],
            {"news_intelligence": mock_intel},
            clear=False,
        ):
            res = router.route_to_vendor("get_news", "AAPL", "2026-08-01", "2026-08-07")
            assert res.startswith("DATA_UNAVAILABLE")
            assert "Endpoint unreachable" in res

    def test_direct_get_news_news_intelligence(self):
        with mock.patch.object(NewsIntelligenceClient, "get_news", return_value="DIRECT_NEWS") as mock_gn:
            res = get_news_news_intelligence("AAPL", "2026-08-01", "2026-08-07")
            assert res == "DIRECT_NEWS"
            mock_gn.assert_called_once_with("AAPL", "2026-08-01", "2026-08-07")

    def test_direct_get_global_news_news_intelligence(self):
        with mock.patch.object(NewsIntelligenceClient, "get_global_news", return_value="DIRECT_GLOBAL") as mock_ggn:
            res = get_global_news_news_intelligence("2026-08-07", look_back_days=7, limit=10)
            assert res == "DIRECT_GLOBAL"
            mock_ggn.assert_called_once_with("2026-08-07", 7, 10)
