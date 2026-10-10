"""Unit tests for the News Intelligence vendor integration.

Verifies:
1. Ticker and instrument symbol mapping (equities, crypto pairs, bare crypto, HKEX).
2. Date windowing and ISO timestamp formatting.
3. Strict bi-temporal point-in-time protections (publishedAt <= cutoff, receivedAt <= cutoff,
   enrichedAt <= cutoff for derived intelligence facts, subsecond precision).
4. News formatting for asset-specific and global macro news.
5. Vendor routing, configuration precedence, disabled-by-default behavior, and fallback chains.
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
)


def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture(autouse=True)
def clean_config():
    _reset_config()
    yield
    _reset_config()


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
        assert to_iso_utc("2026-08-07", end_of_day=True) == "2026-08-07T23:59:59Z"

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


# =====================================================================
# 3. Bi-Temporal Point-in-Time Protections
# =====================================================================


@pytest.mark.unit
class TestBiTemporalProtections:
    """Verifies strict point-in-time invariants:
    1. Old-publication / late-receipt exclusion
    2. Receipt-advancement acceptance
    3. Late-enrichment exclusion (withholding derived intelligence facts)
    4. Enrichment-advancement acceptance
    5. Future publication exclusion
    6. Microsecond boundary precision
    """

    def test_scenario_1_late_receipt_excluded(self):
        # Published Aug 1, received Aug 3, cutoff Aug 2 23:59:59
        event = {
            "id": "ev-1",
            "source": "bloomberg",
            "publishedAt": "2026-08-01T10:00:00Z",
            "receivedAt": "2026-08-03T12:00:00Z",
            "title": "Breaking News",
            "content": "Story content",
        }
        cutoff_dt = parse_utc_timestamp("2026-08-02T23:59:59Z")
        res = filter_and_sanitize_event(event, cutoff_dt=cutoff_dt)
        # MUST BE STRICTLY EXCLUDED
        assert res is None

    def test_scenario_2_receipt_advancement_accepted(self):
        # Cutoff advances to Aug 4 -> now receivedAt <= cutoff
        event = {
            "id": "ev-1",
            "source": "bloomberg",
            "publishedAt": "2026-08-01T10:00:00Z",
            "receivedAt": "2026-08-03T12:00:00Z",
            "title": "Breaking News",
            "content": "Story content",
        }
        cutoff_dt = parse_utc_timestamp("2026-08-04T00:00:00Z")
        res = filter_and_sanitize_event(event, cutoff_dt=cutoff_dt)
        assert res is not None
        assert res["id"] == "ev-1"

    def test_scenario_3_late_enrichment_withheld(self):
        # Article published & received Aug 1. LLM summary & sentiment created Aug 3.
        # Query cutoff is Aug 2. Article is accepted, but derived facts are WITHHELD.
        event = {
            "id": "ev-2",
            "source": "reuters",
            "publishedAt": "2026-08-01T10:00:00Z",
            "receivedAt": "2026-08-01T11:00:00Z",
            "enrichedAt": "2026-08-03T09:00:00Z",
            "title": "NVIDIA Earnings Surprise",
            "content": "Raw body of earnings report...",
            "summary": "Derived LLM summary generated in future",
            "clusterId": "cluster-999",
            "sentiment": {"score": 0.85, "label": "positive"},
        }
        cutoff_dt = parse_utc_timestamp("2026-08-02T23:59:59Z")
        res = filter_and_sanitize_event(event, cutoff_dt=cutoff_dt)

        assert res is not None
        # Article is kept
        assert res["title"] == "NVIDIA Earnings Surprise"
        assert res["content"] == "Raw body of earnings report..."
        # Derived facts must be stripped / withheld
        assert res["summary"] is None
        assert res["sentiment"] is None
        assert res["clusterId"] is None
        assert res["enrichedAt"] is None

    def test_scenario_4_enrichment_advancement_accepted(self):
        # Same event, cutoff advances to Aug 4 -> derived facts are present
        event = {
            "id": "ev-2",
            "source": "reuters",
            "publishedAt": "2026-08-01T10:00:00Z",
            "receivedAt": "2026-08-01T11:00:00Z",
            "enrichedAt": "2026-08-03T09:00:00Z",
            "title": "NVIDIA Earnings Surprise",
            "content": "Raw body of earnings report...",
            "summary": "Derived LLM summary generated in future",
            "clusterId": "cluster-999",
            "sentiment": {"score": 0.85, "label": "positive"},
        }
        cutoff_dt = parse_utc_timestamp("2026-08-04T00:00:00Z")
        res = filter_and_sanitize_event(event, cutoff_dt=cutoff_dt)

        assert res is not None
        assert res["summary"] == "Derived LLM summary generated in future"
        assert res["sentiment"] == {"score": 0.85, "label": "positive"}
        assert res["clusterId"] == "cluster-999"
        assert res["enrichedAt"] == "2026-08-03T09:00:00Z"

    def test_future_publication_excluded(self):
        # Article published in future relative to cutoff
        event = {
            "id": "ev-3",
            "source": "yahoo",
            "publishedAt": "2026-08-05T10:00:00Z",
            "receivedAt": "2026-08-05T10:01:00Z",
            "title": "Future Headline",
        }
        cutoff_dt = parse_utc_timestamp("2026-08-02T23:59:59Z")
        assert filter_and_sanitize_event(event, cutoff_dt=cutoff_dt) is None

    def test_publication_window_bounds(self):
        event_early = {
            "id": "ev-early",
            "publishedAt": "2026-07-31T23:59:59Z",
            "receivedAt": "2026-07-31T23:59:59Z",
            "title": "Early",
        }
        event_in_window = {
            "id": "ev-in",
            "publishedAt": "2026-08-02T10:00:00Z",
            "receivedAt": "2026-08-02T10:05:00Z",
            "title": "In Window",
        }
        event_late = {
            "id": "ev-late",
            "publishedAt": "2026-08-08T00:00:01Z",
            "receivedAt": "2026-08-08T00:00:01Z",
            "title": "Late",
        }
        min_pub_dt = parse_utc_timestamp("2026-08-01T00:00:00Z")
        max_pub_dt = parse_utc_timestamp("2026-08-07T23:59:59Z")
        cutoff_dt = parse_utc_timestamp("2026-08-10T00:00:00Z")

        assert filter_and_sanitize_event(event_early, cutoff_dt, min_pub_dt, max_pub_dt) is None
        assert filter_and_sanitize_event(event_late, cutoff_dt, min_pub_dt, max_pub_dt) is None
        assert filter_and_sanitize_event(event_in_window, cutoff_dt, min_pub_dt, max_pub_dt) is not None

    def test_microsecond_precision_cutoff(self):
        event = {
            "id": "ev-micro",
            "publishedAt": "2026-08-01T10:00:00.123456Z",
            "receivedAt": "2026-08-01T10:00:00.123456Z",
            "title": "Microsecond Event",
        }
        # Cutoff 1 microsecond earlier -> excluded
        cutoff_before = parse_utc_timestamp("2026-08-01T10:00:00.123455Z")
        assert filter_and_sanitize_event(event, cutoff_before) is None

        # Cutoff exact microsecond -> accepted
        cutoff_exact = parse_utc_timestamp("2026-08-01T10:00:00.123456Z")
        assert filter_and_sanitize_event(event, cutoff_exact) is not None

        # Cutoff 1 microsecond later -> accepted
        cutoff_after = parse_utc_timestamp("2026-08-01T10:00:00.123457Z")
        assert filter_and_sanitize_event(event, cutoff_after) is not None


# =====================================================================
# 4. Article Formatting
# =====================================================================


@pytest.mark.unit
class TestArticleFormatting:
    def test_format_article_with_sentiment_and_summary(self):
        event = {
            "title": "Major Tech Acquisition",
            "source": "bloomberg",
            "sentiment": {"label": "positive", "score": 0.75},
            "summary": "Tech giant acquires startup for $2B.",
            "url": "https://example.com/story",
        }
        formatted = format_article(event)
        assert "### Major Tech Acquisition (source: bloomberg)" in formatted
        assert "Sentiment: positive (score: +0.75)" in formatted
        assert "Tech giant acquires startup for $2B." in formatted
        assert "Link: https://example.com/story" in formatted

    def test_format_article_without_sentiment(self):
        event = {
            "title": "Raw Ingestion Story",
            "source": "rss-reuters",
            "sentiment": None,
            "content": "Raw story body text.",
            "url": "https://example.com/raw",
        }
        formatted = format_article(event)
        assert "### Raw Ingestion Story (source: rss-reuters)" in formatted
        assert "Sentiment:" not in formatted
        assert "Raw story body text." in formatted
        assert "Link: https://example.com/raw" in formatted


# =====================================================================
# 5. NewsIntelligenceClient & API Calls
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
                {
                    "id": "btc-1",
                    "source": "binance",
                    "publishedAt": "2026-08-02T10:00:00Z",
                    "receivedAt": "2026-08-02T10:05:00Z",
                    "enrichedAt": "2026-08-02T10:06:00Z",
                    "title": "Bitcoin Surges Past Resistance",
                    "summary": "BTC shows strong buying volume.",
                    "sentiment": {"label": "positive", "score": 0.60},
                    "url": "https://binance.com/news/1",
                }
            ],
        }

        mock_session = mock.Mock()
        mock_session.get.return_value = mock_response

        client = NewsIntelligenceClient(base_url="http://test-server:3001", session=mock_session)
        result = client.get_news("BTC-USD", "2026-08-01", "2026-08-05")

        # Verify query was called with canonical BTC
        mock_session.get.assert_called_once()
        call_url, call_kwargs = mock_session.get.call_args
        assert call_url[0] == "http://test-server:3001/api/v1/news"
        assert call_kwargs["params"]["symbol"] == "BTC"
        assert call_kwargs["params"]["asOf"] == "2026-08-05T23:59:59Z"
        assert call_kwargs["params"]["from"] == "2026-08-01T00:00:00Z"
        assert call_kwargs["params"]["to"] == "2026-08-05T23:59:59Z"

        # Verify report header and contents
        assert "## BTC-USD (resolved to BTC) News, from 2026-08-01 to 2026-08-05:" in result
        assert "### Bitcoin Surges Past Resistance (source: binance)" in result
        assert "Sentiment: positive (score: +0.60)" in result
        assert "Link: https://binance.com/news/1" in result

    def test_get_global_news_success(self):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {
            "count": 1,
            "events": [
                {
                    "id": "macro-1",
                    "source": "reuters",
                    "publishedAt": "2026-08-05T12:00:00Z",
                    "receivedAt": "2026-08-05T12:02:00Z",
                    "enrichedAt": "2026-08-05T12:05:00Z",
                    "title": "Fed Signals Pause in Rate Cuts",
                    "summary": "Federal Reserve indicates policy caution.",
                    "sentiment": {"label": "neutral", "score": 0.0},
                    "url": "https://reuters.com/macro",
                }
            ],
        }

        mock_session = mock.Mock()
        mock_session.get.return_value = mock_response

        client = NewsIntelligenceClient(base_url="http://test-server:3001", session=mock_session)
        result = client.get_global_news("2026-08-07", look_back_days=7, limit=5)

        mock_session.get.assert_called_once()
        call_url, call_kwargs = mock_session.get.call_args
        assert call_url[0] == "http://test-server:3001/api/v1/news/global"
        assert call_kwargs["params"]["asOf"] == "2026-08-07T23:59:59Z"
        assert call_kwargs["params"]["from"] == "2026-07-31T00:00:00Z"
        assert call_kwargs["params"]["limit"] == 5

        assert "## Global Market News, from 2026-07-31 to 2026-08-07:" in result
        assert "### Fed Signals Pause in Rate Cuts (source: reuters)" in result

    def test_empty_news_returns_clear_message(self):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.ok = True
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
        mock_response = mock.Mock()
        mock_response.status_code = 500
        mock_response.ok = False
        mock_response.text = "Internal Server Error"

        mock_session = mock.Mock()
        mock_session.get.return_value = mock_response

        client = NewsIntelligenceClient(base_url="http://test-server:3001", session=mock_session)
        with pytest.raises(VendorUnavailableError) as exc_info:
            client.get_news("AAPL", "2026-08-01", "2026-08-05")
        assert "server error (500)" in str(exc_info.value).lower()

    def test_bearer_token_authorization_header(self):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.ok = True
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
# 6. Vendor Router & Fallback Chain Integration
# =====================================================================


@pytest.mark.unit
class TestVendorRoutingIntegration:
    def test_disabled_by_default(self):
        # Default config uses yfinance, NOT news_intelligence
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
        # Chain: news_intelligence,yfinance
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

    def test_get_insider_transactions_routes_to_yfinance_when_chain_configured(self):
        # news_intelligence does not serve insider transactions, so chain falls back to yfinance
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
