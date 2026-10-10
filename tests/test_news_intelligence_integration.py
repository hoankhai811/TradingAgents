"""Integration and boundary tests for the News Intelligence vendor (T12 / TIP-13).

Comprehensive, network-free integration test coverage around the released
TradingAgents News Intelligence vendor without duplicating existing baseline unit tests.

Coverage areas:
1. Network-free parsing and payload integrity (non-JSON, non-dict payloads, missing/invalid events list,
   offset timestamps, content snippet truncation, title/source defaults).
2. Crypto symbol coverage (expanded USD/USDT/USDC pairs, bare crypto, multi-symbol matching, header resolution).
3. Equity symbol coverage (standard US stocks, dot/hyphen classes, lowercase, foreign/HKEX, header formatting).
4. Point-in-time as-of bi-temporal boundaries (exact microsecond boundaries, plain date vs explicit ISO cutoffs,
   derived facts withholding at cutoff + 1us, global news lookback calculation).
5. Future rejection bi-temporal guarantees (future publication, future receipt, late-ingested historical articles,
   global news withholding).
6. Malformed optional and mandatory fields (optional strings non-str vs None, sentiment schema, mandatory fields,
   mixed-feed selective filtering without fallback).
7. Network errors and HTTP codes (401 Unauthorized, 403 Forbidden, 429 Rate Limited, 502/503 errors,
   connect/read timeouts, custom timeout configuration).
8. Configured fallback chains (multi-vendor chains on 401/403/429/timeout/connection error, global news fallback,
   multi-hop chains, all-vendor failure sentinel).
9. Disabled default and default behavior (default config verification, unconfigured routing to yfinance,
   VendorNotConfiguredError on missing URL, secondary fallback precedence, tool_vendors override).
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from unittest import mock

import pytest
import requests

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import (
    VendorNotConfiguredError,
    VendorUnavailableError,
)
from tradingagents.dataflows.vendors.news_intelligence import (
    NewsIntelligenceClient,
    filter_and_sanitize_event,
    format_article,
    map_ticker_for_news_intelligence,
    parse_utc_timestamp,
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
        "id": "integration-event-1",
        "source": "integration-feed",
        "title": "INTEGRATION HEADLINE",
        "publishedAt": "2026-08-05T12:00:00.000000Z",
        "receivedAt": "2026-08-05T12:01:00.000000Z",
        "enrichedAt": "2026-08-05T12:02:00.000000Z",
        "symbols": ["AAPL"],
        "content": "Original body text for integration testing",
        "summary": "Derived intelligence summary",
        "clusterId": "cluster-int-1",
        "sentiment": {"score": 0.5, "label": "positive", "confidence": 0.9},
        "url": "https://news.example.com/integration-1",
    }
    return {**base, **overrides}


# =====================================================================
# 1. Network-Free Parsing and Payload Integrity
# =====================================================================


class TestNetworkFreeParsingAndPayloadIntegrity:
    """Verifies response parsing error handling and payload integrity."""

    def test_json_decode_error_from_html_or_garbage_raises_vendor_unavailable(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            with pytest.raises(VendorUnavailableError) as exc_info:
                client.get_news("AAPL", "2026-08-01", "2026-08-05")
            assert "invalid JSON" in str(exc_info.value)

    @pytest.mark.parametrize(
        "non_dict_payload",
        [
            [{"id": "event-1"}],
            "not a json object",
            12345,
            True,
            None,
        ],
    )
    def test_json_non_dict_root_payloads_raise_vendor_unavailable(self, non_dict_payload):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = non_dict_payload

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            with pytest.raises(VendorUnavailableError) as exc_info:
                client.get_news("AAPL", "2026-08-01", "2026-08-05")
            assert "invalid JSON" in str(exc_info.value) or "missing or invalid 'events' list" in str(exc_info.value)

    @pytest.mark.parametrize(
        "invalid_events_val",
        [
            "not a list",
            12345,
            None,
            {"nested": "dict"},
            True,
        ],
    )
    def test_events_field_not_a_list_raises_vendor_unavailable(self, invalid_events_val):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"events": invalid_events_val}

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            with pytest.raises(VendorUnavailableError) as exc_info:
                client.get_news("AAPL", "2026-08-01", "2026-08-05")
            assert "missing or invalid 'events' list" in str(exc_info.value)

    def test_parse_utc_timestamp_with_positive_and_negative_offsets(self):
        # Explicit +05:30 offset
        dt_plus = parse_utc_timestamp("2026-08-05T15:30:00+05:30")
        assert dt_plus is not None
        assert dt_plus == datetime(2026, 8, 5, 10, 0, 0, tzinfo=UTC)

        # Explicit -04:00 offset
        dt_minus = parse_utc_timestamp("2026-08-05T07:00:00-04:00")
        assert dt_minus is not None
        assert dt_minus == datetime(2026, 8, 5, 11, 0, 0, tzinfo=UTC)

        # Explicit +00:00 offset
        dt_zero = parse_utc_timestamp("2026-08-05T12:00:00+00:00")
        assert dt_zero is not None
        assert dt_zero == datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)

    def test_parse_utc_timestamp_exact_microsecond_boundary_acceptance_and_excess_rejection(self):
        # 6 digits: accepted
        dt_six = parse_utc_timestamp("2026-08-05T12:00:00.123456Z")
        assert dt_six is not None
        assert dt_six.microsecond == 123456

        # 7 digits: rejected per R5
        assert parse_utc_timestamp("2026-08-05T12:00:00.1234567Z") is None
        # 8 digits: rejected
        assert parse_utc_timestamp("2026-08-05T12:00:00.12345678Z") is None

    def test_article_content_snippet_fallback_and_truncation(self):
        long_content = "Word " * 100  # 500 chars with spaces
        event = canonical_event(summary=None, content=long_content)
        formatted = format_article(event)

        assert "Summary:" not in formatted
        assert "Content: Word Word Word" in formatted
        # Snippet should be truncated to 300 characters plus '...'
        assert formatted.endswith("...")

    def test_format_article_missing_title_or_source_defaults(self):
        event = canonical_event(title="", source="")
        formatted = format_article(event)
        assert "### No title (source: News Intelligence)" in formatted


# =====================================================================
# 2. Crypto Symbol Coverage
# =====================================================================


class TestCryptoSymbolCoverage:
    """Verifies crypto ticker resolution, filtering, and header formatting."""

    @pytest.mark.parametrize(
        ("raw_ticker", "expected_query", "expected_resolved"),
        [
            ("BTC-USDC", "BTC", " (resolved to BTC)"),
            ("ETH-USDC", "ETH", " (resolved to ETH)"),
            ("SOL-USD", "SOL", " (resolved to SOL)"),
            ("DOGE-USD", "DOGE", " (resolved to DOGE)"),
            ("XRP-USD", "XRP", " (resolved to XRP)"),
            ("AVAX-USD", "AVAX", " (resolved to AVAX)"),
            ("DOT-USD", "DOT", " (resolved to DOT)"),
            ("LTC-USD", "LTC", " (resolved to LTC)"),
            ("LTCUSDT", "LTC", " (resolved to LTC)"),
            ("BCH-USD", "BCH", " (resolved to BCH)"),
            ("LINK-USD", "LINK", " (resolved to LINK)"),
            ("LINKUSDT", "LINK", " (resolved to LINK)"),
            ("ethusdt", "ETH", " (resolved to ETH)"),
        ],
    )
    def test_expanded_crypto_pair_mappings(self, raw_ticker, expected_query, expected_resolved):
        query, resolved = map_ticker_for_news_intelligence(raw_ticker)
        assert query == expected_query
        assert resolved == expected_resolved

    @pytest.mark.parametrize(
        "bare_ticker",
        ["DOGE", "XRP", "ADA", "AVAX", "DOT", "LTC", "BCH", "LINK"],
    )
    def test_crypto_bare_symbols_mapping(self, bare_ticker):
        query, resolved = map_ticker_for_news_intelligence(bare_ticker)
        assert query == bare_ticker
        assert resolved == ""

    def test_crypto_get_news_multi_symbol_matching(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        event = canonical_event(symbols=["BTC", "ETH"], title="Bitcoin and Ethereum Rally")
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"events": [event]}

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            # Querying BTC-USD matches
            btc_res = client.get_news("BTC-USD", "2026-08-01", "2026-08-05")
            assert "## BTC-USD (resolved to BTC) News" in btc_res
            assert "Bitcoin and Ethereum Rally" in btc_res

            # Querying ETH-USD matches
            eth_res = client.get_news("ETH-USD", "2026-08-01", "2026-08-05")
            assert "## ETH-USD (resolved to ETH) News" in eth_res
            assert "Bitcoin and Ethereum Rally" in eth_res

            # Querying SOL-USD rejects the event (not in symbols)
            sol_res = client.get_news("SOL-USD", "2026-08-01", "2026-08-05")
            assert "No news found for SOL-USD (resolved to SOL)" in sol_res

    def test_crypto_header_resolution_text_parity(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        event = canonical_event(symbols=["BTC"], title="Bitcoin ETF Inflows")
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"events": [event]}

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            # Ticker with resolution text
            res_pair = client.get_news("BTC-USD", "2026-08-01", "2026-08-05")
            assert res_pair.startswith("## BTC-USD (resolved to BTC) News")

            # Bare ticker without resolution text
            res_bare = client.get_news("BTC", "2026-08-01", "2026-08-05")
            assert res_bare.startswith("## BTC News")
            assert "(resolved to BTC)" not in res_bare


# =====================================================================
# 3. Equity Symbol Coverage
# =====================================================================


class TestEquitySymbolCoverage:
    """Verifies equity ticker resolution, filtering, and header formatting."""

    @pytest.mark.parametrize(
        ("raw_ticker", "expected_query"),
        [
            ("TSLA", "TSLA"),
            ("AMZN", "AMZN"),
            ("META", "META"),
            ("aapl", "AAPL"),
            ("msft", "MSFT"),
            ("9988.HK", "9988.HK"),
            ("NFLX", "NFLX"),
            ("AMD", "AMD"),
            ("005930.KS", "005930.KS"),
        ],
    )
    def test_equity_ticker_normalization(self, raw_ticker, expected_query):
        query, _ = map_ticker_for_news_intelligence(raw_ticker)
        assert query == expected_query

    def test_equity_get_news_symbol_filtering(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        event = canonical_event(symbols=["AAPL", "MSFT"], title="Tech Giants Partnership")
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"events": [event]}

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            aapl_res = client.get_news("AAPL", "2026-08-01", "2026-08-05")
            assert "## AAPL News" in aapl_res
            assert "Tech Giants Partnership" in aapl_res

            msft_res = client.get_news("MSFT", "2026-08-01", "2026-08-05")
            assert "## MSFT News" in msft_res
            assert "Tech Giants Partnership" in msft_res

            nvda_res = client.get_news("NVDA", "2026-08-01", "2026-08-05")
            assert "No news found for NVDA" in nvda_res

    def test_equity_header_formatting(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        event_aapl = canonical_event(symbols=["AAPL"], title="Apple Product Launch")
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"events": [event_aapl]}

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            res_aapl = client.get_news("AAPL", "2026-08-01", "2026-08-05")
            assert res_aapl.startswith("## AAPL News, from 2026-08-01 to 2026-08-05:")
            assert "(resolved to" not in res_aapl

        event_hk = canonical_event(symbols=["0700.HK"], title="Tencent Earnings")
        mock_resp.json.return_value = {"events": [event_hk]}
        with mock.patch.object(client.session, "get", return_value=mock_resp):
            res_hk = client.get_news("700.HK", "2026-08-01", "2026-08-05")
            assert res_hk.startswith("## 700.HK (resolved to 0700.HK) News, from 2026-08-01 to 2026-08-05:")


# =====================================================================
# 4. Point-In-Time As-Of Bi-Temporal Boundaries
# =====================================================================


class TestPointInTimeAsOfBiTemporalBoundaries:
    """Verifies strict microsecond boundary equality and window calculations."""

    def test_explicit_iso_cutoff_vs_plain_date_cutoff(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"events": []}

        with mock.patch.object(client.session, "get", return_value=mock_resp) as mock_get:
            # Plain date expands to microsecond end-of-day
            client.get_news("AAPL", "2026-08-01", "2026-08-05")
            _, kwargs = mock_get.call_args
            params = kwargs["params"]
            assert params["asOf"] == "2026-08-05T23:59:59.999999Z"
            assert params["from"] == "2026-08-01T00:00:00Z"
            assert params["to"] == "2026-08-05T23:59:59.999999Z"

            # Explicit ISO timestamp is preserved
            client.get_news("AAPL", "2026-08-01T08:00:00Z", "2026-08-05T14:30:00Z")
            _, kwargs2 = mock_get.call_args
            params2 = kwargs2["params"]
            assert params2["asOf"] == "2026-08-05T14:30:00Z"
            assert params2["from"] == "2026-08-01T08:00:00Z"
            assert params2["to"] == "2026-08-05T14:30:00Z"

    def test_exact_microsecond_availability_equality(self):
        cutoff = datetime(2026, 8, 5, 12, 0, 0, 500000, tzinfo=UTC)

        # Published and received exactly at cutoff: ACCEPTED
        ev_exact = canonical_event(
            publishedAt="2026-08-05T12:00:00.500000Z",
            receivedAt="2026-08-05T12:00:00.500000Z",
            enrichedAt="2026-08-05T12:00:00.500000Z",
        )
        assert filter_and_sanitize_event(ev_exact, cutoff_dt=cutoff) is not None

        # Received 1 microsecond after cutoff: REJECTED (late receipt lookahead)
        ev_late_rec = canonical_event(
            publishedAt="2026-08-05T12:00:00.500000Z",
            receivedAt="2026-08-05T12:00:00.500001Z",
        )
        assert filter_and_sanitize_event(ev_late_rec, cutoff_dt=cutoff) is None

        # Published 1 microsecond after cutoff: REJECTED (future publication)
        ev_future_pub = canonical_event(
            publishedAt="2026-08-05T12:00:00.500001Z",
            receivedAt="2026-08-05T12:00:00.500000Z",
        )
        assert filter_and_sanitize_event(ev_future_pub, cutoff_dt=cutoff) is None

    def test_exact_microsecond_enrichment_equality(self):
        cutoff = datetime(2026, 8, 5, 12, 0, 0, 500000, tzinfo=UTC)

        # Enriched exactly at cutoff: ACCEPTED with sentiment, summary, and symbols intact
        ev_exact_enr = canonical_event(
            publishedAt="2026-08-05T12:00:00.000000Z",
            receivedAt="2026-08-05T12:00:00.000000Z",
            enrichedAt="2026-08-05T12:00:00.500000Z",
            sentiment={"score": 0.8, "label": "positive"},
            summary="Timely enrichment",
            symbols=["AAPL"],
        )
        res_exact = filter_and_sanitize_event(ev_exact_enr, cutoff_dt=cutoff)
        assert res_exact is not None
        assert res_exact["sentiment"] == {"score": 0.8, "label": "positive"}
        assert res_exact["summary"] == "Timely enrichment"
        assert res_exact["symbols"] == ["AAPL"]

        # Enriched 1 microsecond after cutoff: WITHHELD derived facts
        ev_late_enr = canonical_event(
            publishedAt="2026-08-05T12:00:00.000000Z",
            receivedAt="2026-08-05T12:00:00.000000Z",
            enrichedAt="2026-08-05T12:00:00.500001Z",
            sentiment={"score": 0.8, "label": "positive"},
            summary="Future enrichment",
            symbols=["AAPL"],
        )
        res_late = filter_and_sanitize_event(ev_late_enr, cutoff_dt=cutoff)
        assert res_late is not None
        assert res_late["sentiment"] is None
        assert res_late["summary"] is None
        assert res_late["symbols"] == []

    def test_publication_window_exact_boundaries(self):
        cutoff = datetime(2026, 8, 5, 23, 59, 59, 999999, tzinfo=UTC)
        min_pub = datetime(2026, 8, 1, 0, 0, 0, 0, tzinfo=UTC)
        max_pub = datetime(2026, 8, 5, 23, 59, 59, 999999, tzinfo=UTC)

        # Published at exact min_pub: ACCEPTED
        ev_min = canonical_event(
            publishedAt="2026-08-01T00:00:00.000000Z",
            receivedAt="2026-08-01T00:00:01.000000Z",
        )
        assert filter_and_sanitize_event(ev_min, cutoff_dt=cutoff, min_pub_dt=min_pub, max_pub_dt=max_pub) is not None

        # Published 1 microsecond before min_pub: REJECTED
        ev_before_min = canonical_event(
            publishedAt="2026-07-31T23:59:59.999999Z",
            receivedAt="2026-08-01T00:00:01.000000Z",
        )
        assert filter_and_sanitize_event(ev_before_min, cutoff_dt=cutoff, min_pub_dt=min_pub, max_pub_dt=max_pub) is None

        # Published at exact max_pub: ACCEPTED
        ev_max = canonical_event(
            publishedAt="2026-08-05T23:59:59.999999Z",
            receivedAt="2026-08-05T23:59:59.999999Z",
        )
        assert filter_and_sanitize_event(ev_max, cutoff_dt=cutoff, min_pub_dt=min_pub, max_pub_dt=max_pub) is not None

    def test_global_news_lookback_window_calculation(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"events": []}

        with mock.patch.object(client.session, "get", return_value=mock_resp) as mock_get:
            client.get_global_news("2026-08-10", look_back_days=5, limit=10)
            _, kwargs = mock_get.call_args
            params = kwargs["params"]
            assert params["asOf"] == "2026-08-10T23:59:59.999999Z"
            assert params["from"] == "2026-08-05T00:00:00Z"
            assert params["to"] == "2026-08-10T23:59:59.999999Z"
            assert params["limit"] == 10


# =====================================================================
# 5. Future Rejection Bi-Temporal Guarantees
# =====================================================================


class TestFutureRejectionBiTemporalGuarantees:
    """Verifies future publication, future receipt, and late lookahead rejections."""

    def test_future_published_and_future_received_rejection(self):
        cutoff = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)
        ev_future = canonical_event(
            publishedAt="2026-08-06T12:00:00Z",
            receivedAt="2026-08-06T12:01:00Z",
        )
        assert filter_and_sanitize_event(ev_future, cutoff_dt=cutoff) is None

    def test_past_published_future_received_rejection(self):
        cutoff = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)
        # Article published months ago, but received/ingested tomorrow (late ingestion)
        ev_late_ingest = canonical_event(
            publishedAt="2026-01-01T12:00:00Z",
            receivedAt="2026-08-06T12:00:00Z",
        )
        assert filter_and_sanitize_event(ev_late_ingest, cutoff_dt=cutoff) is None

    def test_global_news_withholds_future_events(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        ev_valid = canonical_event(
            id="ev-valid",
            title="Valid Macro News",
            publishedAt="2026-08-05T10:00:00Z",
            receivedAt="2026-08-05T10:01:00Z",
            symbols=[],
        )
        ev_future = canonical_event(
            id="ev-future",
            title="Future Macro Leak",
            publishedAt="2026-08-06T10:00:00Z",
            receivedAt="2026-08-06T10:01:00Z",
            symbols=[],
        )
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"events": [ev_valid, ev_future]}

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            res = client.get_global_news("2026-08-05")
            assert "Valid Macro News" in res
            assert "Future Macro Leak" not in res

    def test_all_future_articles_returns_no_news_found(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        ev_future = canonical_event(
            publishedAt="2026-08-06T10:00:00Z",
            receivedAt="2026-08-06T10:01:00Z",
        )
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"events": [ev_future]}

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            res = client.get_news("AAPL", "2026-08-01", "2026-08-05")
            assert res == "No news found for AAPL between 2026-08-01 and 2026-08-05"


# =====================================================================
# 6. Malformed Optional and Mandatory Fields
# =====================================================================


class TestMalformedOptionalAndMandatoryFields:
    """Verifies schema contract enforcement across optional and mandatory fields."""

    @pytest.mark.parametrize(
        ("field", "malformed_val"),
        [
            ("summary", 12345),
            ("summary", True),
            ("summary", ["item"]),
            ("clusterId", 999),
            ("clusterId", False),
            ("url", 100),
            ("content", {"nested": "dict"}),
        ],
    )
    def test_malformed_optional_string_fields(self, field, malformed_val):
        ev = canonical_event(**{field: malformed_val})
        assert not validate_canonical_event(ev)

    def test_valid_none_for_all_optional_fields(self):
        ev = canonical_event(
            summary=None,
            clusterId=None,
            url=None,
            content=None,
            sentiment=None,
        )
        assert validate_canonical_event(ev)

    @pytest.mark.parametrize(
        "malformed_sentiment",
        [
            "positive",
            {"score": 0.5},  # missing label
            {"score": 0.5, "label": "bullish"},  # invalid label enum
            {"score": 0.5, "label": "positive", "confidence": -0.1},  # confidence < 0
            {"score": 0.5, "label": "positive", "confidence": 1.5},  # confidence > 1
            {"score": 0.5, "label": "positive", "confidence": "high"},  # non-numeric confidence
            {"score": 0.5, "label": "positive", "confidence": True},  # boolean confidence
            {"score": 2.5, "label": "positive"},  # score > 1.0
            {"score": -1.5, "label": "negative"},  # score < -1.0
        ],
    )
    def test_malformed_sentiment_objects(self, malformed_sentiment):
        ev = canonical_event(sentiment=malformed_sentiment)
        assert not validate_canonical_event(ev)

    @pytest.mark.parametrize(
        ("field", "bad_val"),
        [
            ("id", ""),
            ("id", 123),
            ("source", ""),
            ("source", None),
            ("title", ""),
            ("title", None),
            ("symbols", "AAPL"),  # string instead of list
            ("symbols", [123]),  # list of non-strings
            ("publishedAt", "2026-08-05T12:00:00"),  # timezone-free
            ("publishedAt", "not-a-date"),
            ("receivedAt", "2026-08-05T12:00:00"),  # timezone-free
            ("receivedAt", None),
            ("enrichedAt", "not-a-date"),
        ],
    )
    def test_malformed_mandatory_fields_thorough(self, field, bad_val):
        ev = canonical_event(**{field: bad_val})
        assert not validate_canonical_event(ev)

    def test_mixed_feed_discards_only_malformed_items(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        ev_good = canonical_event(id="good-1", title="Valid Article")
        ev_bad_summary = canonical_event(id="bad-1", title="Bad Summary", summary=123)
        ev_bad_sentiment = canonical_event(id="bad-2", title="Bad Sentiment", sentiment={"score": 5.0, "label": "positive"})

        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"events": [ev_good, ev_bad_summary, ev_bad_sentiment]}

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            res = client.get_news("AAPL", "2026-08-01", "2026-08-05")
            assert "Valid Article" in res
            assert "Bad Summary" not in res
            assert "Bad Sentiment" not in res


# =====================================================================
# 7. Network Errors and HTTP Codes
# =====================================================================


class TestNetworkErrorsAndHttpCodes:
    """Verifies HTTP status codes and network exception handling."""

    def test_http_401_unauthorized_raises_vendor_unavailable(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="bad-token")
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = False
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized: Invalid or expired Bearer token"

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            with pytest.raises(VendorUnavailableError) as exc_info:
                client.get_news("AAPL", "2026-08-01", "2026-08-05")
            assert "401" in str(exc_info.value)

    def test_http_403_forbidden_raises_vendor_unavailable(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = False
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden: Insufficient permissions"

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            with pytest.raises(VendorUnavailableError) as exc_info:
                client.get_news("AAPL", "2026-08-01", "2026-08-05")
            assert "403" in str(exc_info.value)

    def test_http_429_too_many_requests_raises_vendor_unavailable(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = False
        mock_resp.status_code = 429
        mock_resp.text = "Too Many Requests: Rate limit exceeded"

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            with pytest.raises(VendorUnavailableError) as exc_info:
                client.get_news("AAPL", "2026-08-01", "2026-08-05")
            assert "429" in str(exc_info.value)

    def test_http_502_bad_gateway_and_503_service_unavailable(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = False

        for code in (502, 503):
            mock_resp.status_code = code
            mock_resp.text = f"HTTP {code} Server Error"
            with mock.patch.object(client.session, "get", return_value=mock_resp):
                with pytest.raises(VendorUnavailableError) as exc_info:
                    client.get_news("AAPL", "2026-08-01", "2026-08-05")
                assert str(code) in str(exc_info.value)

    def test_connect_and_read_timeout_distinction(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token")

        with mock.patch.object(client.session, "get", side_effect=requests.exceptions.ConnectTimeout("Connection timed out")):
            with pytest.raises(VendorUnavailableError) as exc_info:
                client.get_news("AAPL", "2026-08-01", "2026-08-05")
            assert "timed out" in str(exc_info.value)

        with mock.patch.object(client.session, "get", side_effect=requests.exceptions.ReadTimeout("Read timed out")):
            with pytest.raises(VendorUnavailableError) as exc_info:
                client.get_news("AAPL", "2026-08-01", "2026-08-05")
            assert "timed out" in str(exc_info.value)

    def test_custom_timeout_configuration(self):
        client = NewsIntelligenceClient(base_url="https://news.example.com", token="test-token", timeout=4.5)
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"events": []}

        with mock.patch.object(client.session, "get", return_value=mock_resp) as mock_get:
            client.get_news("AAPL", "2026-08-01", "2026-08-05")
            _, kwargs = mock_get.call_args
            assert kwargs["timeout"] == 4.5


# =====================================================================
# 8. Configured Fallback Chains
# =====================================================================


class TestConfiguredFallbackChains:
    """Verifies router fallback participation across diverse failure modes.

    Leaves the production news_intelligence adapter registration intact in
    router.VENDOR_METHODS and mocks only at the requests.Session.get transport
    boundary, proving real client error handling, adapter wiring, and router
    fallback handoff.
    """

    @pytest.mark.parametrize("method", ["get_news", "get_global_news"])
    @pytest.mark.parametrize(
        ("status_code", "status_text"),
        [
            (401, "Unauthorized"),
            (403, "Forbidden"),
            (429, "Rate limit exceeded"),
            (500, "Internal Server Error"),
            (502, "Bad Gateway"),
            (503, "Service Unavailable"),
        ],
    )
    def test_fallback_chain_triggers_on_http_errors(self, method, status_code, status_text):
        """HTTP error responses from news_intelligence trigger secondary vendor fallback."""
        set_config({
            "data_vendors": {"news_data": "news_intelligence,yfinance"},
            "news_intelligence_url": "https://news.example.com",
        })

        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = False
        mock_resp.status_code = status_code
        mock_resp.text = f"{status_code} {status_text}"

        fallback_mock = mock.Mock(return_value="YFINANCE FALLBACK NEWS")
        args = ("AAPL", "2026-08-01", "2026-08-05") if method == "get_news" else ("2026-08-05",)

        with mock.patch("requests.Session.get", return_value=mock_resp) as mock_get, mock.patch.dict(
            router.VENDOR_METHODS[method],
            {"yfinance": fallback_mock},
            clear=False,
        ):
            res = router.route_to_vendor(method, *args)
            assert res == "YFINANCE FALLBACK NEWS"
            mock_get.assert_called_once()
            fallback_mock.assert_called_once_with(*args)

    @pytest.mark.parametrize("method", ["get_news", "get_global_news"])
    @pytest.mark.parametrize(
        "network_exc",
        [
            requests.exceptions.Timeout("Connection timed out after 10.0s"),
            requests.exceptions.ConnectionError("Connection refused to https://news.example.com"),
        ],
    )
    def test_fallback_chain_triggers_on_network_exceptions(self, method, network_exc):
        """Network timeouts and connection failures trigger secondary vendor fallback."""
        set_config({
            "data_vendors": {"news_data": "news_intelligence,yfinance"},
            "news_intelligence_url": "https://news.example.com",
        })

        fallback_mock = mock.Mock(return_value="YFINANCE NET FALLBACK")
        args = ("AAPL", "2026-08-01", "2026-08-05") if method == "get_news" else ("2026-08-05",)

        with mock.patch("requests.Session.get", side_effect=network_exc) as mock_get, mock.patch.dict(
            router.VENDOR_METHODS[method],
            {"yfinance": fallback_mock},
            clear=False,
        ):
            res = router.route_to_vendor(method, *args)
            assert res == "YFINANCE NET FALLBACK"
            mock_get.assert_called_once()
            fallback_mock.assert_called_once_with(*args)

    @pytest.mark.parametrize("method", ["get_news", "get_global_news"])
    def test_successful_primary_response_does_not_trigger_fallback(self, method):
        """Valid response from news_intelligence serves news without triggering secondary vendor."""
        set_config({
            "data_vendors": {"news_data": "news_intelligence,yfinance"},
            "news_intelligence_url": "https://news.example.com",
        })

        ev = canonical_event(
            symbols=["AAPL"],
            title="Apple Quarterly Earnings",
            published_at="2026-08-03T10:00:00Z",
            received_at="2026-08-03T10:05:00Z",
        )
        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"events": [ev], "count": 1}

        fallback_mock = mock.Mock(return_value="UNEXPECTED FALLBACK")
        args = ("AAPL", "2026-08-01", "2026-08-05") if method == "get_news" else ("2026-08-05",)

        with mock.patch("requests.Session.get", return_value=mock_resp) as mock_get, mock.patch.dict(
            router.VENDOR_METHODS[method],
            {"yfinance": fallback_mock},
            clear=False,
        ):
            res = router.route_to_vendor(method, *args)
            mock_get.assert_called_once()
            fallback_mock.assert_not_called()
            assert "Apple Quarterly Earnings" in res

    @pytest.mark.parametrize("method", ["get_news", "get_global_news"])
    def test_multi_hop_fallback_chain_order_and_arguments(self, method):
        """Verifies multi-hop fallback executes in declared order with preserved arguments."""
        set_config({
            "data_vendors": {"news_data": "news_intelligence,alpha_vantage,yfinance"},
            "news_intelligence_url": "https://news.example.com",
        })

        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = False
        mock_resp.status_code = 429
        mock_resp.text = "Rate limit exceeded"

        call_order = []

        def av_side_effect(*a, **kw):
            call_order.append(("alpha_vantage", a, kw))
            raise VendorUnavailableError("Alpha Vantage rate limit reached")

        def yf_side_effect(*a, **kw):
            call_order.append(("yfinance", a, kw))
            return "MULTI-HOP YF SUCCESS"

        mock_av = mock.Mock(side_effect=av_side_effect)
        mock_yf = mock.Mock(side_effect=yf_side_effect)
        args = ("AAPL", "2026-08-01", "2026-08-05") if method == "get_news" else ("2026-08-05",)

        with mock.patch("requests.Session.get", return_value=mock_resp) as mock_get, mock.patch.dict(
            router.VENDOR_METHODS[method],
            {"alpha_vantage": mock_av, "yfinance": mock_yf},
            clear=False,
        ):
            res = router.route_to_vendor(method, *args)
            assert res == "MULTI-HOP YF SUCCESS"
            mock_get.assert_called_once()
            assert call_order == [
                ("alpha_vantage", args, {}),
                ("yfinance", args, {}),
            ]
            mock_av.assert_called_once_with(*args)
            mock_yf.assert_called_once_with(*args)

    @pytest.mark.parametrize("method", ["get_news", "get_global_news"])
    def test_all_vendors_in_chain_fail_exhaustion(self, method):
        """When all configured vendors fail, router exhausts chain and returns DATA_UNAVAILABLE."""
        set_config({
            "data_vendors": {"news_data": "news_intelligence,yfinance"},
            "news_intelligence_url": "https://news.example.com",
        })

        mock_resp = mock.Mock(spec=requests.Response)
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        mock_yf = mock.Mock(side_effect=VendorUnavailableError("YFinance connection timeout"))
        args = ("AAPL", "2026-08-01", "2026-08-05") if method == "get_news" else ("2026-08-05",)

        with mock.patch("requests.Session.get", return_value=mock_resp) as mock_get, mock.patch.dict(
            router.VENDOR_METHODS[method],
            {"yfinance": mock_yf},
            clear=False,
        ):
            res = router.route_to_vendor(method, *args)
            mock_get.assert_called_once()
            mock_yf.assert_called_once_with(*args)
            assert res.startswith(f"DATA_UNAVAILABLE: no configured vendor could serve {method} right now")


# =====================================================================
# 9. Disabled Default and Default Behavior
# =====================================================================


class TestDisabledDefaultAndDefaultBehavior:
    """Verifies default vendor settings and unconfigured behavior."""

    def test_default_config_news_vendor_is_yfinance(self):
        defaults = default_config.DEFAULT_CONFIG
        assert defaults["data_vendors"]["news_data"] == "yfinance"
        assert defaults["news_intelligence_url"] == "http://localhost:3001"
        assert defaults["news_intelligence_token"] is None
        assert defaults["news_intelligence_timeout"] == 10.0

    def test_router_default_routes_to_yfinance_when_unconfigured(self):
        mock_yf = mock.Mock(return_value="DEFAULT YFINANCE")
        with mock.patch.dict(
            router.VENDOR_METHODS["get_news"],
            {"yfinance": mock_yf},
            clear=False,
        ):
            res = router.route_to_vendor("get_news", "AAPL", "2026-08-01", "2026-08-05")
            assert res == "DEFAULT YFINANCE"
            mock_yf.assert_called_once_with("AAPL", "2026-08-01", "2026-08-05")

    def test_client_instantiation_without_url_raises_vendor_not_configured(self):
        # Empty string URL raises VendorNotConfiguredError
        set_config({"news_intelligence_url": ""})
        with pytest.raises(VendorNotConfiguredError) as exc_info:
            NewsIntelligenceClient()
        assert "not configured" in str(exc_info.value)

    def test_secondary_fallback_chain_yfinance_primary(self):
        set_config({
            "data_vendors": {"news_data": "yfinance,news_intelligence"},
            "news_intelligence_url": "https://news.example.com",
        })

        mock_yf_ok = mock.Mock(return_value="YF PRIMARY SUCCESS")
        ev = canonical_event(symbols=["AAPL"], title="News Intel Secondary Event")
        mock_resp = mock.Mock(spec=requests.Response, ok=True, status_code=200)
        mock_resp.json.return_value = {"events": [ev], "count": 1}

        # When yfinance succeeds, news_intelligence transport is never called
        with mock.patch("requests.Session.get", return_value=mock_resp) as mock_get, mock.patch.dict(
            router.VENDOR_METHODS["get_news"],
            {"yfinance": mock_yf_ok},
            clear=False,
        ):
            res = router.route_to_vendor("get_news", "AAPL", "2026-08-01", "2026-08-05")
            assert res == "YF PRIMARY SUCCESS"
            mock_yf_ok.assert_called_once()
            mock_get.assert_not_called()

        # When yfinance fails, news_intelligence is called as fallback via real adapter and transport
        mock_yf_fail = mock.Mock(side_effect=VendorUnavailableError("YF failed"))

        with mock.patch("requests.Session.get", return_value=mock_resp) as mock_get, mock.patch.dict(
            router.VENDOR_METHODS["get_news"],
            {"yfinance": mock_yf_fail},
            clear=False,
        ):
            res2 = router.route_to_vendor("get_news", "AAPL", "2026-08-01", "2026-08-05")
            assert "News Intel Secondary Event" in res2
            mock_yf_fail.assert_called_once()
            mock_get.assert_called_once()

    def test_tool_vendors_override_takes_precedence_over_data_vendors(self):
        set_config({
            "data_vendors": {"news_data": "yfinance"},
            "tool_vendors": {"get_news": "news_intelligence"},
            "news_intelligence_url": "https://news.example.com",
        })

        ev = canonical_event(symbols=["AAPL"], title="Tool Vendor Override Article")
        mock_resp = mock.Mock(spec=requests.Response, ok=True, status_code=200)
        mock_resp.json.return_value = {"events": [ev], "count": 1}
        mock_yf = mock.Mock(return_value="YFINANCE UNUSED")

        with mock.patch("requests.Session.get", return_value=mock_resp) as mock_get, mock.patch.dict(
            router.VENDOR_METHODS["get_news"],
            {"yfinance": mock_yf},
            clear=False,
        ):
            res = router.route_to_vendor("get_news", "AAPL", "2026-08-01", "2026-08-05")
            assert "Tool Vendor Override Article" in res
            mock_get.assert_called_once()
            mock_yf.assert_not_called()
