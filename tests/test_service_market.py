"""Service market reads keep vendor data truthful and requests bounded."""

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows.vendors.yahoo import lookup


def test_yahoo_lookup_reports_search_and_cache_failures_as_vendor_errors(monkeypatch, tmp_path):
    from tradingagents.dataflows.errors import VendorUnavailableError

    monkeypatch.setattr(
        lookup.yf, "Search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("private"))
    )
    with pytest.raises(VendorUnavailableError):
        lookup.search_quotes("nvda")
    monkeypatch.setattr(lookup.yf, "Search", lambda *a, **k: SimpleNamespace(quotes=None))
    with pytest.raises(VendorUnavailableError):
        lookup.search_quotes("nvda")
    monkeypatch.setattr(
        lookup.yf,
        "set_tz_cache_location",
        lambda _: (_ for _ in ()).throw(ValueError("private")),
    )
    with pytest.raises(VendorUnavailableError):
        lookup.set_cache_location(tmp_path)


def test_search_uses_quotes_only_and_omits_unsupported_or_incomplete_metadata(monkeypatch):
    from tradingagents.service import market

    seen = {}

    def search(query, **options):
        seen.update(query=query, **options)
        return SimpleNamespace(
            quotes=[
                {"quoteType": "EQUITY", "symbol": "NVDA", "shortname": "Nvidia", "exchange": "NMS"},
                {"quoteType": "ETF", "symbol": "SPY"},
                {"quoteType": "CRYPTOCURRENCY", "symbol": "BTC-USD", "shortname": "Bitcoin"},
                {"quoteType": "CRYPTOCURRENCY", "symbol": "BTC-USDT"},
                {"quoteType": "INDEX", "symbol": "^GSPC"},
            ]
        )

    monkeypatch.setattr(lookup.yf, "Search", search)
    assert market.search_assets("  nvda  ") == {
        "query": "nvda",
        "assets": [
            {"ticker": "NVDA", "assetType": "stock", "name": "Nvidia", "exchange": "NMS"},
            {"ticker": "SPY", "assetType": "stock"},
            {"ticker": "BTC-USD", "assetType": "crypto", "name": "Bitcoin", "currency": "USD"},
        ],
    }
    assert seen == {
        "query": "nvda",
        "max_results": 8,
        "news_count": 0,
        "lists_count": 0,
        "recommended": 0,
        "include_cb": False,
        "include_nav_links": False,
        "include_research": False,
        "include_cultural_assets": False,
        "enable_fuzzy_query": False,
        "timeout": 10,
        "raise_errors": True,
    }


def test_search_empty_malformed_and_failure(monkeypatch):
    from tradingagents.dataflows.errors import VendorUnavailableError
    from tradingagents.service import market

    monkeypatch.setattr(lookup.yf, "Search", lambda *a, **k: SimpleNamespace(quotes=[]))
    assert market.search_assets("A") == {"query": "A", "assets": []}
    for query in (" ", "a" * 81):
        with pytest.raises(ValueError):
            market.search_assets(query)
    monkeypatch.setattr(
        lookup.yf, "Search", lambda *a, **k: SimpleNamespace(quotes=[{"quoteType": "EQUITY"}])
    )
    with pytest.raises(market.MarketUnavailable):
        market.search_assets("A")
    monkeypatch.setattr(
        lookup.yf, "Search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("private upstream"))
    )
    with pytest.raises(market.MarketUnavailable) as error:
        market.search_assets("A")
    assert isinstance(error.value.__cause__, VendorUnavailableError)


def test_search_preserves_long_name_when_short_name_is_absent(monkeypatch):
    from tradingagents.service import market

    monkeypatch.setattr(
        lookup.yf,
        "Search",
        lambda *a, **k: SimpleNamespace(
            quotes=[{"quoteType": "EQUITY", "symbol": "NVDA", "longname": "NVIDIA Corporation"}]
        ),
    )
    assert market.search_assets("nvda")["assets"] == [
        {"ticker": "NVDA", "assetType": "stock", "name": "NVIDIA Corporation"}
    ]


@pytest.mark.parametrize("symbol", ["BTC", "BTC-USDT", "BTC-USDC", "UNKNOWNUSD", "../BTC-USD"])
def test_crypto_requires_explicit_verified_usd_pair(symbol, tmp_path):
    from tradingagents.service.market import market_data

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    with pytest.raises(ValueError):
        market_data("crypto", symbol, yesterday, yesterday, {}, tmp_path)


@pytest.mark.parametrize(
    "symbol",
    [
        "BTC-USDT",
        "ETH-USDC",
        "BTCUSD",
        "BTC-USD",
        "PEPE-USDT",
        "PEPE-USDT+",
        "PEPE-USDC+",
        "PEPE-USDT++",
        "PEPE-USDC+++",
        "XAUUSD",
        "EURUSD",
        "SPX500",
        "GC=F",
        "EURUSD=X",
        "^GSPC",
    ],
)
def test_stock_market_rejects_quoted_crypto_and_native_nonstocks_before_vendor(
    symbol, monkeypatch, tmp_path
):
    from tradingagents.service import market

    monkeypatch.setattr(market, "load_ohlcv", lambda *a, **k: pytest.fail("vendor reached"))
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    with pytest.raises(ValueError):
        market.market_data("stock", symbol, yesterday, yesterday, {}, tmp_path)


@pytest.mark.parametrize(
    ("symbol", "canonical"),
    [
        ("BTC", "BTC"),
        ("BRK-B", "BRK-B"),
        ("NVDA", "NVDA"),
        ("700.HK", "0700.HK"),
        ("600519.SH", "600519.SS"),
    ],
)
def test_stock_market_keeps_valid_bare_equity_and_exchange_normalization(
    symbol, canonical, monkeypatch, tmp_path
):
    from tradingagents.service import market

    yesterday = (date.today() - timedelta(days=1)).isoformat()

    def load(actual, cutoff, *, fill_gaps):
        assert (actual, cutoff, fill_gaps) == (canonical, yesterday, False)
        return pd.DataFrame([{"Date": pd.Timestamp(yesterday), "Close": 10.0}])

    monkeypatch.setattr(market, "load_ohlcv", load)
    result = market.market_data("stock", symbol, yesterday, yesterday, {}, tmp_path)
    assert result["asset"] == {"ticker": canonical, "assetType": "stock"}


def test_market_native_nonfilled_dates_and_optional_cells(monkeypatch, tmp_path):
    from tradingagents.service import market

    today = date.today()
    settled_before = min(today, datetime.now(UTC).date())
    start = settled_before - timedelta(days=3)
    yesterday = settled_before - timedelta(days=1)
    seen = {}

    def load(symbol, cutoff, *, fill_gaps):
        from tradingagents.dataflows.config import get_config

        seen.update(
            symbol=symbol, cutoff=cutoff, fill_gaps=fill_gaps, cache=get_config()["data_cache_dir"]
        )
        return pd.DataFrame(
            [
                {"Date": pd.Timestamp(start), "Close": 10.0, "Open": float("nan")},
                {"Date": pd.Timestamp(yesterday), "Close": 12.0, "Open": 11.0},
                {"Date": pd.Timestamp(today), "Close": 13.0, "Open": 12.0},
            ]
        )

    monkeypatch.setattr(market, "load_ohlcv", load)
    result = market.market_data(
        "crypto", "BTC-USD", start.isoformat(), today.isoformat(), {}, tmp_path
    )
    assert result == {
        "asset": {"ticker": "BTC-USD", "assetType": "crypto", "currency": "USD"},
        "startDate": start.isoformat(),
        "endDate": today.isoformat(),
        "bars": [
            {"date": start.isoformat(), "close": 10.0},
            {"date": yesterday.isoformat(), "close": 12.0, "open": 11.0},
        ],
    }
    assert seen == {
        "symbol": "BTC-USD",
        "cutoff": today.isoformat(),
        "fill_gaps": False,
        "cache": str(tmp_path / "read-cache"),
    }


def test_market_excludes_utc_current_day_when_local_calendar_is_ahead(monkeypatch, tmp_path):
    from datetime import UTC

    from tradingagents.service import market

    class LocalDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 14)

    class UTCClock:
        @classmethod
        def now(cls, zone):
            assert zone is UTC
            return cls()

        def date(self):
            return date(2026, 10, 13)

    monkeypatch.setattr(market, "date", LocalDate)
    monkeypatch.setattr(market, "datetime", UTCClock, raising=False)
    monkeypatch.setattr(
        market,
        "load_ohlcv",
        lambda *a, **k: pd.DataFrame(
            [
                {"Date": pd.Timestamp("2026-10-12"), "Close": 10},
                {"Date": pd.Timestamp("2026-10-13"), "Close": 11},
            ]
        ),
    )
    result = market.market_data("crypto", "BTC-USD", "2026-10-12", "2026-10-14", {}, tmp_path)
    assert result["bars"] == [{"date": "2026-10-12", "close": 10.0}]


def test_market_rejects_invalid_ranges_and_surviving_bad_values(monkeypatch, tmp_path):
    from tradingagents.service import market

    today = date.today()
    yesterday = (min(today, datetime.now(UTC).date()) - timedelta(days=1)).isoformat()
    monkeypatch.setattr(market, "load_ohlcv", lambda *a, **k: pytest.fail("vendor reached"))
    for start, end in (
        ("2026-02-30", yesterday),
        (today.isoformat(), yesterday),
        ((date.fromisoformat(yesterday) - timedelta(days=367)).isoformat(), yesterday),
        (yesterday, (today + timedelta(days=1)).isoformat()),
    ):
        with pytest.raises(ValueError):
            market.market_data("stock", "NVDA", start, end, {}, tmp_path)
    with pytest.raises(ValueError):
        market.market_data("stock", "../NVDA", yesterday, yesterday, {}, tmp_path)
    for rows in (
        [{"Date": pd.Timestamp(yesterday), "Close": 0}],
        [{"Date": pd.Timestamp(yesterday), "Close": float("inf")}],
        [{"Date": pd.Timestamp(yesterday), "Close": 1, "High": float("inf")}],
        [
            {"Date": pd.Timestamp(yesterday), "Close": 1},
            {"Date": pd.Timestamp(yesterday), "Close": 2},
        ],
    ):
        monkeypatch.setattr(market, "load_ohlcv", lambda *a, rows=rows, **k: pd.DataFrame(rows))
        with pytest.raises(market.MarketUnavailable):
            market.market_data("stock", "NVDA", yesterday, yesterday, {}, tmp_path)


def test_native_loader_omits_bad_dates_closes_and_does_not_fill_gaps(monkeypatch, tmp_path):
    from tradingagents.dataflows.vendors.yahoo import ohlcv
    from tradingagents.service.market import market_data

    settled_before = min(date.today(), datetime.now(UTC).date())
    days = [(settled_before - timedelta(days=offset)).isoformat() for offset in (4, 3, 2, 1)]
    vendor_rows = pd.DataFrame(
        [
            {"Date": days[0], "Close": "10", "Open": "9"},
            {"Date": days[1], "Close": "bad", "Open": "10"},
            {"Date": "not-a-date", "Close": "11", "Open": "10"},
            {"Date": days[3], "Close": "12", "Open": "bad"},
        ]
    ).set_index("Date")

    class Ticker:
        def __init__(self, symbol):
            assert symbol == "NVDA"

        def history(self, **kwargs):
            assert kwargs["auto_adjust"] is True
            return vendor_rows

    monkeypatch.setattr(ohlcv.yf, "Ticker", Ticker)
    result = market_data("stock", "NVDA", days[0], days[3], {}, tmp_path)
    assert result["bars"] == [
        {"date": days[0], "close": 10.0, "open": 9.0},
        {"date": days[3], "close": 12.0},
    ]


@pytest.mark.parametrize("column", ["Close", "Volume"])
def test_native_boolean_price_cell_is_not_serialized_as_one(monkeypatch, tmp_path, column):
    from tradingagents.dataflows.vendors.yahoo import ohlcv
    from tradingagents.service.market import MarketUnavailable, market_data

    yesterday = (min(date.today(), datetime.now(UTC).date()) - timedelta(days=1)).isoformat()
    row = {"Date": yesterday, "Close": 10.0, "Volume": 100.0}
    row[column] = True

    class Ticker:
        def __init__(self, symbol):
            assert symbol == "NVDA"

        def history(self, **kwargs):
            return pd.DataFrame([row]).set_index("Date")

    monkeypatch.setattr(ohlcv.yf, "Ticker", Ticker)
    with pytest.raises(MarketUnavailable):
        market_data("stock", "NVDA", yesterday, yesterday, {}, tmp_path)


def test_actual_market_payloads_validate_against_shared_openapi(monkeypatch, tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import yaml
    from jsonschema import Draft202012Validator, FormatChecker
    from referencing import Registry, Resource

    from tradingagents.service import market

    contracts = next(
        parent / "trading-contracts"
        for parent in Path(__file__).resolve().parents
        if (parent / "trading-contracts" / "openapi").exists()
    )
    api_file = contracts / "openapi" / "trading-agents.yaml"
    api = yaml.safe_load(api_file.read_text(encoding="utf-8"))
    registry = Registry()
    for file in (contracts / "schemas").glob("*.json"):
        schema = json.loads(file.read_text(encoding="utf-8"))
        resource = Resource.from_contents(schema)
        registry = registry.with_resource(file.as_uri(), resource).with_resource(
            schema["$id"], resource
        )

    def validate(component, value):
        root = {
            "$id": api_file.as_uri(),
            "$ref": f"#/components/schemas/{component}",
            "components": api["components"],
        }
        Draft202012Validator(root, registry=registry, format_checker=FormatChecker()).validate(
            value
        )

    monkeypatch.setattr(
        lookup.yf,
        "Search",
        lambda *a, **k: SimpleNamespace(
            quotes=[
                {"quoteType": "EQUITY", "symbol": "NVDA", "shortname": "Nvidia"},
            ]
        ),
    )
    validate("AssetSearchResponse", market.search_assets("nvda"))
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    monkeypatch.setattr(
        market,
        "load_ohlcv",
        lambda *a, **k: pd.DataFrame(
            [
                {"Date": pd.Timestamp(yesterday), "Close": 10.0, "Volume": 0},
            ]
        ),
    )
    validate(
        "MarketDataResponse",
        market.market_data("stock", "NVDA", yesterday, yesterday, {}, tmp_path),
    )


def test_read_child_places_yfinance_internal_caches_under_service_state(monkeypatch, tmp_path):
    import json

    from tradingagents.service import market

    locations = []
    monkeypatch.setattr(lookup.yf, "set_tz_cache_location", locations.append)
    monkeypatch.setattr(market, "search_assets", lambda query: {"query": query, "assets": []})
    result_file = tmp_path / "result.json"
    market.read_result("search", ("nvda",), {}, tmp_path, result_file)
    assert locations == [str(tmp_path / "yfinance-cache")]
    assert json.loads(result_file.read_text(encoding="utf-8")) == {
        "result": {"query": "nvda", "assets": []}
    }


def test_read_child_caps_oversized_result_without_partial_json(monkeypatch, tmp_path):
    import json

    from tradingagents.service import market

    monkeypatch.setattr(lookup.yf, "set_tz_cache_location", lambda _: None)
    monkeypatch.setattr(
        market,
        "search_assets",
        lambda query: {
            "query": query,
            "assets": [
                {
                    "ticker": "NVDA",
                    "assetType": "stock",
                    "name": "x" * (market.MAX_RESULT_BYTES + 1),
                }
            ],
        },
    )
    result_file = tmp_path / "result.json"
    market.read_result("search", ("nvda",), {}, tmp_path, result_file)
    assert result_file.stat().st_size < market.MAX_RESULT_BYTES
    assert json.loads(result_file.read_text(encoding="utf-8")) == {
        "error": "DEPENDENCY_UNAVAILABLE"
    }


def test_read_child_cache_setup_failure_is_dependency_error(monkeypatch, tmp_path):
    import json

    from tradingagents.service import market

    monkeypatch.setattr(
        lookup.yf,
        "set_tz_cache_location",
        lambda _: (_ for _ in ()).throw(ValueError("private cache failure")),
    )
    result_file = tmp_path / "result.json"
    market.read_result("search", ("nvda",), {}, tmp_path, result_file)
    assert json.loads(result_file.read_text(encoding="utf-8")) == {
        "error": "DEPENDENCY_UNAVAILABLE"
    }


def test_read_child_search_outage_is_dependency_error(monkeypatch, tmp_path):
    import json

    from tradingagents.service import market

    monkeypatch.setattr(lookup.yf, "set_tz_cache_location", lambda _: None)
    monkeypatch.setattr(
        lookup.yf, "Search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("private"))
    )
    result_file = tmp_path / "result.json"
    market.read_result("search", ("nvda",), {}, tmp_path, result_file)
    assert json.loads(result_file.read_text(encoding="utf-8")) == {
        "error": "DEPENDENCY_UNAVAILABLE"
    }
