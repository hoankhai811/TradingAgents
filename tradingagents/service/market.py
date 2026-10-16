"""Bounded quote lookup and truthful native daily market observations."""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.dataflows.config import run_config_context
from tradingagents.dataflows.files import replace_file
from tradingagents.dataflows.symbols import crypto_base, normalize_symbol, safe_ticker_component
from tradingagents.dataflows.vendors.yahoo import lookup
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv


class MarketUnavailable(Exception):
    """A market read did not yield a safe usable answer."""


MAX_RESULT_BYTES = 1024 * 1024


def read_result(kind: str, args: tuple, config: dict, cache_dir: Path, result_path: Path) -> None:
    """Run one read inside a killable child and atomically publish a capped result."""
    try:
        lookup.set_cache_location(Path(cache_dir) / "yfinance-cache")
        result = search_assets(*args) if kind == "search" else market_data(*args, config, cache_dir)
        payload = {"result": result}
    except Exception:
        payload = {"error": "DEPENDENCY_UNAVAILABLE"}
    encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_RESULT_BYTES:
        encoded = b'{"error":"DEPENDENCY_UNAVAILABLE"}'
    replace_file(result_path, lambda path: Path(path).write_bytes(encoded))


def search_assets(query: str) -> dict:
    query = validate_query(query)
    try:
        quotes = lookup.search_quotes(query)
        assets = []
        seen = set()
        for quote in quotes:
            if not isinstance(quote, dict):
                raise MarketUnavailable("malformed quote")
            kind = quote.get("quoteType")
            if kind not in {"EQUITY", "ETF", "CRYPTOCURRENCY"}:
                continue
            symbol = safe_ticker_component(quote.get("symbol"))
            asset_type = "crypto" if kind == "CRYPTOCURRENCY" else "stock"
            if asset_type == "crypto" and not _explicit_usd_pair(symbol):
                continue
            key = (asset_type, symbol)
            if key in seen:
                continue
            seen.add(key)
            asset = {"ticker": symbol, "assetType": asset_type}
            for target, value in (
                ("name", quote.get("shortname") or quote.get("longname")),
                ("exchange", quote.get("exchange")),
                ("currency", quote.get("currency")),
            ):
                if value is not None:
                    if not isinstance(value, str) or not value.strip():
                        raise MarketUnavailable("malformed quote metadata")
                    asset[target] = value.strip()
            if asset_type == "crypto":
                if asset.get("currency", "USD") != "USD":
                    continue
                asset["currency"] = "USD"
            assets.append(asset)
        return {"query": query, "assets": assets[:8]}
    except MarketUnavailable:
        raise
    except Exception as exc:
        raise MarketUnavailable("market search unavailable") from exc


def _explicit_usd_pair(symbol: str) -> bool:
    if not symbol.upper().endswith("-USD"):
        return False
    base = symbol[:-4]
    return bool(base and safe_ticker_component(base))


def validate_query(query: str) -> str:
    query = query.strip()
    if not 1 <= len(query) <= 80:
        raise ValueError("query length is outside the supported range")
    return query


def _day(value: str) -> date:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("date must be canonical YYYY-MM-DD")
    return parsed


def market_data(
    asset_type: str, symbol: str, start_date: str, end_date: str, config: dict, cache_dir: Path
) -> dict:
    canonical, start, end, today = validate_market_request(asset_type, symbol, start_date, end_date)
    settled_before = min(today, datetime.now(UTC).date())
    read_cache = Path(cache_dir) / "read-cache"
    scoped = {**config, "data_cache_dir": str(read_cache)}
    try:
        frame = run_config_context(scoped).run(load_ohlcv, canonical, end_date, fill_gaps=False)
        if not isinstance(frame, pd.DataFrame) or "Date" not in frame or "Close" not in frame:
            raise MarketUnavailable("malformed market data")
        bars = []
        seen = set()
        for row in frame.to_dict("records"):
            day = pd.Timestamp(row["Date"]).date()
            if not start <= day <= end or day >= settled_before:
                continue
            if day in seen:
                raise MarketUnavailable("duplicate market date")
            seen.add(day)
            close = _number(row["Close"], required=True)
            if close <= 0:
                raise MarketUnavailable("invalid close")
            bar = {"date": day.isoformat(), "close": close}
            for source, target in (
                ("Open", "open"),
                ("High", "high"),
                ("Low", "low"),
                ("Volume", "volume"),
            ):
                number = _number(row.get(source))
                if number is not None:
                    if number < 0 or (target != "volume" and number == 0):
                        raise MarketUnavailable("invalid optional price")
                    bar[target] = number
            bars.append(bar)
        bars.sort(key=lambda bar: bar["date"])
    except MarketUnavailable:
        raise
    except Exception as exc:
        raise MarketUnavailable("market data unavailable") from exc
    asset = {"ticker": canonical, "assetType": asset_type}
    if asset_type == "crypto":
        asset["currency"] = "USD"
    return {"asset": asset, "startDate": start_date, "endDate": end_date, "bars": bars}


def validate_market_request(asset_type: str, symbol: str, start_date: str, end_date: str):
    if asset_type not in {"stock", "crypto"}:
        raise ValueError("unsupported asset type")
    safe_ticker_component(symbol)
    if asset_type == "crypto":
        if _explicit_usd_pair(symbol):
            canonical = symbol.upper()
        elif symbol.upper().endswith("USD") and crypto_base(symbol):
            canonical = normalize_symbol(symbol)
        else:
            raise ValueError("crypto requires an explicit USD pair")
    else:
        canonical = normalize_symbol(symbol)
        if (
            crypto_base(symbol) is not None
            or symbol.upper().endswith(("-USD", "-USDT", "-USDC"))
            or canonical.endswith(("-USD", "-USDT", "-USDC", "=F", "=X"))
            or canonical.startswith("^")
        ):
            raise ValueError("stock route does not support this instrument")
    safe_ticker_component(canonical)
    start, end, today = _day(start_date), _day(end_date), date.today()
    if start > end or end > today or (end - start).days > 366:
        raise ValueError("unsupported date range")
    return canonical, start, end, today


def _number(value, *, required: bool = False) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        raise MarketUnavailable("boolean market number")
    if value is None or pd.isna(value):
        if required:
            raise MarketUnavailable("missing close")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MarketUnavailable("invalid market number") from exc
    if not math.isfinite(number):
        raise MarketUnavailable("nonfinite market number")
    return number
