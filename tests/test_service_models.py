"""The HTTP boundary must reject bad paid work before it reaches storage."""

import json
from datetime import date, timedelta

import pytest
from pydantic import ValidationError


def test_analysis_rejects_invalid_calendar_future_date_unknown_field_and_unsafe_ticker():
    from tradingagents.service.models import AnalysisRequest

    base = {"ticker": "NVDA", "tradeDate": "2026-01-05"}
    for change in (
        {"tradeDate": "2026-02-30"},
        {"tradeDate": "2999-01-01"},
        {"provider": "google"},
        {"ticker": "../../etc"},
    ):
        with pytest.raises(ValidationError):
            AnalysisRequest.model_validate(base | change)
    assert AnalysisRequest.model_validate(base).ticker == "NVDA"


def test_backtest_rejects_oversized_grid_and_duplicate_tickers():
    from tradingagents.service.models import BacktestRequest

    base = {"tickers": ["NVDA"], "startDate": "2026-01-01", "endDate": "2026-01-05"}
    for change in (
        {"tickers": ["NVDA", "nvda"]},
        {"tickers": ["NVDA"] * 257},
        {"startDate": "2025-01-01", "endDate": "2026-01-05", "everyNDays": 1},
    ):
        with pytest.raises(ValidationError):
            BacktestRequest.model_validate(base | change)
    assert BacktestRequest.model_validate(base).everyNDays == 7


def test_portfolio_values_and_analysts_are_strict():
    from tradingagents.service.models import AnalysisRequest

    base = {"ticker": "NVDA", "tradeDate": "2026-01-05"}
    for change in (
        {"selectedAnalysts": ["news", "news"]},
        {"selectedAnalysts": ["unregistered"]},
        {"portfolio": {"positions": [{"ticker": "NVDA", "quantity": "a lot"}]}},
        {"portfolio": {"positions": [{"ticker": "NVDA", "quantity": 1, "average_price": -1}]}},
        {"portfolio": {"positions": [], "privateKey": "x"}},
    ):
        with pytest.raises(ValidationError):
            AnalysisRequest.model_validate(base | change)


def test_asof_remains_request_metadata_without_native_valuation_claim():
    from tradingagents.service.models import AnalysisRequest

    payload = {
        "ticker": "NVDA",
        "tradeDate": "2026-01-05",
        "portfolio": {"positions": [], "asOf": "2026-01-01T00:00:00Z"},
    }
    request = AnalysisRequest.model_validate_json(json.dumps(payload))
    assert request.model_dump(mode="json")["portfolio"]["asOf"] == "2026-01-01T00:00:00Z"


def test_asof_requires_timezone_aware_timestamp_from_json():
    from tradingagents.service.models import AnalysisRequest

    base = {"ticker": "NVDA", "tradeDate": "2026-01-05"}
    for as_of in ("2026-01-01T12:00:00", "2026-01-01"):
        with pytest.raises(ValidationError):
            AnalysisRequest.model_validate_json(
                json.dumps(base | {"portfolio": {"positions": [], "asOf": as_of}})
            )
    aware = "2026-01-01T12:00:00+07:00"
    request = AnalysisRequest.model_validate_json(
        json.dumps(base | {"portfolio": {"positions": [], "asOf": aware}})
    )
    assert request.model_dump(mode="json")["portfolio"]["asOf"] == aware


def test_trade_date_uses_native_calendar_day(monkeypatch):
    from tradingagents.service import models

    native_today = (date.today() + timedelta(days=1)).isoformat()
    monkeypatch.setattr(models, "get_current_date", lambda: native_today, raising=False)
    assert (
        models.AnalysisRequest.model_validate(
            {"ticker": "NVDA", "tradeDate": native_today}
        ).tradeDate
        == native_today
    )


def test_huge_grid_is_rejected_before_materializing_dates(monkeypatch):
    from tradingagents.service import models

    monkeypatch.setattr(
        models,
        "iter_grid",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("grid materialized before bound check")
        ),
    )
    with pytest.raises(ValidationError):
        models.BacktestRequest.model_validate(
            {
                "tickers": ["NVDA"],
                "startDate": "0001-01-01",
                "endDate": "2026-01-01",
                "everyNDays": 1,
            }
        )
