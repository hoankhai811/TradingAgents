"""Strict request validation and shared-contract shaped run projections."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingagents.backtest import iter_grid
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import safe_ticker_component

Analyst = Literal["market", "social", "news", "fundamentals"]
DEFAULT_ANALYSTS = ["market", "social", "news", "fundamentals"]
MAX_GRID_CELLS = 256


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _ticker(value: str) -> str:
    return safe_ticker_component(value)


def _date(value: str) -> str:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value or parsed > date.fromisoformat(get_current_date()):
        raise ValueError("date must be a canonical, non-future YYYY-MM-DD date")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PositionInput(StrictModel):
    ticker: str
    quantity: float
    average_price: float | None = Field(default=None, ge=0)

    @field_validator("ticker")
    @classmethod
    def valid_ticker(cls, value: str) -> str:
        return _ticker(value)


class PortfolioInput(StrictModel):
    positions: list[PositionInput]
    cash: float | None = None
    currency: str | None = None
    asOf: AwareDatetime | None = None


class RequestBase(StrictModel):
    assetType: Literal["stock", "crypto"] = "stock"
    selectedAnalysts: list[Analyst] = Field(
        default_factory=lambda: DEFAULT_ANALYSTS.copy(), min_length=1
    )
    portfolio: PortfolioInput | None = None

    @field_validator("selectedAnalysts")
    @classmethod
    def unique_analysts(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("selectedAnalysts must be unique")
        return value


class AnalysisRequest(RequestBase):
    ticker: str
    tradeDate: str
    researchDepth: Literal["quick", "standard", "deep"] = "standard"

    @field_validator("ticker")
    @classmethod
    def valid_ticker(cls, value: str) -> str:
        return _ticker(value)

    @field_validator("tradeDate")
    @classmethod
    def valid_date(cls, value: str) -> str:
        return _date(value)


class BacktestRequest(RequestBase):
    tickers: list[str] = Field(min_length=1)
    startDate: str
    endDate: str
    everyNDays: int = Field(default=7, ge=1)

    @field_validator("tickers")
    @classmethod
    def unique_tickers(cls, value: list[str]) -> list[str]:
        if len(value) != len({ticker.upper() for ticker in value}):
            raise ValueError("tickers must be unique")
        return [_ticker(ticker) for ticker in value]

    @field_validator("startDate", "endDate")
    @classmethod
    def valid_date(cls, value: str) -> str:
        return _date(value)

    @model_validator(mode="after")
    def bounded_grid(self) -> BacktestRequest:
        start, end = date.fromisoformat(self.startDate), date.fromisoformat(self.endDate)
        if end < start:
            raise ValueError("backtest grid ends before it starts")
        if ((end - start).days // self.everyNDays + 1) * len(self.tickers) > MAX_GRID_CELLS:
            raise ValueError(f"backtest grid exceeds {MAX_GRID_CELLS} cells")
        iter_grid(self.startDate, self.endDate, self.everyNDays)
        return self


def pending_run(kind: str, job_id: str, request: dict) -> dict:
    if kind == "analysis":
        result = {
            "runId": job_id,
            "asset": {"ticker": request["ticker"], "assetType": request["assetType"]},
            "tradeDate": request["tradeDate"],
            "status": "pending",
            "selectedAnalysts": request["selectedAnalysts"],
            "researchDepth": request["researchDepth"],
            "createdAt": utc_now(),
        }
    elif kind == "backtest":
        result = {
            "backtestId": job_id,
            "tickers": request["tickers"],
            "startDate": request["startDate"],
            "endDate": request["endDate"],
            "everyNDays": request["everyNDays"],
            "assetType": request["assetType"],
            "selectedAnalysts": request["selectedAnalysts"],
            "status": "pending",
            "cellsTotal": len(
                iter_grid(request["startDate"], request["endDate"], request["everyNDays"])
            )
            * len(request["tickers"]),
            "cellsAttempted": 0,
            "cellsRun": 0,
            "cellsSkipped": 0,
            "createdAt": utc_now(),
        }
    else:
        raise ValueError("unknown job kind")
    if request.get("portfolio") is not None and kind == "analysis":
        result["portfolio"] = request["portfolio"]
    return result
