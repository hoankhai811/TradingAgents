"""One claimed job's native engine invocation and safe contract projection."""

from __future__ import annotations

import copy
import logging
import os

from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.default_config import build_default_config
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV, get_api_key_env
from tradingagents.llm_clients.errors import is_provider_limit_error
from tradingagents.llm_clients.factory import tier_provider
from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS
from tradingagents.portfolio import PortfolioContext
from tradingagents.service.models import utc_now
from tradingagents.service.store import RunStore

logger = logging.getLogger(__name__)


class ProviderNotConfigured(Exception):
    pass


def ensure_provider_configured(config: dict) -> None:
    for tier in ("quick", "deep"):
        try:
            provider = tier_provider(config, tier)
        except (KeyError, AttributeError, TypeError):
            raise ProviderNotConfigured("invalid provider configuration") from None
        if provider not in PROVIDER_API_KEY_ENV:
            raise ProviderNotConfigured("unknown provider")
        spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
        if spec and spec.require_base_url:
            own_url = config.get(f"{tier}_think_backend_url")
            inherited_url = (
                config.get("backend_url") if provider == config["llm_provider"].lower() else None
            )
            if not (own_url or inherited_url):
                raise ProviderNotConfigured("provider endpoint is missing")
        if spec and spec.key_optional:
            continue
        key_env = get_api_key_env(provider)
        if key_env and not os.getenv(key_env):
            raise ProviderNotConfigured("provider credentials are missing")


def _native_portfolio(request: dict):
    payload = request.get("portfolio")
    if payload is None:
        return None
    return PortfolioContext.model_validate(
        {key: value for key, value in payload.items() if key != "asOf"}
    )


def _run_config(base: dict, request: dict, job_id: str, store: RunStore) -> dict:
    config = build_default_config()
    config.update(copy.deepcopy(base))
    root = store.path.parent / "runs" / job_id
    config["results_dir"] = str(root / "results")
    config["data_cache_dir"] = str(root / "cache")
    config["memory_log_path"] = str(root / "memory" / "trading_memory.md")
    if "researchDepth" in request:
        depth = {"quick": 1, "standard": 3, "deep": 5}[request["researchDepth"]]
        for env, key in (
            ("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "max_debate_rounds"),
            ("TRADINGAGENTS_MAX_RISK_ROUNDS", "max_risk_discuss_rounds"),
        ):
            config[key] = build_default_config()[key] if os.getenv(env) else depth
    return config


def _report(state: dict, job_id: str, trade_date: str) -> dict:
    direct = {
        "market": "market_report",
        "news": "news_report",
        "fundamentals": "fundamentals_report",
        "social": "sentiment_report",
        "research": "investment_plan",
        "trader": "trader_investment_plan",
        "decision": "final_trade_decision",
    }
    sections = {
        name: state[path] for name, path in direct.items() if isinstance(state.get(path), str)
    }
    risk = state.get("risk_debate_state")
    if isinstance(risk, dict):
        risk_parts = [
            risk[key]
            for key in ("aggressive_history", "conservative_history", "neutral_history")
            if isinstance(risk.get(key), str)
        ]
        if risk_parts:
            sections["risk"] = "\n\n".join(risk_parts)
    return {"runId": job_id, "tradeDate": trade_date, "sections": sections}


def run_job(store: RunStore, job_id: str, config: dict) -> None:
    job = store.get_job(job_id)
    if not job or job["status"] != "running" or not job["claim_token"]:
        return
    token, request = job["claim_token"], job["request"]
    try:
        run_config = _run_config(config, request, job_id, store)
        ensure_provider_configured(run_config)
        portfolio = _native_portfolio(request)
        if job["kind"] == "analysis":
            graph = TradingAgentsGraph(request["selectedAnalysts"], config=run_config)
            state, rating = graph.propagate(
                request["ticker"], request["tradeDate"], request["assetType"], portfolio=portfolio
            )
            report = _report(state, job_id, request["tradeDate"])
            analyst_reports = {
                key: report["sections"][key]
                for key in ("market", "fundamentals", "news", "social")
                if key in report["sections"]
            }
            decision = {
                "runId": job_id,
                "ticker": request["ticker"],
                "tradeDate": request["tradeDate"],
                "rating": rating,
                "finalTradeDecision": state["final_trade_decision"],
                "generatedAt": utc_now(),
            }
            if analyst_reports:
                decision["analystReports"] = analyst_reports
            result = {
                "decision": decision,
                "_report": report,
            }
            store.finish(job_id, token, "completed", result, None)
        elif job["kind"] == "backtest":
            dates = iter_grid(request["startDate"], request["endDate"], request["everyNDays"])
            native = run_backtest(
                request["tickers"],
                dates,
                run_config,
                asset_type=request["assetType"],
                portfolio=portfolio,
                selected_analysts=request["selectedAnalysts"],
                run_id=job_id,
                stop_on_provider_limit=True,
                progress=lambda index, _total, _ticker, _date: store.update_progress(
                    job_id, token, index
                ),
            )
            summary = summarize(native)
            result = {
                "cellsRun": native.cells_run,
                "cellsSkipped": native.skipped,
                "summary": {
                    "resolved": summary.resolved,
                    "pending": summary.pending,
                    "unscored": summary.unscored,
                    "byRating": {key: score.count for key, score in summary.by_rating.items()},
                },
            }
            partial = bool(native.failures or native.settlement_failures)
            store.finish(
                job_id,
                token,
                "failed" if partial else "completed",
                result,
                "BACKTEST_PARTIAL_FAILURE" if partial else None,
            )
        else:
            store.finish(job_id, token, "failed", None, "INVALID_REQUEST")
    except BaseException as exc:
        logger.error("Run %s failed in native worker", job_id)
        if isinstance(exc, ProviderNotConfigured):
            code = "PROVIDER_NOT_CONFIGURED"
        elif is_provider_limit_error(exc):
            code = "PROVIDER_RATE_LIMITED"
        elif isinstance(exc, (ImportError, ModuleNotFoundError)):
            code = "DEPENDENCY_UNAVAILABLE"
        else:
            code = "INTERNAL_ERROR"
        store.finish(job_id, token, "failed", None, code)
