"""Native boundary tests use a fake graph and backtest, never paid providers."""

from pathlib import Path

import pytest

ANALYSIS = {
    "ticker": "NVDA",
    "tradeDate": "2026-01-05",
    "assetType": "stock",
    "selectedAnalysts": ["market"],
    "researchDepth": "quick",
    "portfolio": {"positions": [], "cash": 10.0, "asOf": "2026-01-01T00:00:00Z"},
}


def test_analysis_calls_native_propagate_with_isolated_config(tmp_path, monkeypatch):
    from tradingagents.service import worker
    from tradingagents.service.store import RunStore

    seen = {}

    class FakeGraph:
        def __init__(self, analysts, config):
            seen["analysts"], seen["config"] = analysts, config

        def propagate(self, ticker, date, asset_type, portfolio):
            seen["args"] = ticker, date, asset_type, portfolio
            return {
                "final_trade_decision": "Buy after research",
                "market_report": "market evidence",
                "news_report": "news evidence",
                "messages": ["secret"],
            }, "Buy"

    monkeypatch.setattr(worker, "TradingAgentsGraph", FakeGraph)
    store = RunStore(tmp_path / "runs.db")
    run = store.create("analysis", ANALYSIS, 2)
    store.claim_next()
    base = {"llm_provider": "ollama", "max_debate_rounds": 3, "max_risk_discuss_rounds": 3}
    worker.run_job(store, run["runId"], base)
    got = store.get(run["runId"])
    assert got["status"] == "completed" and got["decision"]["rating"] == "Buy"
    assert got["decision"]["finalTradeDecision"] == "Buy after research"
    assert got["decision"]["analystReports"] == {
        "market": "market evidence",
        "news": "news evidence",
    }
    assert store.get_report(run["runId"])["sections"] == {
        "market": "market evidence",
        "news": "news evidence",
        "decision": "Buy after research",
    }
    assert seen["analysts"] == ["market"]
    assert seen["args"][:3] == ("NVDA", "2026-01-05", "stock")
    assert not hasattr(seen["args"][3], "asOf")
    assert seen["config"]["max_debate_rounds"] == 1
    assert Path(seen["config"]["results_dir"]).is_relative_to(tmp_path)
    assert base == {"llm_provider": "ollama", "max_debate_rounds": 3, "max_risk_discuss_rounds": 3}


def test_backtest_maps_native_failures_and_pending_summary(tmp_path, monkeypatch):
    from tradingagents.backtest import BacktestResult, BacktestSummary, RatingScore
    from tradingagents.service import worker
    from tradingagents.service.store import RunStore

    seen = {}

    def fake_run(tickers, dates, config, **kwargs):
        seen.update(tickers=tickers, dates=dates, kwargs=kwargs)
        kwargs["progress"](1, 2, "NVDA", "2026-01-05")
        assert store.get(run["backtestId"])["cellsAttempted"] == 1
        return BacktestResult(
            "x",
            tmp_path / "missing",
            cells_run=1,
            skipped=0,
            failures=[("NVDA", "2026-01-12", "secret provider failure")],
        )

    monkeypatch.setattr(worker, "run_backtest", fake_run)
    monkeypatch.setattr(
        worker,
        "summarize",
        lambda result: BacktestSummary(
            resolved=0, pending=1, by_rating={"Buy": RatingScore(1, None, 0.0)}
        ),
    )
    store = RunStore(tmp_path / "runs.db")
    request = {
        "tickers": ["NVDA"],
        "startDate": "2026-01-05",
        "endDate": "2026-01-12",
        "everyNDays": 7,
        "assetType": "stock",
        "selectedAnalysts": ["news"],
    }
    run = store.create("backtest", request, 2)
    store.claim_next()
    worker.run_job(store, run["backtestId"], {"llm_provider": "ollama"})
    got = store.get(run["backtestId"])
    assert seen["kwargs"]["stop_on_provider_limit"] is True
    assert got["status"] == "failed" and got["error"] == "BACKTEST_PARTIAL_FAILURE"
    assert (got["cellsTotal"], got["cellsAttempted"], got["cellsRun"], got["cellsSkipped"]) == (
        2,
        1,
        1,
        0,
    )
    assert got["summary"] == {"resolved": 0, "pending": 1, "unscored": 0, "byRating": {"Buy": 1}}
    assert "secret provider failure" not in str(got)


def test_worker_exception_is_safe_and_terminal(tmp_path, monkeypatch):
    from tradingagents.service import worker
    from tradingagents.service.store import RunStore

    class BadGraph:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("private credential leaked")

    monkeypatch.setattr(worker, "TradingAgentsGraph", BadGraph)
    store = RunStore(tmp_path / "runs.db")
    run = store.create("analysis", ANALYSIS, 2)
    store.claim_next()
    worker.run_job(store, run["runId"], {"llm_provider": "ollama"})
    assert store.get(run["runId"])["status"] == "failed"
    assert "private credential" not in str(store.get(run["runId"]))


def test_missing_credentials_does_not_accept_paid_job(tmp_path, monkeypatch):
    from tradingagents.service.worker import ProviderNotConfigured, ensure_provider_configured

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    for config in (
        {"llm_provider": "openai"},
        {"llm_provider": "openai", "quick_think_provider": "google"},
        {"llm_provider": "google"},
    ):
        with pytest.raises(ProviderNotConfigured):
            ensure_provider_configured(config)
    ensure_provider_configured({"llm_provider": "ollama"})
    ensure_provider_configured(
        {"llm_provider": "openai_compatible", "backend_url": "http://localhost:1234/v1"}
    )
    with pytest.raises(ProviderNotConfigured):
        ensure_provider_configured({"llm_provider": "not-a-provider"})


def test_provider_preflight_checks_both_tiers_and_local_endpoint(monkeypatch):
    from tradingagents.service.worker import ProviderNotConfigured, ensure_provider_configured

    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-placeholder")
    ensure_provider_configured({"llm_provider": "openai"})
    ensure_provider_configured({"llm_provider": "openai", "quick_think_provider": "google"})
    ensure_provider_configured({"llm_provider": "google", "deep_think_provider": "ollama"})
    with pytest.raises(ProviderNotConfigured):
        ensure_provider_configured({"llm_provider": "openai_compatible"})
    for bad in ({}, {"llm_provider": None}):
        with pytest.raises(ProviderNotConfigured):
            ensure_provider_configured(bad)


def test_explicit_round_environment_overrides_depth(tmp_path, monkeypatch):
    from tradingagents.service.store import RunStore
    from tradingagents.service.worker import _run_config

    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "7")
    monkeypatch.setenv("TRADINGAGENTS_MAX_RISK_ROUNDS", "9")
    store = RunStore(tmp_path / "runs.db")
    config = _run_config(
        {"llm_provider": "ollama"},
        ANALYSIS | {"researchDepth": "deep"},
        "00000000-0000-4000-8000-000000000001",
        store,
    )
    assert (config["max_debate_rounds"], config["max_risk_discuss_rounds"]) == (7, 9)


def test_pending_backtest_outcomes_stay_completed_without_invented_metrics(tmp_path, monkeypatch):
    from tradingagents.backtest import BacktestResult, BacktestSummary
    from tradingagents.service import worker
    from tradingagents.service.store import RunStore

    monkeypatch.setattr(
        worker,
        "run_backtest",
        lambda *args, **kwargs: BacktestResult("offline", tmp_path / "unused", cells_run=1),
    )
    monkeypatch.setattr(
        worker, "summarize", lambda result: BacktestSummary(resolved=0, pending=1, by_rating={})
    )
    store = RunStore(tmp_path / "runs.db")
    request = {
        "tickers": ["NVDA"],
        "startDate": "2026-01-05",
        "endDate": "2026-01-05",
        "everyNDays": 7,
        "assetType": "stock",
        "selectedAnalysts": ["market"],
    }
    run = store.create("backtest", request, 1)
    store.claim_next()
    worker.run_job(store, run["backtestId"], {"llm_provider": "ollama"})
    result = store.get(run["backtestId"])
    assert result["status"] == "completed"
    assert result["summary"]["pending"] == 1
    assert "metrics" not in result["summary"]


def test_report_projection_uses_only_native_string_sections():
    from tradingagents.service.worker import _report

    state = {
        "market_report": "market",
        "fundamentals_report": "fundamentals",
        "news_report": "news",
        "sentiment_report": "social",
        "investment_plan": "research",
        "trader_investment_plan": "trader",
        "risk_debate_state": {"aggressive_history": "risk narrative", "neutral_history": 42},
        "final_trade_decision": "decision",
        "messages": ["private raw content"],
        "secret": "private",
    }
    assert _report(state, "id", "2026-01-05")["sections"] == {
        "market": "market",
        "fundamentals": "fundamentals",
        "news": "news",
        "social": "social",
        "research": "research",
        "trader": "trader",
        "risk": "risk narrative",
        "decision": "decision",
    }
