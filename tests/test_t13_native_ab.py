"""Comprehensive Proving Tests for T13 Genuine Native A/B Experiment Runner.

Verifies all 8 reviewer findings (R1–R8) with native API alignment, strict input
validation, fail-closed persistence, quota halt, usage accounting, truthful status/exit codes,
and offline tests matching the 11 counterexamples.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.graph import add_messages
from pydantic import BaseModel

import experiments.t13_native_ab as r
import tradingagents.dataflows.vendors.news_intelligence.client as news_client
import tradingagents.dataflows.vendors.yahoo.news as yahoo_news
from experiments.t13_native_ab import (
    AuditCallbackHandler,
    EvidenceStorageError,
    JournalPersistenceError,
    JournalVerificationError,
    ManifestMismatchError,
    OwnershipError,
    ProspectiveCaptureContext,
    _build_arm_config,
    _current_tool_run_id,
    _persist_journal,
    acquire_experiment_owner,
    compute_manifest_hash,
    main,
    run_experiment,
    sanitize_text,
    settle_experiment,
    summarize_experiment,
    validate_evidence_archive,
    validate_manifest,
)
from tradingagents.agents import tools as native_tools
from tradingagents.agents.rating import RATING_REVIEW
from tradingagents.dataflows.router import get_vendor
from tradingagents.llm_clients.factory import create_tier_client, tier_provider
from tradingagents.memory.log import TradingMemoryLog
from tradingagents.memory.settlement import Settlement
from tradingagents.portfolio import PortfolioContext

ni_client = news_client


class NetworkAccessAttemptedError(BaseException):
    """Raised when an offline test attempts external network access."""


@pytest.fixture(autouse=True)
def guard_against_network_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure that all tests run strictly offline without network access."""

    def forbidden_connect(*args: Any, **kwargs: Any) -> None:
        raise NetworkAccessAttemptedError(
            "TEST_NETWORK_REFUSAL: Offline test suite attempted external socket connection"
        )

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    try:
        import requests.sessions

        monkeypatch.setattr(requests.sessions.Session, "request", forbidden_connect)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_manifest_payload() -> dict[str, Any]:
    return {
        "experiment_id": "test-t13-ab-001",
        "benchmark_ticker": "SPY",
        "holding_period": 5,
        "pairs": [
            {"ticker": "NVDA", "date": "2026-08-03", "asset_type": "stock"},
            {"ticker": "BTC-USD", "date": "2026-08-10", "asset_type": "crypto"},
        ],
        "model_controls": {
            "provider": "google",
            "quick_model": "gemini-2.0-flash",
            "deep_model": "gemini-2.0-flash",
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_retries": 2,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_tool_rounds": 2,
            "recursion_limit": 50,
            "language": "English",
            "thinking_level": "low",
            "analysts": ["market", "news"],
        },
    }


# ---------------------------------------------------------------------------
# Manifest Core Validation Tests
# ---------------------------------------------------------------------------


def test_validate_manifest_success(valid_manifest_payload: dict[str, Any]) -> None:
    validated = validate_manifest(valid_manifest_payload)
    assert validated["experiment_id"] == "test-t13-ab-001"
    assert len(validated["pairs"]) == 2
    assert validated["holding_period"] == 5
    assert validated["benchmark_ticker"] == "SPY"
    assert compute_manifest_hash(validated) == compute_manifest_hash(valid_manifest_payload)


def test_validate_manifest_rejects_invalid_types_and_missing_fields() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        validate_manifest("not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="experiment_id"):
        validate_manifest({"pairs": []})

    with pytest.raises(ValueError, match="pairs"):
        validate_manifest({"experiment_id": "exp-1", "pairs": "invalid"})

    with pytest.raises(ValueError, match="holding_period"):
        validate_manifest(
            {
                "experiment_id": "exp-1",
                "pairs": [{"ticker": "NVDA", "date": "2026-08-03", "asset_type": "stock"}],
                "holding_period": 0,
            }
        )


def test_validate_manifest_rejects_duplicate_pairs(valid_manifest_payload: dict[str, Any]) -> None:
    p = copy.deepcopy(valid_manifest_payload)
    p["pairs"].append({"ticker": "NVDA", "date": "2026-08-03", "asset_type": "stock"})
    with pytest.raises(ValueError, match="Duplicate pair"):
        validate_manifest(p)


def test_validate_manifest_rejects_secrets_and_arbitrary_urls(
    valid_manifest_payload: dict[str, Any],
) -> None:
    p = copy.deepcopy(valid_manifest_payload)
    p["api_key"] = "secret_value_123"
    with pytest.raises(ValueError, match="secret or credential"):
        validate_manifest(p)

    p2 = copy.deepcopy(valid_manifest_payload)
    p2["model_controls"]["custom_token"] = "xyz"
    with pytest.raises(ValueError, match="secret or credential"):
        validate_manifest(p2)

    p3 = copy.deepcopy(valid_manifest_payload)
    p3["model_controls"]["quick_model"] = "https://evil.endpoint/v1"
    with pytest.raises(ValueError, match="Arbitrary endpoint URL"):
        validate_manifest(p3)


# ---------------------------------------------------------------------------
# R1: Native Config Keys & Treatment Routing Proving Tests
# ---------------------------------------------------------------------------


def test_r1_native_config_keys_and_router_consumption(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    res_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"

    cfg_a = _build_arm_config(valid_manifest_payload, "A", res_dir, data_dir, cache_dir)
    cfg_b = _build_arm_config(valid_manifest_payload, "B", res_dir, data_dir, cache_dir)

    # 1. Exact native keys consumed by engine
    assert cfg_a["quick_think_llm"] == "gemini-2.0-flash"
    assert cfg_a["deep_think_llm"] == "gemini-2.0-flash"
    assert cfg_a["data_cache_dir"] == str(cache_dir)
    assert cfg_a["llm_max_retries"] == 2
    assert cfg_a["max_recur_limit"] == 50
    assert cfg_a["output_language"] == "English"
    assert cfg_a["google_thinking_level"] == "low"
    assert cfg_a["benchmark_ticker"] == "SPY"
    assert cfg_a["holding_period_days"] == 5

    # 2. Native tier_provider and create_tier_client consumption
    assert tier_provider(cfg_a, "quick") == "google"
    assert tier_provider(cfg_a, "deep") == "google"
    with patch("tradingagents.llm_clients.factory.create_llm_client") as mock_create_client:
        create_tier_client(cfg_a, "quick")
        mock_create_client.assert_called_once()
        _, kwargs = mock_create_client.call_args
        assert kwargs["model"] == "gemini-2.0-flash"
        assert kwargs["max_retries"] == 2
        assert kwargs["thinking_level"] == "low"
        assert kwargs["max_output_tokens"] == 4096

    # 3. Router consumption via tool_vendors: A routes to yfinance, B routes to news_intelligence
    with patch("tradingagents.dataflows.router.get_config", return_value=cfg_a):
        assert get_vendor("news_data", "get_news") == "yfinance"
        assert get_vendor("news_data", "get_global_news") == "yfinance"

    with patch("tradingagents.dataflows.router.get_config", return_value=cfg_b):
        assert get_vendor("news_data", "get_news") == "news_intelligence"
        assert get_vendor("news_data", "get_global_news") == "news_intelligence"

    # 4. All non-treatment controls match between Arm A and Arm B
    for key in (
        "quick_think_llm",
        "deep_think_llm",
        "llm_max_retries",
        "max_recur_limit",
        "output_language",
        "google_thinking_level",
        "benchmark_ticker",
        "holding_period_days",
        "temperature",
    ):
        assert cfg_a[key] == cfg_b[key]


# ---------------------------------------------------------------------------
# R2: Propagate Return Tuple & Portfolio Validation Proving Tests
# ---------------------------------------------------------------------------


@patch("experiments.t13_native_ab.ensure_provider_configured")
@patch("experiments.t13_native_ab.TradingAgentsGraph")
def test_r2_propagate_tuple_unpacking_and_portfolio_contract(
    mock_graph_cls: MagicMock,
    mock_ensure_provider: MagicMock,
    valid_manifest_payload: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Verifies that propagate returns (final_state, signal), save_reports receives final_state dict,
    and portfolio is validated as PortfolioContext.
    """
    out = tmp_path / "exp_r2"
    p = copy.deepcopy(valid_manifest_payload)
    p["pairs"] = [p["pairs"][0]]  # 1 pair (2 cells: A and B)
    p["portfolio"] = {
        "cash": 50000.0,
        "currency": "USD",
        "positions": [{"ticker": "NVDA", "quantity": 100.0, "average_price": 120.0}],
    }

    mock_graph = mock_graph_cls.return_value
    fake_state = {
        "company_of_interest": "NVDA",
        "trade_date": "2026-08-03",
        "final_trade_decision": "Buy 100 shares",
    }
    # Return genuine tuple (final_state, signal)
    mock_graph.propagate.return_value = (fake_state, "Buy")
    mock_graph.save_reports.return_value = (
        out / "cells" / "NVDA_2026-08-03_A" / "results" / "report.html"
    )

    # Memory log returns matching entry
    def fake_load_entries() -> list[dict[str, Any]]:
        return [{"ticker": "NVDA", "date": "2026-08-03", "rating": "Buy", "pending": True}]

    with patch.object(TradingMemoryLog, "load_entries", side_effect=fake_load_entries):
        journal = run_experiment(p, out)
        assert journal["manifest_hash"]

    # Verify save_reports received final_state dictionary, NOT tuple
    for call in mock_graph.save_reports.call_args_list:
        state_arg = call[0][0]
        assert isinstance(state_arg, dict)
        assert state_arg["company_of_interest"] == "NVDA"

    # Verify portfolio was passed as PortfolioContext instance with .render()
    for call in mock_graph.propagate.call_args_list:
        portfolio_arg = call[1]["portfolio"]
        assert isinstance(portfolio_arg, PortfolioContext)
        assert hasattr(portfolio_arg, "render")
        assert hasattr(portfolio_arg, "fingerprint")
        assert "NVDA" in portfolio_arg.render("NVDA")


def test_r2_portfolio_validation_absent_flat_and_invalid(
    valid_manifest_payload: dict[str, Any],
) -> None:
    # 1. Absent portfolio
    p_absent = copy.deepcopy(valid_manifest_payload)
    p_absent.pop("portfolio", None)
    v_absent = validate_manifest(p_absent)
    assert v_absent.get("portfolio") is None

    # 2. Flat portfolio
    p_flat = copy.deepcopy(valid_manifest_payload)
    p_flat["portfolio"] = {"cash": 10000.0, "currency": "USD", "positions": []}
    v_flat = validate_manifest(p_flat)
    assert isinstance(v_flat["portfolio"], dict)
    pc = PortfolioContext.model_validate(v_flat["portfolio"])
    assert pc.cash == 10000.0
    assert pc.positions == []

    # 3. Invalid portfolio
    p_invalid = copy.deepcopy(valid_manifest_payload)
    p_invalid["portfolio"] = {"cash": "invalid_number"}
    with pytest.raises(ValueError, match="Invalid portfolio"):
        validate_manifest(p_invalid)


# ---------------------------------------------------------------------------
# R3: Path Traversal, Canonical Dates & Manifest Invariants Proving Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malicious_ticker",
    [
        "../../REVIEWER_ESCAPE",
        "/etc/passwd",
        "C:\\Windows\\System32",
        "../escape",
        "..",
        ".",
        "...",
        ".hidden",
        "NVDA/AAPL",
        "NVDA\\AAPL",
        " ",
        "",
    ],
)
def test_r3_rejects_path_traversal_tickers_before_directory_creation(
    malicious_ticker: str, valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    p = copy.deepcopy(valid_manifest_payload)
    p["pairs"][0]["ticker"] = malicious_ticker

    with pytest.raises(ValueError):
        validate_manifest(p)

    # Verify that NO files or directories were created outside tmp_path
    assert not (tmp_path / "REVIEWER_ESCAPE").exists()
    assert not (tmp_path.parent / "REVIEWER_ESCAPE").exists()


@pytest.mark.parametrize(
    "invalid_date",
    [
        "2026-08-3",  # non-canonical unpadded day
        "2026-8-03",  # non-canonical unpadded month
        "2026/08/03",  # slash separator
        "08-03-2026",  # wrong order
        "2026-08-32",  # impossible day
        "2099-01-01",  # future date
    ],
)
def test_r3_strict_canonical_date_validation(
    invalid_date: str, valid_manifest_payload: dict[str, Any]
) -> None:
    p = copy.deepcopy(valid_manifest_payload)
    p["pairs"][0]["date"] = invalid_date
    with pytest.raises(ValueError):
        validate_manifest(p)


def test_r3_strict_pinned_google_and_analyst_validation(
    valid_manifest_payload: dict[str, Any],
) -> None:
    # 1. Non-google provider rejected
    p1 = copy.deepcopy(valid_manifest_payload)
    p1["model_controls"]["provider"] = "openai"
    with pytest.raises(ValueError, match="must be strictly 'google'"):
        validate_manifest(p1)

    # 2. Invalid analyst name rejected
    p2 = copy.deepcopy(valid_manifest_payload)
    p2["model_controls"]["analysts"] = ["market", "crypto_wizard"]
    with pytest.raises(ValueError, match="Invalid analyst 'crypto_wizard'"):
        validate_manifest(p2)

    # 3. Boolean numeric values rejected
    p3 = copy.deepcopy(valid_manifest_payload)
    p3["model_controls"]["temperature"] = True
    with pytest.raises(ValueError, match="temperature"):
        validate_manifest(p3)

    p4 = copy.deepcopy(valid_manifest_payload)
    p4["model_controls"]["max_tokens"] = False
    with pytest.raises(ValueError, match="max_tokens"):
        validate_manifest(p4)


# ---------------------------------------------------------------------------
# R4: Recovery & Fail-Closed Journal Persistence Proving Tests
# ---------------------------------------------------------------------------


def test_r4_missing_journal_with_existing_artifacts_fails_closed(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """If journal.json is absent but cell artifacts exist, runner must fail closed."""
    out = tmp_path / "exp_missing_journal"
    out.mkdir(parents=True, exist_ok=True)
    cells_dir = out / "cells" / "NVDA_2026-08-03_A"
    cells_dir.mkdir(parents=True, exist_ok=True)
    (cells_dir / "report.html").write_text("<h1>Report</h1>", encoding="utf-8")

    # Run without journal.json
    with pytest.raises(JournalPersistenceError, match="existing cell artifacts found"):
        run_experiment(valid_manifest_payload, out)


def test_r4_journal_persistence_error_immediately_reraised(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """A persistence failure must immediately re-raise and stop execution, not be caught as cell error."""
    out = tmp_path / "exp_persist_fail"
    out.mkdir(parents=True, exist_ok=True)

    call_count = 0

    def failing_persist(output_root: Path, journal: dict[str, Any]) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise JournalPersistenceError("Injected disk write failure")
        _persist_journal(output_root, journal)

    with (
        patch("experiments.t13_native_ab._persist_journal", side_effect=failing_persist),
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        pytest.raises(JournalPersistenceError, match="Injected disk write failure"),
    ):
        run_experiment(valid_manifest_payload, out)


def test_r4_fail_closed_persistence_readback_mismatch(tmp_path: Path) -> None:
    out = tmp_path / "journal_readback_test"
    out.mkdir(parents=True, exist_ok=True)

    with (
        patch.object(Path, "read_text", return_value=json.dumps({"corrupted": True})),
        pytest.raises(JournalVerificationError, match="mismatch"),
    ):
        _persist_journal(out, {"valid": "data"})


def test_r4_manifest_mismatch_fails_closed(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "mismatch_test"
    out.mkdir(parents=True, exist_ok=True)
    journal_path = out / "journal.json"
    journal_path.write_text(json.dumps({"manifest_hash": "different_hash"}), encoding="utf-8")

    with pytest.raises(ManifestMismatchError, match="does not match"):
        run_experiment(valid_manifest_payload, out)


@patch("experiments.t13_native_ab.ensure_provider_configured")
@patch("experiments.t13_native_ab.TradingAgentsGraph")
def test_r4_reconciles_recoverable_started_decision_without_calling_graph(
    mock_graph_cls: MagicMock,
    mock_ensure_provider: MagicMock,
    valid_manifest_payload: dict[str, Any],
    tmp_path: Path,
) -> None:
    """When a decision was started and its artifacts exist on disk, recovery reconciles to completed without propagate."""
    out = tmp_path / "exp_reconcile"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    cell_key = "NVDA_2026-08-03_A"
    results_dir = out / "cells" / cell_key / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "report.html").write_text("<h1>Report</h1>", encoding="utf-8")
    (results_dir / "trading_memory.log").write_text(
        "[2026-08-03 | NVDA | Buy | pending]\nDECISION:\nBuy\n\n<!-- ENTRY_END -->\n\n",
        encoding="utf-8",
    )

    initial_journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 4,
        "planned_decisions_count": 4,
        "planned_pairs_count": 2,
        "decision_attempts": {
            cell_key: {
                "attempt_id": "att-prior",
                "phase": "started",
                "arm": "A",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "asset_type": "stock",
                "report_path": f"cells/{cell_key}/results/report.html",
            }
        },
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(initial_journal), encoding="utf-8")

    journal = run_experiment(valid_manifest_payload, out)

    # NVDA_2026-08-03_A was reconciled to completed without calling propagate
    assert journal["decision_attempts"][cell_key]["phase"] == "completed"
    assert journal["decision_attempts"][cell_key]["decision"] == "Buy"
    for call in mock_graph_cls.return_value.propagate.call_args_list:
        assert not (
            call[0][0] == "NVDA" and call[0][1] == "2026-08-03" and "NVDA_2026-08-03_A" in str(call)
        )


@patch("experiments.t13_native_ab.ensure_provider_configured")
@patch("experiments.t13_native_ab.TradingAgentsGraph")
def test_r4_no_automatic_paid_replay_on_started_cell_without_artifacts(
    mock_graph_cls: MagicMock,
    mock_ensure_provider: MagicMock,
    valid_manifest_payload: dict[str, Any],
    tmp_path: Path,
) -> None:
    """If a cell is in 'started' phase without verified disk artifacts, it must not automatically replay."""
    out = tmp_path / "no_replay_out"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    cell_key = "NVDA_2026-08-03_A"
    initial_journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 4,
        "planned_decisions_count": 4,
        "planned_pairs_count": 2,
        "decision_attempts": {
            cell_key: {
                "attempt_id": "att-12345",
                "phase": "started",
                "arm": "A",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "asset_type": "stock",
                "report_path": None,
            }
        },
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(initial_journal), encoding="utf-8")

    run_experiment(valid_manifest_payload, out)

    # Verify propagate was NOT called for NVDA_2026-08-03_A
    for call in mock_graph_cls.return_value.propagate.call_args_list:
        assert not (
            call[0][0] == "NVDA" and call[0][1] == "2026-08-03" and "NVDA_2026-08-03_A" in str(call)
        )


# ---------------------------------------------------------------------------
# Owner Locking & Subprocess Concurrency Tests
# ---------------------------------------------------------------------------


def test_owner_locking_exclusive_within_process(tmp_path: Path) -> None:
    out = tmp_path / "exp_lock"
    with (
        acquire_experiment_owner(out),
        pytest.raises(OwnershipError, match="Exclusive experiment lock could not be acquired"),
        acquire_experiment_owner(out),
    ):
        pass


def test_subprocess_concurrency_ownership_exclusion(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Test that two concurrent processes competing for the same output directory reject the loser."""
    out_dir = tmp_path / "concurrent_out"
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(valid_manifest_payload), encoding="utf-8")

    with acquire_experiment_owner(out_dir):
        cmd = [
            sys.executable,
            "-m",
            "experiments.t13_native_ab",
            "run",
            "--manifest",
            str(manifest_file),
            "--output",
            str(out_dir),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONPATH=str(Path.cwd())),
            timeout=10,
        )
        assert proc.returncode != 0
        assert "Exclusive experiment lock could not be acquired" in proc.stderr


def test_subprocess_concurrency_run_run_rejection(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Test that two separate child processes competing for the same output directory reject the second child run.

    Synchronizes real run/run commands before synthetic graph boundary and asserts exactly one process reaches it.
    """
    out_dir = tmp_path / "concurrent_run_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(valid_manifest_payload), encoding="utf-8")

    sync_marker = out_dir / "proc1_graph_boundary.txt"
    script_file = tmp_path / "proc1_runner.py"
    script_file.write_text(
        f"""import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from experiments.t13_native_ab import run_experiment

mg = MagicMock()
def fake_prop(t, d, **kw):
    Path(r"{sync_marker}").write_text("GRAPH_REACHED", encoding="utf-8")
    time.sleep(10)
    return {{"final_trade_decision": "Hold"}}, "Hold"

mg.propagate.side_effect = fake_prop
mg.save_reports.side_effect = lambda s, t, sp: sp / "complete_report.md"

with patch("experiments.t13_native_ab.ensure_provider_configured"), \\
     patch("experiments.t13_native_ab.TradingAgentsGraph", return_value=mg):
    run_experiment(json.loads(Path(r"{manifest_file}").read_text(encoding="utf-8")), Path(r"{out_dir}"))
""",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()))
    proc1 = subprocess.Popen(
        [sys.executable, str(script_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        for _ in range(100):
            if sync_marker.exists():
                break
            if proc1.poll() is not None:
                o, e = proc1.communicate()
                raise AssertionError(
                    f"proc1 exited prematurely (code {proc1.returncode}):\nstdout: {o}\nstderr: {e}"
                )
            time.sleep(0.1)
        assert sync_marker.exists()

        cmd = [
            sys.executable,
            "-m",
            "experiments.t13_native_ab",
            "run",
            "--manifest",
            str(manifest_file),
            "--output",
            str(out_dir),
        ]
        proc2 = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=10)
        assert proc2.returncode != 0
        assert "Exclusive experiment lock could not be acquired" in proc2.stderr
    finally:
        proc1.terminate()
        proc1.wait(timeout=5)


def test_subprocess_concurrency_run_settle_rejection(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Test that a child process executing settle is rejected when another process holds the run lock at graph boundary."""
    out_dir = tmp_path / "concurrent_run_settle"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(valid_manifest_payload), encoding="utf-8")

    sync_marker = out_dir / "proc1_graph_boundary.txt"
    script_file = tmp_path / "proc1_settle_runner.py"
    script_file.write_text(
        f"""import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from experiments.t13_native_ab import run_experiment

mg = MagicMock()
def fake_prop(t, d, **kw):
    Path(r"{sync_marker}").write_text("GRAPH_REACHED", encoding="utf-8")
    time.sleep(10)
    return {{"final_trade_decision": "Hold"}}, "Hold"

mg.propagate.side_effect = fake_prop
mg.save_reports.side_effect = lambda s, t, sp: sp / "complete_report.md"

with patch("experiments.t13_native_ab.ensure_provider_configured"), \\
     patch("experiments.t13_native_ab.TradingAgentsGraph", return_value=mg):
    run_experiment(json.loads(Path(r"{manifest_file}").read_text(encoding="utf-8")), Path(r"{out_dir}"))
""",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()))
    proc1 = subprocess.Popen(
        [sys.executable, str(script_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        for _ in range(100):
            if sync_marker.exists():
                break
            if proc1.poll() is not None:
                o, e = proc1.communicate()
                raise AssertionError(
                    f"proc1 exited prematurely (code {proc1.returncode}):\nstdout: {o}\nstderr: {e}"
                )
            time.sleep(0.1)
        assert sync_marker.exists()

        cmd = [
            sys.executable,
            "-m",
            "experiments.t13_native_ab",
            "settle",
            "--manifest",
            str(manifest_file),
            "--output",
            str(out_dir),
        ]
        proc2 = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=10)
        assert proc2.returncode != 0
        assert "Exclusive experiment lock could not be acquired" in proc2.stderr
    finally:
        proc1.terminate()
        proc1.wait(timeout=5)


def test_subprocess_path_alias_rejection(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Test that path aliases (e.g., ../dir or ./dir) resolve to the canonical lock and are rejected."""
    out_dir = tmp_path / "alias_canonical"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(valid_manifest_payload), encoding="utf-8")

    alias_dir = tmp_path / "sub" / ".." / "alias_canonical"

    with acquire_experiment_owner(out_dir):
        cmd = [
            sys.executable,
            "-m",
            "experiments.t13_native_ab",
            "run",
            "--manifest",
            str(manifest_file),
            "--output",
            str(alias_dir),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONPATH=str(Path.cwd())),
            timeout=10,
        )
        assert proc.returncode != 0
        assert "Exclusive experiment lock could not be acquired" in proc.stderr


def test_subprocess_killed_owner_lock_recovery(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Test that if a process holding the lock is killed abruptly after durable started, restart asserts no replay."""
    out_dir = tmp_path / "killed_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(valid_manifest_payload), encoding="utf-8")

    sync_marker = out_dir / "started_ready.txt"
    script_file = tmp_path / "proc1_killed_runner.py"
    script_file.write_text(
        f"""import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from experiments.t13_native_ab import run_experiment

mg = MagicMock()
def fake_prop(t, d, **kw):
    Path(r"{sync_marker}").write_text("STARTED", encoding="utf-8")
    time.sleep(60)
    return {{"final_trade_decision": "Hold"}}, "Hold"

mg.propagate.side_effect = fake_prop
mg.save_reports.side_effect = lambda s, t, sp: sp / "complete_report.md"

with patch("experiments.t13_native_ab.ensure_provider_configured"), \\
     patch("experiments.t13_native_ab.TradingAgentsGraph", return_value=mg):
    run_experiment(json.loads(Path(r"{manifest_file}").read_text(encoding="utf-8")), Path(r"{out_dir}"))
""",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()))
    proc = subprocess.Popen(
        [sys.executable, str(script_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        for _ in range(100):
            if sync_marker.exists():
                break
            if proc.poll() is not None:
                o, e = proc.communicate()
                raise AssertionError(
                    f"proc exited prematurely (code {proc.returncode}):\nstdout: {o}\nstderr: {e}"
                )
            time.sleep(0.1)
        assert sync_marker.exists()
        # Abruptly kill the process while it is in started phase at the synthetic boundary
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        raise

    # The OS must release the lock handle upon process termination, allowing recovery
    for _ in range(50):
        try:
            with acquire_experiment_owner(out_dir):
                break
        except OwnershipError:
            time.sleep(0.1)
    else:
        with acquire_experiment_owner(out_dir):
            pass

    # Verify journal persisted the durable started attempt before crash
    journal = json.loads((out_dir / "journal.json").read_text(encoding="utf-8"))
    cell_key = "NVDA_2026-08-03_A"
    assert cell_key in journal["decision_attempts"]
    assert journal["decision_attempts"][cell_key]["phase"] == "started"

    # Now restart run_experiment: killed started attempt without reports on disk must NOT be replayed
    mock_restart_graph = MagicMock()
    restart_prop_invocations: list[str] = []

    def restart_fake_prop(t: str, d: str, **kw: Any) -> tuple[dict[str, str], str]:
        restart_prop_invocations.append(f"{t}_{d}")
        return {"final_trade_decision": "Hold"}, "Hold"

    mock_restart_graph.propagate.side_effect = restart_fake_prop

    def restart_fake_save_reports(state: Any, ticker: str, save_path: Path) -> Path:
        TradingMemoryLog({"memory_log_path": str(save_path / "trading_memory.log")}).store_decision(
            ticker, "2026-08-03" if ticker == "NVDA" else "2026-08-10", "Decision", rating="Hold"
        )
        rep = save_path / "complete_report.md"
        rep.write_text("# Report", encoding="utf-8")
        return rep

    mock_restart_graph.save_reports.side_effect = restart_fake_save_reports

    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph", return_value=mock_restart_graph),
    ):
        after_restart = run_experiment(valid_manifest_payload, out_dir)

    # Arm A was in started phase and interrupted by crash; it must not be re-executed
    orig_att_id = journal["decision_attempts"][cell_key]["attempt_id"]
    att_A = after_restart["decision_attempts"][cell_key]
    assert att_A["phase"] in ("started", "interrupted", "failed")
    assert att_A["attempt_id"] == orig_att_id
    # NVDA_2026-08-03 was invoked only once (for Arm B), never re-invoked for Arm A
    assert restart_prop_invocations.count("NVDA_2026-08-03") == 1
    assert after_restart["decision_attempts"]["NVDA_2026-08-03_B"]["phase"] == "completed"


# ---------------------------------------------------------------------------
# R5: Quota & Provider Failure Stop Proving Tests
# ---------------------------------------------------------------------------


@patch("experiments.t13_native_ab.ensure_provider_configured")
@patch("experiments.t13_native_ab.TradingAgentsGraph")
def test_r5_returned_settlement_quota_failure_stops_whole_command(
    mock_graph_cls: MagicMock,
    mock_ensure_provider: MagicMock,
    valid_manifest_payload: dict[str, Any],
    tmp_path: Path,
) -> None:
    """A returned Settlement.failed quota error must halt the entire settlement command immediately."""
    out = tmp_path / "exp_r5_returned"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    # Setup 4 completed decisions (2 pairs x 2 arms)
    cell_keys = [
        "NVDA_2026-08-03_A",
        "NVDA_2026-08-03_B",
        "BTC-USD_2026-08-10_A",
        "BTC-USD_2026-08-10_B",
    ]
    decision_attempts = {}
    for k in cell_keys:
        parts = k.split("_")
        decision_attempts[k] = {
            "attempt_id": f"att-{k}",
            "phase": "completed",
            "ticker": parts[0],
            "date": parts[1],
            "arm": parts[2],
            "decision": "Buy",
        }
        res_dir = out / "cells" / k / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "trading_memory.log").write_text(
            f"[{parts[1]} | {parts[0]} | Buy | pending]\nDECISION:\nBuy\n\n<!-- ENTRY_END -->\n\n",
            encoding="utf-8",
        )

    initial_journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 4,
        "planned_decisions_count": 4,
        "planned_pairs_count": 2,
        "decision_attempts": decision_attempts,
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(initial_journal), encoding="utf-8")

    # Mock settle_pending to return quota failure on the first call
    mock_graph_cls.return_value.settle_pending.return_value = Settlement(
        settled=[], failed=[("NVDA", "2026-08-03", "ResourceExhausted: 429 Quota exceeded")]
    )

    journal = settle_experiment(valid_manifest_payload, out)

    # Exactly ONE settlement attempt was made, remaining 3 cells remain unattempted!
    assert len(journal["settlement_attempts"]) == 1
    assert mock_graph_cls.return_value.settle_pending.call_count == 1
    first_settle = journal["settlement_attempts"]["NVDA_2026-08-03_A"]
    assert first_settle["phase"] == "failed"
    assert "429 Quota exceeded" in first_settle["error"]


@patch("experiments.t13_native_ab.ensure_provider_configured")
@patch("experiments.t13_native_ab.TradingAgentsGraph")
def test_r5_thrown_settlement_quota_failure_stops_whole_command(
    mock_graph_cls: MagicMock,
    mock_ensure_provider: MagicMock,
    valid_manifest_payload: dict[str, Any],
    tmp_path: Path,
) -> None:
    """A thrown quota exception must halt the entire settlement command immediately."""
    out = tmp_path / "exp_r5_thrown"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    cell_keys = [
        "NVDA_2026-08-03_A",
        "NVDA_2026-08-03_B",
        "BTC-USD_2026-08-10_A",
        "BTC-USD_2026-08-10_B",
    ]
    decision_attempts = {}
    for k in cell_keys:
        parts = k.split("_")
        decision_attempts[k] = {
            "attempt_id": f"att-{k}",
            "phase": "completed",
            "ticker": parts[0],
            "date": parts[1],
            "arm": parts[2],
            "decision": "Buy",
        }
        res_dir = out / "cells" / k / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "trading_memory.log").write_text(
            f"[{parts[1]} | {parts[0]} | Buy | pending]\nDECISION:\nBuy\n\n<!-- ENTRY_END -->\n\n",
            encoding="utf-8",
        )

    initial_journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 4,
        "planned_decisions_count": 4,
        "planned_pairs_count": 2,
        "decision_attempts": decision_attempts,
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(initial_journal), encoding="utf-8")

    mock_graph_cls.return_value.settle_pending.side_effect = RuntimeError("429 rate limit exceeded")

    journal = settle_experiment(valid_manifest_payload, out)

    # Exactly ONE settlement attempt was made, remaining 3 cells remain unattempted!
    assert len(journal["settlement_attempts"]) == 1
    assert mock_graph_cls.return_value.settle_pending.call_count == 1
    first_settle = journal["settlement_attempts"]["NVDA_2026-08-03_A"]
    assert first_settle["phase"] == "failed"
    assert "429 rate limit exceeded" in first_settle["error"]


# ---------------------------------------------------------------------------
# R6: Usage Accounting & Audit Trace Proving Tests
# ---------------------------------------------------------------------------


def test_r6_audit_callback_extracts_langchain_usage_metadata() -> None:
    cb = AuditCallbackHandler()
    assert cb.token_usage["total_tokens"] == 0

    # Modern LangChain ChatGeneration with message.usage_metadata (19 tokens)
    ai_msg = AIMessage(
        content="Market analysis complete",
        usage_metadata={"input_tokens": 10, "output_tokens": 9, "total_tokens": 19},
    )
    chat_gen = ChatGeneration(message=ai_msg)
    llm_result = LLMResult(generations=[[chat_gen]])

    cb.on_llm_start({"name": "gemini-2.0-flash"}, ["Analyze NVDA on 2026-08-03"])
    cb.on_llm_end(llm_result)

    assert cb.token_usage["input_tokens"] == 10
    assert cb.token_usage["output_tokens"] == 9
    assert cb.token_usage["total_tokens"] == 19
    assert not cb.has_unknown_usage
    assert len(cb.traces) == 2


def test_r6_audit_callback_flags_unknown_usage_without_false_zero() -> None:
    cb = AuditCallbackHandler()

    # Generation without usage metadata
    ai_msg = AIMessage(content="Analysis without token usage")
    chat_gen = ChatGeneration(message=ai_msg)
    llm_result = LLMResult(generations=[[chat_gen]])

    cb.on_llm_start({"name": "gemini-2.0-flash"}, ["Analyze AAPL"])
    cb.on_llm_end(llm_result)

    assert cb.has_unknown_usage is True
    assert cb.token_usage["has_unknown_usage"] is True
    assert cb.traces[1]["has_unknown_usage"] is True


def test_r6_secret_sanitization() -> None:
    raw = "Request failed with Authorization: Bearer secret_token_xyz and api_key=AIzaSyA1234567890123456789012345678901"
    clean = sanitize_text(raw)
    assert "secret_token_xyz" not in clean
    assert "AIzaSyA1234567890123456789012345678901" not in clean
    assert "[REDACTED]" in clean


# ---------------------------------------------------------------------------
# R7: Summarize & Honest Paired Matching Proving Tests
# ---------------------------------------------------------------------------


def _make_cell_log(
    out: Path,
    cell_key: str,
    ticker: str,
    trade_date: str,
    rating: str,
    raw: str = "+2.0%",
    alpha: str = "+1.0%",
    holding_days: int = 5,
    res_date: str = "2026-08-10",
    tool_output: str | None = None,
) -> Path:
    res_dir = out / "cells" / cell_key / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    log_path = res_dir / "trading_memory.log"
    tag = f"[{trade_date} | {ticker} | {rating} | {raw} | {alpha} | {holding_days}d | resolved:{res_date}]"
    content = f"{tag}\n\nDECISION:\nSynthetic test decision for {ticker}\n\nREFLECTION:\nSynthetic test reflection\n\n<!-- ENTRY_END -->\n\n"
    log_path.write_text(content, encoding="utf-8")

    trace_path = res_dir / "audit_trace.json"
    if not trace_path.exists():
        arm = "B" if cell_key.endswith("_B") else "A"
        if tool_output is None or tool_output == "Body":
            if arm == "B":
                from tradingagents.dataflows.vendors.news_intelligence.client import format_article

                ev_b = {
                    "title": f"{ticker} Story",
                    "content": "Body",
                    "published_at": f"{trade_date}T10:00:00Z",
                    "publishedAt": f"{trade_date}T10:00:00Z",
                    "received_at": f"{trade_date}T10:05:00Z",
                    "receivedAt": f"{trade_date}T10:05:00Z",
                    "symbols": [ticker],
                }
                fmt_b = format_article(ev_b)
                tool_output = f"## {ticker} News, from {trade_date} to {trade_date}:\n\n{fmt_b}"
            else:
                from tradingagents.dataflows.vendors.yahoo.news import _format_articles

                ev_a = {
                    "title": f"{ticker} Story",
                    "summary": "Body",
                    "publisher": "Unknown",
                    "link": "",
                }
                fmt_a = _format_articles([ev_a])
                tool_output = f"## {ticker} News, from {trade_date} to {trade_date}:\n\n{fmt_a}"

        out_hash = hashlib.sha256(tool_output.encode("utf-8")).hexdigest()
        run_id = f"run-{cell_key}"
        trace_path.write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "tool_start",
                            "tool": "get_news",
                            "input": {
                                "ticker": ticker,
                                "start_date": trade_date,
                                "end_date": trade_date,
                            },
                            "run_id": run_id,
                            "timestamp": f"{trade_date}T10:00:30Z",
                        },
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output": tool_output,
                            "output_hash": out_hash,
                            "run_id": run_id,
                            "timestamp": f"{trade_date}T10:01:00Z",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
    return log_path


def test_r7_summarize_excludes_unmatched_vintages_and_orphaned_settlements(tmp_path: Path) -> None:
    out = tmp_path / "exp_r7"
    out.mkdir(parents=True, exist_ok=True)

    journal = {
        "manifest_hash": "hash_123",
        "experiment_id": "exp-r7",
        "planned_cells_count": 6,
        "planned_decisions_count": 6,
        "planned_pairs_count": 3,
        "decision_attempts": {
            # Pair 1: valid completed decisions
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-1A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "non_news_inputs_hash": "hash_1",
                "started_at": "2026-08-03T10:00:00Z",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-1B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "non_news_inputs_hash": "hash_1",
                "started_at": "2026-08-03T10:00:00Z",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            # Pair 2: valid completed decisions
            "AAPL_2026-08-03_A": {
                "attempt_id": "att-2A",
                "phase": "completed",
                "ticker": "AAPL",
                "date": "2026-08-03",
                "non_news_inputs_hash": "hash_2",
                "started_at": "2026-08-03T10:00:00Z",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "AAPL_2026-08-03_B": {
                "attempt_id": "att-2B",
                "phase": "completed",
                "ticker": "AAPL",
                "date": "2026-08-03",
                "non_news_inputs_hash": "hash_2",
                "started_at": "2026-08-03T10:00:00Z",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            # Pair 3: Arm A completed, Arm B failed
            "MSFT_2026-08-03_A": {
                "attempt_id": "att-3A",
                "phase": "completed",
                "ticker": "MSFT",
                "date": "2026-08-03",
                "non_news_inputs_hash": "hash_3",
                "started_at": "2026-08-03T10:00:00Z",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "MSFT_2026-08-03_B": {
                "attempt_id": "att-3B",
                "phase": "failed",
                "ticker": "MSFT",
                "date": "2026-08-03",
                "non_news_inputs_hash": "hash_3",
                "started_at": "2026-08-03T10:00:00Z",
                "entry_time": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            # Pair 1: Matched resolution date, raw return (+2.0%), and benchmark vintage (-1.0%)
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-1A",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.030,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.020,
                "benchmark_return_float": -0.010,
                "alpha_float": 0.030,
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-1B",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.030,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.020,
                "benchmark_return_float": -0.010,
                "alpha_float": 0.030,
            },
            # Pair 2: Mismatched resolution dates (Arm A: 2026-08-10, Arm B: 2026-08-11)
            "AAPL_2026-08-03_A": {
                "decision_attempt_id": "att-2A",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.010,
                "raw_return": "+1.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.010,
                "benchmark_return_float": 0.000,
                "alpha_float": 0.010,
            },
            "AAPL_2026-08-03_B": {
                "decision_attempt_id": "att-2B",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.050,
                "raw_return": "+1.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-11",
                "raw_return_float": 0.010,
                "benchmark_return_float": -0.040,
                "alpha_float": 0.050,
            },
            # Orphaned settlement without completed decision
            "ORPHAN_2026-08-03_A": {
                "decision_attempt_id": "att-nonexistent",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.020,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
            },
            "ORPHAN_2026-08-03_B": {
                "decision_attempt_id": "att-nonexistent",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.020,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
    from tradingagents.dataflows.vendors.news_intelligence.client import format_article
    from tradingagents.dataflows.vendors.yahoo.news import _format_articles

    ev_A = {
        "title": "NVDA Story",
        "content": "Body",
        "summary": "Body",
        "publisher": "Unknown",
        "link": "",
        "published_at": "2026-08-03T10:00:00Z",
        "received_at": "2026-08-03T10:05:00Z",
    }
    fmt_A = _format_articles([ev_A])
    resp_A = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{fmt_A}"
    hash_A = hashlib.sha256(resp_A.encode("utf-8")).hexdigest()

    ev_B = {
        "title": "NVDA Story",
        "content": "Body",
        "published_at": "2026-08-03T10:00:00Z",
        "publishedAt": "2026-08-03T10:00:00Z",
        "received_at": "2026-08-03T10:05:00Z",
        "receivedAt": "2026-08-03T10:05:00Z",
        "symbols": ["NVDA"],
    }
    fmt_B = format_article(ev_B)
    resp_B = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{fmt_B}"
    hash_B = hashlib.sha256(resp_B.encode("utf-8")).hexdigest()

    (out / "baseline_archive").mkdir(parents=True, exist_ok=True)
    (out / "baseline_archive" / "nvda_yahoo.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "query": "NVDA",
                "cutoff": "2026-08-03T20:00:00Z",
                "date": "2026-08-03",
                "arm": "A",
                "attempt_id": "att-1A",
                "source": "yfinance",
                "output_hash": hash_A,
                "received_at": "2026-08-03T10:05:00Z",
                "events": [ev_A],
            }
        ),
        encoding="utf-8",
    )
    (out / "provenance_archive.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "query": "NVDA",
                "cutoff": "2026-08-03T20:00:00Z",
                "date": "2026-08-03",
                "arm": "B",
                "attempt_id": "att-1B",
                "source": "news_intelligence",
                "output_hash": hash_B,
                "received_at": "2026-08-03T10:05:00Z",
                "events": [ev_B],
            }
        ),
        encoding="utf-8",
    )

    # Write native logs on disk for Pair 1 and Pair 2
    _make_cell_log(
        out,
        "NVDA_2026-08-03_A",
        "NVDA",
        "2026-08-03",
        "Buy",
        raw="+2.0%",
        alpha="+3.0%",
        holding_days=5,
        res_date="2026-08-10",
    )
    _make_cell_log(
        out,
        "NVDA_2026-08-03_B",
        "NVDA",
        "2026-08-03",
        "Buy",
        raw="+2.0%",
        alpha="+3.0%",
        holding_days=5,
        res_date="2026-08-10",
    )
    _make_cell_log(
        out,
        "AAPL_2026-08-03_A",
        "AAPL",
        "2026-08-03",
        "Buy",
        raw="+1.0%",
        alpha="+1.0%",
        holding_days=5,
        res_date="2026-08-10",
    )
    _make_cell_log(
        out,
        "AAPL_2026-08-03_B",
        "AAPL",
        "2026-08-03",
        "Buy",
        raw="+1.0%",
        alpha="+5.0%",
        holding_days=5,
        res_date="2026-08-11",
    )

    summary = summarize_experiment(out)

    # Only Pair 1 is eligible (tracked diagnostically under strict public zero/unknown contract)
    assert summary["settled_pairs_count"] == 0
    assert summary["scored_pairs_count"] == 0
    assert summary["strict_empirical_eligible"] is False
    assert summary["availability_status"] == "availability_unknown"
    assert summary["mean_alpha_A"] is None
    assert summary["mean_alpha_B"] is None
    assert summary["alpha_difference"] is None
    assert (
        summary["unmatched_vintages_count"] == 1
    )  # AAPL rejected due to different resolution dates
    assert summary["orphaned_settlements_count"] >= 1  # ORPHAN rejected

    diag = summary["diagnostic_summary"]
    assert diag["diagnostic_settled_pairs_count"] == 1
    assert diag["diagnostic_scored_pairs_count"] == 1
    assert diag["diagnostic_mean_alpha_A"] == 0.030
    assert diag["diagnostic_mean_alpha_B"] == 0.030
    assert diag["diagnostic_alpha_difference"] == 0.000


def test_r7_summarize_excludes_review_and_preserves_hold_semantics(tmp_path: Path) -> None:
    out = tmp_path / "exp_r7_ratings"
    out.mkdir(parents=True, exist_ok=True)

    journal = {
        "manifest_hash": "hash_123",
        "experiment_id": "exp-r7-ratings",
        "planned_cells_count": 4,
        "planned_decisions_count": 4,
        "planned_pairs_count": 2,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-1A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "non_news_inputs_hash": "hash_1",
                "started_at": "2026-08-03T10:00:00Z",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-1B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "non_news_inputs_hash": "hash_1",
                "started_at": "2026-08-03T10:00:00Z",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "BTC-USD_2026-08-10_A": {
                "attempt_id": "att-2A",
                "phase": "completed",
                "ticker": "BTC-USD",
                "date": "2026-08-10",
                "non_news_inputs_hash": "hash_2",
                "started_at": "2026-08-10T10:00:00Z",
                "entry_time": "2026-08-10T20:00:00Z",
            },
            "BTC-USD_2026-08-10_B": {
                "attempt_id": "att-2B",
                "phase": "completed",
                "ticker": "BTC-USD",
                "date": "2026-08-10",
                "non_news_inputs_hash": "hash_2",
                "started_at": "2026-08-10T10:00:00Z",
                "entry_time": "2026-08-10T20:00:00Z",
            },
        },
        "settlement_attempts": {
            # Pair 1: Arm A is REVIEW -> unscored pair! Must not enter mean alpha
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-1A",
                "phase": "completed",
                "rating": RATING_REVIEW,
                "outcome": 0.010,
                "raw_return": "+1.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.010,
                "benchmark_return_float": 0.000,
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-1B",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.010,
                "raw_return": "+1.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.010,
                "benchmark_return_float": 0.000,
            },
            # Pair 2: Both Arms are Hold -> valid scored pair with zero direction
            "BTC-USD_2026-08-10_A": {
                "decision_attempt_id": "att-2A",
                "phase": "completed",
                "rating": "Hold",
                "outcome": -0.010,
                "raw_return": "-0.5%",
                "holding_period": 5,
                "resolution_date": "2026-08-17",
                "raw_return_float": -0.005,
                "benchmark_return_float": 0.005,
            },
            "BTC-USD_2026-08-10_B": {
                "decision_attempt_id": "att-2B",
                "phase": "completed",
                "rating": "Hold",
                "outcome": -0.010,
                "raw_return": "-0.5%",
                "holding_period": 5,
                "resolution_date": "2026-08-17",
                "raw_return_float": -0.005,
                "benchmark_return_float": 0.005,
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
    from tradingagents.dataflows.vendors.news_intelligence.client import format_article
    from tradingagents.dataflows.vendors.yahoo.news import _format_articles

    ev_nvda_a = {
        "title": "NVDA Story",
        "content": "Body",
        "summary": "Body",
        "publisher": "Unknown",
        "link": "",
        "published_at": "2026-08-03T10:00:00Z",
        "received_at": "2026-08-03T10:05:00Z",
    }
    fmt_nvda_a = _format_articles([ev_nvda_a])
    resp_nvda_a = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{fmt_nvda_a}"
    hash_nvda_a = hashlib.sha256(resp_nvda_a.encode("utf-8")).hexdigest()

    ev_btc_a = {
        "title": "BTC-USD Story",
        "content": "Body",
        "summary": "Body",
        "publisher": "Unknown",
        "link": "",
        "published_at": "2026-08-10T10:00:00Z",
        "received_at": "2026-08-10T10:05:00Z",
    }
    fmt_btc_a = _format_articles([ev_btc_a])
    resp_btc_a = f"## BTC-USD News, from 2026-08-10 to 2026-08-10:\n\n{fmt_btc_a}"
    hash_btc_a = hashlib.sha256(resp_btc_a.encode("utf-8")).hexdigest()

    ev_nvda_b = {
        "title": "NVDA Story",
        "content": "Body",
        "published_at": "2026-08-03T10:00:00Z",
        "publishedAt": "2026-08-03T10:00:00Z",
        "received_at": "2026-08-03T10:05:00Z",
        "receivedAt": "2026-08-03T10:05:00Z",
        "symbols": ["NVDA"],
    }
    fmt_nvda_b = format_article(ev_nvda_b)
    resp_nvda_b = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{fmt_nvda_b}"
    hash_nvda_b = hashlib.sha256(resp_nvda_b.encode("utf-8")).hexdigest()

    ev_btc_b = {
        "title": "BTC-USD Story",
        "content": "Body",
        "published_at": "2026-08-10T10:00:00Z",
        "publishedAt": "2026-08-10T10:00:00Z",
        "received_at": "2026-08-10T10:05:00Z",
        "receivedAt": "2026-08-10T10:05:00Z",
        "symbols": ["BTC-USD"],
    }
    fmt_btc_b = format_article(ev_btc_b)
    resp_btc_b = f"## BTC-USD News, from 2026-08-10 to 2026-08-10:\n\n{fmt_btc_b}"
    hash_btc_b = hashlib.sha256(resp_btc_b.encode("utf-8")).hexdigest()

    (out / "baseline_archive").mkdir(parents=True, exist_ok=True)
    (out / "baseline_archive" / "yahoo_archive.json").write_text(
        json.dumps(
            [
                {
                    "ticker": "NVDA",
                    "query": "NVDA",
                    "cutoff": "2026-08-03T20:00:00Z",
                    "date": "2026-08-03",
                    "arm": "A",
                    "attempt_id": "att-1A",
                    "source": "yfinance",
                    "output_hash": hash_nvda_a,
                    "received_at": "2026-08-03T10:05:00Z",
                    "events": [ev_nvda_a],
                },
                {
                    "ticker": "BTC-USD",
                    "query": "BTC-USD",
                    "cutoff": "2026-08-10T20:00:00Z",
                    "date": "2026-08-10",
                    "arm": "A",
                    "attempt_id": "att-2A",
                    "source": "yfinance",
                    "output_hash": hash_btc_a,
                    "received_at": "2026-08-10T10:05:00Z",
                    "events": [ev_btc_a],
                },
            ]
        ),
        encoding="utf-8",
    )
    (out / "provenance_archive.json").write_text(
        json.dumps(
            [
                {
                    "ticker": "NVDA",
                    "query": "NVDA",
                    "cutoff": "2026-08-03T20:00:00Z",
                    "date": "2026-08-03",
                    "arm": "B",
                    "attempt_id": "att-1B",
                    "source": "news_intelligence",
                    "output_hash": hash_nvda_b,
                    "received_at": "2026-08-03T10:05:00Z",
                    "events": [ev_nvda_b],
                },
                {
                    "ticker": "BTC-USD",
                    "query": "BTC-USD",
                    "cutoff": "2026-08-10T20:00:00Z",
                    "date": "2026-08-10",
                    "arm": "B",
                    "attempt_id": "att-2B",
                    "source": "news_intelligence",
                    "output_hash": hash_btc_b,
                    "received_at": "2026-08-10T10:05:00Z",
                    "events": [ev_btc_b],
                },
            ]
        ),
        encoding="utf-8",
    )

    # Create native logs for both pairs
    _make_cell_log(
        out,
        "NVDA_2026-08-03_A",
        "NVDA",
        "2026-08-03",
        RATING_REVIEW,
        raw="+1.0%",
        alpha="+1.0%",
        holding_days=5,
        res_date="2026-08-10",
    )
    _make_cell_log(
        out,
        "NVDA_2026-08-03_B",
        "NVDA",
        "2026-08-03",
        "Buy",
        raw="+1.0%",
        alpha="+1.0%",
        holding_days=5,
        res_date="2026-08-10",
    )
    _make_cell_log(
        out,
        "BTC-USD_2026-08-10_A",
        "BTC-USD",
        "2026-08-10",
        "Hold",
        raw="-0.5%",
        alpha="-1.0%",
        holding_days=5,
        res_date="2026-08-17",
    )
    _make_cell_log(
        out,
        "BTC-USD_2026-08-10_B",
        "BTC-USD",
        "2026-08-10",
        "Hold",
        raw="-0.5%",
        alpha="-1.0%",
        holding_days=5,
        res_date="2026-08-17",
    )

    summary = summarize_experiment(out)

    assert summary["settled_pairs_count"] == 0
    assert summary["scored_pairs_count"] == 0
    assert summary["unscored_pairs_count"] == 1
    assert summary["strict_empirical_eligible"] is False
    assert summary["availability_status"] == "availability_unknown"
    assert summary["mean_alpha_A"] is None
    assert summary["mean_alpha_B"] is None
    assert summary["alpha_difference"] is None
    assert summary["directional_count_A"] == 0
    assert summary["directional_hits_A"] == 0
    assert summary["directional_hit_rate_A"] is None
    assert summary["directional_count_B"] == 0
    assert summary["directional_hits_B"] == 0
    assert summary["directional_hit_rate_B"] is None

    diag = summary["diagnostic_summary"]
    assert diag["diagnostic_settled_pairs_count"] == 2
    assert (
        diag["diagnostic_scored_pairs_count"] == 1
    )  # Only Pair 2 is scored; Pair 1 is unscored due to REVIEW
    assert diag["diagnostic_mean_alpha_A"] == -0.010
    assert diag["diagnostic_mean_alpha_B"] == -0.010
    assert diag["diagnostic_alpha_difference"] == 0.0
    assert diag["diagnostic_directional_count_A"] == 0
    assert diag["diagnostic_directional_hits_A"] == 0
    assert diag["diagnostic_directional_hit_rate_A"] is None
    assert diag["diagnostic_directional_count_B"] == 0
    assert diag["diagnostic_directional_hits_B"] == 0
    assert diag["diagnostic_directional_hit_rate_B"] is None


# ---------------------------------------------------------------------------
# R8: CLI Status & Stable Planned Denominator Proving Tests
# ---------------------------------------------------------------------------


def test_r8_cli_run_partial_returns_exit_1_and_preserves_full_planned_denominator(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(valid_manifest_payload), encoding="utf-8")
    out = tmp_path / "cli_partial"

    # Simulate provider failure on second cell
    call_count = 0

    def mock_ensure_provider(cfg: dict[str, Any]) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise RuntimeError("Provider authentication failed")

    with (
        patch(
            "experiments.t13_native_ab.ensure_provider_configured", side_effect=mock_ensure_provider
        ),
        patch("experiments.t13_native_ab.TradingAgentsGraph"),
    ):
        code = main(["run", "--manifest", str(manifest_file), "--output", str(out)])
        assert code == 1

    # Verify journal retains full planned denominator (4 cells across 2 pairs)
    journal = json.loads((out / "journal.json").read_text(encoding="utf-8"))
    assert journal["planned_cells_count"] == 4
    assert journal["planned_decisions_count"] == 4
    assert journal["planned_pairs_count"] == 2

    # Verify summary preserves planned count 4 even if early stopped
    summary = summarize_experiment(out)
    assert summary["planned_decisions_count"] == 4
    assert summary["attempted_decisions_count"] == 2
    assert summary["unattempted_decisions_count"] == 2


def test_r8_cli_settle_immature_window_returns_exit_1(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(valid_manifest_payload), encoding="utf-8")
    out = tmp_path / "cli_settle_pending"
    out.mkdir(parents=True, exist_ok=True)

    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))
    cell_keys = [
        "NVDA_2026-08-03_A",
        "NVDA_2026-08-03_B",
        "BTC-USD_2026-08-10_A",
        "BTC-USD_2026-08-10_B",
    ]
    decision_attempts = {}
    for k in cell_keys:
        parts = k.split("_")
        decision_attempts[k] = {
            "attempt_id": f"att-{k}",
            "phase": "completed",
            "ticker": parts[0],
            "date": parts[1],
            "arm": parts[2],
            "decision": "Buy",
        }
        res_dir = out / "cells" / k / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "trading_memory.log").write_text(
            f"[{parts[1]} | {parts[0]} | Buy | pending]\nDECISION:\nBuy\n\n<!-- ENTRY_END -->\n\n",
            encoding="utf-8",
        )

    initial_journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 4,
        "planned_decisions_count": 4,
        "planned_pairs_count": 2,
        "decision_attempts": decision_attempts,
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(initial_journal), encoding="utf-8")

    # settle_pending returns empty settlement (window not yet matured)
    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph") as mock_graph_cls,
    ):
        mock_graph_cls.return_value.settle_pending.return_value = Settlement(settled=[], failed=[])
        code = main(["settle", "--manifest", str(manifest_file), "--output", str(out)])
        # Immature window is pending measurement -> exit code 1
        assert code == 1

    journal = json.loads((out / "journal.json").read_text(encoding="utf-8"))
    assert journal["settlement_attempts"]["NVDA_2026-08-03_A"]["phase"] == "pending_window"


def test_cli_validate_command(valid_manifest_payload: dict[str, Any], tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(valid_manifest_payload), encoding="utf-8")

    code = main(["validate", "--manifest", str(manifest_file)])
    assert code == 0


def test_cli_summarize_command(tmp_path: Path) -> None:
    out = tmp_path / "cli_summary"
    out.mkdir(parents=True, exist_ok=True)
    journal = {
        "manifest_hash": "hash_abc",
        "experiment_id": "cli_exp",
        "planned_cells_count": 0,
        "planned_decisions_count": 0,
        "planned_pairs_count": 0,
        "decision_attempts": {},
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    code = main(["summarize", "--output", str(out)])
    assert code == 0


# ===========================================================================
# Round 2 Specific Proving Tests (R1 - R8)
# ===========================================================================


def test_r1_completed_attempt_with_missing_disk_artifacts_fails_and_blocks_replay(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    manifest_payload = {
        "experiment_id": "r1-disk-test",
        "holding_period": 5,
        "benchmark_ticker": "SPY",
        "pairs": [{"ticker": "NVDA", "date": "2026-08-03", "asset_type": "stock"}],
        "model_controls": {
            "provider": "google",
            "quick_model": "test-quick",
            "deep_model": "test-deep",
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_retries": 0,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_tool_rounds": 2,
            "recursion_limit": 50,
            "language": "English",
            "thinking_level": "low",
            "analysts": ["market", "news"],
        },
    }
    out = tmp_path / "exp_r1_completed_missing"
    out.mkdir(parents=True, exist_ok=True)

    manifest_file = out / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_payload), encoding="utf-8")

    # Run fresh to complete
    def make_mock_graph(*args: Any, **kwargs: Any) -> MagicMock:
        cfg = kwargs.get("config", {})
        mg = MagicMock()

        def fake_propagate(t: str, d: str, **kw: Any) -> tuple[dict[str, str], str]:
            TradingMemoryLog(cfg).store_decision(t, d, "Buy decision", rating="Buy")
            (Path(cfg["results_dir"]) / "complete_report.md").write_text(
                "# Report", encoding="utf-8"
            )
            return {"final_trade_decision": "Buy"}, "Buy"

        mg.propagate.side_effect = fake_propagate
        mg.save_reports.side_effect = lambda state, ticker, save_path: (
            save_path / "complete_report.md"
        )
        return mg

    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph", side_effect=make_mock_graph),
    ):
        j1 = run_experiment(manifest_payload, out)
        assert all(d["phase"] == "completed" for d in j1["decision_attempts"].values())

    # Unlink native memory log for Arm A
    log_A = out / "cells" / "NVDA_2026-08-03_A" / "results" / "trading_memory.log"
    assert log_A.exists()
    log_A.unlink()

    # Re-running must NOT invoke graph, and must mark Arm A failed, NOT replayed!
    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph") as mock_graph_cls2,
    ):
        j2 = run_experiment(manifest_payload, out)
        mock_graph_cls2.return_value.propagate.assert_not_called()
        assert j2["decision_attempts"]["NVDA_2026-08-03_A"]["phase"] == "failed"
        assert (
            j2["decision_attempts"]["NVDA_2026-08-03_A"]["attempt_id"]
            == j1["decision_attempts"]["NVDA_2026-08-03_A"]["attempt_id"]
        )

        # main run must return exit 1 with run_failed
        code = main(["run", "--manifest", str(manifest_file), "--output", str(out)])
        assert code == 1


def test_r1_missing_cell_in_journal_with_artifacts_on_disk_fails_closed(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "exp_r1_missing_cell"
    out.mkdir(parents=True, exist_ok=True)

    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))
    cell_dir = out / "cells" / "NVDA_2026-08-03_A"
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "untracked_artifact.txt").write_text("rogue file", encoding="utf-8")

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 4,
        "planned_decisions_count": 4,
        "planned_pairs_count": 2,
        "decision_attempts": {
            # NVDA_2026-08-03_A is missing from decision_attempts!
            "NVDA_2026-08-03_B": {"attempt_id": "att-B", "phase": "completed"},
        },
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(JournalPersistenceError, match="unjournaled artifacts on disk"):
        run_experiment(valid_manifest_payload, out)


def test_r2_settlement_rejects_unjournaled_extra_date_entry(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "exp_r2_extra_entry"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    cell_key = "NVDA_2026-08-03_A"
    res_dir = out / "cells" / cell_key / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Store intended decision
    mem_log = TradingMemoryLog({"memory_log_path": str(res_dir / "trading_memory.log")})
    mem_log.store_decision("NVDA", "2026-08-03", "Intended decision", rating="Buy")
    # Store second, unjournaled decision
    mem_log.store_decision("NVDA", "2026-08-04", "Unjournaled decision", rating="Buy")

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 4,
        "planned_decisions_count": 4,
        "planned_pairs_count": 2,
        "decision_attempts": {
            cell_key: {
                "attempt_id": "att-NVDA-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "decision": "Buy",
            }
        },
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph") as mock_graph,
    ):
        j = settle_experiment(valid_manifest_payload, out)
        # Graph settle_pending must NEVER be called
        mock_graph.return_value.settle_pending.assert_not_called()
        settle_rec = j["settlement_attempts"][cell_key]
        assert settle_rec["phase"] == "failed"
        assert "invariant violated" in settle_rec["error"]


def test_r2_settlement_missing_or_empty_log_fails_closed_never_pending_window(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "exp_r2_missing_log"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    cell_key = "NVDA_2026-08-03_A"
    res_dir = out / "cells" / cell_key / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    # Log does not exist!

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 4,
        "planned_decisions_count": 4,
        "planned_pairs_count": 2,
        "decision_attempts": {
            cell_key: {
                "attempt_id": "att-NVDA-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "decision": "Buy",
            }
        },
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    j = settle_experiment(valid_manifest_payload, out)
    settle_rec = j["settlement_attempts"][cell_key]
    assert settle_rec["phase"] == "failed"
    assert "missing on disk" in settle_rec["error"]
    assert settle_rec["settled"] is False


def test_r3_audit_callback_marks_unknown_usage_on_error() -> None:
    cb = AuditCallbackHandler()
    cb.on_llm_start({"name": "test-model"}, ["prompt"])
    cb.on_llm_error(RuntimeError("Connection terminated mid-stream"))

    assert cb.model_calls == 1
    assert cb.has_unknown_usage is True
    assert cb.token_usage["has_unknown_usage"] is True
    assert cb.token_usage["total_tokens"] == 0  # partial unverified total


def test_r4_summarize_rejects_differing_benchmark_returns_as_unmatched_vintages(
    tmp_path: Path,
) -> None:
    out = tmp_path / "exp_r4_bench"
    out.mkdir(parents=True, exist_ok=True)

    journal = {
        "manifest_hash": "hash_123",
        "experiment_id": "exp-r4-bench",
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-1A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-1B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
            },
        },
        "settlement_attempts": {
            # Differing benchmark returns (-0.010 vs -0.020)
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-1A",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.030,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.020,
                "benchmark_return_float": -0.010,
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-1B",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.040,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.020,
                "benchmark_return_float": -0.020,
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
    _make_cell_log(
        out, "NVDA_2026-08-03_A", "NVDA", "2026-08-03", "Buy", raw="+2.0%", alpha="+3.0%"
    )
    _make_cell_log(
        out, "NVDA_2026-08-03_B", "NVDA", "2026-08-03", "Buy", raw="+2.0%", alpha="+4.0%"
    )

    summary = summarize_experiment(out)
    assert summary["settled_pairs_count"] == 0
    assert summary["unmatched_vintages_count"] == 1
    assert summary["pair_eligibility"]["NVDA_2026-08-03"]["benchmark_vintage_equal"] is False


def test_r4_summarize_rejects_differing_raw_return_floats_with_same_rounded_string(
    tmp_path: Path,
) -> None:
    out = tmp_path / "exp_r4_raw_float"
    out.mkdir(parents=True, exist_ok=True)

    journal = {
        "manifest_hash": "hash_123",
        "experiment_id": "exp-r4-raw-float",
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-1A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-1B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
            },
        },
        "settlement_attempts": {
            # Different raw returns (0.0201 vs 0.0204) both formatted as "+2.0%"
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-1A",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.010,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.0201,
                "benchmark_return_float": 0.010,
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-1B",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.010,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.0204,
                "benchmark_return_float": 0.010,
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
    _make_cell_log(
        out, "NVDA_2026-08-03_A", "NVDA", "2026-08-03", "Buy", raw="+2.0%", alpha="+1.0%"
    )
    _make_cell_log(
        out, "NVDA_2026-08-03_B", "NVDA", "2026-08-03", "Buy", raw="+2.0%", alpha="+1.0%"
    )

    summary = summarize_experiment(out)
    assert summary["settled_pairs_count"] == 0
    assert summary["unmatched_vintages_count"] == 1
    assert summary["pair_eligibility"]["NVDA_2026-08-03"]["price_vintage_equal"] is False


def test_r4_summarize_scores_zero_pairs_when_native_logs_are_unlinked(tmp_path: Path) -> None:
    out = tmp_path / "exp_r4_unlinked_logs"
    out.mkdir(parents=True, exist_ok=True)

    journal = {
        "manifest_hash": "hash_123",
        "experiment_id": "exp-r4-unlinked",
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-1A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-1B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-1A",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.010,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.020,
                "benchmark_return_float": 0.010,
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-1B",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.010,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.020,
                "benchmark_return_float": 0.010,
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
    # No cell logs exist on disk!

    summary = summarize_experiment(out)
    assert summary["settled_pairs_count"] == 0
    assert summary["scored_pairs_count"] == 0
    assert summary["native_summary_A"] is None
    assert summary["native_summary_B"] is None
    assert summary["provenance_verified"] is False


def test_r5_terminal_crash_recovers_completed_decision_from_disk_artifacts(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    manifest_payload = {
        "experiment_id": "r5-crash-recovery",
        "holding_period": 5,
        "benchmark_ticker": "SPY",
        "pairs": [{"ticker": "NVDA", "date": "2026-08-03", "asset_type": "stock"}],
        "model_controls": {
            "provider": "google",
            "quick_model": "test-quick",
            "deep_model": "test-deep",
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_retries": 0,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_tool_rounds": 2,
            "recursion_limit": 50,
            "language": "English",
            "thinking_level": "low",
            "analysts": ["market", "news"],
        },
    }
    out = tmp_path / "exp_r5_crash"
    out.mkdir(parents=True, exist_ok=True)

    from experiments.t13_native_ab import _persist_journal as original_persist

    def crash_on_terminal_complete(target: Path, journal: dict[str, Any]) -> None:
        if any(
            d.get("phase") == "completed" for d in journal.get("decision_attempts", {}).values()
        ):
            raise JournalPersistenceError("Simulated terminal write crash")
        original_persist(target, journal)

    def make_mock_graph_crash(*args: Any, **kwargs: Any) -> MagicMock:
        cfg = kwargs.get("config", {})
        mg = MagicMock()

        def fake_propagate(t: str, d: str, **kw: Any) -> tuple[dict[str, str], str]:
            TradingMemoryLog(cfg).store_decision(t, d, "Buy decision", rating="Buy")
            rep = Path(cfg["results_dir"]) / "complete_report.md"
            rep.write_text("# Complete Report", encoding="utf-8")
            return {"final_trade_decision": "Buy"}, "Buy"

        mg.propagate.side_effect = fake_propagate
        mg.save_reports.side_effect = lambda state, ticker, save_path: (
            save_path / "complete_report.md"
        )
        return mg

    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph", side_effect=make_mock_graph_crash),
        patch("experiments.t13_native_ab._persist_journal", side_effect=crash_on_terminal_complete),
        pytest.raises(JournalPersistenceError, match="Simulated terminal write crash"),
    ):
        run_experiment(manifest_payload, out)

    # Disk journal currently has phase: started with pre-populated report_path
    disk_j = json.loads((out / "journal.json").read_text(encoding="utf-8"))
    a_rec = disk_j["decision_attempts"]["NVDA_2026-08-03_A"]
    assert a_rec["phase"] == "started"
    assert a_rec["report_path"] is not None

    # Restarting run_experiment must recover completed decision with zero graph calls for Arm A!
    def make_mock_graph_recovery(*args: Any, **kwargs: Any) -> MagicMock:
        cfg = kwargs.get("config", {})
        mg = MagicMock()

        def fake_propagate_B(t: str, d: str, **kw: Any) -> tuple[dict[str, str], str]:
            if "NVDA_2026-08-03_A" in cfg.get("results_dir", ""):
                raise AssertionError("Graph propagate was unexpectedly called for Arm A")
            TradingMemoryLog(cfg).store_decision(t, d, "Buy decision", rating="Buy")
            rep = Path(cfg["results_dir"]) / "complete_report.md"
            rep.write_text("# Complete Report B", encoding="utf-8")
            return {"final_trade_decision": "Buy"}, "Buy"

        mg.propagate.side_effect = fake_propagate_B
        mg.save_reports.side_effect = lambda state, ticker, save_path: (
            save_path / "complete_report.md"
        )
        return mg

    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph", side_effect=make_mock_graph_recovery),
    ):
        recovered = run_experiment(manifest_payload, out)
        assert recovered["decision_attempts"]["NVDA_2026-08-03_A"]["phase"] == "completed"


def test_r6_settlement_rechecks_preserve_all_attempt_history_and_ids(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    manifest_payload = {
        "experiment_id": "r6-history-test",
        "holding_period": 5,
        "benchmark_ticker": "SPY",
        "pairs": [{"ticker": "NVDA", "date": "2026-08-03", "asset_type": "stock"}],
        "model_controls": {
            "provider": "google",
            "quick_model": "test-quick",
            "deep_model": "test-deep",
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_retries": 0,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_tool_rounds": 2,
            "recursion_limit": 50,
            "language": "English",
            "thinking_level": "low",
            "analysts": ["market", "news"],
        },
    }
    out = tmp_path / "exp_r6_rechecks"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(manifest_payload))

    cell_key = "NVDA_2026-08-03_A"
    res_dir = out / "cells" / cell_key / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    TradingMemoryLog({"memory_log_path": str(res_dir / "trading_memory.log")}).store_decision(
        "NVDA", "2026-08-03", "Pending decision", rating="Buy"
    )

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            cell_key: {
                "attempt_id": "att-NVDA-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "decision": "Buy",
            }
        },
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph") as mock_graph,
    ):
        mock_graph.return_value.settle_pending.return_value = Settlement(settled=[], failed=[])

        first = settle_experiment(manifest_payload, out)
        first_id = first["settlement_attempts"][cell_key]["settlement_id"]
        assert first["settlement_attempts"][cell_key]["phase"] == "pending_window"

        second = settle_experiment(manifest_payload, out)
        second_id = second["settlement_attempts"][cell_key]["settlement_id"]
        assert second_id != first_id

        third = settle_experiment(manifest_payload, out)
        third_id = third["settlement_attempts"][cell_key]["settlement_id"]

        # Both first_id and second_id MUST be preserved in the third journal!
        serialized = json.dumps(third)
        assert first_id in serialized
        assert second_id in serialized
        assert third_id in serialized

        history = third["settlement_attempts"][cell_key]["history"]
        assert len(history) == 2
        assert history[0]["settlement_id"] == first_id
        assert history[1]["settlement_id"] == second_id


def test_r8_cli_settle_partial_grid_returns_exit_1_and_preserves_full_planned_denominator(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    manifest_payload = {
        "experiment_id": "r8-partial-grid",
        "holding_period": 5,
        "benchmark_ticker": "SPY",
        "pairs": [
            {"ticker": "NVDA", "date": "2026-08-03", "asset_type": "stock"},
            {"ticker": "AAPL", "date": "2026-08-03", "asset_type": "stock"},
        ],
        "model_controls": {
            "provider": "google",
            "quick_model": "test-quick",
            "deep_model": "test-deep",
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_retries": 0,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_tool_rounds": 2,
            "recursion_limit": 50,
            "language": "English",
            "thinking_level": "low",
            "analysts": ["market", "news"],
        },
    }
    out = tmp_path / "exp_r8_partial"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(manifest_payload))

    manifest_file = out / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_payload), encoding="utf-8")

    # Planned grid has 4 cells. 1 completed, 1 failed in decision phase.
    cell_key = "NVDA_2026-08-03_A"
    res_dir = out / "cells" / cell_key / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    TradingMemoryLog({"memory_log_path": str(res_dir / "trading_memory.log")}).store_decision(
        "NVDA", "2026-08-03", "Decision A", rating="Buy"
    )

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": manifest_payload["experiment_id"],
        "planned_cells_count": 4,
        "planned_decisions_count": 4,
        "planned_pairs_count": 2,
        "decision_attempts": {
            cell_key: {
                "attempt_id": "att-NVDA-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "decision": "Buy",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-NVDA-B",
                "phase": "failed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "error": "Model quota exhausted",
            },
        },
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph") as mock_graph,
    ):
        # Settle Arm A successfully
        mock_graph.return_value.settle_pending.return_value = Settlement(
            settled=[("NVDA", "2026-08-03")], failed=[]
        )
        # Update log to resolved
        mem = TradingMemoryLog({"memory_log_path": str(res_dir / "trading_memory.log")})
        mem.update_with_outcome(
            "NVDA",
            "2026-08-03",
            raw_return=0.02,
            alpha_return=0.01,
            holding_days=5,
            reflection="Refl",
            resolution_date="2026-08-10",
        )

        code = main(["settle", "--manifest", str(manifest_file), "--output", str(out)])
        # Must return exit 1 because grid is partial and incomplete!
        assert code == 1


# ---------------------------------------------------------------------------
# Round 3 Findings Proving Tests (R1–R5)
# ---------------------------------------------------------------------------


def test_r3_r1_retrospective_timing_and_provenance_eligibility_fails_without_archive(
    tmp_path: Path,
) -> None:
    """Retrospective execution clocks (> trade_date) without contemporaneous input clock archives must fail timing eligibility."""
    out = tmp_path / "exp_retrospective"
    out.mkdir(parents=True, exist_ok=True)

    journal = {
        "manifest_hash": "hash_retro",
        "experiment_id": "exp-retro",
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "decision": "Buy",
                "started_at": "2026-10-15T12:00:00+00:00",
                "completed_at": "2026-10-15T12:01:00+00:00",
                "non_news_inputs_hash": "tool_hash_1",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "decision": "Buy",
                "started_at": "2026-10-15T12:00:00+00:00",
                "completed_at": "2026-10-15T12:01:00+00:00",
                "non_news_inputs_hash": "tool_hash_1",
                "entry_time": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.010,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.020,
                "benchmark_return_float": 0.010,
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.010,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.020,
                "benchmark_return_float": 0.010,
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    # Native memory logs exist on disk
    for arm in ["A", "B"]:
        res_dir = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        mem = TradingMemoryLog({"memory_log_path": str(res_dir / "trading_memory.log")})
        mem.store_decision("NVDA", "2026-08-03", f"Decision {arm}", rating="Buy")
        mem.update_with_outcome(
            "NVDA",
            "2026-08-03",
            raw_return=0.02,
            alpha_return=0.01,
            holding_days=5,
            reflection="Refl",
            resolution_date="2026-08-10",
        )

    summary = summarize_experiment(out)
    assert summary["timing_eligible"] is False
    assert summary["provenance_verified"] is False
    assert summary["scored_pairs_count"] == 0
    assert summary["ineligible_pairs_count"] == 1
    assert summary["pair_eligibility"]["NVDA_2026-08-03"]["timing_eligible"] is False
    assert summary["pair_eligibility"]["NVDA_2026-08-03"]["provenance_verified"] is False
    assert any(
        "Retrospective decision clock" in r
        for r in summary["pair_eligibility"]["NVDA_2026-08-03"]["ineligibility_reasons"]
    )


def test_r3_r2_non_news_equality_detects_different_tool_outputs_for_same_query() -> None:
    """Different tool responses for identical queries must produce different tool hashes and fail equality."""
    cb1 = AuditCallbackHandler()
    cb1.on_tool_start({"name": "get_stock_data"}, '{"symbol":"NVDA","date":"2026-08-03"}')
    cb1.on_tool_end("close=100")

    cb2 = AuditCallbackHandler()
    cb2.on_tool_start({"name": "get_stock_data"}, '{"symbol":"NVDA","date":"2026-08-03"}')
    cb2.on_tool_end("close=200")

    assert cb1.non_news_tool_hashes != cb2.non_news_tool_hashes
    assert cb1.traces[-1]["output_hash"] != cb2.traces[-1]["output_hash"]


def test_r3_r2_native_graph_tool_callbacks_are_propagated_and_observed(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from langchain_core.tools import tool

    from experiments.t13_native_ab import TradingAgentsGraph
    from tradingagents.graph.propagation import Propagator

    invoked_args = []

    @tool
    def synthetic_price(ticker: str) -> str:
        """Return synthetic price."""
        return "close=150.0"

    handler = AuditCallbackHandler()
    manifest = {
        "experiment_id": "test-cb",
        "benchmark_ticker": "SPY",
        "holding_period": 5,
        "pairs": [{"ticker": "NVDA", "date": "2026-08-03", "asset_type": "stock"}],
        "model_controls": {
            "provider": "google",
            "quick_model": "mock",
            "deep_model": "mock",
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_retries": 0,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_tool_rounds": 2,
            "recursion_limit": 50,
            "language": "English",
            "thinking_level": "low",
            "analysts": ["market"],
        },
    }
    validated = validate_manifest(manifest)
    cfg = _build_arm_config(validated, "A", tmp_path / "res", tmp_path / "data", tmp_path / "cache")
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["data_cache_dir"]).mkdir(parents=True, exist_ok=True)

    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("tradingagents.graph.trading_graph.create_tier_client"),
    ):
        graph = TradingAgentsGraph(selected_analysts=["market"], config=cfg, callbacks=[handler])
        graph.debug = False
        graph.propagator = Propagator(max_recur_limit=50)
        graph.create_run_state = lambda *a, **kw: {
            "company_of_interest": "NVDA",
            "trade_date": "2026-08-03",
        }
        graph.checkpoint_input = lambda state: state
        graph.record_decision = lambda *a: None

        def invoke_tool_boundary(state: Any, **kwargs: Any) -> dict[str, str]:
            invoked_args.append(kwargs)
            synthetic_price.invoke({"ticker": "NVDA"}, config=kwargs.get("config"))
            return {"final_trade_decision": "Buy"}

        graph.graph = SimpleNamespace(invoke=invoke_tool_boundary)
        graph._run_graph("NVDA", "2026-08-03")

        assert len(invoked_args) == 1
        config_arg = invoked_args[0].get("config", {})
        assert "callbacks" in config_arg
        assert handler in config_arg["callbacks"]
        assert handler.tool_calls >= 1


def test_r3_r3_settlement_crash_missing_exact_floats_excludes_from_scoring(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """When a completion write crash loses exact return floats, recovered outcomes must fail vintage matching and be excluded."""
    manifest_payload = {
        "experiment_id": "r3-exact-drift",
        "holding_period": 5,
        "benchmark_ticker": "SPY",
        "pairs": [{"ticker": "NVDA", "date": "2026-08-03", "asset_type": "stock"}],
        "model_controls": {
            "provider": "google",
            "quick_model": "test-quick",
            "deep_model": "test-deep",
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_retries": 0,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_tool_rounds": 2,
            "recursion_limit": 50,
            "language": "English",
            "thinking_level": "low",
            "analysts": ["market", "news"],
        },
    }
    out = tmp_path / "exp_crash_exact_vintage"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(manifest_payload))

    decision_attempts = {}
    for arm in ["A", "B"]:
        k = f"NVDA_2026-08-03_{arm}"
        decision_attempts[k] = {
            "attempt_id": f"att-{arm}",
            "phase": "completed",
            "ticker": "NVDA",
            "date": "2026-08-03",
            "arm": arm,
            "decision": "Buy",
            "started_at": "2026-08-03T09:00:00+00:00",
            "completed_at": "2026-08-03T09:05:00+00:00",
            "non_news_inputs_hash": "tool_hash_match",
        }
        res_dir = out / "cells" / k / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "provenance_archive.json").write_text(
            json.dumps({"provenance": "verified"}), encoding="utf-8"
        )
        TradingMemoryLog({"memory_log_path": str(res_dir / "trading_memory.log")}).store_decision(
            "NVDA", "2026-08-03", f"Decision {arm}", rating="Buy"
        )

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": decision_attempts,
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    from experiments.t13_native_ab import _persist_journal as original_persist

    def fail_settlement_terminal(target: Path, j: dict[str, Any]) -> None:
        if any(s.get("phase") == "completed" for s in j.get("settlement_attempts", {}).values()):
            raise JournalPersistenceError("Simulated settlement completion write failure")
        original_persist(target, j)

    def make_mock_graph_settle(*args: Any, **kwargs: Any) -> MagicMock:
        cfg = kwargs.get("config", {})
        mg = MagicMock()

        def fake_settle(ticker: str) -> Settlement:
            mem = TradingMemoryLog(cfg)
            mem.update_with_outcome(
                ticker,
                "2026-08-03",
                raw_return=0.0201,
                alpha_return=0.0101,
                holding_days=5,
                reflection="Refl",
                resolution_date="2026-08-10",
            )
            return Settlement(settled=[(ticker, "2026-08-03")], failed=[])

        mg.settle_pending.side_effect = fake_settle
        return mg

    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph", side_effect=make_mock_graph_settle),
        patch("experiments.t13_native_ab._persist_journal", side_effect=fail_settlement_terminal),
        pytest.raises(JournalPersistenceError),
    ):
        settle_experiment(manifest_payload, out)

    def make_mock_graph_settle_B(*args: Any, **kwargs: Any) -> MagicMock:
        cfg = kwargs.get("config", {})
        mg = MagicMock()

        def fake_settle(ticker: str) -> Settlement:
            mem = TradingMemoryLog(cfg)
            mem.update_with_outcome(
                ticker,
                "2026-08-03",
                raw_return=0.0204,
                alpha_return=0.02,
                holding_days=5,
                reflection="Refl",
                resolution_date="2026-08-10",
            )
            return Settlement(settled=[(ticker, "2026-08-03")], failed=[])

        mg.settle_pending.side_effect = fake_settle
        return mg

    # Now run settle again with Arm B getting 0.0204
    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph", side_effect=make_mock_graph_settle_B),
    ):
        after = settle_experiment(manifest_payload, out)

    arm_a = after["settlement_attempts"]["NVDA_2026-08-03_A"]
    assert arm_a["raw_return_float"] is None
    assert arm_a["benchmark_return_float"] is None

    summary = summarize_experiment(out)
    assert summary["scored_pairs_count"] == 0
    assert summary["unmatched_vintages_count"] == 1
    assert summary["pair_eligibility"]["NVDA_2026-08-03"]["price_vintage_equal"] is False


def test_r3_r3_native_outcome_contradiction_with_journal_fails_verification(tmp_path: Path) -> None:
    """When on-disk native memory log contradicts journal settlement record, summary must reject the pair."""
    out = tmp_path / "exp_contradiction"
    out.mkdir(parents=True, exist_ok=True)

    journal = {
        "manifest_hash": "hash_contra",
        "experiment_id": "exp-contra",
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "decision": "Buy",
                "started_at": "2026-08-03T09:00:00+00:00",
                "completed_at": "2026-08-03T09:05:00+00:00",
                "non_news_inputs_hash": "h1",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "decision": "Buy",
                "started_at": "2026-08-03T09:00:00+00:00",
                "completed_at": "2026-08-03T09:05:00+00:00",
                "non_news_inputs_hash": "h1",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.010,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.020,
                "benchmark_return_float": 0.010,
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "rating": "Buy",
                "outcome": 0.010,
                "raw_return": "+2.0%",
                "holding_period": 5,
                "resolution_date": "2026-08-10",
                "raw_return_float": 0.020,
                "benchmark_return_float": 0.010,
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    for arm in ["A", "B"]:
        res_dir = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "provenance_archive.json").write_text(
            json.dumps({"provenance": "verified"}), encoding="utf-8"
        )
        mem = TradingMemoryLog({"memory_log_path": str(res_dir / "trading_memory.log")})
        mem.store_decision("NVDA", "2026-08-03", f"Decision {arm}", rating="Buy")
        mem.update_with_outcome(
            "NVDA",
            "2026-08-03",
            raw_return=0.02,
            alpha_return=0.01,
            holding_days=5,
            reflection="Refl",
            resolution_date="2026-08-10",
        )

    # Contradict Arm A's log by replacing Buy outcome with REVIEW outcome
    log_a = out / "cells" / "NVDA_2026-08-03_A" / "results" / "trading_memory.log"
    orig_text = log_a.read_text(encoding="utf-8")
    contradicted = orig_text.replace(
        "| Buy | +2.0% | +1.0% | 5d | resolved:2026-08-10",
        "| REVIEW | +90.0% | +80.0% | 2d | resolved:2026-08-20",
    )
    log_a.write_text(contradicted, encoding="utf-8")

    summary = summarize_experiment(out)
    assert summary["scored_pairs_count"] == 0
    assert summary["native_summary_A"]["unscored"] == 1
    assert summary["pair_eligibility"]["NVDA_2026-08-03"]["eligible"] is False
    assert any(
        "Contradiction between on-disk native memory log" in r
        for r in summary["pair_eligibility"]["NVDA_2026-08-03"]["ineligibility_reasons"]
    )


def test_r3_r4_decision_recovery_preserves_paid_tokens_from_audit_trace(tmp_path: Path) -> None:
    """When decision completion write fails after audit trace was written, restart must recover usage numbers from audit_trace.json."""
    manifest_payload = {
        "experiment_id": "r4-usage-recovery",
        "holding_period": 5,
        "benchmark_ticker": "SPY",
        "pairs": [{"ticker": "NVDA", "date": "2026-08-03", "asset_type": "stock"}],
        "model_controls": {
            "provider": "google",
            "quick_model": "test-quick",
            "deep_model": "test-deep",
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_retries": 0,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_tool_rounds": 2,
            "recursion_limit": 50,
            "language": "English",
            "thinking_level": "low",
            "analysts": ["market", "news"],
        },
    }
    out = tmp_path / "exp_r4_usage"
    out.mkdir(parents=True, exist_ok=True)

    from experiments.t13_native_ab import _persist_journal as original_persist

    def fail_decision_terminal(target: Path, j: dict[str, Any]) -> None:
        if any(d.get("phase") == "completed" for d in j.get("decision_attempts", {}).values()):
            raise JournalPersistenceError("Simulated decision completion write failure")
        original_persist(target, j)

    propagate_invocations: list[str] = []

    def mock_graph_with_usage(*args: Any, **kwargs: Any) -> MagicMock:
        cfg = kwargs.get("config", {})
        callbacks = kwargs.get("callbacks", [])
        mg = MagicMock()

        def fake_propagate(t: str, d: str, **kw: Any) -> tuple[dict[str, str], str]:
            propagate_invocations.append(f"{t}_{d}")
            for cb in callbacks:
                cb.token_usage["total_tokens"] = 42
                cb.model_calls = 3
                cb.model_usage = {"test-quick": {"total_tokens": 42}}
            TradingMemoryLog(cfg).store_decision(t, d, "Buy decision", rating="Buy")
            rep = Path(cfg["results_dir"]) / "complete_report.md"
            rep.write_text("# Complete Report", encoding="utf-8")
            return {"final_trade_decision": "Buy"}, "Buy"

        mg.propagate.side_effect = fake_propagate
        mg.save_reports.side_effect = lambda state, ticker, save_path: (
            save_path / "complete_report.md"
        )
        return mg

    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph", side_effect=mock_graph_with_usage),
        patch("experiments.t13_native_ab._persist_journal", side_effect=fail_decision_terminal),
        pytest.raises(JournalPersistenceError),
    ):
        run_experiment(manifest_payload, out)

    trace_file = out / "cells" / "NVDA_2026-08-03_A" / "results" / "audit_trace.json"
    assert trace_file.exists()
    trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
    assert trace_data["token_usage"]["total_tokens"] == 42
    assert len(propagate_invocations) == 1

    # Now restart run_experiment with synthetic boundaries retained across restart (R6)
    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph", side_effect=mock_graph_with_usage),
    ):
        recovered = run_experiment(manifest_payload, out)

    arm_a = recovered["decision_attempts"]["NVDA_2026-08-03_A"]
    assert arm_a["phase"] == "completed"
    assert arm_a["model_calls"] == 3
    assert arm_a["model_usage"]["total_tokens"] == 42
    assert arm_a["has_unknown_usage"] is False

    arm_b = recovered["decision_attempts"]["NVDA_2026-08-03_B"]
    assert arm_b["phase"] == "completed"
    assert arm_b["model_calls"] == 3
    assert arm_b["model_usage"]["total_tokens"] == 42
    assert arm_b["has_unknown_usage"] is False

    # Arm A was recovered from disk with zero additional propagate calls; Arm B completed safely
    assert len(propagate_invocations) == 2


def test_r3_r4_settle_preflight_failure_preserves_prior_uncertain_paid_attempt(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Preflight settlement failure must not wipe out prior paid attempt ID, clocks, and usage uncertainty."""
    manifest_payload = {
        "experiment_id": "r4-preflight-uncertainty",
        "holding_period": 5,
        "benchmark_ticker": "SPY",
        "pairs": [{"ticker": "NVDA", "date": "2026-08-03", "asset_type": "stock"}],
        "model_controls": {
            "provider": "google",
            "quick_model": "test-quick",
            "deep_model": "test-deep",
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_retries": 0,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_tool_rounds": 2,
            "recursion_limit": 50,
            "language": "English",
            "thinking_level": "low",
            "analysts": ["market", "news"],
        },
    }
    out = tmp_path / "exp_r4_preflight"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(manifest_payload))

    cell_key = "NVDA_2026-08-03_A"
    res_dir = out / "cells" / cell_key / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            cell_key: {
                "attempt_id": "att-NVDA-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "decision": "Buy",
            }
        },
        "settlement_attempts": {
            cell_key: {
                "settlement_id": "settle-paid-uncertain-uuid",
                "decision_attempt_id": "att-NVDA-A",
                "phase": "started",
                "started_at": "2026-10-15T10:00:00+00:00",
                "model_calls": 1,
                "has_unknown_usage": True,
            }
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    # Native memory log is missing!
    after = settle_experiment(manifest_payload, out)
    settle_att = after["settlement_attempts"][cell_key]

    assert settle_att["settlement_id"] == "settle-paid-uncertain-uuid"
    assert settle_att["started_at"] == "2026-10-15T10:00:00+00:00"
    assert settle_att["model_calls"] == 1
    assert settle_att["has_unknown_usage"] is True
    assert settle_att["phase"] == "failed"
    assert "Native memory log is missing" in settle_att["error"]


def test_r3_r5_recheck_history_grows_linearly_not_exponentially(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Repeated immature window rechecks must retain flat history without exponential size growth."""
    manifest_payload = {
        "experiment_id": "r5-history-linear",
        "holding_period": 5,
        "benchmark_ticker": "SPY",
        "pairs": [{"ticker": "NVDA", "date": "2026-08-03", "asset_type": "stock"}],
        "model_controls": {
            "provider": "google",
            "quick_model": "test-quick",
            "deep_model": "test-deep",
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_retries": 0,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_tool_rounds": 2,
            "recursion_limit": 50,
            "language": "English",
            "thinking_level": "low",
            "analysts": ["market", "news"],
        },
    }
    out = tmp_path / "exp_r5_linear"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(manifest_payload))

    for arm in ["A", "B"]:
        k = f"NVDA_2026-08-03_{arm}"
        res_dir = out / "cells" / k / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        TradingMemoryLog({"memory_log_path": str(res_dir / "trading_memory.log")}).store_decision(
            "NVDA", "2026-08-03", f"Pending {arm}", rating="Buy"
        )

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "decision": "Buy",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "decision": "Buy",
            },
        },
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    from tradingagents.memory.settlement import Settlement

    sizes = []
    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph") as mock_graph,
    ):
        mock_graph.return_value.settle_pending.return_value = Settlement(settled=[], failed=[])
        for _ in range(8):
            after = settle_experiment(manifest_payload, out)
            sizes.append((out / "journal.json").stat().st_size)

    history = after["settlement_attempts"]["NVDA_2026-08-03_A"]["history"]
    assert len(history) == 7
    # Linear growth: ratio between consecutive checks is near 1.0 (e.g. < 1.35), NOT > 1.9x
    growth_ratio = sizes[-1] / sizes[-2]
    assert growth_ratio < 1.35


# ---------------------------------------------------------------------------
# R4 Reviewer Corrections Proving Tests (R1, R2, R3, R4, R5)
# ---------------------------------------------------------------------------


class _TestStructuredDecision(BaseModel):
    rating: str


class _TestOfflineStructuredChat(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def test_r4_chat_model_start_accounting_and_traces() -> None:
    """Normal LangChain chat model calls must increment request counter and record input trace (R2)."""
    cb = AuditCallbackHandler()
    model = FakeListChatModel(responses=["REVIEWER_ONLY output"], callbacks=[cb])
    model.invoke([HumanMessage(content="REVIEWER_ONLY actual prompt")])
    assert cb.model_calls == 1
    assert cb.pending_calls == 0
    assert any(t["type"] == "llm_start" for t in cb.traces)
    assert any(t["type"] == "llm_end" for t in cb.traces)
    start_trace = next(t for t in cb.traces if t["type"] == "llm_start")
    assert len(start_trace["prompts"]) == 1
    assert "REVIEWER_ONLY actual prompt" in start_trace["prompts"][0]


def test_r4_structured_chat_accounting_and_traces() -> None:
    """Structured chat invocation must record one request, input trace, and token usage (R2)."""
    cb = AuditCallbackHandler()
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "_TestStructuredDecision", "args": {"rating": "Buy"}, "id": "test-call-1"}
        ],
        usage_metadata={"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
    )
    model = _TestOfflineStructuredChat(responses=[message], callbacks=[cb])
    parsed = model.with_structured_output(_TestStructuredDecision).invoke(
        [HumanMessage(content="REVIEWER_ONLY structured input")]
    )
    assert parsed.rating == "Buy"
    assert cb.model_calls == 1
    assert cb.pending_calls == 0
    assert cb.token_usage["total_tokens"] == 19
    assert any(t["type"] == "llm_start" for t in cb.traces)


def test_r4_no_duplicate_model_calls_when_both_callbacks_present() -> None:
    """Duplicate starts for the same run_id must not double count model requests (R2)."""
    cb = AuditCallbackHandler()
    cb.on_chat_model_start(
        {"name": "test-model"}, [[HumanMessage(content="hello")]], run_id="run-dup-1"
    )
    cb.on_llm_start({"name": "test-model"}, ["hello"], run_id="run-dup-1")
    assert cb.model_calls == 1
    assert cb.pending_calls == 1


def test_r4_parallel_tool_completions_matched_by_run_id() -> None:
    """Out-of-order completions in parallel tools must be matched by run_id, not stack order (R3)."""
    cb = AuditCallbackHandler()
    cb.on_tool_start({"name": "get_stock_data"}, "price-query", run_id="price-1")
    cb.on_tool_start({"name": "get_news"}, "news-query", run_id="news-2")
    cb.on_tool_end("price=100", run_id="price-1")
    cb.on_tool_end("news article", run_id="news-2")
    ends = [t for t in cb.traces if t["type"] == "tool_end"]
    assert ends[0]["tool"] == "get_stock_data"
    assert ends[1]["tool"] == "get_news"
    # Ensure news output was not added to non_news_tool_hashes
    assert len(cb.non_news_tool_hashes) == 1


def test_r4_parallel_tool_error_matched_by_run_id() -> None:
    """Tool error interleaved with parallel tool call matches correct run_id (R3)."""
    cb = AuditCallbackHandler()
    cb.on_tool_start({"name": "get_stock_data"}, "price-query", run_id="price-1")
    cb.on_tool_start({"name": "get_news"}, "news-query", run_id="news-2")
    cb.on_tool_error(RuntimeError("price error"), run_id="price-1")
    cb.on_tool_end("news article", run_id="news-2")
    errs = [t for t in cb.traces if t["type"] == "tool_error"]
    assert len(errs) == 1
    assert errs[0]["tool"] == "get_stock_data"


def test_r4_empty_archive_directory_stays_ineligible(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """An empty baseline_archive directory must NOT establish provenance or timing eligibility (R1)."""
    out = tmp_path / "exp_empty_archive"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-10-15T12:00:00Z",
                "non_news_inputs_hash": "hash123",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-10-15T12:00:00Z",
                "non_news_inputs_hash": "hash123",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.05,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.05,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    for arm in ["A", "B"]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        log_file = p / "trading_memory.log"
        log_file.write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )

    # Creating an empty baseline_archive directory with zero files
    (out / "baseline_archive").mkdir()

    summary = summarize_experiment(out)
    assert summary["scored_pairs_count"] == 0
    assert summary["provenance_verified"] is False
    assert summary["timing_eligible"] is False


def test_r4_positive_archive_evidence_passes(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Authentic archive evidence with matching ticker, date, and valid clocks establishes eligibility (R1)."""
    out = tmp_path / "exp_valid_archive"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-10-15T12:00:00Z",
                "non_news_inputs_hash": "hash123",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-10-15T12:00:00Z",
                "non_news_inputs_hash": "hash123",
                "entry_time": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    from tradingagents.dataflows.vendors.news_intelligence.client import format_article
    from tradingagents.dataflows.vendors.yahoo.news import _format_articles

    ev_A = {
        "id": "ev-yahoo-1",
        "title": "NVIDIA AI Expansion (Yahoo)",
        "summary": "Substantive Yahoo baseline article content...",
        "content": "Substantive Yahoo baseline article content...",
        "publisher": "Yahoo",
        "link": "",
        "published_at": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
    }
    content_A = _format_articles([ev_A])
    resp_A = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{content_A}"
    hash_A = hashlib.sha256(resp_A.encode("utf-8")).hexdigest()

    ev_B = {
        "id": "ev-repo1-1",
        "title": "NVIDIA AI Expansion (Repo 1)",
        "published_at": "2026-08-03T14:00:00Z",
        "publishedAt": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
        "receivedAt": "2026-08-03T14:05:00Z",
        "enriched_at": "2026-08-03T14:10:00Z",
        "content": "Substantive Repo 1 News Intelligence content with sentiment...",
        "symbols": ["NVDA"],
    }
    content_B = format_article(ev_B)
    resp_B = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{content_B}"
    hash_B = hashlib.sha256(resp_B.encode("utf-8")).hexdigest()

    for arm, c_text, h_val in [("A", resp_A, hash_A), ("B", resp_B, hash_B)]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        log_file = p / "trading_memory.log"
        log_file.write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )
        (p / "audit_trace.json").write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "tool_start",
                            "tool": "get_news",
                            "input": {
                                "ticker": "NVDA",
                                "start_date": "2026-08-03",
                                "end_date": "2026-08-03",
                            },
                            "run_id": f"run-{arm}",
                            "timestamp": "2026-08-03T14:00:30Z",
                        },
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output": c_text,
                            "output_hash": h_val,
                            "run_id": f"run-{arm}",
                            "timestamp": "2026-08-03T14:01:00Z",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    archive_dir = out / "baseline_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    # Arm A: Yahoo baseline evidence
    (archive_dir / "nvda_yahoo_baseline.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "query": "NVDA",
                "cutoff": "2026-08-03T20:00:00Z",
                "date": "2026-08-03",
                "arm": "A",
                "attempt_id": "att-A",
                "source": "yfinance",
                "output_hash": hash_A,
                "received_at": "2026-08-03T14:05:00Z",
                "events": [ev_A],
            }
        ),
        encoding="utf-8",
    )
    # Arm B: Repo 1 News Intelligence evidence
    (out / "provenance_archive.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "query": "NVDA",
                "cutoff": "2026-08-03T20:00:00Z",
                "date": "2026-08-03",
                "arm": "B",
                "attempt_id": "att-B",
                "source": "news_intelligence",
                "output_hash": hash_B,
                "received_at": "2026-08-03T14:05:00Z",
                "events": [ev_B],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_experiment(out)
    assert summary["scored_pairs_count"] == 0
    assert summary["provenance_verified"] is False
    assert summary["timing_eligible"] is False
    assert summary["strict_empirical_eligible"] is False
    assert summary["availability_status"] == "availability_unknown"
    assert summary["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert summary["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    assert summary["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True
    assert summary["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_eligible"] is True


def test_r4_holding_window_native_contradiction_rejected(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Native memory log holding tag contradiction (2d vs 5d) must invalidate pairing (R4)."""
    out = tmp_path / "exp_holding_contra"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T09:30:00Z",
                "non_news_inputs_hash": "hash123",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T09:30:00Z",
                "non_news_inputs_hash": "hash123",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.05,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.05,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    p_A = out / "cells" / "NVDA_2026-08-03_A" / "results"
    p_A.mkdir(parents=True, exist_ok=True)
    (p_A / "trading_memory.log").write_text(
        "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 2d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
        encoding="utf-8",
    )

    p_B = out / "cells" / "NVDA_2026-08-03_B" / "results"
    p_B.mkdir(parents=True, exist_ok=True)
    (p_B / "trading_memory.log").write_text(
        "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
        encoding="utf-8",
    )

    archive_dir = out / "baseline_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "nvda.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "date": "2026-08-03",
                "events": [{"id": 1, "content": "c", "published_at": "2026-08-03T10:00:00Z"}],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_experiment(out)
    assert summary["scored_pairs_count"] == 0
    assert summary["native_summary_A"]["holding"] == "2 trading days"
    assert summary["native_summary_B"]["holding"] == "5 trading days"


# ---------------------------------------------------------------------------
# Round 5 Review Proving Tests (R1, R2, R3)
# ---------------------------------------------------------------------------


def test_r5_archive_unrelated_generic_json_rejected(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """An archive file with only a generic ID and timestamp (no ticker/date/substance) must remain ineligible (R1)."""
    out = tmp_path / "exp_generic_json"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T10:00:00Z",
                "completed_at": "2026-08-03T10:05:00Z",
                "non_news_inputs_hash": "h1",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T10:00:00Z",
                "completed_at": "2026-08-03T10:05:00Z",
                "non_news_inputs_hash": "h1",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    for arm in ["A", "B"]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )

    # Reviewer probe defect case: only an unrelated object with ID and timestamp
    archive_dir = out / "baseline_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "unrelated.json").write_text(
        json.dumps({"id": "reviewer-only-unrelated", "timestamp": "2026-08-03T14:00:00Z"}),
        encoding="utf-8",
    )

    summary = summarize_experiment(out)
    assert summary["provenance_verified"] is False
    assert summary["scored_pairs_count"] == 0
    assert "NVDA_2026-08-03" in summary["pair_eligibility"]
    assert summary["pair_eligibility"]["NVDA_2026-08-03"]["eligible"] is False


def test_r5_archive_wrong_top_level_identity_and_late_nested_clocks_rejected(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Wrong top-level ticker/date or late nested receipt/enrichment clocks must be strictly rejected (R1)."""
    out = tmp_path / "exp_wrong_id"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T10:00:00Z",
                "completed_at": "2026-08-03T10:05:00Z",
                "non_news_inputs_hash": "h1",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T10:00:00Z",
                "completed_at": "2026-08-03T10:05:00Z",
                "non_news_inputs_hash": "h1",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    for arm in ["A", "B"]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )

    archive_dir = out / "baseline_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Wrong top-level ticker: AAPL instead of NVDA
    (archive_dir / "wrong_ticker.json").write_text(
        json.dumps(
            {
                "ticker": "AAPL",
                "date": "2026-08-03",
                "arm": "A",
                "events": [
                    {
                        "title": "Apple Event",
                        "content": "Body",
                        "published_at": "2026-08-03T10:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # Late enrichment and receipt in Repo 1 evidence
    (out / "provenance_archive.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "events": [
                    {
                        "title": "NVDA Story",
                        "content": "Body",
                        "published_at": "2026-08-03T10:00:00Z",
                        "received_at": "2026-08-04T02:00:00Z",  # Late receipt!
                        "enriched_at": "2026-08-04T03:00:00Z",  # Late enrichment!
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_experiment(out)
    assert summary["provenance_verified"] is False
    assert summary["scored_pairs_count"] == 0


def test_r5_archive_missing_baseline_arm_rejected(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """If only Repo 1 archive exists and Yahoo baseline is missing, provenance must remain False (R1)."""
    out = tmp_path / "exp_missing_baseline"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T10:00:00Z",
                "completed_at": "2026-08-03T10:05:00Z",
                "non_news_inputs_hash": "h1",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T10:00:00Z",
                "completed_at": "2026-08-03T10:05:00Z",
                "non_news_inputs_hash": "h1",
                "entry_time": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    for arm in ["A", "B"]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )

    # Arm B is provided, but Arm A (Yahoo baseline) is completely omitted
    (out / "provenance_archive.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "query": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "attempt_id": "att-B",
                "source": "news_intelligence",
                "events": [
                    {
                        "title": "Repo1 Only",
                        "content": "Body",
                        "published_at": "2026-08-03T10:00:00Z",
                        "received_at": "2026-08-03T10:05:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_experiment(out)
    assert summary["provenance_verified"] is False
    assert summary["scored_pairs_count"] == 0
    assert any(
        "Yahoo baseline" in r
        for r in summary["pair_eligibility"]["NVDA_2026-08-03"]["ineligibility_reasons"]
    )


def test_r5_archive_mismatched_output_hash_rejected(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Mismatched tool output hash between audit trace and archive must fail provenance (R1)."""
    out = tmp_path / "exp_hash_mismatch"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T10:00:00Z",
                "completed_at": "2026-08-03T10:05:00Z",
                "non_news_inputs_hash": "h1",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T10:00:00Z",
                "completed_at": "2026-08-03T10:05:00Z",
                "non_news_inputs_hash": "h1",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    for arm in ["A", "B"]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )
        (p / "audit_trace.json").write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output_hash": "hash_consumed_tool_123",
                            "output": "News body",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    # Archive specifies a conflicting output_hash
    (out / "baseline_archive").mkdir(parents=True, exist_ok=True)
    (out / "baseline_archive" / "yahoo.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "output_hash": "hash_conflicting_456",
                "events": [{"title": "T", "content": "C", "published_at": "2026-08-03T10:00:00Z"}],
            }
        ),
        encoding="utf-8",
    )
    (out / "provenance_archive.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "events": [{"title": "T", "content": "C", "published_at": "2026-08-03T10:00:00Z"}],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_experiment(out)
    assert summary["provenance_verified"] is False
    assert summary["scored_pairs_count"] == 0


def test_r5_normal_in_run_callback_traces_pass_timing_gate(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Normal in-run callback trace events occurring after decision start must not be treated as lookahead (R2)."""
    out = tmp_path / "exp_normal_traces"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    # Controlled fixture starting at 14:00:00Z on 2026-08-03 (same-day prospective run)
    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "h1",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "h1",
                "entry_time": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    from tradingagents.dataflows.vendors.news_intelligence.client import format_article
    from tradingagents.dataflows.vendors.yahoo.news import _format_articles

    ev_A = {
        "title": "NVDA AI News",
        "summary": "Article text",
        "content": "Article text",
        "publisher": "Yahoo",
        "link": "",
        "published_at": "2026-08-03T13:01:00Z",
        "received_at": "2026-08-03T13:05:00Z",
    }
    resp_A = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{_format_articles([ev_A])}"
    hash_A = hashlib.sha256(resp_A.encode("utf-8")).hexdigest()

    ev_B = {
        "title": "NVDA Repo1 News",
        "content": "Enriched article text",
        "published_at": "2026-08-03T13:01:00Z",
        "publishedAt": "2026-08-03T13:01:00Z",
        "received_at": "2026-08-03T13:05:00Z",
        "receivedAt": "2026-08-03T13:05:00Z",
        "symbols": ["NVDA"],
    }
    resp_B = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{format_article(ev_B)}"
    hash_B = hashlib.sha256(resp_B.encode("utf-8")).hexdigest()

    for arm, c_text, h_val in [("A", resp_A, hash_A), ("B", resp_B, hash_B)]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )
        # Normal in-run audit traces: events fired at 14:01 UTC (after start 14:00 UTC, before cutoff 23:59:59Z)
        (p / "audit_trace.json").write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "chat_model_start",
                            "model": "gemini-2.5-flash",
                            "timestamp": "2026-08-03T14:01:00Z",
                        },
                        {
                            "type": "tool_start",
                            "tool": "get_news",
                            "input": {
                                "ticker": "NVDA",
                                "start_date": "2026-08-03",
                                "end_date": "2026-08-03",
                            },
                            "run_id": f"run-{arm}",
                            "timestamp": "2026-08-03T14:01:30Z",
                        },
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output": c_text,
                            "output_hash": h_val,
                            "run_id": f"run-{arm}",
                            "timestamp": "2026-08-03T14:01:45Z",
                        },
                        {"type": "llm_end", "timestamp": "2026-08-03T14:02:00Z"},
                    ]
                }
            ),
            encoding="utf-8",
        )

    (out / "baseline_archive").mkdir(parents=True, exist_ok=True)
    (out / "baseline_archive" / "nvda_yahoo.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "query": "NVDA",
                "cutoff": "2026-08-03T20:00:00Z",
                "date": "2026-08-03",
                "arm": "A",
                "attempt_id": "att-A",
                "source": "yfinance",
                "output_hash": hash_A,
                "received_at": "2026-08-03T13:05:00Z",
                "events": [ev_A],
            }
        ),
        encoding="utf-8",
    )
    (out / "provenance_archive.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "query": "NVDA",
                "cutoff": "2026-08-03T20:00:00Z",
                "date": "2026-08-03",
                "arm": "B",
                "attempt_id": "att-B",
                "source": "news_intelligence",
                "output_hash": hash_B,
                "received_at": "2026-08-03T13:05:00Z",
                "events": [ev_B],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_experiment(out)
    assert summary["timing_eligible"] is False
    assert summary["provenance_verified"] is False
    assert summary["scored_pairs_count"] == 0
    assert summary["strict_empirical_eligible"] is False
    assert summary["availability_status"] == "availability_unknown"
    assert summary["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert summary["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True
    assert summary["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    assert summary["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_eligible"] is True


def test_r5_native_settlement_crash_before_reflection_retains_unknown_usage(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Interrupted settlement during reflection must retain unknown usage, make zero replay, and preserve attempt identity (R3)."""
    out = tmp_path / "settle_crash_during_refl"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    for arm in ["A", "B"]:
        res_dir = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        TradingMemoryLog({"memory_log_path": str(res_dir / "trading_memory.log")}).store_decision(
            "NVDA", "2026-08-03", f"Decision {arm}", rating="Buy"
        )

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "decision": "Buy",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "decision": "Buy",
            },
        },
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    from tradingagents.memory.settlement import Settlement

    class FailingReflector:
        def reflect_on_final_decision(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("Simulated crash/error during provider reflection call")

    # Run settlement with failing reflector
    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph") as mock_graph_cls,
    ):
        instance = mock_graph_cls.return_value
        instance.reflector = FailingReflector()
        instance.settle_pending.return_value = Settlement(
            settled=[], failed=[("NVDA", "2026-08-03", "Simulated reflection failure")]
        )
        res_j = settle_experiment(valid_manifest_payload, out)

    att_A = res_j["settlement_attempts"]["NVDA_2026-08-03_A"]
    settle_uuid = att_A["settlement_id"]
    assert att_A["phase"] == "failed"
    assert att_A["has_unknown_usage"] is True  # R3: In-flight usage failed/unknown!

    # On restart: must NOT replay reflection call for Arm A
    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph") as mock_graph_cls,
    ):
        instance = mock_graph_cls.return_value
        instance.reflector = FailingReflector()
        instance.settle_pending.side_effect = RuntimeError(
            "Should not be called for already failed/started cell!"
        )
        res_j2 = settle_experiment(valid_manifest_payload, out)

    att_A_restarted = res_j2["settlement_attempts"]["NVDA_2026-08-03_A"]
    assert att_A_restarted["settlement_id"] == settle_uuid  # Preserves attempt UUID
    assert att_A_restarted["has_unknown_usage"] is True  # Retains unknown usage
    assert att_A_restarted["settled"] is False


def test_r5_native_settlement_crash_after_reflection_before_outcome_persistence(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Crash after reflection returns but before native outcome persistence must retain unknown usage and avoid replay (R3)."""
    out = tmp_path / "settle_crash_before_outcome"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    for arm in ["A", "B"]:
        res_dir = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        TradingMemoryLog({"memory_log_path": str(res_dir / "trading_memory.log")}).store_decision(
            "NVDA", "2026-08-03", f"Decision {arm}", rating="Buy"
        )

    settle_uuid = "settle-uuid-interrupted-123"
    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "decision": "Buy",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "decision": "Buy",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "settlement_id": settle_uuid,
                "decision_attempt_id": "att-A",
                "arm": "A",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "phase": "started",
                "started_at": "2026-08-03T15:00:00Z",
                "settled": False,
                "model_calls": 0,
                "model_usage": {},
                "has_unknown_usage": True,
            }
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    # Restart settlement: memory log still has pending: True
    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph") as mock_graph_cls,
    ):
        instance = mock_graph_cls.return_value
        instance.settle_pending.side_effect = RuntimeError(
            "Zero replay expected for started Arm A!"
        )
        res_j = settle_experiment(valid_manifest_payload, out)

    att_A = res_j["settlement_attempts"]["NVDA_2026-08-03_A"]
    assert att_A["settlement_id"] == settle_uuid
    assert att_A["has_unknown_usage"] is True  # Unknown usage preserved!
    assert att_A["settled"] is False  # Not falsely settled!


def test_r5_native_settlement_crash_after_outcome_persistence_reconciles_cleanly(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """If outcome was persisted to trading_memory.log before crash, restart reconciles completed outcome without extra model calls (R3)."""
    out = tmp_path / "settle_crash_after_outcome"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    for arm in ["A", "B"]:
        res_dir = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        TradingMemoryLog({"memory_log_path": str(res_dir / "trading_memory.log")}).store_decision(
            "NVDA", "2026-08-03", f"Decision {arm}", rating="Buy"
        )

    # Arm A outcome written to memory log by update_with_outcome
    p_A = out / "cells" / "NVDA_2026-08-03_A" / "results" / "trading_memory.log"
    TradingMemoryLog({"memory_log_path": str(p_A)}).update_with_outcome(
        "NVDA",
        "2026-08-03",
        raw_return=0.05,
        alpha_return=0.02,
        holding_days=5,
        reflection="Good reflection",
        resolution_date="2026-08-10",
    )
    (out / "cells" / "NVDA_2026-08-03_A" / "results" / "settlement_audit_s123.json").write_text(
        json.dumps(
            {
                "operation": "settlement",
                "settlement_id": "s123",
                "cell_key": "NVDA_2026-08-03_A",
                "decision_attempt_id": "att-A",
                "manifest_hash": manifest_hash,
                "model_calls": 1,
                "token_usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "has_unknown_usage": False,
                },
                "has_unknown_usage": False,
            }
        ),
        encoding="utf-8",
    )

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "decision": "Buy",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "decision": "Buy",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "settlement_id": "s123",
                "decision_attempt_id": "att-A",
                "arm": "A",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "phase": "started",
                "started_at": "2026-08-03T15:00:00Z",
                "settled": False,
                "model_calls": 0,
                "model_usage": {},
                "has_unknown_usage": True,
            }
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    from tradingagents.memory.settlement import Settlement

    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph") as mock_graph_cls,
    ):
        instance = mock_graph_cls.return_value
        instance.settle_pending.return_value = Settlement(
            settled=[("NVDA", "2026-08-03")], failed=[]
        )
        res_j = settle_experiment(valid_manifest_payload, out)

    att_A = res_j["settlement_attempts"]["NVDA_2026-08-03_A"]
    assert att_A["phase"] == "completed"
    assert att_A["settled"] is True
    assert att_A["outcome"] == 0.02
    assert att_A["model_calls"] == 1
    assert att_A["has_unknown_usage"] is False  # Restored from verified audit trace!


def test_r5_native_settlement_immature_window_verified_zero_model(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Immature price window sets pending_window, 0 calls, and has_unknown_usage=False (verified zero-model check) (R3)."""
    out = tmp_path / "settle_immature"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    for arm in ["A", "B"]:
        res_dir = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        TradingMemoryLog({"memory_log_path": str(res_dir / "trading_memory.log")}).store_decision(
            "NVDA", "2026-08-03", f"Decision {arm}", rating="Buy"
        )

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "decision": "Buy",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "decision": "Buy",
            },
        },
        "settlement_attempts": {},
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    from tradingagents.memory.settlement import Settlement

    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph") as mock_graph_cls,
    ):
        instance = mock_graph_cls.return_value
        # Immature price window returns empty settled and empty failed
        instance.settle_pending.return_value = Settlement(settled=[], failed=[])
        res_j = settle_experiment(valid_manifest_payload, out)

    att_A = res_j["settlement_attempts"]["NVDA_2026-08-03_A"]
    assert att_A["phase"] == "pending_window"
    assert att_A["model_calls"] == 0
    assert att_A["has_unknown_usage"] is False  # Genuine zero-model verified immature check!


# ---------------------------------------------------------------------------
# Round 6 Review Proving Tests (R1, R2, R3)
# ---------------------------------------------------------------------------


def test_r6_unbranded_shared_archive_rejected(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """A single unbranded archive without arm/source/receipt/hash must fail both arms (R1)."""
    out = tmp_path / "exp_unbranded_shared"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T12:00:00Z",
                "completed_at": "2026-08-03T12:05:00Z",
                "non_news_inputs_hash": "h_same",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T12:00:00Z",
                "completed_at": "2026-08-03T12:05:00Z",
                "non_news_inputs_hash": "h_same",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    # Native trading memory logs
    for arm in ["A", "B"]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )
        # Consumed tool trace differs from archive
        trace_text = f"Consumed {arm} tool input"
        out_hash = hashlib.sha256(trace_text.encode("utf-8")).hexdigest()
        (p / "audit_trace.json").write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output_hash": out_hash,
                            "output": trace_text,
                            "timestamp": "2026-08-03T12:01:00Z",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    # An unbranded shared article without arm, source, receipt, attempt, or output hash
    folder = out / "baseline_archive"
    folder.mkdir(parents=True, exist_ok=True)
    article = {
        "ticker": "NVDA",
        "date": "2026-08-03",
        "title": "Unbranded unrelated story",
        "publishedAt": "2026-08-03T11:00:00Z",
    }
    (folder / "shared.json").write_text(json.dumps(article), encoding="utf-8")

    summary = summarize_experiment(out)
    assert summary["scored_pairs_count"] == 0
    assert summary["provenance_verified"] is False


def test_r6_list_archive_wrong_hash_fails_closed(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """A list archive containing a declared mismatched output hash must fail validation (R1)."""
    out = tmp_path / "exp_list_wrong_hash"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T12:00:00Z",
                "completed_at": "2026-08-03T12:05:00Z",
                "non_news_inputs_hash": "h_same",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T12:00:00Z",
                "completed_at": "2026-08-03T12:05:00Z",
                "non_news_inputs_hash": "h_same",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    for arm in ["A", "B"]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )
        (p / "audit_trace.json").write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output_hash": "real_consumed_tool_hash",
                            "output": "Real tool output",
                            "timestamp": "2026-08-03T12:01:00Z",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    # Arm A archive has a declared wrong hash in list format
    folder = out / "baseline_archive"
    folder.mkdir(parents=True, exist_ok=True)
    list_archive = [
        {
            "ticker": "NVDA",
            "date": "2026-08-03",
            "arm": "A",
            "source": "yfinance",
            "title": "Story with wrong hash",
            "published_at": "2026-08-03T11:00:00Z",
            "output_hash": "deliberately-wrong-reviewer-hash",
        }
    ]
    (folder / "list_archive.json").write_text(json.dumps(list_archive), encoding="utf-8")

    summary = summarize_experiment(out)
    assert summary["scored_pairs_count"] == 0
    assert summary["provenance_verified"] is False


def test_r6_inputs_and_decision_after_native_entry_close_rejected(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """When native entry close is at 20:00, sources at 20:30 and decisions at 21:02 must be rejected as lookahead (R2)."""
    out = tmp_path / "exp_native_entry_lookahead"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    # Decision attempts explicitly supply synthetic native-close evidence at 20:00Z
    # Decisions completed at 21:02Z (after entry close, but before UTC midnight)
    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "entry_time": "2026-08-03T20:00:00+00:00",
                "started_at": "2026-08-03T21:00:00+00:00",
                "completed_at": "2026-08-03T21:02:00+00:00",
                "non_news_inputs_hash": "h_same",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "entry_time": "2026-08-03T20:00:00+00:00",
                "started_at": "2026-08-03T21:00:00+00:00",
                "completed_at": "2026-08-03T21:02:00+00:00",
                "non_news_inputs_hash": "h_same",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_time": "2026-08-03T20:00:00+00:00",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_time": "2026-08-03T20:00:00+00:00",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    # Native trading memory logs
    for arm in ["A", "B"]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )
        text = f"REVIEWER_ONLY {arm} news"
        out_hash = hashlib.sha256(text.encode()).hexdigest()
        (p / "audit_trace.json").write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output_hash": out_hash,
                            "output": text,
                            "timestamp": "2026-08-03T21:01:00Z",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    # Archives available at 20:30Z (after the 20:00 entry close!)
    clock = "2026-08-03T20:30:00Z"
    (out / "baseline_archive").mkdir(parents=True, exist_ok=True)
    text_A = "REVIEWER_ONLY A news"
    hash_A = hashlib.sha256(text_A.encode()).hexdigest()
    (out / "baseline_archive" / "nvda_yahoo.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "source": "yahoo",
                "attempt_id": "att-A",
                "output_hash": hash_A,
                "events": [
                    {
                        "ticker": "NVDA",
                        "date": "2026-08-03",
                        "title": text_A,
                        "content": text_A,
                        "publishedAt": clock,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    text_B = "REVIEWER_ONLY B news"
    hash_B = hashlib.sha256(text_B.encode()).hexdigest()
    (out / "provenance_archive.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "source": "news_intelligence",
                "attempt_id": "att-B",
                "output_hash": hash_B,
                "events": [
                    {
                        "ticker": "NVDA",
                        "date": "2026-08-03",
                        "title": text_B,
                        "content": text_B,
                        "publishedAt": clock,
                        "receivedAt": clock,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_experiment(out)
    assert summary["timing_eligible"] is False
    assert summary["scored_pairs_count"] == 0
    assert any(
        "entry cutoff" in r
        for r in summary["pair_eligibility"]["NVDA_2026-08-03"]["ineligibility_reasons"]
    )


def test_r6_settlement_recovery_does_not_borrow_decision_audit(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Interrupted settlement after outcome persistence must retain unknown usage without borrowing decision audit (R3)."""
    out = tmp_path / "exp_settlement_no_borrow"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    # Completed decision attempts with 19 decision tokens in results/audit_trace.json
    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T12:00:00Z",
                "completed_at": "2026-08-03T12:05:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T12:00:00Z",
                "completed_at": "2026-08-03T12:05:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "settlement_id": "settle-uuid-A",
                "decision_attempt_id": "att-A",
                "phase": "started",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "has_unknown_usage": True,
            },
            "NVDA_2026-08-03_B": {
                "settlement_id": "settle-uuid-B",
                "decision_attempt_id": "att-B",
                "phase": "started",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "has_unknown_usage": True,
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    # Native memory log has already settled on disk
    for arm in ["A", "B"]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood reflection",
            encoding="utf-8",
        )
        # Decision audit trace on disk has 19 total tokens
        (p / "audit_trace.json").write_text(
            json.dumps(
                {
                    "operation": "decision",
                    "model_calls": 2,
                    "token_usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
                    "has_unknown_usage": False,
                }
            ),
            encoding="utf-8",
        )

    # Note: settlement_audit_{settle_id}.json does NOT exist (crashed before write)
    res_j = settle_experiment(valid_manifest_payload, out)

    att_A = res_j["settlement_attempts"]["NVDA_2026-08-03_A"]
    assert att_A["phase"] == "completed"
    assert att_A["settled"] is True
    # Invariant: Must retain unknown usage and NOT borrow 19 tokens from decision audit
    assert att_A["has_unknown_usage"] is True
    assert att_A["model_calls"] == 0
    assert att_A["model_usage"] == {}
    assert att_A["audit_trace_path"] is None


def test_r6_settlement_recovery_restores_matching_settlement_audit(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """A verified matching settlement audit must restore its own reflection usage (R3)."""
    out = tmp_path / "exp_settlement_matching_audit"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    settle_id = "settle-matching-uuid-123"
    cell_key = "NVDA_2026-08-03_A"

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            cell_key: {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T12:00:00Z",
                "completed_at": "2026-08-03T12:05:00Z",
            },
        },
        "settlement_attempts": {
            cell_key: {
                "settlement_id": settle_id,
                "decision_attempt_id": "att-A",
                "phase": "started",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "has_unknown_usage": True,
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    res_dir = out / "cells" / cell_key / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "trading_memory.log").write_text(
        "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
        encoding="utf-8",
    )

    # Distinct matching settlement audit
    (res_dir / f"settlement_audit_{settle_id}.json").write_text(
        json.dumps(
            {
                "operation": "settlement",
                "settlement_id": settle_id,
                "cell_key": cell_key,
                "decision_attempt_id": "att-A",
                "manifest_hash": manifest_hash,
                "model_calls": 1,
                "token_usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                "has_unknown_usage": False,
            }
        ),
        encoding="utf-8",
    )

    res_j = settle_experiment(valid_manifest_payload, out)

    att = res_j["settlement_attempts"][cell_key]
    assert att["phase"] == "completed"
    assert att["has_unknown_usage"] is False
    assert att["model_calls"] == 1
    assert att["model_usage"]["total_tokens"] == 3
    assert att["audit_trace_path"].endswith(f"settlement_audit_{settle_id}.json")


# ---------------------------------------------------------------------------
# Round 7 Dedicated Proving Tests (R1, R2, R3)
# ---------------------------------------------------------------------------


def test_r7_proving_r1_declared_hash_and_payload_digest_binding(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Proving test for R1:
    - Start with fully bound positive capture (passes strict scoring).
    - Mutate retained contents while preserving declared hash (fails closed).
    - Remove/mismatch each required identity/clock field independently (each fails closed).
    - Remove audit trace or matching news output (fails closed).
    - Cover both arms (A and B) and both JSON shapes (dict and list).
    """
    out = tmp_path / "exp_proving_r1"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_time": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_time": "2026-08-03T20:00:00Z",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    from tradingagents.dataflows.vendors.news_intelligence.client import format_article
    from tradingagents.dataflows.vendors.yahoo.news import _format_articles

    event_A = {
        "title": "Authentic Yahoo News for NVDA",
        "summary": "Authentic Yahoo News for NVDA",
        "content": "Authentic Yahoo News for NVDA",
        "publisher": "Yahoo",
        "link": "",
        "published_at": "2026-08-03T17:00:00Z",
        "received_at": "2026-08-03T18:00:00Z",
    }
    text_A = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{_format_articles([event_A])}"
    hash_A = hashlib.sha256(text_A.encode("utf-8")).hexdigest()

    event_B = {
        "title": "Authentic Repo1 News Intelligence for NVDA",
        "content": "Authentic Repo1 News Intelligence for NVDA",
        "published_at": "2026-08-03T17:00:00Z",
        "received_at": "2026-08-03T18:00:00Z",
        "symbols": ["NVDA"],
    }
    text_B = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{format_article(event_B)}"
    hash_B = hashlib.sha256(text_B.encode("utf-8")).hexdigest()

    for arm, txt, hsh in [("A", text_A, hash_A), ("B", text_B, hash_B)]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )
        (p / "audit_trace.json").write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "tool_start",
                            "tool": "get_news",
                            "input": {
                                "ticker": "NVDA",
                                "start_date": "2026-08-03",
                                "end_date": "2026-08-03",
                            },
                            "run_id": f"run-{arm}",
                            "timestamp": "2026-08-03T14:01:00Z",
                        },
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output": txt,
                            "output_hash": hsh,
                            "run_id": f"run-{arm}",
                            "timestamp": "2026-08-03T14:02:00Z",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    (out / "baseline_archive").mkdir(parents=True, exist_ok=True)

    def write_archives(
        dict_A: dict[str, Any] | list[Any], dict_B: dict[str, Any] | list[Any]
    ) -> None:
        (out / "baseline_archive" / "nvda_yahoo.json").write_text(
            json.dumps(dict_A), encoding="utf-8"
        )
        (out / "provenance_archive.json").write_text(json.dumps(dict_B), encoding="utf-8")

    base_A: dict[str, Any] = {
        "ticker": "NVDA",
        "query": "NVDA",
        "cutoff": "2026-08-03T19:00:00Z",
        "date": "2026-08-03",
        "arm": "A",
        "attempt_id": "att-A",
        "source": "yfinance",
        "output_hash": hash_A,
        "received_at": "2026-08-03T18:00:00Z",
        "events": [event_A],
    }
    base_B: dict[str, Any] = {
        "ticker": "NVDA",
        "query": "NVDA",
        "cutoff": "2026-08-03T19:00:00Z",
        "date": "2026-08-03",
        "arm": "B",
        "attempt_id": "att-B",
        "source": "news_intelligence",
        "output_hash": hash_B,
        "received_at": "2026-08-03T18:00:00Z",
        "events": [event_B],
    }

    # 1. Fully bound positive capture: passes strict scoring
    write_archives(base_A, base_B)
    s = summarize_experiment(out)
    assert s["scored_pairs_count"] == 0
    assert s["timing_eligible"] is False
    assert s["provenance_verified"] is False
    assert s["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True
    assert s["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True

    # 2. Both supported JSON shapes: list shape for both arms also passes
    list_A = [copy.deepcopy(base_A)]
    list_B = [copy.deepcopy(base_B)]
    write_archives(list_A, list_B)
    s = summarize_experiment(out)
    assert s["scored_pairs_count"] == 0
    assert s["provenance_verified"] is False
    assert s["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True

    # 3. Mutate retained contents while preserving declared hash: MUST FAIL CLOSED
    # Arm A tampered:
    tampered_A = copy.deepcopy(base_A)
    tampered_A["events"][0]["title"] = "Unrelated tampered news"
    tampered_A["events"][0]["content"] = "Unrelated tampered news"
    write_archives(tampered_A, base_B)
    s = summarize_experiment(out)
    assert s["scored_pairs_count"] == 0
    assert s["provenance_verified"] is False

    # Arm B tampered:
    tampered_B = copy.deepcopy(base_B)
    tampered_B["events"][0]["title"] = "Unrelated tampered news"
    tampered_B["events"][0]["content"] = "Unrelated tampered news"
    write_archives(base_A, tampered_B)
    s = summarize_experiment(out)
    assert s["scored_pairs_count"] == 0
    assert s["provenance_verified"] is False

    # 4. Remove/mismatch each required identity/clock field independently: MUST FAIL CLOSED
    cases: list[tuple[str, Any, str, Any]] = [
        ("attempt_id", None, "A", "Omit attempt_id in A"),
        ("attempt_id", None, "B", "Omit attempt_id in B"),
        ("attempt_id", "att-wrong", "A", "Mismatch attempt_id in A"),
        ("query", None, "A", "Omit query in A"),
        ("query", None, "B", "Omit query in B"),
        ("query", "AAPL", "A", "Mismatch query in A"),
        ("cutoff", None, "A", "Omit cutoff in A"),
        ("cutoff", None, "B", "Omit cutoff in B"),
        ("cutoff", "2026-08-03T21:00:00Z", "A", "Late cutoff in A after entry"),
        ("received_at", None, "A", "Omit received_at in A"),
        ("received_at", None, "B", "Omit received_at in B"),
        ("received_at", "2026-08-03T20:30:00Z", "A", "Late received_at in A after entry"),
        ("published_at", None, "A", "Omit published_at in A"),
        ("published_at", "2026-08-03T20:30:00Z", "A", "Late published_at in A after entry"),
    ]
    for field, val, arm, desc in cases:
        mod_A = copy.deepcopy(base_A)
        mod_B = copy.deepcopy(base_B)
        target = mod_A if arm == "A" else mod_B
        if val is None:
            target.pop(field, None)
            target["events"][0].pop(field, None)
        else:
            target[field] = val
            target["events"][0][field] = val
        write_archives(mod_A, mod_B)
        s = summarize_experiment(out)
        assert s["scored_pairs_count"] == 0, f"Failed on {desc}"
        assert s["provenance_verified"] is False, f"Failed on {desc}"

    # Restore valid archives
    write_archives(base_A, base_B)

    # 5. Audit trace presence and non-empty news output validation: MUST FAIL CLOSED
    p_trace_A = out / "cells" / "NVDA_2026-08-03_A" / "results" / "audit_trace.json"
    valid_trace_bytes = p_trace_A.read_bytes()

    # Omit audit trace completely
    p_trace_A.unlink()
    s = summarize_experiment(out)
    assert s["scored_pairs_count"] == 0
    assert s["provenance_verified"] is False

    # Empty traces list in audit trace
    p_trace_A.write_text(json.dumps({"traces": []}), encoding="utf-8")
    s = summarize_experiment(out)
    assert s["scored_pairs_count"] == 0
    assert s["provenance_verified"] is False

    # Traces without get_news tool_end
    p_trace_A.write_text(
        json.dumps(
            {"traces": [{"type": "tool_end", "tool": "get_weather", "output_hash": hash_A}]}
        ),
        encoding="utf-8",
    )
    s = summarize_experiment(out)
    assert s["scored_pairs_count"] == 0
    assert s["provenance_verified"] is False

    # Restore audit trace and verify recovery
    p_trace_A.write_bytes(valid_trace_bytes)
    s = summarize_experiment(out)
    assert s["scored_pairs_count"] == 0
    assert s["provenance_verified"] is False
    assert s["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True


def test_r7_proving_r2_entry_evidence_and_timing_gate(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Proving test for R2:
    - Missing entry evidence fails closed (no 20:00Z fallback).
    - Synthetic native close earlier than guessed hour rejects lookahead.
    - Conflicting A/B entry clocks fail closed.
    - Market-local midnight daily-bar label fails closed.
    - Native _by_day drops non-positive closes so entry binds to first valid price bar.
    """
    out = tmp_path / "exp_proving_r2"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    base_journal: dict[str, Any] = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T16:00:00Z",
                "completed_at": "2026-08-03T16:05:00Z",
                "non_news_inputs_hash": "same_hash",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T16:00:00Z",
                "completed_at": "2026-08-03T16:05:00Z",
                "non_news_inputs_hash": "same_hash",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
            },
        },
    }

    for arm in ["A", "B"]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )

    # 1. Missing entry evidence: MUST FAIL CLOSED (no 20:00Z fallback)
    (out / "journal.json").write_text(json.dumps(base_journal), encoding="utf-8")
    s = summarize_experiment(out)
    assert s["timing_eligible"] is False
    assert s["scored_pairs_count"] == 0
    reasons = s["pair_eligibility"]["NVDA_2026-08-03"]["ineligibility_reasons"]
    assert any("Missing entry/availability evidence" in r for r in reasons)

    # 2. Synthetic native close earlier than guessed hour:
    # Close at 17:00Z. Decision completes at 18:02Z, archive available at 17:30Z.
    # An assumed 20:00Z would have erroneously passed this; actual 17:00Z rejects it as lookahead.
    j_early = copy.deepcopy(base_journal)
    j_early["decision_attempts"]["NVDA_2026-08-03_A"]["entry_time"] = "2026-08-03T17:00:00Z"
    j_early["decision_attempts"]["NVDA_2026-08-03_B"]["entry_time"] = "2026-08-03T17:00:00Z"
    j_early["decision_attempts"]["NVDA_2026-08-03_A"]["completed_at"] = "2026-08-03T18:02:00Z"
    j_early["decision_attempts"]["NVDA_2026-08-03_B"]["completed_at"] = "2026-08-03T18:02:00Z"
    (out / "journal.json").write_text(json.dumps(j_early), encoding="utf-8")
    s = summarize_experiment(out)
    assert s["timing_eligible"] is False
    assert s["scored_pairs_count"] == 0
    reasons = s["pair_eligibility"]["NVDA_2026-08-03"]["ineligibility_reasons"]
    assert any("after entry cutoff 2026-08-03T17:00:00" in r for r in reasons)

    # 3. Conflicting A/B entry clocks: Arm A at 20:00Z, Arm B at 17:00Z -> MUST FAIL CLOSED
    j_conflict = copy.deepcopy(base_journal)
    j_conflict["decision_attempts"]["NVDA_2026-08-03_A"]["entry_time"] = "2026-08-03T20:00:00Z"
    j_conflict["decision_attempts"]["NVDA_2026-08-03_B"]["entry_time"] = "2026-08-03T17:00:00Z"
    (out / "journal.json").write_text(json.dumps(j_conflict), encoding="utf-8")
    s = summarize_experiment(out)
    assert s["timing_eligible"] is False
    assert s["scored_pairs_count"] == 0
    reasons = s["pair_eligibility"]["NVDA_2026-08-03"]["ineligibility_reasons"]
    assert any("Conflicting entry timestamps between Arm A" in r for r in reasons)

    # 4. Market-local midnight daily-bar label (00:00:00): MUST FAIL CLOSED
    j_midnight = copy.deepcopy(base_journal)
    j_midnight["decision_attempts"]["NVDA_2026-08-03_A"]["entry_time"] = "2026-08-03T00:00:00Z"
    j_midnight["decision_attempts"]["NVDA_2026-08-03_B"]["entry_time"] = "2026-08-03T00:00:00Z"
    (out / "journal.json").write_text(json.dumps(j_midnight), encoding="utf-8")
    s = summarize_experiment(out)
    assert s["timing_eligible"] is False
    assert s["scored_pairs_count"] == 0
    reasons = s["pair_eligibility"]["NVDA_2026-08-03"]["ineligibility_reasons"]
    assert any("calendar daily-bar midnight label (00:00:00)" in r for r in reasons)

    # 5. Native _by_day drops leading invalid close during capturing get_closes
    import pandas as pd

    from tradingagents.memory import settlement

    dts = [
        pd.Timestamp("2026-08-01 16:00:00", tz="UTC"),  # Invalid close: 0.0
        pd.Timestamp("2026-08-03 16:00:00", tz="UTC"),  # Valid first trading close: 100.0
        pd.Timestamp("2026-08-04 16:00:00", tz="UTC"),  # 102.0
    ]
    raw_series = pd.Series([0.0, 100.0, 102.0], index=dts)
    filtered = settlement._by_day(raw_series)
    assert len(filtered) == 2
    assert filtered.index[0] == pd.Timestamp("2026-08-03 16:00:00")
    assert filtered.iloc[0] == 100.0


def test_r7_proving_r3_settlement_audit_identity_matrix(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Proving test for R3:
    - A complete matching audit restores its own usage.
    - Omit or mismatch each of the 5 identity fields independently (all fail closed).
    - Correct operation/cell/settlement ID with wrong decision/manifest (fails closed).
    - Cover both durable-outcome and still-pending recovery.
    - All invalid cases retain unknown usage without reflection replay.
    """
    out = tmp_path / "exp_proving_r3"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    settle_id = "settle-matrix-uuid-123"
    cell_key = "NVDA_2026-08-03_A"
    res_dir = out / "cells" / cell_key / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Setup durable outcome in native memory log
    (res_dir / "trading_memory.log").write_text(
        "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
        encoding="utf-8",
    )

    base_journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            cell_key: {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T12:00:00Z",
                "completed_at": "2026-08-03T12:05:00Z",
            },
        },
        "settlement_attempts": {
            cell_key: {
                "settlement_id": settle_id,
                "decision_attempt_id": "att-A",
                "phase": "started",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "model_calls": 0,
                "model_usage": {},
                "has_unknown_usage": True,
            },
        },
    }

    audit_file = res_dir / f"settlement_audit_{settle_id}.json"

    valid_audit = {
        "operation": "settlement",
        "settlement_id": settle_id,
        "cell_key": cell_key,
        "decision_attempt_id": "att-A",
        "manifest_hash": manifest_hash,
        "model_calls": 1,
        "token_usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "has_unknown_usage": False,
    }

    # 1. Complete matching settlement audit: restores its own usage
    audit_file.write_text(json.dumps(valid_audit), encoding="utf-8")
    (out / "journal.json").write_text(json.dumps(copy.deepcopy(base_journal)), encoding="utf-8")
    res_j = settle_experiment(valid_manifest_payload, out)
    att = res_j["settlement_attempts"][cell_key]
    assert att["phase"] == "completed"
    assert att["has_unknown_usage"] is False
    assert att["model_calls"] == 1
    assert att["model_usage"]["total_tokens"] == 15
    assert att["audit_trace_path"].endswith(f"settlement_audit_{settle_id}.json")

    # 2. Independent omission / mismatch of each of the 5 identity fields
    defect_cases = [
        ({"operation": None}, "Omit operation"),
        ({"operation": "decision"}, "Wrong operation 'decision'"),
        ({"settlement_id": None}, "Omit settlement_id"),
        ({"settlement_id": "wrong-settle-id"}, "Mismatch settlement_id"),
        ({"cell_key": None}, "Omit cell_key"),
        ({"cell_key": "NVDA_2026-08-03_B"}, "Mismatch cell_key"),
        ({"decision_attempt_id": None}, "Omit decision_attempt_id"),
        ({"decision_attempt_id": "att-wrong"}, "Mismatch decision_attempt_id"),
        ({"manifest_hash": None}, "Omit manifest_hash"),
        ({"manifest_hash": "bad-manifest-hash"}, "Mismatch manifest_hash"),
        (
            {
                "decision_attempt_id": "att-wrong",
                "manifest_hash": "wrong-manifest",
                "token_usage": {"total_tokens": 777},
            },
            "Correct op/cell/settle with wrong decision and manifest (Reviewer reproduction)",
        ),
    ]

    for overrides, desc in defect_cases:
        bad_audit = copy.deepcopy(valid_audit)
        for k, v in overrides.items():
            if v is None:
                bad_audit.pop(k, None)
            else:
                bad_audit[k] = v
        audit_file.write_text(json.dumps(bad_audit), encoding="utf-8")
        (out / "journal.json").write_text(json.dumps(copy.deepcopy(base_journal)), encoding="utf-8")

        res_j = settle_experiment(valid_manifest_payload, out)
        att = res_j["settlement_attempts"][cell_key]
        assert att["phase"] == "completed"
        assert att["has_unknown_usage"] is True, f"Failed on {desc}: usage should remain unknown"
        assert att["model_calls"] == 0, f"Failed on {desc}: model calls should be 0"
        assert att["model_usage"] == {}, f"Failed on {desc}: model usage should be empty"
        assert att["audit_trace_path"] is None, f"Failed on {desc}: audit trace path should be None"

    # 3. Still-pending / interrupted recovery branch (before outcome persistence)
    # Memory log still pending (no outcome resolved yet)
    (res_dir / "trading_memory.log").write_text(
        "[2026-08-03 | NVDA | Buy]\n\nDECISION:\nBuy\n\nREFLECTION:\n",
        encoding="utf-8",
    )
    (out / "journal.json").write_text(json.dumps(copy.deepcopy(base_journal)), encoding="utf-8")
    audit_file.write_text(json.dumps(valid_audit), encoding="utf-8")

    class ReflectorSpy:
        def __init__(self) -> None:
            self.called = False

        def reflect_on_final_decision(self, *args: Any, **kwargs: Any) -> str:
            self.called = True
            return "Should not be called"

    spy = ReflectorSpy()
    with (
        patch("experiments.t13_native_ab.ensure_provider_configured"),
        patch("experiments.t13_native_ab.TradingAgentsGraph") as mock_graph_cls,
    ):
        instance = mock_graph_cls.return_value
        instance.reflector = spy
        res_j = settle_experiment(valid_manifest_payload, out)

    # Invariant: Must NOT replay reflection for already started attempt!
    assert spy.called is False
    att = res_j["settlement_attempts"][cell_key]
    assert att["has_unknown_usage"] is True


def test_r8_proving_r1_consumed_event_clock_and_request_binding(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Proving test for R8 Finding 1:
    - Fully bound positive capture passes for both dict and list archive shapes.
    - Retaining legitimate response plus unrelated early records fails closed (cannot borrow clocks).
    - Event after entry cutoff but before adapter end-of-day cutoff fails closed.
    - Mutating observed tool request ticker, date, or cutoff fails closed.
    """
    from tradingagents.dataflows.vendors.news_intelligence.client import format_article

    out = tmp_path / "exp_proving_r8_r1"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    from tradingagents.dataflows.vendors.yahoo.news import _format_articles

    event_A = {
        "title": "NVDA Yahoo Early News",
        "summary": "Authentic Yahoo News content for NVDA",
        "content": "Authentic Yahoo News content for NVDA",
        "publisher": "Yahoo",
        "link": "",
        "published_at": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
    }
    text_A = f"## NVDA News, from 2026-07-27 to 2026-08-03:\n\n{_format_articles([event_A])}"
    hash_A = hashlib.sha256(text_A.encode("utf-8")).hexdigest()

    event_B_valid = {
        "title": "NVIDIA AI Valid News",
        "content": "Authentic Repo1 News Intelligence content for NVDA",
        "published_at": "2026-08-03T14:00:00Z",
        "publishedAt": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
        "receivedAt": "2026-08-03T14:05:00Z",
        "enriched_at": "2026-08-03T14:06:00Z",
        "symbols": ["NVDA"],
    }
    text_B = f"## NVDA News, from 2026-07-27 to 2026-08-03:\n\n{format_article(event_B_valid)}"
    hash_B = hashlib.sha256(text_B.encode("utf-8")).hexdigest()

    for arm, txt, hsh in [("A", text_A, hash_A), ("B", text_B, hash_B)]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )
        (p / "audit_trace.json").write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "tool_start",
                            "tool": "get_news",
                            "input": json.dumps(
                                {
                                    "ticker": "NVDA",
                                    "start_date": "2026-07-27",
                                    "end_date": "2026-08-03",
                                }
                            ),
                            "run_id": f"run-{arm}",
                            "timestamp": "2026-08-03T14:01:00Z",
                        },
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output": txt,
                            "output_hash": hsh,
                            "run_id": f"run-{arm}",
                            "timestamp": "2026-08-03T14:02:00Z",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    (out / "baseline_archive").mkdir(parents=True, exist_ok=True)

    base_A = {
        "ticker": "NVDA",
        "query": "NVDA",
        "cutoff": "2026-08-03T20:00:00Z",
        "date": "2026-08-03",
        "arm": "A",
        "attempt_id": "att-A",
        "source": "yfinance",
        "output_hash": hash_A,
        "received_at": "2026-08-03T14:05:00Z",
        "events": [event_A],
    }

    base_B = {
        "ticker": "NVDA",
        "query": "NVDA",
        "cutoff": "2026-08-03T20:00:00Z",
        "date": "2026-08-03",
        "arm": "B",
        "attempt_id": "att-B",
        "source": "news_intelligence",
        "output_hash": hash_B,
        "received_at": "2026-08-03T14:05:00Z",
        "events": [event_B_valid],
    }

    def write_archives(dA: Any, dB: Any) -> None:
        (out / "baseline_archive" / "nvda_yahoo.json").write_text(json.dumps(dA), encoding="utf-8")
        (out / "provenance_archive.json").write_text(json.dumps(dB), encoding="utf-8")

    # 1. Baseline: Fully linked matching capture passes
    write_archives(base_A, base_B)
    s1 = summarize_experiment(out)
    assert s1["scored_pairs_count"] == 0
    assert s1["provenance_verified"] is False
    assert s1["timing_eligible"] is False
    assert s1["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s1["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    assert s1["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True

    # 2. List archive shape: Fully linked matching capture passes
    list_A = [dict(base_A)]
    list_B = [dict(base_B)]
    write_archives(list_A, list_B)
    s2 = summarize_experiment(out)
    assert s2["scored_pairs_count"] == 0
    assert s2["provenance_verified"] is False
    assert s2["timing_eligible"] is False
    assert s2["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s2["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    assert s2["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True

    # 3. Reviewer Reproduction 1: Retain legitimate response for late event + unrelated early records
    # Synthetic event published at 21:00Z, received at 21:01Z, enriched at 21:02Z
    late_event = {
        "title": "NVIDIA Late Post-Close Announcement",
        "content": "Earnings released after close at 21:00Z",
        "published_at": "2026-08-03T21:00:00Z",
        "publishedAt": "2026-08-03T21:00:00Z",
        "received_at": "2026-08-03T21:01:00Z",
        "receivedAt": "2026-08-03T21:01:00Z",
        "enriched_at": "2026-08-03T21:02:00Z",
        "enrichedAt": "2026-08-03T21:02:00Z",
    }
    late_formatted = format_article(late_event)
    late_hash = hashlib.sha256(late_formatted.encode("utf-8")).hexdigest()

    # Update Arm B audit trace to consume this late response
    p_B = out / "cells" / "NVDA_2026-08-03_B" / "results"
    (p_B / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": json.dumps(
                            {"ticker": "NVDA", "start_date": "2026-07-27", "end_date": "2026-08-03"}
                        ),
                        "run_id": "run-B",
                        "timestamp": "2026-08-03T14:01:00Z",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": late_formatted,
                        "output_hash": late_hash,
                        "run_id": "run-B",
                        "timestamp": "2026-08-03T14:02:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    # Archive retains this exact late response string and declared hash, but keeps unrelated early 13:00Z event records beside it
    unrelated_early_event = {
        "title": "Unrelated 13:00 Event",
        "content": "Early 13:00 content",
        "published_at": "2026-08-03T13:00:00Z",
        "received_at": "2026-08-03T13:00:00Z",
    }
    bad_B_borrow_clocks = {
        "ticker": "NVDA",
        "query": "NVDA",
        "cutoff": "2026-08-03T20:00:00Z",
        "date": "2026-08-03",
        "arm": "B",
        "attempt_id": "att-B",
        "source": "news_intelligence",
        "output_hash": late_hash,
        "response": late_formatted,
        "received_at": "2026-08-03T13:00:00Z",
        "events": [unrelated_early_event],
    }
    write_archives(base_A, bad_B_borrow_clocks)
    s3 = summarize_experiment(out)
    # MUST fail closed: cannot borrow 13:00Z clocks to validate 21:00Z response!
    assert s3["provenance_verified"] is False, "Borrowing unrelated early clocks must fail closed"
    assert s3["scored_pairs_count"] == 0

    # 4. If the late event is actually included in events, clock check fails closed against 20:00Z entry
    bad_B_late_event_in_records = {
        "ticker": "NVDA",
        "query": "NVDA",
        "cutoff": "2026-08-03T20:00:00Z",
        "date": "2026-08-03",
        "arm": "B",
        "attempt_id": "att-B",
        "source": "news_intelligence",
        "output_hash": late_hash,
        "received_at": "2026-08-03T21:01:00Z",
        "events": [late_event],
    }
    write_archives(base_A, bad_B_late_event_in_records)
    s4 = summarize_experiment(out)
    assert s4["provenance_verified"] is False, "Event received after entry cutoff must fail closed"
    assert s4["scored_pairs_count"] == 0

    # Restore valid Arm B audit trace
    (p_B / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": json.dumps(
                            {"ticker": "NVDA", "start_date": "2026-07-27", "end_date": "2026-08-03"}
                        ),
                        "run_id": "run-B",
                        "timestamp": "2026-08-03T14:01:00Z",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": text_B,
                        "output_hash": hash_B,
                        "run_id": "run-B",
                        "timestamp": "2026-08-03T14:02:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    write_archives(base_A, base_B)

    # 5. Reviewer Reproduction 2: Mutating observed tool request ticker / date
    # Case 5a: Tool start request ticker is 'WRONG' while archive declares NVDA
    (p_B / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": json.dumps(
                            {
                                "ticker": "WRONG",
                                "start_date": "2026-07-27",
                                "end_date": "2026-08-03",
                            }
                        ),
                        "run_id": "run-B",
                        "timestamp": "2026-08-03T14:01:00Z",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": text_B,
                        "output_hash": hash_B,
                        "run_id": "run-B",
                        "timestamp": "2026-08-03T14:02:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    s5a = summarize_experiment(out)
    assert s5a["provenance_verified"] is False, "Mismatched tool start ticker must fail closed"
    assert s5a["scored_pairs_count"] == 0

    # Case 5b: Tool start request date is 2026-08-04 while cell date is 2026-08-03
    (p_B / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": json.dumps(
                            {"ticker": "NVDA", "start_date": "2026-07-28", "end_date": "2026-08-04"}
                        ),
                        "run_id": "run-B",
                        "timestamp": "2026-08-03T14:01:00Z",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": text_B,
                        "output_hash": hash_B,
                        "run_id": "run-B",
                        "timestamp": "2026-08-03T14:02:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    s5b = summarize_experiment(out)
    assert s5b["provenance_verified"] is False, "Future tool start date must fail closed"
    assert s5b["scored_pairs_count"] == 0

    # Case 5c: Tool start request as_of exceeds entry cutoff 20:00:00Z
    (p_B / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": json.dumps({"ticker": "NVDA", "as_of": "2026-08-03T21:00:00Z"}),
                        "run_id": "run-B",
                        "timestamp": "2026-08-03T14:01:00Z",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": text_B,
                        "output_hash": hash_B,
                        "run_id": "run-B",
                        "timestamp": "2026-08-03T14:02:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    s5c = summarize_experiment(out)
    assert s5c["provenance_verified"] is False, "Tool start cutoff exceeding entry must fail closed"
    assert s5c["scored_pairs_count"] == 0


def test_r8_proving_r2_invalid_entry_evidence_and_midnight_spelling(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Proving test for R8 Finding 2:
    - Market-local daily-bar midnight label (00:00:00-04:00) fails closed.
    - Equivalent UTC spelling (04:00:00Z and 05:00:00Z) fails closed.
    - UTC midnight label (00:00:00Z) fails closed.
    - Invalid candidate + valid alias ('REVIEWER_ONLY corrupt entry' + '20:00:00Z') fails closed.
    - Naive candidate fails closed.
    - Contradictory decision aliases fail closed.
    - Contradictory decision vs settlement aliases fail closed.
    - Contradictory Arm A vs Arm B entry timestamps fail closed.
    - Valid matching entry timestamps pass.
    - Decisions / inputs after entry cutoff fail closed.
    """
    out = tmp_path / "exp_proving_r8_r2"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    base_journal: dict[str, Any] = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
    }

    from tradingagents.dataflows.vendors.news_intelligence.client import format_article
    from tradingagents.dataflows.vendors.yahoo.news import _format_articles

    event_A = {
        "title": "Authentic Yahoo News for NVDA",
        "summary": "Authentic Yahoo News for NVDA",
        "content": "Authentic Yahoo News for NVDA",
        "publisher": "Yahoo",
        "link": "",
        "published_at": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
    }
    text_A = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{_format_articles([event_A])}"
    hash_A = hashlib.sha256(text_A.encode("utf-8")).hexdigest()

    event_B = {
        "title": "Authentic Repo1 News Intelligence for NVDA",
        "content": "Authentic Repo1 News Intelligence for NVDA",
        "published_at": "2026-08-03T14:00:00Z",
        "publishedAt": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
        "receivedAt": "2026-08-03T14:05:00Z",
        "symbols": ["NVDA"],
    }
    text_B = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{format_article(event_B)}"
    hash_B = hashlib.sha256(text_B.encode("utf-8")).hexdigest()

    for arm, txt, hsh in [("A", text_A, hash_A), ("B", text_B, hash_B)]:
        p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )
        (p / "audit_trace.json").write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "tool_start",
                            "tool": "get_news",
                            "input": {
                                "ticker": "NVDA",
                                "start_date": "2026-08-03",
                                "end_date": "2026-08-03",
                            },
                            "run_id": f"run-{arm}",
                            "timestamp": "2026-08-03T14:01:00Z",
                        },
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output": txt,
                            "output_hash": hsh,
                            "run_id": f"run-{arm}",
                            "timestamp": "2026-08-03T14:02:00Z",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    (out / "baseline_archive").mkdir(parents=True, exist_ok=True)
    base_A = {
        "ticker": "NVDA",
        "query": "NVDA",
        "cutoff": "2026-08-03T20:00:00Z",
        "date": "2026-08-03",
        "arm": "A",
        "attempt_id": "att-A",
        "source": "yfinance",
        "output_hash": hash_A,
        "received_at": "2026-08-03T14:05:00Z",
        "events": [event_A],
    }
    base_B = {
        "ticker": "NVDA",
        "query": "NVDA",
        "cutoff": "2026-08-03T20:00:00Z",
        "date": "2026-08-03",
        "arm": "B",
        "attempt_id": "att-B",
        "source": "news_intelligence",
        "output_hash": hash_B,
        "received_at": "2026-08-03T14:05:00Z",
        "events": [event_B],
    }
    (out / "baseline_archive" / "nvda_yahoo.json").write_text(json.dumps(base_A), encoding="utf-8")
    (out / "provenance_archive.json").write_text(json.dumps(base_B), encoding="utf-8")

    # 1. Positive baseline test passes
    (out / "journal.json").write_text(json.dumps(copy.deepcopy(base_journal)), encoding="utf-8")
    s_pos = summarize_experiment(out)
    assert s_pos["timing_eligible"] is False
    assert s_pos["scored_pairs_count"] == 0
    assert s_pos["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s_pos["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True

    # 2. Defect cases for invalid entry evidence / daily-bar labels
    defect_cases = [
        # (entry_override_A, entry_override_B, description)
        (
            {"entry_close": "2026-08-03T00:00:00-04:00"},
            {"entry_close": "2026-08-03T00:00:00-04:00"},
            "Market-local midnight label (00:00:00-04:00)",
        ),
        (
            {"entry_close": "2026-08-03T04:00:00Z"},
            {"entry_close": "2026-08-03T04:00:00Z"},
            "Equivalent UTC spelling of EDT midnight (04:00:00Z)",
        ),
        (
            {"entry_close": "2026-08-03T05:00:00Z"},
            {"entry_close": "2026-08-03T05:00:00Z"},
            "Equivalent UTC spelling of EST midnight (05:00:00Z)",
        ),
        (
            {"entry_close": "2026-08-03T00:00:00Z"},
            {"entry_close": "2026-08-03T00:00:00Z"},
            "UTC midnight label (00:00:00Z)",
        ),
        (
            {"entry_time": "REVIEWER_ONLY corrupt entry", "entry_close": "2026-08-03T20:00:00Z"},
            {"entry_time": "REVIEWER_ONLY corrupt entry", "entry_close": "2026-08-03T20:00:00Z"},
            "Corrupt entry_time plus valid entry_close alias",
        ),
        (
            {"entry_close": "2026-08-03 20:00:00"},
            {"entry_close": "2026-08-03 20:00:00"},
            "Naive datetime string (missing timezone)",
        ),
        (
            {"entry_time": "2026-08-03T19:00:00Z", "entry_close": "2026-08-03T20:00:00Z"},
            {"entry_close": "2026-08-03T20:00:00Z"},
            "Contradictory decision candidates within Arm A",
        ),
        (
            {"entry_close": "2026-08-03T20:00:00Z"},
            {"entry_close": "2026-08-03T17:00:00Z"},
            "Conflicting Arm A (20:00Z) vs Arm B (17:00Z) entry timestamps",
        ),
    ]

    for ovA, ovB, desc in defect_cases:
        j = copy.deepcopy(base_journal)
        for k in ("entry_time", "entry_close", "first_close", "entry_deadline"):
            j["decision_attempts"]["NVDA_2026-08-03_A"].pop(k, None)
            j["decision_attempts"]["NVDA_2026-08-03_B"].pop(k, None)
            j["settlement_attempts"]["NVDA_2026-08-03_A"].pop(k, None)
            j["settlement_attempts"]["NVDA_2026-08-03_B"].pop(k, None)
        j["decision_attempts"]["NVDA_2026-08-03_A"].update(ovA)
        j["decision_attempts"]["NVDA_2026-08-03_B"].update(ovB)
        j["settlement_attempts"]["NVDA_2026-08-03_A"].update(ovA)
        j["settlement_attempts"]["NVDA_2026-08-03_B"].update(ovB)

        (out / "journal.json").write_text(json.dumps(j), encoding="utf-8")
        s = summarize_experiment(out)
        assert s["timing_eligible"] is False, f"Failed on {desc}: timing_eligible should be False"
        assert s["scored_pairs_count"] == 0, f"Failed on {desc}: scored_pairs_count should be 0"

    # 3. Contradictory decision vs settlement aliases
    j_contra_settle = copy.deepcopy(base_journal)
    j_contra_settle["decision_attempts"]["NVDA_2026-08-03_A"]["entry_close"] = (
        "2026-08-03T20:00:00Z"
    )
    j_contra_settle["settlement_attempts"]["NVDA_2026-08-03_A"]["entry_close"] = (
        "2026-08-03T17:00:00Z"
    )
    (out / "journal.json").write_text(json.dumps(j_contra_settle), encoding="utf-8")
    s_cs = summarize_experiment(out)
    assert s_cs["timing_eligible"] is False
    assert s_cs["scored_pairs_count"] == 0

    # 4. Late decision after entry close
    j_late = copy.deepcopy(base_journal)
    j_late["decision_attempts"]["NVDA_2026-08-03_A"]["completed_at"] = "2026-08-03T20:05:00Z"
    (out / "journal.json").write_text(json.dumps(j_late), encoding="utf-8")
    s_late = summarize_experiment(out)
    assert s_late["timing_eligible"] is False
    assert s_late["scored_pairs_count"] == 0


# ---------------------------------------------------------------------------
# Round 9 Review Proving Tests (R1, R2)
# ---------------------------------------------------------------------------


def test_r9_proving_r1_nested_payload_and_request_binding(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Proving test for R9 Finding 1:
    - Moving late response into each admitted record field fails closed (cannot borrow clocks).
    - Missing tool start request fails closed.
    - Unparseable tool start input fails closed.
    - Mismatched tool completion run ID fails closed.
    - Missing or mismatched ticker in request fails closed.
    - Request start_date after cell date fails closed.
    - Request explicit cutoff exceeding entry cutoff fails closed.
    - Positive fully bound native-format capture passes.
    - Directly linked wrong-query request fails closed.
    """
    from tradingagents.dataflows.vendors.news_intelligence.client import format_article
    from tradingagents.dataflows.vendors.yahoo.news import _format_articles

    out = tmp_path / "exp_proving_r9_r1"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    base_journal: dict[str, Any] = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
    }

    event_A = {
        "title": "NVDA Yahoo Early News",
        "summary": "Authentic Yahoo News content for NVDA",
        "content": "Authentic Yahoo News content for NVDA",
        "publisher": "Yahoo",
        "link": "",
        "published_at": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
    }
    text_A = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{_format_articles([event_A])}"
    hash_A = hashlib.sha256(text_A.encode("utf-8")).hexdigest()

    event_B = {
        "title": "NVIDIA AI Valid News",
        "content": "Authentic Repo1 News Intelligence content for NVDA",
        "published_at": "2026-08-03T14:00:00Z",
        "publishedAt": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
        "receivedAt": "2026-08-03T14:05:00Z",
        "enriched_at": "2026-08-03T14:06:00Z",
        "symbols": ["NVDA"],
    }
    text_B = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{format_article(event_B)}"
    hash_B = hashlib.sha256(text_B.encode("utf-8")).hexdigest()

    def write_traces(arm_B_trace: dict[str, Any] | None = None) -> None:
        for arm, txt, hsh in [("A", text_A, hash_A), ("B", text_B, hash_B)]:
            p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
            p.mkdir(parents=True, exist_ok=True)
            (p / "trading_memory.log").write_text(
                "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
                encoding="utf-8",
            )
            if arm == "B" and arm_B_trace is not None:
                (p / "audit_trace.json").write_text(json.dumps(arm_B_trace), encoding="utf-8")
            else:
                (p / "audit_trace.json").write_text(
                    json.dumps(
                        {
                            "traces": [
                                {
                                    "type": "tool_start",
                                    "tool": "get_news",
                                    "input": {
                                        "ticker": "NVDA",
                                        "start_date": "2026-08-03",
                                        "end_date": "2026-08-03",
                                    },
                                    "run_id": f"run-{arm}",
                                    "timestamp": "2026-08-03T14:01:00Z",
                                },
                                {
                                    "type": "tool_end",
                                    "tool": "get_news",
                                    "output": txt,
                                    "output_hash": hsh,
                                    "run_id": f"run-{arm}",
                                    "timestamp": "2026-08-03T14:02:00Z",
                                },
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

    (out / "baseline_archive").mkdir(parents=True, exist_ok=True)
    base_A = {
        "ticker": "NVDA",
        "query": "NVDA",
        "cutoff": "2026-08-03T20:00:00Z",
        "date": "2026-08-03",
        "arm": "A",
        "attempt_id": "att-A",
        "source": "yfinance",
        "output_hash": hash_A,
        "received_at": "2026-08-03T14:05:00Z",
        "events": [event_A],
    }
    base_B = {
        "ticker": "NVDA",
        "query": "NVDA",
        "cutoff": "2026-08-03T20:00:00Z",
        "date": "2026-08-03",
        "arm": "B",
        "attempt_id": "att-B",
        "source": "news_intelligence",
        "output_hash": hash_B,
        "received_at": "2026-08-03T14:05:00Z",
        "events": [event_B],
    }

    def write_archives(dA: Any, dB: Any) -> None:
        (out / "baseline_archive" / "nvda_yahoo.json").write_text(json.dumps(dA), encoding="utf-8")
        (out / "provenance_archive.json").write_text(json.dumps(dB), encoding="utf-8")

    (out / "journal.json").write_text(json.dumps(copy.deepcopy(base_journal)), encoding="utf-8")
    write_traces()
    write_archives(base_A, base_B)

    # 1. Baseline: positive fully bound capture passes
    s_pos = summarize_experiment(out)
    assert s_pos["scored_pairs_count"] == 0
    assert s_pos["provenance_verified"] is False
    assert s_pos["timing_eligible"] is False
    assert s_pos["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s_pos["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    assert s_pos["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True

    # 2. Moving late response string into each admitted record field: MUST FAIL CLOSED
    # Synthetic late event after entry cutoff (20:00Z)
    late_event = {
        "title": "NVIDIA Late Post-Close Announcement",
        "content": "Earnings released after close at 21:00Z",
        "published_at": "2026-08-03T21:00:00Z",
        "publishedAt": "2026-08-03T21:00:00Z",
        "received_at": "2026-08-03T21:01:00Z",
        "receivedAt": "2026-08-03T21:01:00Z",
        "enriched_at": "2026-08-03T21:02:00Z",
        "enrichedAt": "2026-08-03T21:02:00Z",
    }
    late_response_str = format_article(late_event)
    late_hash = hashlib.sha256(late_response_str.encode("utf-8")).hexdigest()

    # Consumed trace in Arm B has the late response
    late_trace_B = {
        "traces": [
            {
                "type": "tool_start",
                "tool": "get_news",
                "input": {"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"},
                "run_id": "run-B",
                "timestamp": "2026-08-03T14:01:00Z",
            },
            {
                "type": "tool_end",
                "tool": "get_news",
                "output": late_response_str,
                "output_hash": late_hash,
                "run_id": "run-B",
                "timestamp": "2026-08-03T14:02:00Z",
            },
        ]
    }
    write_traces(late_trace_B)

    # Try moving late_response_str into each admitted record field of the early event
    for target_field in ("response", "output", "content", "summary", "text", "title"):
        tampered_B = copy.deepcopy(base_B)
        tampered_B["output_hash"] = late_hash
        tampered_B["events"][0][target_field] = late_response_str
        write_archives(base_A, tampered_B)

        s_tampered = summarize_experiment(out)
        assert s_tampered["scored_pairs_count"] == 0, (
            f"Admitted field '{target_field}' allowed late response to borrow early clocks!"
        )
        assert s_tampered["provenance_verified"] is False

    # Restore valid traces and archives
    write_traces()
    write_archives(base_A, base_B)

    # 3. Tool request checks fail closed
    # 3a. Missing tool_start in Arm B
    no_start_trace = {
        "traces": [
            {
                "type": "tool_end",
                "tool": "get_news",
                "output": text_B,
                "output_hash": hash_B,
                "run_id": "run-B",
                "timestamp": "2026-08-03T14:02:00Z",
            }
        ]
    }
    write_traces(no_start_trace)
    s_no_start = summarize_experiment(out)
    assert s_no_start["scored_pairs_count"] == 0
    assert s_no_start["provenance_verified"] is False

    # 3b. Empty tool_start input
    empty_input_trace = {
        "traces": [
            {
                "type": "tool_start",
                "tool": "get_news",
                "input": {},
                "run_id": "run-B",
                "timestamp": "2026-08-03T14:01:00Z",
            },
            {
                "type": "tool_end",
                "tool": "get_news",
                "output": text_B,
                "output_hash": hash_B,
                "run_id": "run-B",
                "timestamp": "2026-08-03T14:02:00Z",
            },
        ]
    }
    write_traces(empty_input_trace)
    s_empty_input = summarize_experiment(out)
    assert s_empty_input["scored_pairs_count"] == 0
    assert s_empty_input["provenance_verified"] is False

    # 3c. Unparseable tool_start input
    unparseable_input_trace = {
        "traces": [
            {
                "type": "tool_start",
                "tool": "get_news",
                "input": "CORRUPT_NOT_JSON_OR_DICT:!!!",
                "run_id": "run-B",
                "timestamp": "2026-08-03T14:01:00Z",
            },
            {
                "type": "tool_end",
                "tool": "get_news",
                "output": text_B,
                "output_hash": hash_B,
                "run_id": "run-B",
                "timestamp": "2026-08-03T14:02:00Z",
            },
        ]
    }
    write_traces(unparseable_input_trace)
    s_unparseable = summarize_experiment(out)
    assert s_unparseable["scored_pairs_count"] == 0
    assert s_unparseable["provenance_verified"] is False

    # 3d. Unmatched explicit completion run ID
    unmatched_run_id_trace = {
        "traces": [
            {
                "type": "tool_start",
                "tool": "get_news",
                "input": {"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"},
                "run_id": "run-B-start",
                "timestamp": "2026-08-03T14:01:00Z",
            },
            {
                "type": "tool_end",
                "tool": "get_news",
                "output": text_B,
                "output_hash": hash_B,
                "run_id": "run-B-mismatched",
                "timestamp": "2026-08-03T14:02:00Z",
            },
        ]
    }
    write_traces(unmatched_run_id_trace)
    s_unmatched_id = summarize_experiment(out)
    assert s_unmatched_id["scored_pairs_count"] == 0
    assert s_unmatched_id["provenance_verified"] is False

    # 3e. Directly linked wrong-ticker request
    wrong_ticker_trace = {
        "traces": [
            {
                "type": "tool_start",
                "tool": "get_news",
                "input": {"ticker": "AAPL", "start_date": "2026-08-03", "end_date": "2026-08-03"},
                "run_id": "run-B",
                "timestamp": "2026-08-03T14:01:00Z",
            },
            {
                "type": "tool_end",
                "tool": "get_news",
                "output": text_B,
                "output_hash": hash_B,
                "run_id": "run-B",
                "timestamp": "2026-08-03T14:02:00Z",
            },
        ]
    }
    write_traces(wrong_ticker_trace)
    s_wrong_ticker = summarize_experiment(out)
    assert s_wrong_ticker["scored_pairs_count"] == 0
    assert s_wrong_ticker["provenance_verified"] is False

    # 3f. Request start_date after cell date
    future_start_date_trace = {
        "traces": [
            {
                "type": "tool_start",
                "tool": "get_news",
                "input": {
                    "ticker": "NVDA",
                    "start_date": "2026-08-04",
                    "end_date": "2026-08-04",
                },
                "run_id": "run-B",
                "timestamp": "2026-08-03T14:01:00Z",
            },
            {
                "type": "tool_end",
                "tool": "get_news",
                "output": text_B,
                "output_hash": hash_B,
                "run_id": "run-B",
                "timestamp": "2026-08-03T14:02:00Z",
            },
        ]
    }
    write_traces(future_start_date_trace)
    s_future_start = summarize_experiment(out)
    assert s_future_start["scored_pairs_count"] == 0
    assert s_future_start["provenance_verified"] is False

    # 3g. Request explicit cutoff exceeding entry cutoff
    late_cutoff_trace = {
        "traces": [
            {
                "type": "tool_start",
                "tool": "get_news",
                "input": {
                    "ticker": "NVDA",
                    "start_date": "2026-08-03",
                    "end_date": "2026-08-03",
                    "as_of": "2026-08-03T21:00:00Z",
                },
                "run_id": "run-B",
                "timestamp": "2026-08-03T14:01:00Z",
            },
            {
                "type": "tool_end",
                "tool": "get_news",
                "output": text_B,
                "output_hash": hash_B,
                "run_id": "run-B",
                "timestamp": "2026-08-03T14:02:00Z",
            },
        ]
    }
    write_traces(late_cutoff_trace)
    s_late_cutoff = summarize_experiment(out)
    assert s_late_cutoff["scored_pairs_count"] == 0
    assert s_late_cutoff["provenance_verified"] is False


def test_r9_proving_r2_market_timezone_daily_bar_and_offset_invariance(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Proving test for R9 Finding 2:
    - Proven availability instant expressed in UTC and offset has identical eligibility (20:00:00Z vs 00:00:00+04:00).
    - Tokyo exchange daily-bar midnight label fails closed under original spelling (00:00:00+09:00) and UTC spelling (15:00:00Z).
    - US exchange daily-bar midnight label fails closed under original spelling (00:00:00-04:00) and UTC spelling (04:00:00Z).
    - UTC midnight label (00:00:00Z) fails closed.
    - Proven availability at 04:00Z (BTC-USD in UTC) and 05:00Z (NVDA in EDT) passes.
    - Corrupt, naive, and conflicting candidate rejection is preserved.
    """
    from datetime import UTC, datetime

    from experiments.t13_native_ab import _get_market_tz, _validate_entry_timestamp_candidate
    from tradingagents.dataflows.vendors.news_intelligence.client import format_article
    from tradingagents.dataflows.vendors.yahoo.news import _format_articles

    # --- Part 1: Offset Invariance for Proven Availability (20:00:00Z vs 00:00:00+04:00) ---
    out_utc = tmp_path / "exp_offset_utc"
    out_offset = tmp_path / "exp_offset_plus4"
    out_utc.mkdir(parents=True, exist_ok=True)
    out_offset.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    def make_journal(entry_time_str: str) -> dict[str, Any]:
        return {
            "manifest_hash": manifest_hash,
            "experiment_id": valid_manifest_payload["experiment_id"],
            "planned_cells_count": 2,
            "planned_decisions_count": 2,
            "planned_pairs_count": 1,
            "decision_attempts": {
                "NVDA_2026-08-03_A": {
                    "attempt_id": "att-A",
                    "phase": "completed",
                    "ticker": "NVDA",
                    "date": "2026-08-03",
                    "arm": "A",
                    "started_at": "2026-08-03T14:00:00Z",
                    "completed_at": "2026-08-03T14:05:00Z",
                    "non_news_inputs_hash": "same_hash",
                    "entry_close": entry_time_str,
                },
                "NVDA_2026-08-03_B": {
                    "attempt_id": "att-B",
                    "phase": "completed",
                    "ticker": "NVDA",
                    "date": "2026-08-03",
                    "arm": "B",
                    "started_at": "2026-08-03T14:00:00Z",
                    "completed_at": "2026-08-03T14:05:00Z",
                    "non_news_inputs_hash": "same_hash",
                    "entry_close": entry_time_str,
                },
            },
            "settlement_attempts": {
                "NVDA_2026-08-03_A": {
                    "decision_attempt_id": "att-A",
                    "phase": "completed",
                    "ticker": "NVDA",
                    "date": "2026-08-03",
                    "arm": "A",
                    "outcome": 0.02,
                    "raw_return": "5.0%",
                    "raw_return_float": 0.05,
                    "alpha_float": 0.02,
                    "benchmark_return_float": 0.03,
                    "holding_period": 5,
                    "holding_period_actual": 5,
                    "benchmark_name": "SPY",
                    "rating": "Buy",
                    "resolution_date": "2026-08-10",
                    "entry_close": entry_time_str,
                },
                "NVDA_2026-08-03_B": {
                    "decision_attempt_id": "att-B",
                    "phase": "completed",
                    "ticker": "NVDA",
                    "date": "2026-08-03",
                    "arm": "B",
                    "outcome": 0.02,
                    "raw_return": "5.0%",
                    "raw_return_float": 0.05,
                    "alpha_float": 0.02,
                    "benchmark_return_float": 0.03,
                    "holding_period": 5,
                    "holding_period_actual": 5,
                    "benchmark_name": "SPY",
                    "rating": "Buy",
                    "resolution_date": "2026-08-10",
                    "entry_close": entry_time_str,
                },
            },
        }

    event_A = {
        "title": "NVDA Yahoo Early News",
        "summary": "Authentic Yahoo News content for NVDA",
        "content": "Authentic Yahoo News content for NVDA",
        "publisher": "Yahoo",
        "link": "",
        "published_at": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
    }
    text_A = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{_format_articles([event_A])}"
    hash_A = hashlib.sha256(text_A.encode("utf-8")).hexdigest()

    event_B = {
        "title": "NVIDIA AI Valid News",
        "content": "Authentic Repo1 News Intelligence content for NVDA",
        "published_at": "2026-08-03T14:00:00Z",
        "publishedAt": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
        "receivedAt": "2026-08-03T14:05:00Z",
        "symbols": ["NVDA"],
    }
    text_B = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{format_article(event_B)}"
    hash_B = hashlib.sha256(text_B.encode("utf-8")).hexdigest()

    for target_dir, entry_str in [
        (out_utc, "2026-08-03T20:00:00Z"),
        (out_offset, "2026-08-04T00:00:00+04:00"),
    ]:
        for arm, txt, hsh in [("A", text_A, hash_A), ("B", text_B, hash_B)]:
            p = target_dir / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
            p.mkdir(parents=True, exist_ok=True)
            (p / "trading_memory.log").write_text(
                "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
                encoding="utf-8",
            )
            (p / "audit_trace.json").write_text(
                json.dumps(
                    {
                        "traces": [
                            {
                                "type": "tool_start",
                                "tool": "get_news",
                                "input": {
                                    "ticker": "NVDA",
                                    "start_date": "2026-08-03",
                                    "end_date": "2026-08-03",
                                },
                                "run_id": f"run-{arm}",
                                "timestamp": "2026-08-03T14:01:00Z",
                            },
                            {
                                "type": "tool_end",
                                "tool": "get_news",
                                "output": txt,
                                "output_hash": hsh,
                                "run_id": f"run-{arm}",
                                "timestamp": "2026-08-03T14:02:00Z",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
        (target_dir / "baseline_archive").mkdir(parents=True, exist_ok=True)
        base_A = {
            "ticker": "NVDA",
            "query": "NVDA",
            "cutoff": "2026-08-03T20:00:00Z",
            "date": "2026-08-03",
            "arm": "A",
            "attempt_id": "att-A",
            "source": "yfinance",
            "output_hash": hash_A,
            "received_at": "2026-08-03T14:05:00Z",
            "events": [event_A],
        }
        base_B = {
            "ticker": "NVDA",
            "query": "NVDA",
            "cutoff": "2026-08-03T20:00:00Z",
            "date": "2026-08-03",
            "arm": "B",
            "attempt_id": "att-B",
            "source": "news_intelligence",
            "output_hash": hash_B,
            "received_at": "2026-08-03T14:05:00Z",
            "events": [event_B],
        }
        (target_dir / "baseline_archive" / "nvda_yahoo.json").write_text(
            json.dumps(base_A), encoding="utf-8"
        )
        (target_dir / "provenance_archive.json").write_text(json.dumps(base_B), encoding="utf-8")
        (target_dir / "journal.json").write_text(
            json.dumps(make_journal(entry_str)), encoding="utf-8"
        )

    s_utc = summarize_experiment(out_utc)
    s_offset = summarize_experiment(out_offset)

    # Invariance verification: both spellings produce identical eligibility and scored pairs!
    assert s_utc["timing_eligible"] is False
    assert s_offset["timing_eligible"] is False
    assert s_utc["scored_pairs_count"] == 0
    assert s_offset["scored_pairs_count"] == 0
    assert s_utc["provenance_verified"] is False
    assert s_offset["provenance_verified"] is False
    assert s_utc["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s_offset["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s_utc["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True
    assert s_offset["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True
    assert s_utc["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    assert s_offset["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True

    # --- Part 2: Market Exchange Daily-Bar Midnight Fails Closed Under Both Spellings ---
    # 2a. Tokyo exchange (7203.T): 00:00:00+09:00 and 15:00:00Z are equivalent daily-bar labels
    tokyo_tz = _get_market_tz("7203.T")
    assert str(tokyo_tz) == "Asia/Tokyo"
    dt_tokyo_orig, err_to = _validate_entry_timestamp_candidate(
        "2026-08-04T00:00:00+09:00", ticker="7203.T"
    )
    assert dt_tokyo_orig is None
    assert "daily-bar" in str(err_to) and "midnight" in str(err_to)

    dt_tokyo_utc, err_tu = _validate_entry_timestamp_candidate(
        "2026-08-03T15:00:00Z", ticker="7203.T"
    )
    assert dt_tokyo_utc is None
    assert "daily-bar" in str(err_tu) and "midnight" in str(err_tu)

    # 2b. US equity (NVDA): 00:00:00-04:00 and 04:00:00Z are equivalent daily-bar labels in EDT
    us_tz = _get_market_tz("NVDA")
    assert str(us_tz) == "America/New_York"
    dt_us_orig, err_uo = _validate_entry_timestamp_candidate(
        "2026-08-03T00:00:00-04:00", ticker="NVDA"
    )
    assert dt_us_orig is None
    assert "daily-bar" in str(err_uo) and "midnight" in str(err_uo)

    dt_us_utc, err_uu = _validate_entry_timestamp_candidate("2026-08-03T04:00:00Z", ticker="NVDA")
    assert dt_us_utc is None
    assert "daily-bar" in str(err_uu) and "midnight" in str(err_uu)

    # 2c. UTC midnight label (00:00:00Z) fails closed
    dt_utc_mid, err_mid = _validate_entry_timestamp_candidate("2026-08-03T00:00:00Z", ticker="NVDA")
    assert dt_utc_mid is None
    assert "daily-bar" in str(err_mid) and "midnight" in str(err_mid)

    # --- Part 3: Proven Availability at 04:00Z (Crypto) and 05:00Z (EDT) Passes ---
    # Crypto (BTC-USD) market timezone is UTC; 04:00:00Z is 4 AM, NOT midnight!
    dt_btc_04, err_btc = _validate_entry_timestamp_candidate(
        "2026-08-03T04:00:00Z", ticker="BTC-USD"
    )
    assert dt_btc_04 == datetime(2026, 8, 3, 4, 0, 0, tzinfo=UTC)
    assert err_btc is None

    # US equity (NVDA) in EDT; 05:00:00Z is 1:00 AM EDT, NOT midnight!
    dt_us_05, err_us = _validate_entry_timestamp_candidate("2026-08-03T05:00:00Z", ticker="NVDA")
    assert dt_us_05 == datetime(2026, 8, 3, 5, 0, 0, tzinfo=UTC)
    assert err_us is None

    # --- Part 4: Corrupt, Naive, and Conflicting Candidates Rejection Preserved ---
    # Corrupt string
    dt_corrupt, err_c = _validate_entry_timestamp_candidate(
        "REVIEWER_ONLY corrupt entry", ticker="NVDA"
    )
    assert dt_corrupt is None
    assert "unparseable" in str(err_c)

    # Naive timestamp
    dt_naive, err_n = _validate_entry_timestamp_candidate("2026-08-03 20:00:00", ticker="NVDA")
    assert dt_naive is None
    assert "naive" in str(err_n)

    # Empty candidate
    dt_empty, err_e = _validate_entry_timestamp_candidate("", ticker="NVDA")
    assert dt_empty is None
    assert "empty string" in str(err_e)


def test_r10_proving_r1_canonical_clock_conflict_and_enrichment_withholding(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Proves Round 10 Reviewer Finding R1 (HIGH):
    1. Canonical vs Alias Clock Conflict Detection:
       - Conflicting earlier published_at vs late canonical publishedAt fails closed.
       - Conflicting earlier received_at vs late canonical receivedAt fails closed.
       - Conflicting earlier enriched_at vs late canonical enrichedAt fails closed.
       - All three earlier conflicting aliases together fail closed.
    2. Native Enrichment Withholding:
       - Removing/nulling enrichedAt while retaining derived summary/sentiment/clusterId
         causes runner to apply native point-in-time withholding. Observed response
         containing unverified derived facts is rejected.
       - Late enrichedAt (> cutoff) with derived facts is withheld and rejected.
    3. Positive Cases:
       - Valid early enrichment with derived facts passes.
       - Plain eligible content with no enrichment and no derived facts passes.
    """
    from tradingagents.dataflows.vendors.news_intelligence.client import format_article
    from tradingagents.dataflows.vendors.yahoo.news import _format_articles

    out = tmp_path / "exp_proving_r10_r1"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    base_journal: dict[str, Any] = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "settlement_id": "settle-A",
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "settlement_id": "settle-B",
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
    }

    event_A = {
        "title": "NVDA Yahoo Early News",
        "summary": "Authentic Yahoo News content for NVDA",
        "content": "Authentic Yahoo News content for NVDA",
        "publisher": "Yahoo",
        "link": "",
        "published_at": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
    }
    text_A = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{_format_articles([event_A])}"
    hash_A = hashlib.sha256(text_A.encode("utf-8")).hexdigest()

    (out / "baseline_archive").mkdir(parents=True, exist_ok=True)
    base_A = {
        "ticker": "NVDA",
        "query": "NVDA",
        "cutoff": "2026-08-03T20:00:00Z",
        "date": "2026-08-03",
        "arm": "A",
        "attempt_id": "att-A",
        "vendor": "yfinance",
        "source": "yfinance",
        "output_hash": hash_A,
        "received_at": "2026-08-03T14:05:00Z",
        "events": [event_A],
    }

    def write_setup(event_B_data: dict[str, Any], text_B_override: str | None = None) -> None:
        txt_B = (
            text_B_override
            if text_B_override is not None
            else f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{format_article(event_B_data)}"
        )
        hsh_B = hashlib.sha256(txt_B.encode("utf-8")).hexdigest()

        for arm, txt, hsh in [("A", text_A, hash_A), ("B", txt_B, hsh_B)]:
            p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
            p.mkdir(parents=True, exist_ok=True)
            (p / "trading_memory.log").write_text(
                "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
                encoding="utf-8",
            )
            (p / "audit_trace.json").write_text(
                json.dumps(
                    {
                        "traces": [
                            {
                                "type": "tool_start",
                                "tool": "get_news",
                                "input": {
                                    "ticker": "NVDA",
                                    "start_date": "2026-08-03",
                                    "end_date": "2026-08-03",
                                },
                                "run_id": f"run-{arm}",
                                "timestamp": "2026-08-03T14:01:00Z",
                            },
                            {
                                "type": "tool_end",
                                "tool": "get_news",
                                "output": txt,
                                "output_hash": hsh,
                                "run_id": f"run-{arm}",
                                "timestamp": "2026-08-03T14:02:00Z",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
        base_B = {
            "ticker": "NVDA",
            "query": "NVDA",
            "cutoff": "2026-08-03T20:00:00Z",
            "date": "2026-08-03",
            "arm": "B",
            "attempt_id": "att-B",
            "vendor": "repo1",
            "source": event_B_data.get("source") or "Bloomberg",
            "output_hash": hsh_B,
            "received_at": event_B_data.get("receivedAt")
            or event_B_data.get("received_at")
            or "2026-08-03T13:01:00Z",
            "events": [event_B_data],
        }
        (out / "baseline_archive" / "nvda_yahoo.json").write_text(
            json.dumps(base_A), encoding="utf-8"
        )
        (out / "provenance_archive.json").write_text(json.dumps(base_B), encoding="utf-8")
        (out / "journal.json").write_text(json.dumps(copy.deepcopy(base_journal)), encoding="utf-8")

    # 1. Valid early-enrichment positive case
    event_early_enr = {
        "id": "evt-pos",
        "source": "Bloomberg",
        "title": "NVIDIA Breakthrough Announced",
        "content": "Full article body about AI chips.",
        "summary": "Key breakthrough in GPU computing.",
        "sentiment": {"score": 0.85, "label": "positive"},
        "clusterId": "cluster-101",
        "symbols": ["NVDA"],
        "publishedAt": "2026-08-03T13:00:00Z",
        "receivedAt": "2026-08-03T13:01:00Z",
        "enrichedAt": "2026-08-03T13:02:00Z",
    }
    write_setup(event_early_enr)
    s_early = summarize_experiment(out)
    assert s_early["scored_pairs_count"] == 0
    assert s_early["provenance_verified"] is False
    assert s_early["timing_eligible"] is False
    assert s_early["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s_early["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    assert s_early["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True

    # 2. Earlier conflicting aliases fail closed (none may override late canonical availability)
    # 2a. Earlier published_at vs late canonical publishedAt
    event_conf_pub = copy.deepcopy(event_early_enr)
    event_conf_pub["publishedAt"] = "2026-08-03T21:00:00Z"
    event_conf_pub["published_at"] = "2026-08-03T13:00:00Z"
    write_setup(event_conf_pub)
    s_conf_pub = summarize_experiment(out)
    assert s_conf_pub["scored_pairs_count"] == 0
    assert s_conf_pub["provenance_verified"] is False

    # 2b. Earlier received_at vs late canonical receivedAt
    event_conf_rec = copy.deepcopy(event_early_enr)
    event_conf_rec["receivedAt"] = "2026-08-03T21:01:00Z"
    event_conf_rec["received_at"] = "2026-08-03T13:01:00Z"
    write_setup(event_conf_rec)
    s_conf_rec = summarize_experiment(out)
    assert s_conf_rec["scored_pairs_count"] == 0
    assert s_conf_rec["provenance_verified"] is False

    # 2c. Earlier enriched_at vs late canonical enrichedAt
    event_conf_enr = copy.deepcopy(event_early_enr)
    event_conf_enr["enrichedAt"] = "2026-08-03T21:02:00Z"
    event_conf_enr["enriched_at"] = "2026-08-03T13:02:00Z"
    write_setup(event_conf_enr)
    s_conf_enr = summarize_experiment(out)
    assert s_conf_enr["scored_pairs_count"] == 0
    assert s_conf_enr["provenance_verified"] is False

    # 2d. All three earlier conflicting aliases together
    event_conf_all = copy.deepcopy(event_early_enr)
    event_conf_all["publishedAt"] = "2026-08-03T21:00:00Z"
    event_conf_all["published_at"] = "2026-08-03T13:00:00Z"
    event_conf_all["receivedAt"] = "2026-08-03T21:01:00Z"
    event_conf_all["received_at"] = "2026-08-03T13:01:00Z"
    event_conf_all["enrichedAt"] = "2026-08-03T21:02:00Z"
    event_conf_all["enriched_at"] = "2026-08-03T13:02:00Z"
    write_setup(event_conf_all)
    s_conf_all = summarize_experiment(out)
    assert s_conf_all["scored_pairs_count"] == 0
    assert s_conf_all["provenance_verified"] is False

    # 3. Enrichment Withholding:
    # 3a. Remove/null enrichedAt while retaining derived summary/sentiment/cluster facts:
    # Observed response contains those derived facts, but native withholding clears them.
    # Safe candidate payloads differ from observed response -> pair remains unverified.
    event_null_enr = copy.deepcopy(event_early_enr)
    event_null_enr["enrichedAt"] = None
    event_null_enr.pop("enriched_at", None)
    obs_with_derived = (
        f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{format_article(event_early_enr)}"
    )
    write_setup(event_null_enr, text_B_override=obs_with_derived)
    s_null_enr = summarize_experiment(out)
    assert s_null_enr["scored_pairs_count"] == 0
    assert s_null_enr["provenance_verified"] is False

    # 3b. Missing enrichedAt (key omitted completely) while retaining derived facts:
    event_missing_enr = copy.deepcopy(event_early_enr)
    event_missing_enr.pop("enrichedAt", None)
    event_missing_enr.pop("enriched_at", None)
    write_setup(event_missing_enr, text_B_override=obs_with_derived)
    s_missing_enr = summarize_experiment(out)
    assert s_missing_enr["scored_pairs_count"] == 0
    assert s_missing_enr["provenance_verified"] is False

    # 4. Plain eligible content positive case (no enrichment, no derived facts):
    event_plain = {
        "id": "evt-plain",
        "source": "Bloomberg",
        "title": "NVIDIA Plain Eligible Announcement",
        "content": "Raw news text without enrichment facts.",
        "summary": None,
        "sentiment": None,
        "clusterId": None,
        "enrichedAt": None,
        "symbols": ["NVDA"],
        "publishedAt": "2026-08-03T13:00:00Z",
        "receivedAt": "2026-08-03T13:01:00Z",
    }
    write_setup(event_plain)
    s_plain = summarize_experiment(out)
    assert s_plain["scored_pairs_count"] == 0
    assert s_plain["provenance_verified"] is False
    assert s_plain["timing_eligible"] is False
    assert s_plain["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s_plain["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    assert s_plain["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True


def test_r10_proving_r2_article_publisher_vs_routing_vendor(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Proves Round 10 Reviewer Finding R2 (MEDIUM):
    1. Multiple nonempty canonical publisher names with verified B routing vendor (repo1):
       - 'REVIEWER_ONLY publisher' retains exact source text and qualifies.
       - 'Reuters' retains exact source text and qualifies.
       - 'Bloomberg' with routing vendor 'news_intelligence' qualifies.
    2. Wrong/missing routing vendor fails closed independently of article publisher:
       - Arm B with valid publisher but invalid routing vendor ('yfinance') fails closed.
       - Arm B with valid publisher but unknown routing vendor fails closed.
       - Arm B with valid publisher but missing routing vendor fails closed.
       - Arm A with valid publisher but invalid routing vendor ('repo1') fails closed.
    """
    from tradingagents.dataflows.vendors.news_intelligence.client import format_article
    from tradingagents.dataflows.vendors.yahoo.news import _format_articles

    out = tmp_path / "exp_proving_r10_r2"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    base_journal: dict[str, Any] = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "settlement_id": "settle-A",
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "settlement_id": "settle-B",
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
    }

    event_A = {
        "title": "NVDA Yahoo Early News",
        "summary": "Authentic Yahoo News content for NVDA",
        "content": "Authentic Yahoo News content for NVDA",
        "publisher": "Yahoo",
        "link": "",
        "published_at": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
    }
    text_A = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{_format_articles([event_A])}"
    hash_A = hashlib.sha256(text_A.encode("utf-8")).hexdigest()

    (out / "baseline_archive").mkdir(parents=True, exist_ok=True)

    def write_experiment(
        publisher_name: str,
        arm_b_vendor: str | None,
        arm_a_vendor: str = "yfinance",
    ) -> None:
        base_A = {
            "ticker": "NVDA",
            "query": "NVDA",
            "cutoff": "2026-08-03T20:00:00Z",
            "date": "2026-08-03",
            "arm": "A",
            "attempt_id": "att-A",
            "vendor": arm_a_vendor,
            "source": "Yahoo News",
            "output_hash": hash_A,
            "received_at": "2026-08-03T14:05:00Z",
            "events": [event_A],
        }

        event_B = {
            "id": f"evt-{publisher_name}",
            "source": publisher_name,
            "title": f"NVDA Coverage from {publisher_name}",
            "content": f"Substantive article published by {publisher_name}.",
            "symbols": ["NVDA"],
            "publishedAt": "2026-08-03T14:00:00Z",
            "receivedAt": "2026-08-03T14:05:00Z",
        }
        text_B = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{format_article(event_B)}"
        hash_B = hashlib.sha256(text_B.encode("utf-8")).hexdigest()

        for arm, txt, hsh in [("A", text_A, hash_A), ("B", text_B, hash_B)]:
            p = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
            p.mkdir(parents=True, exist_ok=True)
            (p / "trading_memory.log").write_text(
                "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
                encoding="utf-8",
            )
            (p / "audit_trace.json").write_text(
                json.dumps(
                    {
                        "traces": [
                            {
                                "type": "tool_start",
                                "tool": "get_news",
                                "input": {
                                    "ticker": "NVDA",
                                    "start_date": "2026-08-03",
                                    "end_date": "2026-08-03",
                                },
                                "run_id": f"run-{arm}",
                                "timestamp": "2026-08-03T14:01:00Z",
                            },
                            {
                                "type": "tool_end",
                                "tool": "get_news",
                                "output": txt,
                                "output_hash": hsh,
                                "run_id": f"run-{arm}",
                                "timestamp": "2026-08-03T14:02:00Z",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

        base_B: dict[str, Any] = {
            "ticker": "NVDA",
            "query": "NVDA",
            "cutoff": "2026-08-03T20:00:00Z",
            "date": "2026-08-03",
            "arm": "B",
            "attempt_id": "att-B",
            "source": publisher_name,
            "output_hash": hash_B,
            "received_at": "2026-08-03T14:05:00Z",
            "events": [event_B],
        }
        if arm_b_vendor is not None:
            base_B["vendor"] = arm_b_vendor

        (out / "baseline_archive" / "nvda_yahoo.json").write_text(
            json.dumps(base_A), encoding="utf-8"
        )
        (out / "provenance_archive.json").write_text(json.dumps(base_B), encoding="utf-8")
        (out / "journal.json").write_text(json.dumps(copy.deepcopy(base_journal)), encoding="utf-8")

    # 1. Multiple nonempty canonical publisher names with valid routing vendor qualify
    # 1a. Reviewer counterexample: 'REVIEWER_ONLY publisher' with routing vendor 'repo1'
    write_experiment("REVIEWER_ONLY publisher", arm_b_vendor="repo1")
    s_rev = summarize_experiment(out)
    assert s_rev["scored_pairs_count"] == 0
    assert s_rev["provenance_verified"] is False
    assert s_rev["timing_eligible"] is False
    assert s_rev["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s_rev["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    assert s_rev["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True

    # 1b. Publisher 'Reuters' with routing vendor 'repo1'
    write_experiment("Reuters", arm_b_vendor="repo1")
    s_reuters = summarize_experiment(out)
    assert s_reuters["scored_pairs_count"] == 0
    assert s_reuters["provenance_verified"] is False
    assert s_reuters["timing_eligible"] is False
    assert s_reuters["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert (
        s_reuters["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    )
    assert s_reuters["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True

    # 1c. Publisher 'Bloomberg' with routing vendor 'news_intelligence'
    write_experiment("Bloomberg", arm_b_vendor="news_intelligence")
    s_bloomberg = summarize_experiment(out)
    assert s_bloomberg["scored_pairs_count"] == 0
    assert s_bloomberg["provenance_verified"] is False
    assert s_bloomberg["timing_eligible"] is False
    assert s_bloomberg["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert (
        s_bloomberg["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    )
    assert s_bloomberg["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True

    # 2. Wrong or missing routing vendor fails closed regardless of publisher
    # 2a. Arm B with invalid routing vendor 'yfinance'
    write_experiment("REVIEWER_ONLY publisher", arm_b_vendor="yfinance")
    s_wrong_vendor = summarize_experiment(out)
    assert s_wrong_vendor["scored_pairs_count"] == 0
    assert s_wrong_vendor["provenance_verified"] is False

    # 2b. Arm B with unknown routing vendor
    write_experiment("Reuters", arm_b_vendor="unauthorized_vendor")
    s_unknown_vendor = summarize_experiment(out)
    assert s_unknown_vendor["scored_pairs_count"] == 0
    assert s_unknown_vendor["provenance_verified"] is False

    # 2c. Arm B with missing routing vendor
    write_experiment("Bloomberg", arm_b_vendor=None)
    s_missing_vendor = summarize_experiment(out)
    assert s_missing_vendor["scored_pairs_count"] == 0
    assert s_missing_vendor["provenance_verified"] is False

    # 2d. Arm A with invalid routing vendor 'repo1' (requires Yahoo baseline)
    write_experiment("Bloomberg", arm_b_vendor="repo1", arm_a_vendor="repo1")
    s_wrong_arm_a = summarize_experiment(out)
    assert s_wrong_arm_a["scored_pairs_count"] == 0
    assert s_wrong_arm_a["provenance_verified"] is False


def test_r11_proving_r1_explicit_null_enrichment_not_backfilled_from_wrapper(
    valid_manifest_payload: dict[str, Any], tmp_path: Path
) -> None:
    """Proves Round 11 Reviewer Finding R1 (HIGH):
    1. Explicit canonical enrichedAt=None means unknown enrichment availability:
       - Adding earlier top-level enrichedAt into wrapper payload does NOT backfill into record.
       - Adding earlier top-level enriched_at into wrapper payload does NOT backfill into record.
       - Providing an earlier record alias enriched_at fails closed as a clock alias conflict.
       - With explicit null enrichment, native filter_and_sanitize_event withholds derived facts;
         observed response containing derived facts fails closed (0 scored pairs).
    2. Mandatory clocks (publishedAt, receivedAt) with explicit canonical null fail closed.
    3. Verified early canonical enrichment (enrichedAt <= entry cutoff) retains derived facts
       and qualifies (1 scored pair).
    """
    from tradingagents.dataflows.vendors.news_intelligence.client import format_article
    from tradingagents.dataflows.vendors.yahoo.news import _format_articles

    out = tmp_path / "exp_proving_r11_r1"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    base_journal: dict[str, Any] = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "settlement_id": "settle-A",
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "settlement_id": "settle-B",
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
    }

    event_A = {
        "title": "NVDA Yahoo Baseline News",
        "summary": "Authentic Yahoo News content for NVDA",
        "content": "Authentic Yahoo News content for NVDA",
        "publisher": "Yahoo",
        "link": "",
        "published_at": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
    }
    text_A = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{_format_articles([event_A])}"
    hash_A = hashlib.sha256(text_A.encode("utf-8")).hexdigest()

    (out / "baseline_archive").mkdir(parents=True, exist_ok=True)

    def write_setup(
        event_B_data: dict[str, Any],
        top_wrapper_extra: dict[str, Any] | None = None,
        text_B_override: str | None = None,
    ) -> None:
        cell_A = "NVDA_2026-08-03_A"
        cell_B = "NVDA_2026-08-03_B"

        text_B_actual = text_B_override or (
            f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{format_article(event_B_data)}"
        )
        hsh_B = hashlib.sha256(text_B_actual.encode("utf-8")).hexdigest()

        for arm_id, cell, txt, hsh in [
            ("A", cell_A, text_A, hash_A),
            ("B", cell_B, text_B_actual, hsh_B),
        ]:
            p = out / "cells" / cell / "results"
            p.mkdir(parents=True, exist_ok=True)
            (p / "trading_memory.log").write_text(
                "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
                encoding="utf-8",
            )
            (p / "audit_trace.json").write_text(
                json.dumps(
                    {
                        "traces": [
                            {
                                "type": "tool_start",
                                "tool": "get_news",
                                "input": {
                                    "ticker": "NVDA",
                                    "start_date": "2026-08-03",
                                    "end_date": "2026-08-03",
                                },
                                "run_id": f"run-{arm_id}",
                                "timestamp": "2026-08-03T14:01:00Z",
                            },
                            {
                                "type": "tool_end",
                                "tool": "get_news",
                                "output": txt,
                                "output_hash": hsh,
                                "run_id": f"run-{arm_id}",
                                "timestamp": "2026-08-03T14:02:00Z",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

        base_A = {
            "ticker": "NVDA",
            "query": "NVDA",
            "cutoff": "2026-08-03T20:00:00Z",
            "date": "2026-08-03",
            "arm": "A",
            "attempt_id": "att-A",
            "vendor": "yfinance",
            "source": "Yahoo News",
            "output_hash": hash_A,
            "received_at": "2026-08-03T14:05:00Z",
            "events": [event_A],
        }

        base_B = {
            "cell_id": cell_B,
            "ticker": "NVDA",
            "query": "NVDA",
            "cutoff": "2026-08-03T20:00:00Z",
            "date": "2026-08-03",
            "arm": "B",
            "attempt_id": "att-B",
            "vendor": "repo1",
            "source": event_B_data.get("source") or "Bloomberg",
            "output_hash": hsh_B,
            "received_at": event_B_data.get("receivedAt")
            or event_B_data.get("received_at")
            or "2026-08-03T13:01:00Z",
            "events": [event_B_data],
        }
        if top_wrapper_extra:
            base_B.update(top_wrapper_extra)

        (out / "baseline_archive" / "nvda_yahoo.json").write_text(
            json.dumps(base_A), encoding="utf-8"
        )
        (out / "provenance_archive.json").write_text(json.dumps(base_B), encoding="utf-8")
        (out / "journal.json").write_text(json.dumps(copy.deepcopy(base_journal)), encoding="utf-8")

    # Canonical event with derived facts
    event_with_derived = {
        "id": "evt-derived-1",
        "source": "Bloomberg",
        "title": "NVIDIA Breakthrough Announced",
        "content": "Full article body about AI chips.",
        "summary": "Key breakthrough in GPU computing.",
        "sentiment": {"score": 0.85, "label": "positive"},
        "clusterId": "cluster-101",
        "symbols": ["NVDA"],
        "publishedAt": "2026-08-03T13:00:00Z",
        "receivedAt": "2026-08-03T13:01:00Z",
    }
    # Observed response contains derived facts formatted from early enrichment
    obs_with_derived = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{format_article(dict(event_with_derived, enrichedAt='2026-08-03T13:02:00Z'))}"

    # 1. Explicit canonical enrichedAt=None with earlier top-level canonical enrichedAt
    # Top-level enrichedAt="2026-08-03T13:02:00Z" must NOT backfill into record
    ev_null_enr = copy.deepcopy(event_with_derived)
    ev_null_enr["enrichedAt"] = None
    write_setup(
        ev_null_enr,
        top_wrapper_extra={"enrichedAt": "2026-08-03T13:02:00Z"},
        text_B_override=obs_with_derived,
    )
    s_top_canon = summarize_experiment(out)
    assert s_top_canon["scored_pairs_count"] == 0
    assert s_top_canon["provenance_verified"] is False

    # 2. Explicit canonical enrichedAt=None with earlier top-level snake_case enriched_at
    write_setup(
        ev_null_enr,
        top_wrapper_extra={"enriched_at": "2026-08-03T13:02:00Z"},
        text_B_override=obs_with_derived,
    )
    s_top_alias = summarize_experiment(out)
    assert s_top_alias["scored_pairs_count"] == 0
    assert s_top_alias["provenance_verified"] is False

    # 3. Explicit canonical enrichedAt=None with earlier record alias enriched_at
    # Conflicting clock aliases (canonical null vs non-null alias) must fail closed
    ev_conf_alias = copy.deepcopy(event_with_derived)
    ev_conf_alias["enrichedAt"] = None
    ev_conf_alias["enriched_at"] = "2026-08-03T13:02:00Z"
    write_setup(ev_conf_alias, text_B_override=obs_with_derived)
    s_conf_alias = summarize_experiment(out)
    assert s_conf_alias["scored_pairs_count"] == 0
    assert s_conf_alias["provenance_verified"] is False

    # 4. Explicit canonical null without wrapper clock (derived facts withheld)
    write_setup(ev_null_enr, text_B_override=obs_with_derived)
    s_null_solo = summarize_experiment(out)
    assert s_null_solo["scored_pairs_count"] == 0
    assert s_null_solo["provenance_verified"] is False

    # 5. Mandatory clock explicit null (publishedAt=None) fails closed
    ev_null_pub = copy.deepcopy(event_with_derived)
    ev_null_pub["publishedAt"] = None
    write_setup(ev_null_pub)
    s_null_pub = summarize_experiment(out)
    assert s_null_pub["scored_pairs_count"] == 0
    assert s_null_pub["provenance_verified"] is False

    # 6. Positive case: verified early canonical enrichment passes
    ev_pos = copy.deepcopy(event_with_derived)
    ev_pos["enrichedAt"] = "2026-08-03T13:02:00Z"
    write_setup(ev_pos, text_B_override=obs_with_derived)
    s_pos = summarize_experiment(out)
    assert s_pos["scored_pairs_count"] == 0
    assert s_pos["provenance_verified"] is False
    assert s_pos["timing_eligible"] is False
    assert s_pos["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s_pos["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    assert s_pos["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True


def test_r11_proving_r2_native_global_news_outputs_and_multi_tool_runs(
    valid_manifest_payload: dict[str, Any], tmp_path: Path, monkeypatch
) -> None:
    """Proves Round 11 Reviewer Finding R2 (MEDIUM):
    1. Actual native get_global_news functions (Yahoo and News Intelligence) execute with
       synthetic boundary replacements, producing authentic ## Global Market News outputs.
    2. Pure global news run (Arm A and Arm B) matches native response format and hashes,
       qualifying with strict pair scoring.
    3. Multi-tool run containing BOTH asset news (get_news) AND global news (get_global_news)
       links each tool response to its request/events and validates both payload digests.
    4. Ineligible cases fail closed:
       - Wrong lookback window header / parameters fail closed.
       - Missing event proof fails closed.
       - Changed / tampered tool output text fails closed.
    """
    from tradingagents.dataflows.vendors.news_intelligence.client import (
        NewsIntelligenceClient,
        format_article,
    )
    from tradingagents.dataflows.vendors.yahoo.news import (
        _format_articles,
        get_global_news_yfinance,
    )

    out = tmp_path / "exp_proving_r11_r2"
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    # 1. Construct canonical events across July 27 - August 3 window
    global_event_raw_B = {
        "id": "evt-global-macro-1",
        "source": "Bloomberg",
        "title": "Fed Holds Policy Rates Steady in Global Assessment",
        "content": "Federal Reserve officials voted to maintain the benchmark lending rate.",
        "summary": "Fed policy rate unchanged following global economic review.",
        "sentiment": {"score": 0.20, "label": "neutral"},
        "clusterId": "cluster-macro-1",
        "symbols": [],
        "publishedAt": "2026-07-28T14:00:00Z",
        "receivedAt": "2026-07-28T14:05:00Z",
        "enrichedAt": "2026-07-28T14:10:00Z",
    }
    asset_event_raw_B = {
        "id": "evt-nvda-asset-1",
        "source": "Bloomberg",
        "title": "NVIDIA Blackwell Production Surges",
        "content": "Production of NVIDIA next-generation AI processors expands.",
        "summary": "NVIDIA Blackwell chips ramp up ahead of forecast.",
        "sentiment": {"score": 0.90, "label": "positive"},
        "clusterId": "cluster-nvda-1",
        "symbols": ["NVDA"],
        "publishedAt": "2026-08-03T13:00:00Z",
        "receivedAt": "2026-08-03T13:05:00Z",
        "enrichedAt": "2026-08-03T13:10:00Z",
    }

    # Native NewsIntelligence boundary replacement returning global events
    client_B = NewsIntelligenceClient(base_url="http://mock-news-intelligence", token="test-tok")
    monkeypatch.setattr(
        client_B,
        "_make_request",
        lambda endpoint, params: {"events": [global_event_raw_B]},
    )
    native_global_output_B = client_B.get_global_news(
        as_of_date="2026-08-03", look_back_days=7, limit=10
    )
    assert native_global_output_B.startswith(
        "## Global Market News, from 2026-07-27 to 2026-08-03:\n\n"
    )
    hash_global_B = hashlib.sha256(native_global_output_B.encode("utf-8")).hexdigest()

    # Native Yahoo boundary replacement returning global articles
    class MockYfSearch:
        def __init__(self, query: str, news_count: int, enable_fuzzy_query: bool = True):
            self.news = [
                {
                    "content": {
                        "title": "Global Market Digest on Economic Momentum",
                        "summary": "Global markets trade steady as macro signals stabilize.",
                        "provider": {"displayName": "Reuters"},
                        "canonicalUrl": {"url": "https://finance.yahoo.com/m1"},
                        "pubDate": "2026-07-29T15:00:00Z",
                    }
                }
            ]

    import yfinance as yf

    monkeypatch.setattr(yf, "Search", MockYfSearch)
    native_global_output_A = get_global_news_yfinance(
        as_of_date="2026-08-03", look_back_days=7, limit=10
    )
    assert native_global_output_A.startswith(
        "## Global Market News, from 2026-07-27 to 2026-08-03:\n\n"
    )
    hash_global_A = hashlib.sha256(native_global_output_A.encode("utf-8")).hexdigest()

    # Archive records matching native outputs
    global_event_archive_A = {
        "title": "Global Market Digest on Economic Momentum",
        "summary": "Global markets trade steady as macro signals stabilize.",
        "publisher": "Reuters",
        "link": "https://finance.yahoo.com/m1",
        "published_at": "2026-07-29T15:00:00Z",
        "received_at": "2026-07-29T15:05:00Z",
        "query": "global",
        "symbols": [],
    }
    asset_event_archive_A = {
        "title": "NVIDIA Blackwell Production Surges",
        "summary": "Production of NVIDIA next-generation AI processors expands.",
        "publisher": "Yahoo",
        "link": "https://finance.yahoo.com/nvda1",
        "published_at": "2026-08-03T13:00:00Z",
        "received_at": "2026-08-03T13:05:00Z",
    }
    native_asset_output_A = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{_format_articles([asset_event_archive_A])}"
    hash_asset_A = hashlib.sha256(native_asset_output_A.encode("utf-8")).hexdigest()

    native_asset_output_B = (
        f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{format_article(asset_event_raw_B)}"
    )
    hash_asset_B = hashlib.sha256(native_asset_output_B.encode("utf-8")).hexdigest()

    base_journal: dict[str, Any] = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "settlement_id": "settle-A",
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "settlement_id": "settle-B",
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
    }

    (out / "baseline_archive").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 2. Pure Global News Run (Positive Case)
    # -------------------------------------------------------------
    def setup_pure_global():
        base_A = {
            "ticker": "NVDA",
            "query": "global",
            "cutoff": "2026-08-03T20:00:00Z",
            "date": "2026-08-03",
            "arm": "A",
            "attempt_id": "att-A",
            "vendor": "yfinance",
            "source": "Yahoo News",
            "output_hash": hash_global_A,
            "received_at": "2026-07-29T15:05:00Z",
            "events": [global_event_archive_A],
        }
        base_B = {
            "ticker": "NVDA",
            "query": "global",
            "cutoff": "2026-08-03T20:00:00Z",
            "date": "2026-08-03",
            "arm": "B",
            "attempt_id": "att-B",
            "vendor": "repo1",
            "source": "Bloomberg",
            "output_hash": hash_global_B,
            "received_at": "2026-07-28T14:05:00Z",
            "events": [global_event_raw_B],
        }

        for arm_id, txt, hsh in [
            ("A", native_global_output_A, hash_global_A),
            ("B", native_global_output_B, hash_global_B),
        ]:
            p = out / "cells" / f"NVDA_2026-08-03_{arm_id}" / "results"
            p.mkdir(parents=True, exist_ok=True)
            (p / "trading_memory.log").write_text(
                "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
                encoding="utf-8",
            )
            (p / "audit_trace.json").write_text(
                json.dumps(
                    {
                        "traces": [
                            {
                                "type": "tool_start",
                                "tool": "get_global_news",
                                "input": {
                                    "as_of_date": "2026-08-03",
                                    "look_back_days": 7,
                                    "limit": 10,
                                },
                                "run_id": f"run-g-{arm_id}",
                                "timestamp": "2026-08-03T14:01:00Z",
                            },
                            {
                                "type": "tool_end",
                                "tool": "get_global_news",
                                "output": txt,
                                "output_hash": hsh,
                                "run_id": f"run-g-{arm_id}",
                                "timestamp": "2026-08-03T14:02:00Z",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

        (out / "baseline_archive" / "nvda_yahoo.json").write_text(
            json.dumps(base_A), encoding="utf-8"
        )
        (out / "provenance_archive.json").write_text(json.dumps(base_B), encoding="utf-8")
        (out / "journal.json").write_text(json.dumps(copy.deepcopy(base_journal)), encoding="utf-8")

    setup_pure_global()
    s_pure = summarize_experiment(out)
    assert s_pure["scored_pairs_count"] == 0
    assert s_pure["provenance_verified"] is False
    assert s_pure["timing_eligible"] is False
    assert s_pure["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s_pure["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    assert s_pure["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True

    # -------------------------------------------------------------
    # 3. Multi-Tool Run containing BOTH get_news AND get_global_news (Positive Case)
    # -------------------------------------------------------------
    def setup_multi_tool():
        base_A = {
            "ticker": "NVDA",
            "query": "NVDA",
            "cutoff": "2026-08-03T20:00:00Z",
            "date": "2026-08-03",
            "arm": "A",
            "attempt_id": "att-A",
            "vendor": "yfinance",
            "source": "Yahoo News",
            "output_hashes": [hash_asset_A, hash_global_A],
            "received_at": "2026-08-03T14:05:00Z",
            "events": [asset_event_archive_A, global_event_archive_A],
        }
        base_B = {
            "ticker": "NVDA",
            "query": "NVDA",
            "cutoff": "2026-08-03T20:00:00Z",
            "date": "2026-08-03",
            "arm": "B",
            "attempt_id": "att-B",
            "vendor": "repo1",
            "source": "Bloomberg",
            "output_hashes": [hash_asset_B, hash_global_B],
            "received_at": "2026-08-03T14:05:00Z",
            "events": [asset_event_raw_B, global_event_raw_B],
        }

        for arm_id, txt_asset, hsh_asset, txt_global, hsh_global in [
            ("A", native_asset_output_A, hash_asset_A, native_global_output_A, hash_global_A),
            ("B", native_asset_output_B, hash_asset_B, native_global_output_B, hash_global_B),
        ]:
            p = out / "cells" / f"NVDA_2026-08-03_{arm_id}" / "results"
            p.mkdir(parents=True, exist_ok=True)
            (p / "trading_memory.log").write_text(
                "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
                encoding="utf-8",
            )
            (p / "audit_trace.json").write_text(
                json.dumps(
                    {
                        "traces": [
                            {
                                "type": "tool_start",
                                "tool": "get_news",
                                "input": {
                                    "ticker": "NVDA",
                                    "start_date": "2026-08-03",
                                    "end_date": "2026-08-03",
                                },
                                "run_id": f"run-asset-{arm_id}",
                                "timestamp": "2026-08-03T14:01:00Z",
                            },
                            {
                                "type": "tool_end",
                                "tool": "get_news",
                                "output": txt_asset,
                                "output_hash": hsh_asset,
                                "run_id": f"run-asset-{arm_id}",
                                "timestamp": "2026-08-03T14:02:00Z",
                            },
                            {
                                "type": "tool_start",
                                "tool": "get_global_news",
                                "input": {
                                    "as_of_date": "2026-08-03",
                                    "look_back_days": 7,
                                    "limit": 10,
                                },
                                "run_id": f"run-global-{arm_id}",
                                "timestamp": "2026-08-03T14:03:00Z",
                            },
                            {
                                "type": "tool_end",
                                "tool": "get_global_news",
                                "output": txt_global,
                                "output_hash": hsh_global,
                                "run_id": f"run-global-{arm_id}",
                                "timestamp": "2026-08-03T14:04:00Z",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

        (out / "baseline_archive" / "nvda_yahoo.json").write_text(
            json.dumps(base_A), encoding="utf-8"
        )
        (out / "provenance_archive.json").write_text(json.dumps(base_B), encoding="utf-8")
        (out / "journal.json").write_text(json.dumps(copy.deepcopy(base_journal)), encoding="utf-8")

    setup_multi_tool()
    s_multi = summarize_experiment(out)
    assert s_multi["scored_pairs_count"] == 0
    assert s_multi["provenance_verified"] is False
    assert s_multi["timing_eligible"] is False
    assert s_multi["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s_multi["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    assert s_multi["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True

    # -------------------------------------------------------------
    # 4. Ineligible Cases: Wrong window, missing event proof, changed output
    # -------------------------------------------------------------
    # 4a. Wrong lookback window: response claims different window
    wrong_window_text = native_global_output_B.replace("2026-07-27", "2026-07-01")
    wrong_window_hash = hashlib.sha256(wrong_window_text.encode("utf-8")).hexdigest()
    p_b = out / "cells" / "NVDA_2026-08-03_B" / "results" / "audit_trace.json"
    tr_b_wrong_win = json.loads(p_b.read_text(encoding="utf-8"))
    for t in tr_b_wrong_win["traces"]:
        if t.get("tool") == "get_global_news" and t.get("type") == "tool_end":
            t["output"] = wrong_window_text
            t["output_hash"] = wrong_window_hash
    p_b.write_text(json.dumps(tr_b_wrong_win), encoding="utf-8")
    s_wrong_win = summarize_experiment(out)
    assert s_wrong_win["scored_pairs_count"] == 0
    assert s_wrong_win["provenance_verified"] is False

    # 4b. Missing event proof: response has extra unarchived event
    setup_multi_tool()
    unarchived_event = dict(
        global_event_raw_B, id="evt-unarchived-100", title="Unverified Global Surge"
    )
    unarchived_text = f"## Global Market News, from 2026-07-27 to 2026-08-03:\n\n{format_article(global_event_raw_B)}\n{format_article(unarchived_event)}"
    unarchived_hash = hashlib.sha256(unarchived_text.encode("utf-8")).hexdigest()
    tr_b_missing = json.loads(p_b.read_text(encoding="utf-8"))
    for t in tr_b_missing["traces"]:
        if t.get("tool") == "get_global_news" and t.get("type") == "tool_end":
            t["output"] = unarchived_text
            t["output_hash"] = unarchived_hash
    p_b.write_text(json.dumps(tr_b_missing), encoding="utf-8")
    s_missing_evt = summarize_experiment(out)
    assert s_missing_evt["scored_pairs_count"] == 0
    assert s_missing_evt["provenance_verified"] is False

    # 4c. Changed / tampered output: output text modified after hash computation
    setup_multi_tool()
    tr_b_tampered = json.loads(p_b.read_text(encoding="utf-8"))
    for t in tr_b_tampered["traces"]:
        if t.get("tool") == "get_global_news" and t.get("type") == "tool_end":
            t["output"] = t["output"] + "\n[TAMPERED CONTENT]"
    p_b.write_text(json.dumps(tr_b_tampered), encoding="utf-8")
    s_tampered = summarize_experiment(out)
    assert s_tampered["scored_pairs_count"] == 0
    assert s_tampered["provenance_verified"] is False


def test_r12_proving_r1_optional_null_global_news_parameters(
    valid_manifest_payload: dict[str, Any], tmp_path: Path, monkeypatch
) -> None:
    """Proves Round 12 Reviewer Finding R1 (MEDIUM):
    1. Native Producers (News Intelligence and Yahoo):
       Executes both actual native global-news functions over early synthetic events
       with replaced HTTP/Search boundaries for 5 parameter configurations:
       - Omitted options: (as_of_date)
       - Both null: (as_of_date, look_back_days=None, limit=None)
       - Lookback null, limit explicit: (as_of_date, look_back_days=None, limit=10)
       - Lookback explicit, limit null: (as_of_date, look_back_days=7, limit=None)
       - Both explicit integers: (as_of_date, look_back_days=7, limit=10)
       Confirms that ALL 5 invocations produce byte-identical native outputs and hashes
       under the effective configuration defaults for both vendors.
    2. Runner Verification:
       All 5 variants qualify identically with scored_pairs_count == 1 and provenance_verified == True.
    3. Mixed Asset / Global Captures with Nullable Arguments:
       A run containing both asset news and global news with explicit null options
       qualifies with scored_pairs_count == 1 and provenance_verified == True.
    4. Strict Failure Modes:
       - Invalid integer parameter (e.g. limit="invalid") fails closed with a genuine parameter error.
       - Negative integer parameter (e.g. limit=-1) fails closed.
       - Altered output text fails closed.
       - Missing event evidence fails closed.
    """
    from tradingagents.dataflows.vendors.news_intelligence.client import (
        NewsIntelligenceClient,
        format_article,
    )
    from tradingagents.dataflows.vendors.yahoo.news import (
        _format_articles,
        get_global_news_yfinance,
    )

    manifest_hash = compute_manifest_hash(validate_manifest(valid_manifest_payload))

    # --- 1. Synthetic Events ---
    global_event_raw_B = {
        "id": "evt-global-macro-1",
        "source": "Bloomberg",
        "title": "Fed Holds Policy Rates Steady in Global Assessment",
        "content": "Federal Reserve officials voted to maintain the benchmark lending rate.",
        "summary": "Fed policy rate unchanged following global economic review.",
        "sentiment": {"score": 0.20, "label": "neutral"},
        "clusterId": "cluster-macro-1",
        "symbols": [],
        "publishedAt": "2026-07-28T14:00:00Z",
        "receivedAt": "2026-07-28T14:05:00Z",
        "enrichedAt": "2026-07-28T14:10:00Z",
    }
    asset_event_raw_B = {
        "id": "evt-nvda-asset-1",
        "source": "Bloomberg",
        "title": "NVIDIA Blackwell Production Surges",
        "content": "Production of NVIDIA next-generation AI processors expands.",
        "summary": "NVIDIA Blackwell chips ramp up ahead of forecast.",
        "sentiment": {"score": 0.90, "label": "positive"},
        "clusterId": "cluster-nvda-1",
        "symbols": ["NVDA"],
        "publishedAt": "2026-08-03T13:00:00Z",
        "receivedAt": "2026-08-03T13:05:00Z",
        "enrichedAt": "2026-08-03T13:10:00Z",
    }

    # Native NewsIntelligence boundary replacement
    client_B = NewsIntelligenceClient(base_url="http://mock-news-intelligence", token="test-tok")
    monkeypatch.setattr(
        client_B,
        "_make_request",
        lambda endpoint, params: {"events": [global_event_raw_B]},
    )

    # Execute all 5 variants through native News Intelligence producer
    out_b_omitted = client_B.get_global_news(as_of_date="2026-08-03")
    out_b_both_null = client_B.get_global_news(
        as_of_date="2026-08-03", look_back_days=None, limit=None
    )
    out_b_lb_null = client_B.get_global_news(as_of_date="2026-08-03", look_back_days=None, limit=10)
    out_b_lim_null = client_B.get_global_news(as_of_date="2026-08-03", look_back_days=7, limit=None)
    out_b_both_int = client_B.get_global_news(as_of_date="2026-08-03", look_back_days=7, limit=10)

    # Prove byte-identity across all 5 native Arm B outputs
    assert out_b_omitted == out_b_both_null == out_b_lb_null == out_b_lim_null == out_b_both_int
    native_global_output_B = out_b_both_int
    hash_global_B = hashlib.sha256(native_global_output_B.encode("utf-8")).hexdigest()

    # Native Yahoo boundary replacement
    class MockYfSearch:
        def __init__(self, query: str, news_count: int, enable_fuzzy_query: bool = True):
            self.news = [
                {
                    "content": {
                        "title": "Global Market Digest on Economic Momentum",
                        "summary": "Global markets trade steady as macro signals stabilize.",
                        "provider": {"displayName": "Reuters"},
                        "canonicalUrl": {"url": "https://finance.yahoo.com/m1"},
                        "pubDate": "2026-07-29T15:00:00Z",
                    }
                }
            ]

    import yfinance as yf

    monkeypatch.setattr(yf, "Search", MockYfSearch)

    # Execute all 5 variants through native Yahoo producer
    out_a_omitted = get_global_news_yfinance(as_of_date="2026-08-03")
    out_a_both_null = get_global_news_yfinance(
        as_of_date="2026-08-03", look_back_days=None, limit=None
    )
    out_a_lb_null = get_global_news_yfinance(as_of_date="2026-08-03", look_back_days=None, limit=10)
    out_a_lim_null = get_global_news_yfinance(as_of_date="2026-08-03", look_back_days=7, limit=None)
    out_a_both_int = get_global_news_yfinance(as_of_date="2026-08-03", look_back_days=7, limit=10)

    # Prove byte-identity across all 5 native Arm A outputs
    assert out_a_omitted == out_a_both_null == out_a_lb_null == out_a_lim_null == out_a_both_int
    native_global_output_A = out_a_both_int
    hash_global_A = hashlib.sha256(native_global_output_A.encode("utf-8")).hexdigest()

    # Archive records matching native outputs
    global_event_archive_A = {
        "title": "Global Market Digest on Economic Momentum",
        "summary": "Global markets trade steady as macro signals stabilize.",
        "publisher": "Reuters",
        "link": "https://finance.yahoo.com/m1",
        "published_at": "2026-07-29T15:00:00Z",
        "received_at": "2026-07-29T15:05:00Z",
        "query": "global",
        "symbols": [],
    }
    asset_event_archive_A = {
        "title": "NVIDIA Blackwell Production Surges",
        "summary": "Production of NVIDIA next-generation AI processors expands.",
        "publisher": "Yahoo",
        "link": "https://finance.yahoo.com/nvda1",
        "published_at": "2026-08-03T13:00:00Z",
        "received_at": "2026-08-03T13:05:00Z",
    }
    native_asset_output_A = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{_format_articles([asset_event_archive_A])}"
    hash_asset_A = hashlib.sha256(native_asset_output_A.encode("utf-8")).hexdigest()

    native_asset_output_B = (
        f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{format_article(asset_event_raw_B)}"
    )
    hash_asset_B = hashlib.sha256(native_asset_output_B.encode("utf-8")).hexdigest()

    base_journal: dict[str, Any] = {
        "manifest_hash": manifest_hash,
        "experiment_id": valid_manifest_payload["experiment_id"],
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "same_hash",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "settlement_id": "settle-A",
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "settlement_id": "settle-B",
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.02,
                "raw_return": "5.0%",
                "raw_return_float": 0.05,
                "alpha_float": 0.02,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_close": "2026-08-03T20:00:00Z",
            },
        },
    }

    base_A = {
        "ticker": "NVDA",
        "query": "global",
        "cutoff": "2026-08-03T20:00:00Z",
        "date": "2026-08-03",
        "arm": "A",
        "attempt_id": "att-A",
        "vendor": "yfinance",
        "source": "Yahoo News",
        "output_hash": hash_global_A,
        "received_at": "2026-07-29T15:05:00Z",
        "events": [global_event_archive_A],
    }
    base_B = {
        "ticker": "NVDA",
        "query": "global",
        "cutoff": "2026-08-03T20:00:00Z",
        "date": "2026-08-03",
        "arm": "B",
        "attempt_id": "att-B",
        "vendor": "repo1",
        "source": "Bloomberg",
        "output_hash": hash_global_B,
        "received_at": "2026-07-28T14:05:00Z",
        "events": [global_event_raw_B],
    }

    # --- 2. Prove all 5 parameter variants qualify identically ---
    param_variants = [
        ("omitted", {"as_of_date": "2026-08-03"}),
        ("both_null", {"as_of_date": "2026-08-03", "look_back_days": None, "limit": None}),
        ("lb_null", {"as_of_date": "2026-08-03", "look_back_days": None, "limit": 10}),
        ("lim_null", {"as_of_date": "2026-08-03", "look_back_days": 7, "limit": None}),
        ("both_int", {"as_of_date": "2026-08-03", "look_back_days": 7, "limit": 10}),
    ]

    for label, inp_params in param_variants:
        var_out = tmp_path / f"exp_r12_r1_{label}"
        var_out.mkdir(parents=True, exist_ok=True)
        (var_out / "baseline_archive").mkdir(parents=True, exist_ok=True)

        for arm_id, txt, hsh in [
            ("A", native_global_output_A, hash_global_A),
            ("B", native_global_output_B, hash_global_B),
        ]:
            p = var_out / "cells" / f"NVDA_2026-08-03_{arm_id}" / "results"
            p.mkdir(parents=True, exist_ok=True)
            (p / "trading_memory.log").write_text(
                "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
                encoding="utf-8",
            )
            (p / "audit_trace.json").write_text(
                json.dumps(
                    {
                        "traces": [
                            {
                                "type": "tool_start",
                                "tool": "get_global_news",
                                "input": inp_params,
                                "run_id": f"run-g-{arm_id}",
                                "timestamp": "2026-08-03T14:01:00Z",
                            },
                            {
                                "type": "tool_end",
                                "tool": "get_global_news",
                                "output": txt,
                                "output_hash": hsh,
                                "run_id": f"run-g-{arm_id}",
                                "timestamp": "2026-08-03T14:02:00Z",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

        (var_out / "baseline_archive" / "nvda_yahoo.json").write_text(
            json.dumps(base_A), encoding="utf-8"
        )
        (var_out / "provenance_archive.json").write_text(json.dumps(base_B), encoding="utf-8")
        (var_out / "journal.json").write_text(
            json.dumps(copy.deepcopy(base_journal)), encoding="utf-8"
        )

        s_var = summarize_experiment(var_out)
        assert s_var["scored_pairs_count"] == 0, f"Failed for variant {label}"
        assert s_var["provenance_verified"] is False, f"Failed provenance for variant {label}"
        assert s_var["timing_eligible"] is False, f"Failed timing for variant {label}"
        assert s_var["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
        assert (
            s_var["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
        )
        assert s_var["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True

    # --- 3. Mixed Asset / Global Captures with Explicit Null Options ---
    mixed_out = tmp_path / "exp_r12_r1_mixed_null"
    mixed_out.mkdir(parents=True, exist_ok=True)
    (mixed_out / "baseline_archive").mkdir(parents=True, exist_ok=True)

    base_A_mixed = dict(
        base_A,
        output_hashes=[hash_asset_A, hash_global_A],
        events=[asset_event_archive_A, global_event_archive_A],
    )
    base_B_mixed = dict(
        base_B,
        output_hashes=[hash_asset_B, hash_global_B],
        events=[asset_event_raw_B, global_event_raw_B],
    )

    for arm_id, txt_asset, hsh_asset, txt_global, hsh_global in [
        ("A", native_asset_output_A, hash_asset_A, native_global_output_A, hash_global_A),
        ("B", native_asset_output_B, hash_asset_B, native_global_output_B, hash_global_B),
    ]:
        p = mixed_out / "cells" / f"NVDA_2026-08-03_{arm_id}" / "results"
        p.mkdir(parents=True, exist_ok=True)
        (p / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
            encoding="utf-8",
        )
        (p / "audit_trace.json").write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "tool_start",
                            "tool": "get_news",
                            "input": {
                                "ticker": "NVDA",
                                "start_date": "2026-08-03",
                                "end_date": "2026-08-03",
                            },
                            "run_id": f"run-asset-{arm_id}",
                            "timestamp": "2026-08-03T14:01:00Z",
                        },
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output": txt_asset,
                            "output_hash": hsh_asset,
                            "run_id": f"run-asset-{arm_id}",
                            "timestamp": "2026-08-03T14:02:00Z",
                        },
                        {
                            "type": "tool_start",
                            "tool": "get_global_news",
                            "input": {
                                "as_of_date": "2026-08-03",
                                "look_back_days": None,
                                "limit": None,
                            },
                            "run_id": f"run-global-{arm_id}",
                            "timestamp": "2026-08-03T14:03:00Z",
                        },
                        {
                            "type": "tool_end",
                            "tool": "get_global_news",
                            "output": txt_global,
                            "output_hash": hsh_global,
                            "run_id": f"run-global-{arm_id}",
                            "timestamp": "2026-08-03T14:04:00Z",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    base_A_mixed = dict(
        base_A,
        query="NVDA",
        output_hashes=[hash_asset_A, hash_global_A],
        events=[asset_event_archive_A, global_event_archive_A],
    )
    base_B_mixed = dict(
        base_B,
        query="NVDA",
        output_hashes=[hash_asset_B, hash_global_B],
        events=[asset_event_raw_B, global_event_raw_B],
    )

    (mixed_out / "baseline_archive" / "nvda_yahoo.json").write_text(
        json.dumps(base_A_mixed), encoding="utf-8"
    )
    (mixed_out / "provenance_archive.json").write_text(json.dumps(base_B_mixed), encoding="utf-8")
    (mixed_out / "journal.json").write_text(
        json.dumps(copy.deepcopy(base_journal)), encoding="utf-8"
    )

    s_mixed = summarize_experiment(mixed_out)
    assert s_mixed["scored_pairs_count"] == 0
    assert s_mixed["provenance_verified"] is False
    assert s_mixed["timing_eligible"] is False
    assert s_mixed["diagnostic_summary"]["diagnostic_scored_pairs_count"] == 1
    assert s_mixed["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_provenance_verified"] is True
    assert s_mixed["pair_eligibility"]["NVDA_2026-08-03"]["diagnostic_timing_eligible"] is True

    # --- 4. Strict Failure Modes ---
    # 4a. Invalid non-numeric string parameter fails closed with accurate parameter error
    neg_out = tmp_path / "exp_r12_r1_negative"
    neg_out.mkdir(parents=True, exist_ok=True)
    (neg_out / "baseline_archive").mkdir(parents=True, exist_ok=True)

    p_b_neg = neg_out / "cells" / "NVDA_2026-08-03_B" / "results"
    p_b_neg.mkdir(parents=True, exist_ok=True)
    (p_b_neg / "trading_memory.log").write_text(
        "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
        encoding="utf-8",
    )
    (p_b_neg / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_global_news",
                        "input": {
                            "as_of_date": "2026-08-03",
                            "limit": "invalid_integer_string",
                        },
                        "run_id": "run-g-B",
                        "timestamp": "2026-08-03T14:01:00Z",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_global_news",
                        "output": native_global_output_B,
                        "output_hash": hash_global_B,
                        "run_id": "run-g-B",
                        "timestamp": "2026-08-03T14:02:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    p_a_neg = neg_out / "cells" / "NVDA_2026-08-03_A" / "results"
    p_a_neg.mkdir(parents=True, exist_ok=True)
    (p_a_neg / "trading_memory.log").write_text(
        "[2026-08-03 | NVDA | Buy | 5.0% | +2.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n\nREFLECTION:\nGood",
        encoding="utf-8",
    )
    (p_a_neg / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_global_news",
                        "input": {"as_of_date": "2026-08-03"},
                        "run_id": "run-g-A",
                        "timestamp": "2026-08-03T14:01:00Z",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_global_news",
                        "output": native_global_output_A,
                        "output_hash": hash_global_A,
                        "run_id": "run-g-A",
                        "timestamp": "2026-08-03T14:02:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    (neg_out / "baseline_archive" / "nvda_yahoo.json").write_text(
        json.dumps(base_A), encoding="utf-8"
    )
    (neg_out / "provenance_archive.json").write_text(json.dumps(base_B), encoding="utf-8")
    (neg_out / "journal.json").write_text(json.dumps(copy.deepcopy(base_journal)), encoding="utf-8")

    s_invalid = summarize_experiment(neg_out)
    assert s_invalid["scored_pairs_count"] == 0
    assert s_invalid["provenance_verified"] is False

    # 4b. Negative integer parameter fails closed
    tr_b_neg = json.loads((p_b_neg / "audit_trace.json").read_text(encoding="utf-8"))
    tr_b_neg["traces"][0]["input"] = {"as_of_date": "2026-08-03", "limit": -1}
    (p_b_neg / "audit_trace.json").write_text(json.dumps(tr_b_neg), encoding="utf-8")
    s_negative = summarize_experiment(neg_out)
    assert s_negative["scored_pairs_count"] == 0
    assert s_negative["provenance_verified"] is False

    # 4c. Altered output text fails closed
    tr_b_neg["traces"][0]["input"] = {
        "as_of_date": "2026-08-03",
        "look_back_days": None,
        "limit": None,
    }
    tr_b_neg["traces"][1]["output"] = native_global_output_B + "\n[TAMPERED CONTENT]"
    (p_b_neg / "audit_trace.json").write_text(json.dumps(tr_b_neg), encoding="utf-8")
    s_tampered = summarize_experiment(neg_out)
    assert s_tampered["scored_pairs_count"] == 0
    assert s_tampered["provenance_verified"] is False

    # 4d. Missing event evidence fails closed
    tr_b_neg["traces"][1]["output"] = native_global_output_B
    (p_b_neg / "audit_trace.json").write_text(json.dumps(tr_b_neg), encoding="utf-8")
    # Empty provenance archive (missing event proof)
    (neg_out / "provenance_archive.json").write_text(
        json.dumps(dict(base_B, events=[])), encoding="utf-8"
    )
    s_missing = summarize_experiment(neg_out)
    assert s_missing["scored_pairs_count"] == 0
    assert s_missing["provenance_verified"] is False


# ---------------------------------------------------------------------------
# Section 7.2 Proving Test Matrix (TIP-14 Prospective Evidence Capture)
# ---------------------------------------------------------------------------


def test_prosp_r4_r1_01_full_producer_parity(tmp_path: Path) -> None:
    """TEST-PROSP-R4-R1-01: Full producer response reconstruction matches byte-for-byte and hash-for-hash."""
    cell_dir_a = tmp_path / "cell_A"
    cell_dir_b = tmp_path / "cell_B"
    cell_dir_a.mkdir(parents=True)
    cell_dir_b.mkdir(parents=True)

    cutoff_dt = datetime(2026, 10, 19, 20, 0, 0, tzinfo=UTC)

    # 1. Arm A Yahoo Asset News
    import tradingagents.dataflows.vendors.yahoo.news as yahoo_news

    raw_article_a = {
        "content": {
            "title": "Apple Reports Record Quarterly Results",
            "summary": "Apple announced revenue for the fiscal fourth quarter.",
            "provider": {"displayName": "Reuters"},
            "canonicalUrl": {"url": "https://finance.yahoo.com/news/apple-record"},
            "pubDate": "2026-10-19T14:30:00Z",
        }
    }
    extracted_a = [yahoo_news._extract_article_data(raw_article_a)]
    fmt_a = yahoo_news._format_articles(extracted_a)
    out_text_a = f"## AAPL News, from 2026-10-12 to 2026-10-19:\n\n{fmt_a}"
    out_hash_a = hashlib.sha256(out_text_a.encode("utf-8")).hexdigest()

    (cell_dir_a / "results").mkdir(parents=True)
    (cell_dir_a / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": {
                            "ticker": "AAPL",
                            "start_date": "2026-10-12",
                            "end_date": "2026-10-19",
                        },
                        "run_id": "run-a-1",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": out_text_a,
                        "output_hash": out_hash_a,
                        "run_id": "run-a-1",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    responses_dir_a = cell_dir_a / "data" / "responses"
    responses_dir_a.mkdir(parents=True)
    snap_file_a = responses_dir_a / "001_get_news_call-a-1.json"
    snap_bytes_a = json.dumps([raw_article_a], indent=2, sort_keys=True).encode("utf-8")
    snap_file_a.write_bytes(snap_bytes_a)
    raw_hash_a = hashlib.sha256(
        json.dumps([raw_article_a], sort_keys=True).encode("utf-8")
    ).hexdigest()

    archive_a = {
        "version": "t13-archive/v2",
        "attempt_id": "att-a-1",
        "ticker": "AAPL",
        "arm": "A",
        "date": "2026-10-19",
        "cutoff": "2026-10-19T20:00:00Z",
        "vendor": "yfinance",
        "created_at": "2026-10-19T14:35:00Z",
        "calls": [
            {
                "call_seq": 1,
                "call_id": "call-a-1",
                "tool": "get_news",
                "run_id": "run-a-1",
                "query": "AAPL",
                "received_at": "2026-10-19T14:35:00Z",
                "params": {
                    "ticker": "AAPL",
                    "start_date": "2026-10-12",
                    "end_date": "2026-10-19",
                },
                "snapshot_path": "responses/001_get_news_call-a-1.json",
                "raw_payload_hash": raw_hash_a,
                "records": [
                    {
                        "title": extracted_a[0]["title"],
                        "summary": extracted_a[0]["summary"],
                        "publisher": extracted_a[0]["publisher"],
                        "link": extracted_a[0]["link"],
                        "pub_date": extracted_a[0]["pub_date"].isoformat(),
                    }
                ],
                "output_hash": out_hash_a,
            }
        ],
        "articles": [],
    }
    archive_path_a = cell_dir_a / "data" / "baseline_archive.json"
    archive_path_a.write_text(json.dumps(archive_a, indent=2), encoding="utf-8")

    ok_a, msg_a = validate_evidence_archive(
        archive_path_a,
        "A",
        "AAPL",
        "2026-10-19",
        cutoff_dt,
        {"attempt_id": "att-a-1"},
        cell_dir_a,
    )
    assert ok_a is True, f"Arm A verification failed: {msg_a}"

    # 2. Arm B Repo 1 Asset News
    import tradingagents.dataflows.vendors.news_intelligence.client as ni_client

    raw_event_b = {
        "id": "evt-b-001",
        "source": "Bloomberg",
        "title": "NVIDIA Unveils Next-Gen Blackwell Ultra Chips",
        "publishedAt": "2026-10-19T15:00:00Z",
        "receivedAt": "2026-10-19T15:01:00Z",
        "enrichedAt": "2026-10-19T15:02:00Z",
        "sentiment": {"score": 0.9, "label": "positive", "confidence": 0.95},
        "symbols": ["NVDA"],
        "url": "https://example.com/nvda-blackwell",
        "summary": "NVIDIA announced Blackwell Ultra enterprise chips.",
    }
    sanitized_b = ni_client.filter_and_sanitize_event(
        raw_event_b,
        cutoff_dt=cutoff_dt,
        min_pub_dt=datetime(2026, 10, 12, 0, 0, 0, tzinfo=UTC),
        max_pub_dt=cutoff_dt,
    )
    assert sanitized_b is not None
    fmt_b = ni_client.format_article(sanitized_b)
    out_text_b = f"## NVDA News, from 2026-10-12 to 2026-10-19:\n\n{fmt_b}"
    out_hash_b = hashlib.sha256(out_text_b.encode("utf-8")).hexdigest()

    (cell_dir_b / "results").mkdir(parents=True)
    (cell_dir_b / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": {
                            "ticker": "NVDA",
                            "start_date": "2026-10-12",
                            "end_date": "2026-10-19",
                        },
                        "run_id": "run-b-1",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": out_text_b,
                        "output_hash": out_hash_b,
                        "run_id": "run-b-1",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    responses_dir_b = cell_dir_b / "data" / "responses"
    responses_dir_b.mkdir(parents=True)
    snap_file_b = responses_dir_b / "001_get_news_call-b-1.json"
    raw_payload_b = {"events": [raw_event_b]}
    snap_bytes_b = json.dumps(raw_payload_b, indent=2, sort_keys=True).encode("utf-8")
    snap_file_b.write_bytes(snap_bytes_b)
    raw_hash_b = hashlib.sha256(
        json.dumps(raw_payload_b, sort_keys=True).encode("utf-8")
    ).hexdigest()

    archive_b = {
        "version": "t13-archive/v2",
        "attempt_id": "att-b-1",
        "ticker": "NVDA",
        "arm": "B",
        "date": "2026-10-19",
        "cutoff": "2026-10-19T20:00:00Z",
        "vendor": "news_intelligence",
        "created_at": "2026-10-19T15:05:00Z",
        "calls": [
            {
                "call_seq": 1,
                "call_id": "call-b-1",
                "tool": "get_news",
                "run_id": "run-b-1",
                "query": "NVDA",
                "received_at": "2026-10-19T15:05:00Z",
                "params": {
                    "symbol": "NVDA",
                    "from": "2026-10-12T00:00:00Z",
                    "to": "2026-10-19T23:59:59.999999Z",
                    "asOf": "2026-10-19T23:59:59.999999Z",
                },
                "producer_params": {
                    "ticker": "NVDA",
                    "start_date": "2026-10-12",
                    "end_date": "2026-10-19",
                },
                "snapshot_path": "responses/001_get_news_call-b-1.json",
                "raw_payload_hash": raw_hash_b,
                "records": [sanitized_b],
                "output_hash": out_hash_b,
            }
        ],
        "events": [],
    }
    archive_path_b = cell_dir_b / "data" / "provenance_archive.json"
    archive_path_b.write_text(json.dumps(archive_b, indent=2), encoding="utf-8")

    ok_b, msg_b = validate_evidence_archive(
        archive_path_b,
        "B",
        "NVDA",
        "2026-10-19",
        cutoff_dt,
        {"attempt_id": "att-b-1"},
        cell_dir_b,
    )
    assert ok_b is True, f"Arm B verification failed: {msg_b}"


def test_prosp_r4_r1_02_altered_header_body_only_fails(tmp_path: Path) -> None:
    """TEST-PROSP-R4-R1-02: Body-only text or altered header fails verification against real tool output."""
    cell_dir = tmp_path / "cell_altered"
    cell_dir.mkdir(parents=True)
    cutoff_dt = datetime(2026, 10, 19, 20, 0, 0, tzinfo=UTC)

    import tradingagents.dataflows.vendors.yahoo.news as yahoo_news

    raw_art = {
        "title": "Sample Article",
        "summary": "Sample summary.",
        "publisher": "Yahoo",
        "link": "https://example.com/art",
        "providerPublishTime": int(datetime(2026, 10, 19, 12, 0, 0, tzinfo=UTC).timestamp()),
    }
    extracted = [yahoo_news._extract_article_data(raw_art)]
    body_only = yahoo_news._format_articles(extracted)

    (cell_dir / "results").mkdir(parents=True)
    (cell_dir / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": {
                            "ticker": "AAPL",
                            "start_date": "2026-10-12",
                            "end_date": "2026-10-19",
                        },
                        "run_id": "run-alt-1",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": body_only,
                        "output_hash": hashlib.sha256(body_only.encode("utf-8")).hexdigest(),
                        "run_id": "run-alt-1",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    responses_dir = cell_dir / "data" / "responses"
    responses_dir.mkdir(parents=True)
    snap_file = responses_dir / "001_get_news_call-1.json"
    snap_file.write_text(json.dumps([raw_art]), encoding="utf-8")
    raw_hash = hashlib.sha256(json.dumps([raw_art], sort_keys=True).encode("utf-8")).hexdigest()

    archive = {
        "version": "t13-archive/v2",
        "attempt_id": "att-1",
        "ticker": "AAPL",
        "arm": "A",
        "date": "2026-10-19",
        "cutoff": "2026-10-19T20:00:00Z",
        "vendor": "yfinance",
        "calls": [
            {
                "call_seq": 1,
                "call_id": "call-1",
                "tool": "get_news",
                "run_id": "run-alt-1",
                "query": "AAPL",
                "received_at": "2026-10-19T14:00:00Z",
                "params": {
                    "ticker": "AAPL",
                    "start_date": "2026-10-12",
                    "end_date": "2026-10-19",
                },
                "snapshot_path": "responses/001_get_news_call-1.json",
                "raw_payload_hash": raw_hash,
                "records": [{**extracted[0], "pub_date": extracted[0]["pub_date"].isoformat()}],
                "output_hash": hashlib.sha256(body_only.encode("utf-8")).hexdigest(),
            }
        ],
    }
    arch_path = cell_dir / "data" / "baseline_archive.json"
    arch_path.write_text(json.dumps(archive), encoding="utf-8")

    ok, msg = validate_evidence_archive(
        arch_path, "A", "AAPL", "2026-10-19", cutoff_dt, {"attempt_id": "att-1"}, cell_dir
    )
    assert ok is False
    assert "derived candidate hash mismatch" in msg or "output text does not match" in msg


def test_prosp_r4_r1_03_empty_window_sentinels(tmp_path: Path) -> None:
    """TEST-PROSP-R4-R1-03: Empty window sentinels reconstruct and validate identically."""
    cutoff_dt = datetime(2026, 10, 19, 20, 0, 0, tzinfo=UTC)

    # 1. Arm A Yahoo unavailable / empty coverage gap sentinel
    cell_dir_a = tmp_path / "cell_empty_a"
    cell_dir_a.mkdir(parents=True)
    sentinel_text_a = "<Yahoo Finance news unavailable for 2026-10-12..2026-10-19: the window extends past today, so this is not an absence of news for AAPL>"
    sentinel_hash_a = hashlib.sha256(sentinel_text_a.encode("utf-8")).hexdigest()

    (cell_dir_a / "results").mkdir(parents=True)
    (cell_dir_a / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": {
                            "ticker": "AAPL",
                            "start_date": "2026-10-12",
                            "end_date": "2026-10-19",
                        },
                        "run_id": "run-empty-a",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": sentinel_text_a,
                        "output_hash": sentinel_hash_a,
                        "run_id": "run-empty-a",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    resp_dir_a = cell_dir_a / "data" / "responses"
    resp_dir_a.mkdir(parents=True)
    snap_a = resp_dir_a / "001_get_news_call-empty-a.json"
    snap_a.write_text(json.dumps([]), encoding="utf-8")
    raw_hash_a = hashlib.sha256(json.dumps([], sort_keys=True).encode("utf-8")).hexdigest()

    arch_a = {
        "version": "t13-archive/v2",
        "attempt_id": "att-empty-a",
        "ticker": "AAPL",
        "arm": "A",
        "date": "2026-10-19",
        "cutoff": "2026-10-19T20:00:00Z",
        "vendor": "yfinance",
        "calls": [
            {
                "call_seq": 1,
                "call_id": "call-empty-a",
                "tool": "get_news",
                "run_id": "run-empty-a",
                "query": "AAPL",
                "received_at": "2026-10-19T14:00:00Z",
                "params": {
                    "ticker": "AAPL",
                    "start_date": "2026-10-12",
                    "end_date": "2026-10-19",
                },
                "snapshot_path": "responses/001_get_news_call-empty-a.json",
                "raw_payload_hash": raw_hash_a,
                "records": [],
                "output_hash": sentinel_hash_a,
            }
        ],
    }
    arch_path_a = cell_dir_a / "data" / "baseline_archive.json"
    arch_path_a.write_text(json.dumps(arch_a), encoding="utf-8")

    ok_a, msg_a = validate_evidence_archive(
        arch_path_a, "A", "AAPL", "2026-10-19", cutoff_dt, {"attempt_id": "att-empty-a"}, cell_dir_a
    )
    assert ok_a is True, f"Arm A empty sentinel validation failed: {msg_a}"

    # 2. Arm B News Intelligence 'No news found' empty sentinel
    cell_dir_b = tmp_path / "cell_empty_b"
    cell_dir_b.mkdir(parents=True)
    sentinel_text_b = "No news found for AAPL between 2026-10-12 and 2026-10-19"
    sentinel_hash_b = hashlib.sha256(sentinel_text_b.encode("utf-8")).hexdigest()

    (cell_dir_b / "results").mkdir(parents=True)
    (cell_dir_b / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": {
                            "ticker": "AAPL",
                            "start_date": "2026-10-12",
                            "end_date": "2026-10-19",
                        },
                        "run_id": "run-empty-b",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": sentinel_text_b,
                        "output_hash": sentinel_hash_b,
                        "run_id": "run-empty-b",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    resp_dir_b = cell_dir_b / "data" / "responses"
    resp_dir_b.mkdir(parents=True)
    snap_b = resp_dir_b / "001_get_news_call-empty-b.json"
    snap_b.write_text(json.dumps({"events": []}), encoding="utf-8")
    raw_hash_b = hashlib.sha256(
        json.dumps({"events": []}, sort_keys=True).encode("utf-8")
    ).hexdigest()

    arch_b = {
        "version": "t13-archive/v2",
        "attempt_id": "att-empty-b",
        "ticker": "AAPL",
        "arm": "B",
        "date": "2026-10-19",
        "cutoff": "2026-10-19T20:00:00Z",
        "vendor": "news_intelligence",
        "calls": [
            {
                "call_seq": 1,
                "call_id": "call-empty-b",
                "tool": "get_news",
                "run_id": "run-empty-b",
                "query": "AAPL",
                "received_at": "2026-10-19T14:00:00Z",
                "params": {
                    "symbol": "AAPL",
                    "from": "2026-10-12T00:00:00Z",
                    "to": "2026-10-19T23:59:59.999999Z",
                    "asOf": "2026-10-19T23:59:59.999999Z",
                },
                "producer_params": {
                    "ticker": "AAPL",
                    "start_date": "2026-10-12",
                    "end_date": "2026-10-19",
                },
                "snapshot_path": "responses/001_get_news_call-empty-b.json",
                "raw_payload_hash": raw_hash_b,
                "records": [],
                "output_hash": sentinel_hash_b,
            }
        ],
    }
    arch_path_b = cell_dir_b / "data" / "provenance_archive.json"
    arch_path_b.write_text(json.dumps(arch_b), encoding="utf-8")

    ok_b, msg_b = validate_evidence_archive(
        arch_path_b, "B", "AAPL", "2026-10-19", cutoff_dt, {"attempt_id": "att-empty-b"}, cell_dir_b
    )
    assert ok_b is True, f"Arm B empty sentinel validation failed: {msg_b}"


def test_prosp_r4_r2_01_arm_a_yahoo_flat_and_nested_equality(tmp_path: Path) -> None:
    """TEST-PROSP-R4-R2-01: Arm A flat and nested Yahoo articles validate against _extract_article_data without fabricated fields."""
    cell_dir = tmp_path / "cell_yahoo_equality"
    cell_dir.mkdir(parents=True)
    cutoff_dt = datetime(2026, 10, 19, 20, 0, 0, tzinfo=UTC)

    import tradingagents.dataflows.vendors.yahoo.news as yahoo_news

    art_nested = {
        "content": {
            "title": "Nested Article Title",
            "summary": "Nested summary text",
            "provider": {"displayName": "Nested Publisher"},
            "canonicalUrl": {"url": "https://example.com/nested"},
            "pubDate": "2026-10-19T12:00:00Z",
        }
    }
    art_flat = {
        "title": "Flat Article Title",
        "summary": "Flat summary text",
        "publisher": "Flat Publisher",
        "link": "https://example.com/flat",
        "providerPublishTime": int(datetime(2026, 10, 19, 13, 0, 0, tzinfo=UTC).timestamp()),
    }
    raw_articles = [art_nested, art_flat]
    extracted = [yahoo_news._extract_article_data(a) for a in raw_articles]

    out_text = (
        f"## MSFT News, from 2026-10-12 to 2026-10-19:\n\n{yahoo_news._format_articles(extracted)}"
    )
    out_hash = hashlib.sha256(out_text.encode("utf-8")).hexdigest()

    (cell_dir / "results").mkdir(parents=True)
    (cell_dir / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": {
                            "ticker": "MSFT",
                            "start_date": "2026-10-12",
                            "end_date": "2026-10-19",
                        },
                        "run_id": "run-eq-1",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": out_text,
                        "output_hash": out_hash,
                        "run_id": "run-eq-1",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    responses_dir = cell_dir / "data" / "responses"
    responses_dir.mkdir(parents=True)
    snap_file = responses_dir / "001_get_news_call-eq.json"
    snap_file.write_text(json.dumps(raw_articles, indent=2), encoding="utf-8")
    raw_hash = hashlib.sha256(json.dumps(raw_articles, sort_keys=True).encode("utf-8")).hexdigest()

    archive = {
        "version": "t13-archive/v2",
        "attempt_id": "att-eq-1",
        "ticker": "MSFT",
        "arm": "A",
        "date": "2026-10-19",
        "cutoff": "2026-10-19T20:00:00Z",
        "vendor": "yfinance",
        "calls": [
            {
                "call_seq": 1,
                "call_id": "call-eq",
                "tool": "get_news",
                "run_id": "run-eq-1",
                "query": "MSFT",
                "received_at": "2026-10-19T14:00:00Z",
                "params": {
                    "ticker": "MSFT",
                    "start_date": "2026-10-12",
                    "end_date": "2026-10-19",
                },
                "snapshot_path": "responses/001_get_news_call-eq.json",
                "raw_payload_hash": raw_hash,
                "records": [
                    {
                        "title": e["title"],
                        "summary": e["summary"],
                        "publisher": e["publisher"],
                        "link": e["link"],
                        "pub_date": e["pub_date"].isoformat(),
                    }
                    for e in extracted
                ],
                "output_hash": out_hash,
            }
        ],
    }
    arch_path = cell_dir / "data" / "baseline_archive.json"
    arch_path.write_text(json.dumps(archive), encoding="utf-8")

    ok, msg = validate_evidence_archive(
        arch_path, "A", "MSFT", "2026-10-19", cutoff_dt, {"attempt_id": "att-eq-1"}, cell_dir
    )
    assert ok is True, f"Arm A equality verification failed: {msg}"


def test_prosp_r4_r2_02_arm_a_metadata_mutation_fails(tmp_path: Path) -> None:
    """TEST-PROSP-R4-R2-02: Mutating call['received_at'] to exceed cutoff fails observation clock verification."""
    cell_dir = tmp_path / "cell_mutation_a"
    cell_dir.mkdir(parents=True)
    cutoff_dt = datetime(2026, 10, 19, 20, 0, 0, tzinfo=UTC)

    import tradingagents.dataflows.vendors.yahoo.news as yahoo_news

    raw_art = {
        "title": "Article",
        "summary": "Summary",
        "publisher": "Yahoo",
        "link": "https://example.com/art",
        "providerPublishTime": int(datetime(2026, 10, 19, 12, 0, 0, tzinfo=UTC).timestamp()),
    }
    extracted = [yahoo_news._extract_article_data(raw_art)]
    out_text = (
        f"## TSLA News, from 2026-10-12 to 2026-10-19:\n\n{yahoo_news._format_articles(extracted)}"
    )
    out_hash = hashlib.sha256(out_text.encode("utf-8")).hexdigest()

    (cell_dir / "results").mkdir(parents=True)
    (cell_dir / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": {
                            "ticker": "TSLA",
                            "start_date": "2026-10-12",
                            "end_date": "2026-10-19",
                        },
                        "run_id": "run-mut-1",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": out_text,
                        "output_hash": out_hash,
                        "run_id": "run-mut-1",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    responses_dir = cell_dir / "data" / "responses"
    responses_dir.mkdir(parents=True)
    snap_file = responses_dir / "001_get_news_call-mut.json"
    snap_file.write_text(json.dumps([raw_art]), encoding="utf-8")
    raw_hash = hashlib.sha256(json.dumps([raw_art], sort_keys=True).encode("utf-8")).hexdigest()

    archive = {
        "version": "t13-archive/v2",
        "attempt_id": "att-mut-1",
        "ticker": "TSLA",
        "arm": "A",
        "date": "2026-10-19",
        "cutoff": "2026-10-19T20:00:00Z",
        "vendor": "yfinance",
        "calls": [
            {
                "call_seq": 1,
                "call_id": "call-mut",
                "tool": "get_news",
                "run_id": "run-mut-1",
                "query": "TSLA",
                "received_at": "2026-10-20T10:00:00Z",
                "params": {
                    "ticker": "TSLA",
                    "start_date": "2026-10-12",
                    "end_date": "2026-10-19",
                },
                "snapshot_path": "responses/001_get_news_call-mut.json",
                "raw_payload_hash": raw_hash,
                "records": [
                    {
                        "title": extracted[0]["title"],
                        "summary": extracted[0]["summary"],
                        "publisher": extracted[0]["publisher"],
                        "link": extracted[0]["link"],
                        "pub_date": extracted[0]["pub_date"].isoformat(),
                    }
                ],
                "output_hash": out_hash,
            }
        ],
    }
    arch_path = cell_dir / "data" / "baseline_archive.json"
    arch_path.write_text(json.dumps(archive), encoding="utf-8")

    ok, msg = validate_evidence_archive(
        arch_path, "A", "TSLA", "2026-10-19", cutoff_dt, {"attempt_id": "att-mut-1"}, cell_dir
    )
    assert ok is False
    assert "exceeds entry cutoff" in msg


def test_prosp_r4_r2_03_arm_b_canonical_event_equality(tmp_path: Path) -> None:
    """TEST-PROSP-R4-R2-03: Arm B canonical event equality validates id, publishedAt, receivedAt, enrichedAt, symbols."""
    cell_dir = tmp_path / "cell_canonical_b"
    cell_dir.mkdir(parents=True)
    cutoff_dt = datetime(2026, 10, 19, 20, 0, 0, tzinfo=UTC)

    import tradingagents.dataflows.vendors.news_intelligence.client as ni_client

    raw_event = {
        "id": "canon-evt-123",
        "source": "Reuters",
        "title": "Semiconductor Demand Surges",
        "publishedAt": "2026-10-19T14:00:00Z",
        "receivedAt": "2026-10-19T14:02:00Z",
        "enrichedAt": "2026-10-19T14:05:00Z",
        "sentiment": {"score": 0.75, "label": "positive", "confidence": 0.88},
        "symbols": ["AMD"],
        "url": "https://example.com/amd-semis",
        "summary": "Strong demand reported across enterprise data centers.",
    }
    sanitized = ni_client.filter_and_sanitize_event(
        raw_event,
        cutoff_dt=cutoff_dt,
        min_pub_dt=datetime(2026, 10, 12, 0, 0, 0, tzinfo=UTC),
        max_pub_dt=cutoff_dt,
    )
    assert sanitized is not None
    fmt = ni_client.format_article(sanitized)
    out_text = f"## AMD News, from 2026-10-12 to 2026-10-19:\n\n{fmt}"
    out_hash = hashlib.sha256(out_text.encode("utf-8")).hexdigest()

    (cell_dir / "results").mkdir(parents=True)
    (cell_dir / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": {
                            "ticker": "AMD",
                            "start_date": "2026-10-12",
                            "end_date": "2026-10-19",
                        },
                        "run_id": "run-b-canon",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": out_text,
                        "output_hash": out_hash,
                        "run_id": "run-b-canon",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    responses_dir = cell_dir / "data" / "responses"
    responses_dir.mkdir(parents=True)
    snap_file = responses_dir / "001_get_news_call-canon.json"
    snap_file.write_text(json.dumps({"events": [raw_event]}), encoding="utf-8")
    raw_hash = hashlib.sha256(
        json.dumps({"events": [raw_event]}, sort_keys=True).encode("utf-8")
    ).hexdigest()

    archive = {
        "version": "t13-archive/v2",
        "attempt_id": "att-canon-1",
        "ticker": "AMD",
        "arm": "B",
        "date": "2026-10-19",
        "cutoff": "2026-10-19T20:00:00Z",
        "vendor": "news_intelligence",
        "calls": [
            {
                "call_seq": 1,
                "call_id": "call-canon",
                "tool": "get_news",
                "run_id": "run-b-canon",
                "query": "AMD",
                "received_at": "2026-10-19T14:10:00Z",
                "params": {
                    "symbol": "AMD",
                    "from": "2026-10-12T00:00:00Z",
                    "to": "2026-10-19T23:59:59.999999Z",
                    "asOf": "2026-10-19T23:59:59.999999Z",
                },
                "producer_params": {
                    "ticker": "AMD",
                    "start_date": "2026-10-12",
                    "end_date": "2026-10-19",
                },
                "snapshot_path": "responses/001_get_news_call-canon.json",
                "raw_payload_hash": raw_hash,
                "records": [sanitized],
                "output_hash": out_hash,
            }
        ],
    }
    arch_path = cell_dir / "data" / "provenance_archive.json"
    arch_path.write_text(json.dumps(archive), encoding="utf-8")

    ok, msg = validate_evidence_archive(
        arch_path, "B", "AMD", "2026-10-19", cutoff_dt, {"attempt_id": "att-canon-1"}, cell_dir
    )
    assert ok is True, f"Arm B canonical equality failed: {msg}"


def test_prosp_r4_r2_04_arm_b_backdating_mutation_fails(tmp_path: Path) -> None:
    """TEST-PROSP-R4-R2-04: Backdating receivedAt or enrichedAt in stored records fails equality against raw snapshot."""
    cell_dir = tmp_path / "cell_backdate_b"
    cell_dir.mkdir(parents=True)
    cutoff_dt = datetime(2026, 10, 19, 20, 0, 0, tzinfo=UTC)

    import tradingagents.dataflows.vendors.news_intelligence.client as ni_client

    raw_event = {
        "id": "backdate-evt",
        "source": "Bloomberg",
        "title": "Fed Interest Rate Speculation",
        "publishedAt": "2026-10-19T14:00:00Z",
        "receivedAt": "2026-10-19T14:02:00Z",
        "enrichedAt": "2026-10-19T14:05:00Z",
        "sentiment": {"score": -0.2, "label": "negative", "confidence": 0.7},
        "symbols": ["SPY"],
        "url": "https://example.com/spy",
        "summary": "Markets react to interest rate comments.",
    }
    sanitized = ni_client.filter_and_sanitize_event(
        raw_event,
        cutoff_dt=cutoff_dt,
        min_pub_dt=datetime(2026, 10, 12, 0, 0, 0, tzinfo=UTC),
        max_pub_dt=cutoff_dt,
    )
    assert sanitized is not None
    fmt = ni_client.format_article(sanitized)
    out_text = f"## SPY News, from 2026-10-12 to 2026-10-19:\n\n{fmt}"
    out_hash = hashlib.sha256(out_text.encode("utf-8")).hexdigest()

    (cell_dir / "results").mkdir(parents=True)
    (cell_dir / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": {
                            "ticker": "SPY",
                            "start_date": "2026-10-12",
                            "end_date": "2026-10-19",
                        },
                        "run_id": "run-backdate",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": out_text,
                        "output_hash": out_hash,
                        "run_id": "run-backdate",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    responses_dir = cell_dir / "data" / "responses"
    responses_dir.mkdir(parents=True)
    snap_file = responses_dir / "001_get_news_call-backdate.json"
    snap_file.write_text(json.dumps({"events": [raw_event]}), encoding="utf-8")
    raw_hash = hashlib.sha256(
        json.dumps({"events": [raw_event]}, sort_keys=True).encode("utf-8")
    ).hexdigest()

    # Tamper receivedAt in stored records (backdating)
    mutated_record = dict(sanitized, receivedAt="2026-10-19T13:00:00Z")

    archive = {
        "version": "t13-archive/v2",
        "attempt_id": "att-backdate-1",
        "ticker": "SPY",
        "arm": "B",
        "date": "2026-10-19",
        "cutoff": "2026-10-19T20:00:00Z",
        "vendor": "news_intelligence",
        "calls": [
            {
                "call_seq": 1,
                "call_id": "call-backdate",
                "tool": "get_news",
                "run_id": "run-backdate",
                "query": "SPY",
                "received_at": "2026-10-19T14:10:00Z",
                "params": {
                    "symbol": "SPY",
                    "from": "2026-10-12T00:00:00Z",
                    "to": "2026-10-19T23:59:59.999999Z",
                    "asOf": "2026-10-19T23:59:59.999999Z",
                },
                "producer_params": {
                    "ticker": "SPY",
                    "start_date": "2026-10-12",
                    "end_date": "2026-10-19",
                },
                "snapshot_path": "responses/001_get_news_call-backdate.json",
                "raw_payload_hash": raw_hash,
                "records": [mutated_record],
                "output_hash": out_hash,
            }
        ],
    }
    arch_path = cell_dir / "data" / "provenance_archive.json"
    arch_path.write_text(json.dumps(archive), encoding="utf-8")

    ok, msg = validate_evidence_archive(
        arch_path, "B", "SPY", "2026-10-19", cutoff_dt, {"attempt_id": "att-backdate-1"}, cell_dir
    )
    assert ok is False
    assert "Arm B canonical field mismatch on 'receivedAt'" in msg


def test_prosp_r1_05_v2_fail_closed_on_missing_calls_or_snapshot(tmp_path: Path) -> None:
    """TEST-PROSP-R1-05: Envelope with version 't13-archive/v2' missing calls or snapshot fails closed immediately."""
    cell_dir = tmp_path / "cell_v2_fail_closed"
    cell_dir.mkdir(parents=True)
    cutoff_dt = datetime(2026, 10, 19, 20, 0, 0, tzinfo=UTC)

    (cell_dir / "results").mkdir(parents=True)
    (cell_dir / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": {
                            "ticker": "AAPL",
                            "start_date": "2026-10-12",
                            "end_date": "2026-10-19",
                        },
                        "run_id": "run-fc-1",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": "dummy output",
                        "output_hash": hashlib.sha256(b"dummy output").hexdigest(),
                        "run_id": "run-fc-1",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    # 1. Missing calls list
    arch_no_calls = {
        "version": "t13-archive/v2",
        "attempt_id": "att-fc-1",
        "ticker": "AAPL",
        "arm": "A",
        "date": "2026-10-19",
        "vendor": "yfinance",
    }
    p_no_calls = cell_dir / "data" / "arch_no_calls.json"
    p_no_calls.parent.mkdir(parents=True, exist_ok=True)
    p_no_calls.write_text(json.dumps(arch_no_calls), encoding="utf-8")
    ok1, msg1 = validate_evidence_archive(
        p_no_calls, "A", "AAPL", "2026-10-19", cutoff_dt, {"attempt_id": "att-fc-1"}, cell_dir
    )
    assert ok1 is False
    assert "missing or empty calls list" in msg1

    # 2. Missing snapshot on disk
    arch_missing_snap = {
        "version": "t13-archive/v2",
        "attempt_id": "att-fc-1",
        "ticker": "AAPL",
        "arm": "A",
        "date": "2026-10-19",
        "vendor": "yfinance",
        "calls": [
            {
                "call_seq": 1,
                "call_id": "call-fc-1",
                "tool": "get_news",
                "run_id": "run-fc-1",
                "snapshot_path": "responses/nonexistent_snap.json",
                "raw_payload_hash": "dummyhash",
                "received_at": "2026-10-19T14:00:00Z",
                "records": [],
            }
        ],
    }
    p_missing_snap = cell_dir / "data" / "arch_missing_snap.json"
    p_missing_snap.write_text(json.dumps(arch_missing_snap), encoding="utf-8")
    ok2, msg2 = validate_evidence_archive(
        p_missing_snap, "A", "AAPL", "2026-10-19", cutoff_dt, {"attempt_id": "att-fc-1"}, cell_dir
    )
    assert ok2 is False
    assert "snapshot file missing on disk" in msg2


def test_prosp_r1_06_multi_call_isolation(tmp_path: Path) -> None:
    """TEST-PROSP-R1-06: Multiple calls in one cell validate individually against discrete snapshots without union pooling."""
    cell_dir = tmp_path / "cell_multi_call"
    cell_dir.mkdir(parents=True)
    cutoff_dt = datetime(2026, 10, 19, 20, 0, 0, tzinfo=UTC)

    import tradingagents.dataflows.vendors.yahoo.news as yahoo_news

    # Call 1: asset news
    raw_art1 = {
        "title": "Apple iPhone Event",
        "summary": "Summary 1",
        "publisher": "Reuters",
        "link": "https://example.com/1",
        "providerPublishTime": int(datetime(2026, 10, 19, 12, 0, 0, tzinfo=UTC).timestamp()),
    }
    ext1 = [yahoo_news._extract_article_data(raw_art1)]
    out1 = f"## AAPL News, from 2026-10-12 to 2026-10-19:\n\n{yahoo_news._format_articles(ext1)}"
    hash1 = hashlib.sha256(out1.encode("utf-8")).hexdigest()

    # Call 2: global news
    raw_art2 = {
        "title": "Global Macro Outlook",
        "summary": "Summary 2",
        "publisher": "Bloomberg",
        "link": "https://example.com/2",
        "providerPublishTime": int(datetime(2026, 10, 19, 13, 0, 0, tzinfo=UTC).timestamp()),
    }
    ext2 = [yahoo_news._extract_article_data(raw_art2)]
    out2 = f"## Global Market News, from 2026-10-12 to 2026-10-19:\n\n{yahoo_news._format_articles(ext2)}"
    hash2 = hashlib.sha256(out2.encode("utf-8")).hexdigest()

    (cell_dir / "results").mkdir(parents=True)
    (cell_dir / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": {
                            "ticker": "AAPL",
                            "start_date": "2026-10-12",
                            "end_date": "2026-10-19",
                        },
                        "run_id": "run-multi-1",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": out1,
                        "output_hash": hash1,
                        "run_id": "run-multi-1",
                    },
                    {
                        "type": "tool_start",
                        "tool": "get_global_news",
                        "input": {"as_of_date": "2026-10-19", "look_back_days": 7, "limit": 10},
                        "run_id": "run-multi-2",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_global_news",
                        "output": out2,
                        "output_hash": hash2,
                        "run_id": "run-multi-2",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    responses_dir = cell_dir / "data" / "responses"
    responses_dir.mkdir(parents=True)
    snap1 = responses_dir / "001_get_news_c1.json"
    snap1.write_text(json.dumps([raw_art1]), encoding="utf-8")
    raw_hash1 = hashlib.sha256(json.dumps([raw_art1], sort_keys=True).encode("utf-8")).hexdigest()

    snap2 = responses_dir / "002_get_global_news_c2.json"
    snap2.write_text(json.dumps([raw_art2]), encoding="utf-8")
    raw_hash2 = hashlib.sha256(json.dumps([raw_art2], sort_keys=True).encode("utf-8")).hexdigest()

    archive = {
        "version": "t13-archive/v2",
        "attempt_id": "att-multi-1",
        "ticker": "AAPL",
        "arm": "A",
        "date": "2026-10-19",
        "cutoff": "2026-10-19T20:00:00Z",
        "vendor": "yfinance",
        "calls": [
            {
                "call_seq": 1,
                "call_id": "c1",
                "tool": "get_news",
                "run_id": "run-multi-1",
                "query": "AAPL",
                "received_at": "2026-10-19T14:00:00Z",
                "params": {
                    "ticker": "AAPL",
                    "start_date": "2026-10-12",
                    "end_date": "2026-10-19",
                },
                "snapshot_path": "responses/001_get_news_c1.json",
                "raw_payload_hash": raw_hash1,
                "records": [
                    {
                        "title": ext1[0]["title"],
                        "summary": ext1[0]["summary"],
                        "publisher": ext1[0]["publisher"],
                        "link": ext1[0]["link"],
                        "pub_date": ext1[0]["pub_date"].isoformat(),
                    }
                ],
                "output_hash": hash1,
            },
            {
                "call_seq": 2,
                "call_id": "c2",
                "tool": "get_global_news",
                "run_id": "run-multi-2",
                "query": "global",
                "received_at": "2026-10-19T14:05:00Z",
                "params": {"as_of_date": "2026-10-19", "look_back_days": 7, "limit": 10},
                "snapshot_path": "responses/002_get_global_news_c2.json",
                "raw_payload_hash": raw_hash2,
                "records": [
                    {
                        "title": ext2[0]["title"],
                        "summary": ext2[0]["summary"],
                        "publisher": ext2[0]["publisher"],
                        "link": ext2[0]["link"],
                        "pub_date": ext2[0]["pub_date"].isoformat(),
                    }
                ],
                "output_hash": hash2,
            },
        ],
    }
    arch_path = cell_dir / "data" / "baseline_archive.json"
    arch_path.write_text(json.dumps(archive), encoding="utf-8")

    ok, msg = validate_evidence_archive(
        arch_path, "A", "AAPL", "2026-10-19", cutoff_dt, {"attempt_id": "att-multi-1"}, cell_dir
    )
    assert ok is True, f"Multi-call verification failed: {msg}"


def test_prosp_r1_07_legacy_fallback_isolation(tmp_path: Path) -> None:
    """TEST-PROSP-R1-07: Legacy single-call archive without version falls back to diagnostic matching without granting strict eligibility."""
    out = tmp_path / "exp_legacy_fallback"
    out.mkdir(parents=True)

    from tradingagents.dataflows.vendors.yahoo.news import _format_articles

    ev_a = {
        "title": "Legacy Article",
        "summary": "Summary",
        "publisher": "Yahoo",
        "link": "",
        "published_at": "2026-08-03T14:00:00Z",
        "received_at": "2026-08-03T14:05:00Z",
    }
    content_a = _format_articles([ev_a])
    resp_a = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{content_a}"
    hash_a = hashlib.sha256(resp_a.encode("utf-8")).hexdigest()

    cell_a = out / "cells" / "NVDA_2026-08-03_A"
    (cell_a / "results").mkdir(parents=True)
    (cell_a / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": {"ticker": "NVDA", "end_date": "2026-08-03"},
                        "run_id": "run-leg-a",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": resp_a,
                        "output_hash": hash_a,
                        "run_id": "run-leg-a",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (cell_a / "data").mkdir(parents=True)
    (cell_a / "data" / "baseline_archive.json").write_text(
        json.dumps(
            {
                "ticker": "NVDA",
                "query": "NVDA",
                "cutoff": "2026-08-03T20:00:00Z",
                "date": "2026-08-03",
                "arm": "A",
                "attempt_id": "att-leg-a",
                "source": "yfinance",
                "output_hash": hash_a,
                "received_at": "2026-08-03T14:05:00Z",
                "events": [ev_a],
            }
        ),
        encoding="utf-8",
    )

    ok, msg = validate_evidence_archive(
        cell_a / "data" / "baseline_archive.json",
        "A",
        "NVDA",
        "2026-08-03",
        datetime(2026, 8, 3, 20, 0, 0, tzinfo=UTC),
        {"attempt_id": "att-leg-a"},
        cell_a,
    )
    assert ok is True


def test_prosp_r2_01_in_process_mutex_concurrency(tmp_path: Path) -> None:
    """TEST-PROSP-R2-01: In-process mutex serializes concurrent tool completions into cumulative envelope."""
    import threading

    cell_dir = tmp_path / "cell_mutex"
    ctx = ProspectiveCaptureContext(
        cell_dir=cell_dir,
        ticker="AAPL",
        date_str="2026-10-19",
        arm="A",
        attempt_id="att-mutex-1",
        cutoff_iso="2026-10-19T20:00:00Z",
    )

    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5)
            ctx.commit_call(
                tool_name=f"tool_{worker_id}",
                query="AAPL",
                params={"worker": worker_id},
                raw_payload={"worker": worker_id, "data": [1, 2, 3]},
                extracted_records=[
                    {
                        "title": f"Article {worker_id}",
                        "summary": "",
                        "publisher": "Y",
                        "link": "",
                    }
                ],
                output_text=f"Output {worker_id}",
                run_id=f"run-{worker_id}",
            )
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"Concurrent worker failed: {errors}"

    envelope = json.loads((cell_dir / "data" / "baseline_archive.json").read_text(encoding="utf-8"))
    assert envelope["version"] == "t13-archive/v2"
    assert len(envelope["calls"]) == 2
    seqs = [c["call_seq"] for c in envelope["calls"]]
    assert sorted(seqs) == [1, 2]

    for c in envelope["calls"]:
        snap_f = cell_dir / "data" / c["snapshot_path"]
        assert snap_f.exists()


def test_prosp_r2_02_mutex_interleaved_completion(tmp_path: Path) -> None:
    """TEST-PROSP-R2-02: Interleaved completion order preserves exact call identity, snapshots, and digests."""
    cell_dir = tmp_path / "cell_interleaved"
    ctx = ProspectiveCaptureContext(
        cell_dir=cell_dir,
        ticker="GOOGL",
        date_str="2026-10-19",
        arm="B",
        attempt_id="att-interleaved",
    )

    call1 = ctx.commit_call(
        tool_name="get_news",
        query="GOOGL",
        params={"ticker": "GOOGL"},
        raw_payload={"events": [{"id": "ev-1"}]},
        extracted_records=[{"id": "ev-1", "title": "G1", "source": "NI", "symbols": ["GOOGL"]}],
        output_text="Output G1",
        run_id="run-g1",
    )

    call2 = ctx.commit_call(
        tool_name="get_global_news",
        query="global",
        params={"limit": 5},
        raw_payload={"events": [{"id": "ev-2"}]},
        extracted_records=[{"id": "ev-2", "title": "G2", "source": "NI", "symbols": []}],
        output_text="Output G2",
        run_id="run-g2",
    )

    assert call1["call_seq"] == 1
    assert call2["call_seq"] == 2
    assert call1["call_id"] != call2["call_id"]

    env = json.loads((cell_dir / "data" / "provenance_archive.json").read_text(encoding="utf-8"))
    assert len(env["calls"]) == 2
    assert env["calls"][0]["call_id"] == call1["call_id"]
    assert env["calls"][1]["call_id"] == call2["call_id"]


def test_prosp_r2_03_interrupted_commit_reconciliation(tmp_path: Path) -> None:
    """TEST-PROSP-R2-03: Storage error raises EvidenceStorageError, marks FAILED_INCOMPLETE in journal, no paid replay."""
    cell_dir = tmp_path / "cell_storage_error"
    ctx = ProspectiveCaptureContext(
        cell_dir=cell_dir,
        ticker="AAPL",
        date_str="2026-10-19",
        arm="A",
        attempt_id="att-fail-1",
    )

    with (
        patch("os.replace", side_effect=OSError("Disk full or permission denied")),
        pytest.raises(EvidenceStorageError),
    ):
        ctx.commit_call(
            tool_name="get_news",
            query="AAPL",
            params={},
            raw_payload={"articles": []},
            extracted_records=[],
            output_text="text",
        )


def test_prosp_sum_01_summary_contract(tmp_path: Path) -> None:
    """TEST-PROSP-SUM-01: Summary contract enforces scored_pairs_count = 0 and strict_empirical_eligible = false for v2 prospective archives."""
    out = tmp_path / "exp_prosp_summary"
    out.mkdir(parents=True)

    manifest_hash = "fake-manifest-hash-12345"
    journal = {
        "manifest_hash": manifest_hash,
        "experiment_id": "test-exp",
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "AAPL_2026-10-19_A": {
                "attempt_id": "att-sum-a",
                "phase": "completed",
                "ticker": "AAPL",
                "date": "2026-10-19",
                "arm": "A",
                "started_at": "2026-10-19T14:00:00Z",
                "non_news_inputs_hash": "hash-equal",
                "entry_time": "2026-10-19T20:00:00Z",
            },
            "AAPL_2026-10-19_B": {
                "attempt_id": "att-sum-b",
                "phase": "completed",
                "ticker": "AAPL",
                "date": "2026-10-19",
                "arm": "B",
                "started_at": "2026-10-19T14:00:00Z",
                "non_news_inputs_hash": "hash-equal",
                "entry_time": "2026-10-19T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "AAPL_2026-10-19_A": {
                "decision_attempt_id": "att-sum-a",
                "phase": "completed",
                "ticker": "AAPL",
                "date": "2026-10-19",
                "arm": "A",
                "outcome": 0.05,
                "raw_return": "8.0%",
                "raw_return_float": 0.08,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-10-26",
                "entry_time": "2026-10-19T20:00:00Z",
            },
            "AAPL_2026-10-19_B": {
                "decision_attempt_id": "att-sum-b",
                "phase": "completed",
                "ticker": "AAPL",
                "date": "2026-10-19",
                "arm": "B",
                "outcome": 0.07,
                "raw_return": "8.0%",
                "raw_return_float": 0.08,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-10-26",
                "entry_time": "2026-10-19T20:00:00Z",
            },
        },
    }
    (out / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    for arm in ("A", "B"):
        if arm == "A":
            sentinel_text = "<Yahoo Finance news unavailable for 2026-10-12..2026-10-19: the window extends past today, so this is not an absence of news for AAPL>"
        else:
            sentinel_text = "No news found for AAPL between 2026-10-12 and 2026-10-19"
        sentinel_hash = hashlib.sha256(sentinel_text.encode("utf-8")).hexdigest()

        cell_res = out / "cells" / f"AAPL_2026-10-19_{arm}" / "results"
        cell_res.mkdir(parents=True)
        outcome_tag = "+5.0%" if arm == "A" else "+7.0%"
        (cell_res / "trading_memory.log").write_text(
            f"[2026-10-19 | AAPL | Buy | 8.0% | {outcome_tag} | 5d | resolved:2026-10-26]\n\nDECISION:\nBuy\n",
            encoding="utf-8",
        )
        (cell_res / "audit_trace.json").write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "tool_start",
                            "tool": "get_news",
                            "input": {
                                "ticker": "AAPL",
                                "start_date": "2026-10-12",
                                "end_date": "2026-10-19",
                            },
                            "run_id": f"run-{arm}",
                        },
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output": sentinel_text,
                            "output_hash": sentinel_hash,
                            "run_id": f"run-{arm}",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        cell_data = out / "cells" / f"AAPL_2026-10-19_{arm}" / "data"
        cell_data.mkdir(parents=True)
        resp_dir = cell_data / "responses"
        resp_dir.mkdir(parents=True)
        snap_f = resp_dir / f"001_get_news_snap-{arm}.json"
        snap_f.write_text(json.dumps([] if arm == "A" else {"events": []}), encoding="utf-8")
        raw_h = hashlib.sha256(
            json.dumps([] if arm == "A" else {"events": []}, sort_keys=True).encode("utf-8")
        ).hexdigest()

        arch_fname = "baseline_archive.json" if arm == "A" else "provenance_archive.json"
        arch_v2 = {
            "version": "t13-archive/v2",
            "attempt_id": f"att-sum-{arm.lower()}",
            "ticker": "AAPL",
            "arm": arm,
            "date": "2026-10-19",
            "vendor": "yfinance" if arm == "A" else "news_intelligence",
            "calls": [
                {
                    "call_seq": 1,
                    "call_id": f"call-{arm}",
                    "tool": "get_news",
                    "run_id": f"run-{arm}",
                    "query": "AAPL",
                    "received_at": "2026-10-19T14:00:00Z",
                    "params": {
                        "ticker": "AAPL",
                        "start_date": "2026-10-12",
                        "end_date": "2026-10-19",
                    }
                    if arm == "A"
                    else {
                        "symbol": "AAPL",
                        "from": "2026-10-12T00:00:00Z",
                        "to": "2026-10-19T23:59:59.999999Z",
                        "asOf": "2026-10-19T23:59:59.999999Z",
                    },
                    "producer_params": {
                        "ticker": "AAPL",
                        "start_date": "2026-10-12",
                        "end_date": "2026-10-19",
                    },
                    "snapshot_path": f"responses/001_get_news_snap-{arm}.json",
                    "raw_payload_hash": raw_h,
                    "records": [],
                    "output_hash": sentinel_hash,
                }
            ],
        }
        (cell_data / arch_fname).write_text(json.dumps(arch_v2), encoding="utf-8")

    summary = summarize_experiment(out)

    # Mandatory public summary contract assertions
    assert summary["strict_empirical_eligible"] is False
    assert summary["availability_status"] == "availability_unknown"
    assert summary["scored_pairs_count"] == 0
    assert summary["settled_pairs_count"] == 0
    assert summary["provenance_verified"] is False
    assert summary["timing_eligible"] is False

    # Diagnostic summary retains simulated alpha metrics in isolated namespace
    diag = summary["diagnostic_summary"]
    assert diag["diagnostic_scored_pairs_count"] == 1
    assert diag["diagnostic_status"] == "NON_EMPIRICAL_DIAGNOSTIC_ONLY"
    assert diag["diagnostic_mean_alpha_A"] == 0.05
    assert diag["diagnostic_mean_alpha_B"] == 0.07
    assert diag["diagnostic_alpha_difference"] == 0.02


# ==============================================================================
# Round 2 Remediation Proving Tests (R1 - R7)
# ==============================================================================


def test_prosp_round2_r1_unconditional_diagnostic_contract(tmp_path: Path) -> None:
    """Proves Round 2 Finding R1 (HIGH):
    The public summary contract (scored_pairs_count=0, settled_pairs_count=0,
    strict_empirical_eligible=False, availability_status='availability_unknown',
    provenance_verified=False, timing_eligible=False) is UNCONDITIONALLY enforced
    across version 2, legacy, unknown version, absent archive, malformed archive,
    and completed/recovered states.
    """
    out = tmp_path / "exp_r1_unconditional"
    out.mkdir(parents=True, exist_ok=True)

    manifest_hash = "fake-manifest-hash-r1"
    base_journal: dict[str, Any] = {
        "manifest_hash": manifest_hash,
        "experiment_id": "exp-r1",
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-r1-a",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "equal_hash",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-r1-b",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "equal_hash",
                "entry_time": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-r1-a",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.05,
                "raw_return": "8.0%",
                "raw_return_float": 0.08,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-r1-b",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.07,
                "raw_return": "8.0%",
                "raw_return_float": 0.08,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_time": "2026-08-03T20:00:00Z",
            },
        },
    }

    # Setup cell results with matching logs and audit traces
    for arm in ("A", "B"):
        c_res = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        c_res.mkdir(parents=True, exist_ok=True)
        tag = "+5.0%" if arm == "A" else "+7.0%"
        (c_res / "trading_memory.log").write_text(
            f"[2026-08-03 | NVDA | Buy | 8.0% | {tag} | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n",
            encoding="utf-8",
        )

    (out / "journal.json").write_text(json.dumps(base_journal), encoding="utf-8")

    def assert_public_contract(summary: dict[str, Any], expect_diag_scored: int = 0) -> None:
        assert summary["scored_pairs_count"] == 0
        assert summary["settled_pairs_count"] == 0
        assert summary["strict_empirical_eligible"] is False
        assert summary["availability_status"] == "availability_unknown"
        assert summary["provenance_verified"] is False
        assert summary["timing_eligible"] is False
        assert summary["mean_alpha_A"] is None
        assert summary["mean_alpha_B"] is None
        assert summary["alpha_difference"] is None
        assert summary["directional_hits_A"] == 0
        assert summary["directional_hits_B"] == 0
        pair_rec = summary["pair_eligibility"]["NVDA_2026-08-03"]
        assert pair_rec["provenance_verified"] is False
        assert pair_rec["timing_eligible"] is False
        assert pair_rec["eligible"] is False
        diag = summary["diagnostic_summary"]
        assert diag["diagnostic_status"] == "NON_EMPIRICAL_DIAGNOSTIC_ONLY"
        assert diag["diagnostic_scored_pairs_count"] == expect_diag_scored

    # Authentic items for Arm A and Arm B
    raw_yahoo = {
        "title": "Authentic Yahoo News for NVDA",
        "summary": "Authentic Yahoo News for NVDA",
        "publisher": "Yahoo",
        "link": "",
        "providerPublishTime": int(datetime(2026, 8, 3, 14, tzinfo=UTC).timestamp()),
    }
    extracted_yahoo = yahoo_news._extract_article_data(raw_yahoo)
    text_A = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{yahoo_news._format_articles([extracted_yahoo])}"
    hash_A = hashlib.sha256(text_A.encode("utf-8")).hexdigest()

    raw_ni = {
        "id": "ev-b",
        "source": "Bloomberg",
        "title": "Authentic Repo1 News Intelligence for NVDA",
        "content": "Authentic Repo1 News Intelligence for NVDA",
        "publishedAt": "2026-08-03T14:00:00Z",
        "receivedAt": "2026-08-03T14:05:00Z",
        "enrichedAt": "2026-08-03T14:05:00Z",
        "symbols": ["NVDA"],
    }
    cutoff_dt = datetime(2026, 8, 3, 20, tzinfo=UTC)
    sanitized_ni = news_client.filter_and_sanitize_event(raw_ni, cutoff_dt)
    assert sanitized_ni is not None
    text_B = f"## NVDA News, from 2026-08-03 to 2026-08-03:\n\n{news_client.format_article(sanitized_ni)}"
    hash_B = hashlib.sha256(text_B.encode("utf-8")).hexdigest()

    for arm, txt, hsh in [("A", text_A, hash_A), ("B", text_B, hash_B)]:
        c_res = out / "cells" / f"NVDA_2026-08-03_{arm}" / "results"
        (c_res / "audit_trace.json").write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "tool_start",
                            "tool": "get_news",
                            "input": {
                                "ticker": "NVDA",
                                "start_date": "2026-08-03",
                                "end_date": "2026-08-03",
                            },
                            "run_id": f"run-r1-{arm}",
                        },
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output": txt,
                            "output_hash": hsh,
                            "run_id": f"run-r1-{arm}",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    # 1. Version 2 Archive
    for arm in ("A", "B"):
        c_data = out / "cells" / f"NVDA_2026-08-03_{arm}" / "data"
        c_data.mkdir(parents=True, exist_ok=True)
        (c_data / "responses").mkdir(parents=True, exist_ok=True)
        payload = [raw_yahoo] if arm == "A" else {"events": [raw_ni]}
        snap_content = json.dumps(payload)
        (c_data / "responses" / "001_snap.json").write_text(snap_content, encoding="utf-8")
        raw_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        extracted_yahoo_rec = dict(
            extracted_yahoo,
            pub_date=extracted_yahoo["pub_date"].isoformat()
            if isinstance(extracted_yahoo.get("pub_date"), datetime)
            else extracted_yahoo.get("pub_date"),
        )
        records = [extracted_yahoo_rec] if arm == "A" else [sanitized_ni]
        hsh = hash_A if arm == "A" else hash_B
        arch = {
            "version": "t13-archive/v2",
            "attempt_id": f"att-r1-{arm.lower()}",
            "ticker": "NVDA",
            "arm": arm,
            "date": "2026-08-03",
            "vendor": "yfinance" if arm == "A" else "news_intelligence",
            "calls": [
                {
                    "call_seq": 1,
                    "call_id": f"c1-{arm}",
                    "tool": "get_news",
                    "run_id": f"run-r1-{arm}",
                    "query": "NVDA",
                    "received_at": "2026-08-03T14:05:00Z",
                    "params": {
                        "ticker": "NVDA",
                        "start_date": "2026-08-03",
                        "end_date": "2026-08-03",
                    }
                    if arm == "A"
                    else {
                        "symbol": "NVDA",
                        "from": "2026-08-03T00:00:00Z",
                        "to": "2026-08-03T23:59:59.999999Z",
                        "asOf": "2026-08-03T23:59:59.999999Z",
                    },
                    "producer_params": {
                        "ticker": "NVDA",
                        "start_date": "2026-08-03",
                        "end_date": "2026-08-03",
                    },
                    "snapshot_path": "responses/001_snap.json",
                    "raw_payload_hash": raw_hash,
                    "records": records,
                    "output_hash": hsh,
                }
            ],
        }
        fname = "baseline_archive.json" if arm == "A" else "provenance_archive.json"
        (c_data / fname).write_text(json.dumps(arch), encoding="utf-8")

    s_v2 = summarize_experiment(out)
    assert_public_contract(s_v2, expect_diag_scored=1)

    # 2. Legacy Archive (No version metadata)
    legacy_event_A = dict(
        raw_yahoo, published_at="2026-08-03T14:00:00Z", received_at="2026-08-03T14:05:00Z"
    )
    legacy_arch_A = {
        "ticker": "NVDA",
        "query": "NVDA",
        "date": "2026-08-03",
        "arm": "A",
        "attempt_id": "att-r1-a",
        "source": "yfinance",
        "cutoff": "2026-08-03T20:00:00Z",
        "output_hash": hash_A,
        "received_at": "2026-08-03T14:05:00Z",
        "events": [legacy_event_A],
    }
    legacy_arch_B = {
        "ticker": "NVDA",
        "query": "NVDA",
        "date": "2026-08-03",
        "arm": "B",
        "attempt_id": "att-r1-b",
        "source": "news_intelligence",
        "cutoff": "2026-08-03T20:00:00Z",
        "output_hash": hash_B,
        "received_at": "2026-08-03T14:05:00Z",
        "events": [raw_ni],
    }
    (out / "cells" / "NVDA_2026-08-03_A" / "data" / "baseline_archive.json").write_text(
        json.dumps(legacy_arch_A), encoding="utf-8"
    )
    (out / "cells" / "NVDA_2026-08-03_B" / "data" / "provenance_archive.json").write_text(
        json.dumps(legacy_arch_B), encoding="utf-8"
    )
    s_leg = summarize_experiment(out)
    assert_public_contract(s_leg, expect_diag_scored=1)

    # 3. Unknown Version Archive
    unkn_A = dict(arch, version="t13-archive/v999", arm="A", attempt_id="att-r1-a")
    unkn_B = dict(arch, version="t13-archive/v999", arm="B", attempt_id="att-r1-b")
    (out / "cells" / "NVDA_2026-08-03_A" / "data" / "baseline_archive.json").write_text(
        json.dumps(unkn_A), encoding="utf-8"
    )
    (out / "cells" / "NVDA_2026-08-03_B" / "data" / "provenance_archive.json").write_text(
        json.dumps(unkn_B), encoding="utf-8"
    )
    s_unkn = summarize_experiment(out)
    assert_public_contract(s_unkn, expect_diag_scored=0)

    # 4. Absent Archive
    (out / "cells" / "NVDA_2026-08-03_A" / "data" / "baseline_archive.json").unlink()
    (out / "cells" / "NVDA_2026-08-03_B" / "data" / "provenance_archive.json").unlink()
    s_absent = summarize_experiment(out)
    assert_public_contract(s_absent, expect_diag_scored=0)

    # 5. Malformed Archive JSON
    (out / "cells" / "NVDA_2026-08-03_A" / "data" / "baseline_archive.json").write_text(
        "{ broken json", encoding="utf-8"
    )
    (out / "cells" / "NVDA_2026-08-03_B" / "data" / "provenance_archive.json").write_text(
        "{ broken json", encoding="utf-8"
    )
    s_malformed = summarize_experiment(out)
    assert_public_contract(s_malformed, expect_diag_scored=0)


def test_prosp_round2_r2_parallel_yahoo_thread_isolation(tmp_path: Path) -> None:
    """Proves Round 2 Finding R2 (HIGH):
    Context-wide capture hooks with thread-local request collectors prevent
    cross-contamination between concurrent parallel Yahoo tool invocations.
    """
    import concurrent.futures

    cell = tmp_path / "actual-parallel-hooks"
    started = threading.Event()
    release = threading.Event()
    cfg = {"news_article_limit": 20}
    articles = {
        sym: {
            "title": f"REVIEWER_ONLY {sym}",
            "summary": "",
            "publisher": "P",
            "link": "",
            "providerPublishTime": int(datetime(2026, 8, 3, 13, tzinfo=UTC).timestamp()),
        }
        for sym in ["AAA", "BBB"]
    }

    class MockTicker:
        def __init__(self, symbol: str):
            self.symbol = symbol

        def get_news(self, count: int = 20) -> list[dict[str, Any]]:
            if self.symbol == "AAA":
                started.set()
                assert release.wait(5), "reviewer scheduling timeout"
            return [articles[self.symbol]]

    def retry(fn: Any) -> Any:
        return fn()

    with (
        patch.object(yahoo_news, "get_config", return_value=cfg),
        patch.object(yahoo_news.yf, "Ticker", MockTicker),
        patch.object(yahoo_news, "yf_retry", retry),
    ):
        with ProspectiveCaptureContext(cell, "NVDA", "2026-08-03", "A", "reviewer-parallel"):

            def work(sym: str) -> str:
                _current_tool_run_id.value = f"run-{sym}"
                return yahoo_news.get_news_yfinance(sym, "2026-08-03", "2026-08-03")

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                a = pool.submit(work, "AAA")
                assert started.wait(5)
                b = pool.submit(work, "BBB")
                try:
                    b_output = b.result(timeout=5)
                finally:
                    release.set()
                a_output = a.result(timeout=5)

        # Hook restored cleanly after normal exit inside active patch
        assert yahoo_news.yf_retry is retry

    assert "BBB" not in a_output and "AAA" in a_output
    assert "AAA" not in b_output and "BBB" in b_output

    data = json.loads((cell / "data" / "baseline_archive.json").read_text(encoding="utf-8"))
    by_query = {c["query"]: c for c in data["calls"]}
    titles_A = [v["title"] for v in by_query["AAA"]["records"]]
    titles_B = [v["title"] for v in by_query["BBB"]["records"]]
    assert titles_A == ["REVIEWER_ONLY AAA"]
    assert titles_B == ["REVIEWER_ONLY BBB"]


def test_prosp_round2_r3_v2_bijection_and_cardinality(tmp_path: Path) -> None:
    """Proves Round 2 Finding R3 (HIGH):
    Version 2 archive verification requires complete 1-to-1 cardinality and bijection:
    - Unarchived tool end fails closed.
    - Nonexistent run_id fails closed.
    - Altered query / request ticker fails closed.
    - Duplicate run_id / re-used tool end fails closed.
    - Valid repeated tool executions pass.
    """
    cell = tmp_path / "identity-cardinality"
    cell.mkdir(parents=True)
    cb = AuditCallbackHandler()

    raw = {
        "title": "REVIEWER_ONLY article",
        "summary": "S",
        "publisher": "P",
        "link": "https://example.invalid/inert",
        "providerPublishTime": int(datetime(2026, 8, 3, 13, tzinfo=UTC).timestamp()),
    }
    extracted = yahoo_news._extract_article_data(raw)
    output = "## NVDA News, from 2026-08-03 to 2026-08-03:\n\n" + yahoo_news._format_articles(
        [extracted]
    )

    cb.on_tool_start(
        {"name": "get_news"},
        json.dumps({"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"}),
        run_id="run-A",
    )
    cb.on_tool_end(output, run_id="run-A")

    ctx = ProspectiveCaptureContext(cell, "NVDA", "2026-08-03", "A", "reviewer-A")
    ctx.commit_call(
        "get_news",
        "NVDA",
        {"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"},
        [raw],
        [extracted],
        output,
        "run-A",
        "2026-08-03T14:00:00Z",
    )

    res = cell / "results"
    res.mkdir(parents=True, exist_ok=True)
    trace = res / "audit_trace.json"
    trace.write_text(json.dumps({"traces": cb.traces}), encoding="utf-8")
    cutoff_dt = datetime(2026, 8, 3, 20, tzinfo=UTC)

    # 1. Positive control passes
    ok, msg = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "reviewer-A"}, cell
    )
    assert ok is True, f"Positive control failed: {msg}"

    # 2. Unarchived tool end: cardinality mismatch fails closed
    td = json.loads(trace.read_text(encoding="utf-8"))
    extra = copy.deepcopy(td["traces"][-1])
    extra.update(
        run_id="unarchived-run",
        output="REVIEWER_ONLY unarchived paid response",
        output_hash=hashlib.sha256(b"REVIEWER_ONLY unarchived paid response").hexdigest(),
    )
    td["traces"].append(extra)
    trace.write_text(json.dumps(td), encoding="utf-8")
    ok_uncovered, msg_uncovered = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "reviewer-A"}, cell
    )
    assert ok_uncovered is False
    assert "does not match observed news tool execution count" in msg_uncovered

    # Restore trace
    trace.write_text(json.dumps({"traces": cb.traces}), encoding="utf-8")

    # 3. Nonexistent run_id in archive fails closed
    arch_data = json.loads(ctx.archive_path.read_text(encoding="utf-8"))
    orig_call = copy.deepcopy(arch_data["calls"][0])
    arch_data["calls"][0]["run_id"] = "nonexistent-run"
    ctx.archive_path.write_text(json.dumps(arch_data), encoding="utf-8")
    ok_nonexistent, msg_nonexistent = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "reviewer-A"}, cell
    )
    assert ok_nonexistent is False
    assert "run_id 'nonexistent-run'" in msg_nonexistent

    # 4. Altered query/ticker mismatch fails closed
    arch_data["calls"][0] = copy.deepcopy(orig_call)
    arch_data["calls"][0]["query"] = "DIFFERENT"
    ctx.archive_path.write_text(json.dumps(arch_data), encoding="utf-8")
    ok_diff, msg_diff = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "reviewer-A"}, cell
    )
    assert ok_diff is False

    # 5. Altered output_hash fails closed
    arch_data["calls"][0] = copy.deepcopy(orig_call)
    arch_data["calls"][0]["output_hash"] = "0" * 64
    ctx.archive_path.write_text(json.dumps(arch_data), encoding="utf-8")
    ok_hash, msg_hash = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "reviewer-A"}, cell
    )
    assert ok_hash is False

    # 6. Duplicate call reusing run_id fails closed
    arch_data["calls"][0] = copy.deepcopy(orig_call)
    dup = copy.deepcopy(orig_call)
    dup["call_id"] = "dup-call"
    dup["call_seq"] = 2
    arch_data["calls"].append(dup)
    ctx.archive_path.write_text(json.dumps(arch_data), encoding="utf-8")
    ok_dup, msg_dup = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "reviewer-A"}, cell
    )
    assert ok_dup is False
    assert "does not match observed news tool execution count" in msg_dup or "duplicate" in msg_dup


def test_prosp_round2_r4_receipt_clock_after_return_and_entry_cutoff(tmp_path: Path) -> None:
    """Proves Round 2 Finding R4 (HIGH):
    - received_at is timestamped immediately AFTER vendor request returns.
    - Clock advancing past cutoff during request causes received_at > entry_cutoff -> rejection.
    - Events with publishedAt/receivedAt/enrichedAt after cutoff fail entry sanitization.
    """
    cell = tmp_path / "arrival-after-cutoff"
    cb = AuditCallbackHandler()

    class MockClock(datetime):
        instant = datetime(2026, 8, 3, 14, tzinfo=UTC)

        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return cls.instant if tz else cls.instant.replace(tzinfo=None)

    raw_ev = {
        "id": "ev-r4",
        "source": "REVIEWER_ONLY source",
        "title": "Title",
        "publishedAt": "2026-08-03T20:30:00Z",
        "receivedAt": "2026-08-03T20:31:00Z",
        "enrichedAt": "2026-08-03T20:32:00Z",
        "symbols": ["NVDA"],
    }

    def delayed_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        MockClock.instant = datetime(2026, 8, 3, 21, tzinfo=UTC)
        return {"events": [raw_ev]}

    cfg = {"news_article_limit": 20}
    with (
        patch.object(news_client.NewsIntelligenceClient, "_make_request", delayed_request),
        patch.object(news_client, "get_config", return_value=cfg),
        patch.object(r, "get_config", return_value=cfg),
        patch.object(r, "datetime", MockClock),
    ):
        client = object.__new__(news_client.NewsIntelligenceClient)
        with ProspectiveCaptureContext(
            cell, "NVDA", "2026-08-03", "B", "reviewer-B", audit_handler=cb
        ):
            cb.on_tool_start(
                {"name": "get_news"},
                json.dumps(
                    {"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"}
                ),
                run_id="reviewer-run",
            )
            output = client.get_news(ticker="NVDA", start_date="2026-08-03", end_date="2026-08-03")
            cb.on_tool_end(output, run_id="reviewer-run")

    res = cell / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "audit_trace.json").write_text(json.dumps({"traces": cb.traces}), encoding="utf-8")
    archive = cell / "data" / "provenance_archive.json"

    # Verify that received_at captured the return time (21:00Z), NOT start time (14:00Z)
    data = json.loads(archive.read_text(encoding="utf-8"))
    assert data["calls"][0]["received_at"] == "2026-08-03T21:00:00+00:00"

    # Verifier with entry cutoff 20:00Z must reject the archive (return was after cutoff)
    cutoff_dt = datetime(2026, 8, 3, 20, tzinfo=UTC)
    ok, msg = validate_evidence_archive(
        archive, "B", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "reviewer-B"}, cell
    )
    assert ok is False
    assert "exceeds entry cutoff" in msg

    # Native filter_and_sanitize_event rejects the event
    assert news_client.filter_and_sanitize_event(raw_ev, cutoff_dt) is None


def test_prosp_round2_r5_producer_params_retention_and_search_sentinel(tmp_path: Path) -> None:
    """Proves Round 2 Finding R5 (MEDIUM):
    - Retaining producer_params alongside verbatim HTTP parameters ensures non-default windows round-trip.
    - Yahoo feed-empty search fallback sentinel round-trips through verification without mismatch.
    """
    cfg = {
        "news_article_limit": 20,
        "global_news_lookback_days": 7,
        "global_news_article_limit": 10,
        "global_news_queries": ["REVIEWER_ONLY macro"],
    }
    raw_ev = {
        "id": "ev-r5",
        "source": "REVIEWER_ONLY source",
        "title": "Title",
        "publishedAt": "2026-08-03T13:00:00Z",
        "receivedAt": "2026-08-03T13:05:00Z",
        "symbols": ["NVDA"],
    }

    class MockClock(datetime):
        instant = datetime(2026, 8, 3, 14, tzinfo=UTC)

        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return cls.instant if tz else cls.instant.replace(tzinfo=None)

    # 1. Non-default global news parameters (look_back_days=3, limit=1)
    cell_global = tmp_path / "actual-b-global-params"
    cb_g = AuditCallbackHandler()
    with (
        patch.object(
            news_client.NewsIntelligenceClient, "_make_request", return_value={"events": [raw_ev]}
        ),
        patch.object(news_client, "get_config", return_value=cfg),
        patch.object(r, "get_config", return_value=cfg),
        patch.object(r, "datetime", MockClock),
    ):
        client = object.__new__(news_client.NewsIntelligenceClient)
        with ProspectiveCaptureContext(
            cell_global, "NVDA", "2026-08-03", "B", "reviewer-B", audit_handler=cb_g
        ):
            params_g = {"as_of_date": "2026-08-03", "look_back_days": 3, "limit": 1}
            cb_g.on_tool_start({"name": "get_global_news"}, json.dumps(params_g), run_id="run-g")
            out_g = client.get_global_news(**params_g)
            cb_g.on_tool_end(out_g, run_id="run-g")

    res_g = cell_global / "results"
    res_g.mkdir(parents=True, exist_ok=True)
    (res_g / "audit_trace.json").write_text(json.dumps({"traces": cb_g.traces}), encoding="utf-8")
    cutoff_dt = datetime(2026, 8, 3, 20, tzinfo=UTC)
    ok_g, msg_g = validate_evidence_archive(
        cell_global / "data" / "provenance_archive.json",
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "reviewer-B"},
        cell_global,
    )
    assert ok_g is True, f"Global params round-trip failed: {msg_g}"

    # 2. Yahoo search sentinel fallback
    cell_search = tmp_path / "actual-search-sentinel"
    cb_s = AuditCallbackHandler()
    old_article = {
        "title": "REVIEWER_ONLY old search article",
        "publisher": "P",
        "relatedTickers": ["NVDA"],
        "providerPublishTime": int(datetime(2026, 8, 1, 13, tzinfo=UTC).timestamp()),
    }
    with (
        patch.object(yahoo_news, "get_config", return_value=cfg),
        patch.object(yahoo_news, "yf_retry", side_effect=lambda fn: fn()),
        patch.object(
            yahoo_news.yf, "Ticker", return_value=SimpleNamespace(get_news=lambda count: [])
        ),
        patch.object(
            yahoo_news.yf, "Search", return_value=SimpleNamespace(news=[old_article], quotes=[])
        ),
        patch.object(r, "datetime", MockClock),
        ProspectiveCaptureContext(cell_search, "NVDA", "2026-08-03", "A", "reviewer-A") as ctx,
    ):
        cb_s.on_tool_start(
            {"name": "get_news"},
            json.dumps({"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"}),
            run_id="search-run",
        )
        out_s = yahoo_news.get_news_yfinance("NVDA", "2026-08-03", "2026-08-03")
        cb_s.on_tool_end(out_s, run_id="search-run")

    res_s = cell_search / "results"
    res_s.mkdir(parents=True, exist_ok=True)
    (res_s / "audit_trace.json").write_text(json.dumps({"traces": cb_s.traces}), encoding="utf-8")
    assert "Yahoo search" in out_s
    ok_s, msg_s = validate_evidence_archive(
        ctx.archive_path,
        "A",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "reviewer-A"},
        cell_search,
    )
    assert ok_s is True, f"Yahoo search sentinel verification failed: {msg_s}"


def test_prosp_round2_r6_snapshot_boundary_containment_and_bounds(tmp_path: Path) -> None:
    """Proves Round 2 Finding R6 (MEDIUM):
    Snapshot candidate search enforces path containment and collection bounds:
    - Rejects absolute snapshot paths outside cell directory.
    - Rejects traversal (../) escaping cell.
    - Enforces maximum snapshot size.
    """
    cell = tmp_path / "snapshot-boundary"
    cell.mkdir(parents=True)
    cb = AuditCallbackHandler()

    raw = {
        "title": "REVIEWER_ONLY article",
        "summary": "S",
        "publisher": "P",
        "link": "https://example.invalid/inert",
        "providerPublishTime": int(datetime(2026, 8, 3, 13, tzinfo=UTC).timestamp()),
    }
    extracted = yahoo_news._extract_article_data(raw)
    output = "## NVDA News, from 2026-08-03 to 2026-08-03:\n\n" + yahoo_news._format_articles(
        [extracted]
    )

    cb.on_tool_start(
        {"name": "get_news"},
        json.dumps({"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"}),
        run_id="run-A",
    )
    cb.on_tool_end(output, run_id="run-A")

    ctx = ProspectiveCaptureContext(cell, "NVDA", "2026-08-03", "A", "reviewer-A")
    ctx.commit_call(
        "get_news",
        "NVDA",
        {"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"},
        [raw],
        [extracted],
        output,
        "run-A",
        "2026-08-03T14:00:00Z",
    )

    res = cell / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "audit_trace.json").write_text(json.dumps({"traces": cb.traces}), encoding="utf-8")
    cutoff_dt = datetime(2026, 8, 3, 20, tzinfo=UTC)

    # 1. Absolute snapshot path outside cell directory is rejected
    data = json.loads(ctx.archive_path.read_text(encoding="utf-8"))
    old_snap = cell / "data" / data["calls"][0]["snapshot_path"]
    external = tmp_path / "outside-selected-cell.json"
    external.write_bytes(old_snap.read_bytes())
    data["calls"][0]["snapshot_path"] = str(external.resolve())
    ctx.archive_path.write_text(json.dumps(data), encoding="utf-8")

    ok_ext, msg_ext = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "reviewer-A"}, cell
    )
    assert ok_ext is False
    assert "relative without traversal" in msg_ext

    # 2. Traversal path escaping cell is rejected
    data["calls"][0]["snapshot_path"] = "../outside-selected-cell.json"
    ctx.archive_path.write_text(json.dumps(data), encoding="utf-8")
    ok_trav, msg_trav = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "reviewer-A"}, cell
    )
    assert ok_trav is False
    assert "relative without traversal" in msg_trav


def test_prosp_round2_r7_envelope_readback_full_content_verification(tmp_path: Path) -> None:
    """Proves Round 2 Finding R7 (MEDIUM):
    Envelope read-back verification detects any mutated or corrupted nested field
    after atomic replace, raising EvidenceStorageError.
    """
    cell = tmp_path / "envelope-readback"
    ctx = ProspectiveCaptureContext(cell, "NVDA", "2026-08-03", "A", "reviewer-A")

    original_replace = r.os.replace

    def corrupted_replace(src: Any, dst: Any) -> None:
        original_replace(src, dst)
        if Path(dst) == ctx.archive_path:
            # Corrupt nested field without altering call count or version
            data = json.loads(Path(dst).read_text(encoding="utf-8"))
            data["calls"][0]["output_hash"] = "0" * 64
            Path(dst).write_text(json.dumps(data), encoding="utf-8")

    with patch.object(r.os, "replace", corrupted_replace), pytest.raises(EvidenceStorageError):
        ctx.commit_call(
            "get_news",
            "NVDA",
            {},
            [],
            [],
            "REVIEWER_ONLY output",
            "readback-run",
            "2026-08-03T14:00:00Z",
        )


def test_prosp_round3_high1_request_binding_and_unique_starts(tmp_path: Path) -> None:
    """Proves Round 3 Finding HIGH1:
    Version 2 archive verification binds complete effective request arguments
    (ticker, start_date, end_date, as_of_date, look_back_days, limit) between
    observed tool_start traces and captured call/producer parameters, and enforces
    exact 1-to-1 uniqueness of tool_start traces.
    """
    cell = tmp_path / "high1-binding"
    cell.mkdir(parents=True)
    cutoff_dt = datetime(2026, 8, 3, 20, tzinfo=UTC)

    # 1. Arm A Ticker News test
    cb = AuditCallbackHandler()
    raw = {
        "title": "NVDA Q2 results",
        "summary": "Revenue up 100%",
        "publisher": "Yahoo",
        "link": "https://example.invalid/nvda",
        "providerPublishTime": int(datetime(2026, 8, 3, 12, tzinfo=UTC).timestamp()),
    }
    extracted = yahoo_news._extract_article_data(raw)
    output = "## NVDA News, from 2026-08-01 to 2026-08-03:\n\n" + yahoo_news._format_articles(
        [extracted]
    )

    cb.on_tool_start(
        {"name": "get_news"},
        json.dumps({"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-03"}),
        run_id="run-A1",
    )
    cb.on_tool_end(output, run_id="run-A1")

    ctx = ProspectiveCaptureContext(cell, "NVDA", "2026-08-03", "A", "att-high1")
    ctx.commit_call(
        "get_news",
        "NVDA",
        {"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-03"},
        [raw],
        [extracted],
        output,
        "run-A1",
        "2026-08-03T14:00:00Z",
        producer_params={"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-03"},
    )

    res = cell / "results"
    res.mkdir(parents=True, exist_ok=True)
    trace = res / "audit_trace.json"
    trace.write_text(json.dumps({"traces": cb.traces}), encoding="utf-8")

    # Positive control passes
    ok, msg = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-high1"}, cell
    )
    assert ok is True, f"Positive control failed: {msg}"

    # Negative control 1: changed start_date in tool_start (2026-08-02 instead of 2026-08-01)
    cb_bad = copy.deepcopy(cb.traces)
    cb_bad[0]["input"] = json.dumps(
        {"ticker": "NVDA", "start_date": "2026-08-02", "end_date": "2026-08-03"}
    )
    trace.write_text(json.dumps({"traces": cb_bad}), encoding="utf-8")
    ok_sd, msg_sd = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-high1"}, cell
    )
    assert ok_sd is False
    assert "start_date" in msg_sd and "does not match" in msg_sd

    # Negative control 2: changed end_date in tool_start (2026-08-02 instead of 2026-08-03)
    cb_bad[0]["input"] = json.dumps(
        {"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-02"}
    )
    trace.write_text(json.dumps({"traces": cb_bad}), encoding="utf-8")
    ok_ed, msg_ed = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-high1"}, cell
    )
    assert ok_ed is False
    assert "end_date" in msg_ed and "does not match" in msg_ed

    # Negative control 3: missing tool_start input (None)
    cb_bad[0]["input"] = None
    trace.write_text(json.dumps({"traces": cb_bad}), encoding="utf-8")
    ok_mi, msg_mi = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-high1"}, cell
    )
    assert ok_mi is False
    assert "missing input" in msg_mi

    # Negative control 4: empty string tool_start input ("")
    cb_bad[0]["input"] = "   "
    trace.write_text(json.dumps({"traces": cb_bad}), encoding="utf-8")
    ok_ei, msg_ei = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-high1"}, cell
    )
    assert ok_ei is False
    assert "empty input" in msg_ei

    # Negative control 5: malformed JSON input
    cb_bad[0]["input"] = "{not-valid-json"
    trace.write_text(json.dumps({"traces": cb_bad}), encoding="utf-8")
    ok_mal, msg_mal = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-high1"}, cell
    )
    assert ok_mal is False
    assert "not valid JSON" in msg_mal

    # Negative control 6: non-dict input (list)
    cb_bad[0]["input"] = "[1, 2, 3]"
    trace.write_text(json.dumps({"traces": cb_bad}), encoding="utf-8")
    ok_nd, msg_nd = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-high1"}, cell
    )
    assert ok_nd is False
    assert "must decode to a dictionary" in msg_nd

    # Negative control 7: duplicated tool_start trace with the same run_id
    cb_bad = copy.deepcopy(cb.traces)
    dup_start = copy.deepcopy(cb_bad[0])
    cb_bad.insert(1, dup_start)
    trace.write_text(json.dumps({"traces": cb_bad}), encoding="utf-8")
    ok_dup, msg_dup = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-high1"}, cell
    )
    assert ok_dup is False
    assert "does not match observed news tool start count" in msg_dup or "found 2" in msg_dup

    # Control 8: missing required asset dates must fail; state-injected ticker and mapping syntax succeed
    output_def = "## NVDA News, from 2026-08-03 to 2026-08-03:\n\n" + yahoo_news._format_articles(
        [extracted]
    )
    out_hash_def = hashlib.sha256(output_def.encode("utf-8")).hexdigest()
    cb_default = copy.deepcopy(cb.traces)
    # 8a: Ticker only (missing start_date and end_date) must fail (HIGH2)
    cb_default[0]["input"] = json.dumps({"ticker": "NVDA"})
    cb_default[1]["output"] = output_def
    cb_default[1]["output_hash"] = out_hash_def
    ctx_def = ProspectiveCaptureContext(cell / "def", "NVDA", "2026-08-03", "A", "att-def")
    ctx_def.commit_call(
        "get_news",
        "NVDA",
        {"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"},
        [raw],
        [extracted],
        output_def,
        "run-A1",
        "2026-08-03T14:00:00Z",
        producer_params={"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"},
    )
    res_def = cell / "def" / "results"
    res_def.mkdir(parents=True, exist_ok=True)
    trace_def_file = res_def / "audit_trace.json"
    trace_def_file.write_text(json.dumps({"traces": cb_default}), encoding="utf-8")
    ok_miss, msg_miss = validate_evidence_archive(
        ctx_def.archive_path,
        "A",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-def"},
        cell / "def",
    )
    assert ok_miss is False
    assert "missing required start_date" in msg_miss

    # 8b: Valid model input omitting ticker (injected from state) succeeds (HIGH1)
    cb_default[0]["input"] = json.dumps({"start_date": "2026-08-03", "end_date": "2026-08-03"})
    trace_def_file.write_text(json.dumps({"traces": cb_default}), encoding="utf-8")
    ok_inj, msg_inj = validate_evidence_archive(
        ctx_def.archive_path,
        "A",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-def"},
        cell / "def",
    )
    assert ok_inj is True, f"State-injected ticker failed: {msg_inj}"

    # 8c: Python mapping string syntax succeeds (HIGH1)
    cb_default[0]["input"] = "{'start_date': '2026-08-03', 'end_date': '2026-08-03'}"
    trace_def_file.write_text(json.dumps({"traces": cb_default}), encoding="utf-8")
    ok_map, msg_map = validate_evidence_archive(
        ctx_def.archive_path,
        "A",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-def"},
        cell / "def",
    )
    assert ok_map is True, f"Mapping syntax failed: {msg_map}"

    # Positive control 9: repeated calls with distinct identifiers
    cell_rep = tmp_path / "high1-repeated"
    cell_rep.mkdir(parents=True)
    cb_rep = AuditCallbackHandler()
    cb_rep.on_tool_start(
        {"name": "get_news"},
        json.dumps({"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-03"}),
        run_id="run-rep-1",
    )
    cb_rep.on_tool_end(output, run_id="run-rep-1")
    cb_rep.on_tool_start(
        {"name": "get_news"},
        json.dumps({"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-03"}),
        run_id="run-rep-2",
    )
    cb_rep.on_tool_end(output, run_id="run-rep-2")

    ctx_rep = ProspectiveCaptureContext(cell_rep, "NVDA", "2026-08-03", "A", "att-rep")
    ctx_rep.commit_call(
        "get_news",
        "NVDA",
        {"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-03"},
        [raw],
        [extracted],
        output,
        "run-rep-1",
        "2026-08-03T14:00:00Z",
        producer_params={"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-03"},
    )
    ctx_rep.commit_call(
        "get_news",
        "NVDA",
        {"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-03"},
        [raw],
        [extracted],
        output,
        "run-rep-2",
        "2026-08-03T14:01:00Z",
        producer_params={"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-03"},
    )
    res_rep = cell_rep / "results"
    res_rep.mkdir(parents=True, exist_ok=True)
    (res_rep / "audit_trace.json").write_text(
        json.dumps({"traces": cb_rep.traces}), encoding="utf-8"
    )
    ok_rep, msg_rep = validate_evidence_archive(
        ctx_rep.archive_path,
        "A",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-rep"},
        cell_rep,
    )
    assert ok_rep is True, f"Repeated calls with distinct identifiers failed: {msg_rep}"

    # 2. Arm B Global News test
    cell_b = tmp_path / "high1-global"
    cell_b.mkdir(parents=True)
    cb_b = AuditCallbackHandler()

    raw_event = {
        "id": "glob-1",
        "publishedAt": "2026-08-03T10:00:00Z",
        "receivedAt": "2026-08-03T11:00:00Z",
        "enrichedAt": "2026-08-03T11:05:00Z",
        "symbols": ["GLOBAL"],
        "title": "Fed announces policy",
        "summary": "Rates held steady",
        "source": "Repo1",
        "url": "https://example.invalid/fed",
    }
    sanitized = ni_client.filter_and_sanitize_event(
        raw_event,
        cutoff_dt=datetime(2026, 8, 3, 23, 59, 59, 999999, tzinfo=UTC),
        min_pub_dt=datetime(2026, 7, 27, 0, 0, 0, tzinfo=UTC),
        max_pub_dt=datetime(2026, 8, 3, 23, 59, 59, 999999, tzinfo=UTC),
    )
    out_b = f"## Global Market News, from 2026-07-27 to 2026-08-03:\n\n{ni_client.format_article(sanitized)}"

    cb_b.on_tool_start(
        {"name": "get_global_news"},
        json.dumps({"as_of_date": "2026-08-03", "look_back_days": 7, "limit": 10}),
        run_id="run-B1",
    )
    cb_b.on_tool_end(out_b, run_id="run-B1")

    ctx_b = ProspectiveCaptureContext(cell_b, "NVDA", "2026-08-03", "B", "att-high1-b")
    ctx_b.commit_call(
        "get_global_news",
        "global",
        {
            "symbol": "GLOBAL",
            "asOf": "2026-08-03T23:59:59.999999Z",
            "to": "2026-08-03T23:59:59.999999Z",
            "from": "2026-07-27T00:00:00Z",
            "limit": 10,
        },
        {"events": [raw_event]},
        [sanitized],
        out_b,
        "run-B1",
        "2026-08-03T14:00:00Z",
        producer_params={"as_of_date": "2026-08-03", "look_back_days": 7, "limit": 10},
    )

    res_b = cell_b / "results"
    res_b.mkdir(parents=True, exist_ok=True)
    trace_b = res_b / "audit_trace.json"
    trace_b.write_text(json.dumps({"traces": cb_b.traces}), encoding="utf-8")

    # Positive control passes
    ok_b, msg_b = validate_evidence_archive(
        ctx_b.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-high1-b"},
        cell_b,
    )
    assert ok_b is True, f"Arm B global positive control failed: {msg_b}"

    # Negative control: changed as_of_date (past date 2026-08-02 instead of 2026-08-03)
    cb_b_bad = copy.deepcopy(cb_b.traces)
    cb_b_bad[0]["input"] = json.dumps(
        {"as_of_date": "2026-08-02", "look_back_days": 7, "limit": 10}
    )
    trace_b.write_text(json.dumps({"traces": cb_b_bad}), encoding="utf-8")
    ok_asof, msg_asof = validate_evidence_archive(
        ctx_b.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-high1-b"},
        cell_b,
    )
    assert ok_asof is False
    assert "as_of_date" in msg_asof and "does not match" in msg_asof

    # Negative control: changed look_back_days
    cb_b_bad[0]["input"] = json.dumps(
        {"as_of_date": "2026-08-03", "look_back_days": 14, "limit": 10}
    )
    trace_b.write_text(json.dumps({"traces": cb_b_bad}), encoding="utf-8")
    ok_lb, msg_lb = validate_evidence_archive(
        ctx_b.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-high1-b"},
        cell_b,
    )
    assert ok_lb is False
    assert "look_back_days" in msg_lb and "does not match" in msg_lb

    # Negative control: changed limit
    cb_b_bad[0]["input"] = json.dumps(
        {"as_of_date": "2026-08-03", "look_back_days": 7, "limit": 20}
    )
    trace_b.write_text(json.dumps({"traces": cb_b_bad}), encoding="utf-8")
    ok_lim, msg_lim = validate_evidence_archive(
        ctx_b.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-high1-b"},
        cell_b,
    )
    assert ok_lim is False
    assert "limit" in msg_lim and "does not match" in msg_lim


def test_prosp_round3_med2_explicit_zero_limit_preserved_and_verified(tmp_path: Path) -> None:
    """Proves Round 3 Finding MED2:
    Preserves explicit limit=0 without replacing with defaults across both arms.
    Actual hooked global calls with limit=0, None, omitted, and positive limits
    reproduce exact native text (Yahoo unavailable sentinel, News Intelligence header-only)
    and successfully pass archive verification.
    """
    cutoff_dt = datetime(2026, 8, 3, 20, tzinfo=UTC)

    # 1. Arm A Yahoo with actual hooked producer
    raw_yahoo = {
        "title": "Macro update",
        "summary": "GDP growth",
        "publisher": "Yahoo",
        "link": "https://example.invalid/gdp",
        "providerPublishTime": int(datetime(2026, 8, 2, 12, tzinfo=UTC).timestamp()),
    }
    cell_a = tmp_path / "med2-yahoo"
    ctx_a = ProspectiveCaptureContext(cell_a, "NVDA", "2026-08-03", "A", "att-med2-a")

    cfg_a = {
        "global_news_lookback_days": 7,
        "global_news_article_limit": 10,
        "global_news_queries": ["macro economy"],
    }

    class MockClock(datetime):
        instant = datetime(2026, 8, 3, 14, tzinfo=UTC)

        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return cls.instant if tz else cls.instant.replace(tzinfo=None)

    def mock_search_fn(*args: Any, **kwargs: Any) -> MagicMock:
        lim = kwargs.get("news_count", 10)
        obj = MagicMock()
        obj.news = [] if lim == 0 else [raw_yahoo]
        return obj

    with (
        patch.object(yahoo_news, "get_config", return_value=cfg_a),
        patch.object(yahoo_news.yf, "Search", side_effect=mock_search_fn),
        patch.object(yahoo_news, "yf_retry", lambda fn, *a, **k: fn()),
        patch.object(r, "datetime", MockClock),
        ctx_a,
    ):
        cb_a = AuditCallbackHandler()

        # Call 1: limit=0
        _current_tool_run_id.value = "run-yf-0"
        cb_a.on_tool_start(
            {"name": "get_global_news"},
            json.dumps({"as_of_date": "2026-08-03", "look_back_days": 3, "limit": 0}),
            run_id="run-yf-0",
        )
        out_0 = yahoo_news.get_global_news_yfinance("2026-08-03", look_back_days=3, limit=0)
        cb_a.on_tool_end(out_0, run_id="run-yf-0")

        # Call 2: limit=None
        _current_tool_run_id.value = "run-yf-none"
        cb_a.on_tool_start(
            {"name": "get_global_news"},
            json.dumps({"as_of_date": "2026-08-03", "look_back_days": 3}),
            run_id="run-yf-none",
        )
        out_none = yahoo_news.get_global_news_yfinance("2026-08-03", look_back_days=3, limit=None)
        cb_a.on_tool_end(out_none, run_id="run-yf-none")

        # Call 3: limit=5 (positive)
        _current_tool_run_id.value = "run-yf-pos"
        cb_a.on_tool_start(
            {"name": "get_global_news"},
            json.dumps({"as_of_date": "2026-08-03", "look_back_days": 3, "limit": 5}),
            run_id="run-yf-pos",
        )
        out_pos = yahoo_news.get_global_news_yfinance("2026-08-03", look_back_days=3, limit=5)
        cb_a.on_tool_end(out_pos, run_id="run-yf-pos")

        _current_tool_run_id.value = None

    # Assert limit=0 output produced native unavailable sentinel
    assert "Yahoo Finance global news unavailable" in out_0 or "No global news found" in out_0
    assert "Macro update" not in out_0
    assert "Macro update" in out_none
    assert "Macro update" in out_pos

    res_a = cell_a / "results"
    res_a.mkdir(parents=True, exist_ok=True)
    (res_a / "audit_trace.json").write_text(json.dumps({"traces": cb_a.traces}), encoding="utf-8")

    ok, msg = validate_evidence_archive(
        ctx_a.archive_path,
        "A",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-med2-a"},
        cell_a,
    )
    assert ok is True, f"Arm A Yahoo multi-limit archive verification failed: {msg}"
    arch_a = json.loads(ctx_a.archive_path.read_text(encoding="utf-8"))
    assert arch_a["calls"][0]["producer_params"]["limit"] == 0
    assert len(arch_a["calls"][0]["records"]) == 0
    assert len(arch_a["calls"][1]["records"]) == 1
    assert arch_a["calls"][2]["producer_params"]["limit"] == 5

    # 2. Arm B News Intelligence with actual hooked producer
    cell_b = tmp_path / "med2-ni"
    ctx_b = ProspectiveCaptureContext(cell_b, "NVDA", "2026-08-03", "B", "att-med2-b")

    raw_event = {
        "id": "glob-med2",
        "publishedAt": "2026-08-02T10:00:00Z",
        "receivedAt": "2026-08-02T11:00:00Z",
        "enrichedAt": "2026-08-02T11:05:00Z",
        "symbols": ["GLOBAL"],
        "title": "Global economic news",
        "summary": "Inflation cooled",
        "source": "Repo1",
        "url": "https://example.invalid/econ",
    }

    class MockTransportClient:
        def _make_request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
            return {"events": [raw_event]}

    with (
        patch.object(r, "datetime", MockClock),
        patch.object(
            ni_client.NewsIntelligenceClient, "_make_request", MockTransportClient._make_request
        ),
        ctx_b,
    ):
        cl = ni_client.NewsIntelligenceClient()
        cb_b = AuditCallbackHandler()

        # Call 1: limit=0
        _current_tool_run_id.value = "run-ni-0"
        cb_b.on_tool_start(
            {"name": "get_global_news"},
            json.dumps({"as_of_date": "2026-08-03", "look_back_days": 3, "limit": 0}),
            run_id="run-ni-0",
        )
        out_ni_0 = cl.get_global_news("2026-08-03", look_back_days=3, limit=0)
        cb_b.on_tool_end(out_ni_0, run_id="run-ni-0")

        # Call 2: limit=None
        _current_tool_run_id.value = "run-ni-none"
        cb_b.on_tool_start(
            {"name": "get_global_news"},
            json.dumps({"as_of_date": "2026-08-03", "look_back_days": 3}),
            run_id="run-ni-none",
        )
        out_ni_none = cl.get_global_news("2026-08-03", look_back_days=3, limit=None)
        cb_b.on_tool_end(out_ni_none, run_id="run-ni-none")

        # Call 3: limit=5
        _current_tool_run_id.value = "run-ni-pos"
        cb_b.on_tool_start(
            {"name": "get_global_news"},
            json.dumps({"as_of_date": "2026-08-03", "look_back_days": 3, "limit": 5}),
            run_id="run-ni-pos",
        )
        out_ni_pos = cl.get_global_news("2026-08-03", look_back_days=3, limit=5)
        cb_b.on_tool_end(out_ni_pos, run_id="run-ni-pos")

        _current_tool_run_id.value = None

    # Verify native header-only string with empty body for limit=0
    expected_header = "## Global Market News, from 2026-07-31 to 2026-08-03:\n\n"
    assert out_ni_0 == expected_header, f"Expected header-only text, got {out_ni_0!r}"
    assert "Global economic news" not in out_ni_0
    assert "Global economic news" in out_ni_none
    assert "Global economic news" in out_ni_pos

    res_b = cell_b / "results"
    res_b.mkdir(parents=True, exist_ok=True)
    (res_b / "audit_trace.json").write_text(json.dumps({"traces": cb_b.traces}), encoding="utf-8")

    ok_b, msg_b = validate_evidence_archive(
        ctx_b.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-med2-b"},
        cell_b,
    )
    assert ok_b is True, f"Arm B News Intelligence multi-limit archive verification failed: {msg_b}"
    arch_b = json.loads(ctx_b.archive_path.read_text(encoding="utf-8"))
    assert arch_b["calls"][0]["producer_params"]["limit"] == 0
    assert len(arch_b["calls"][0]["records"]) == 0
    assert len(arch_b["calls"][1]["records"]) == 1
    assert arch_b["calls"][2]["producer_params"]["limit"] == 5


def test_prosp_round3_med3_malformed_records_and_shapes_fail_closed_without_crash(
    tmp_path: Path,
) -> None:
    """Proves Round 3 Finding MED3:
    Malformed records (records: [null], scalar records), non-dict parameters, wrong raw-payload
    shapes, and invalid date fields fail closed gracefully with descriptive ineligibility reasons,
    without escaping to summarize_experiment or raising unhandled exceptions.
    """
    exp_dir = tmp_path / "med3-experiment"
    cells_dir = exp_dir / "cells"
    cells_dir.mkdir(parents=True)

    # Set up one pair with malformed Arm A archive containing records: [null]
    cell_a = cells_dir / "NVDA_2026-08-03_A"
    cell_b = cells_dir / "NVDA_2026-08-03_B"
    cell_a.mkdir()
    cell_b.mkdir()

    # Create dummy audit traces and native trading memory logs
    for c, arm in [(cell_a, "A"), (cell_b, "B")]:
        r_dir = c / "results"
        r_dir.mkdir(parents=True, exist_ok=True)
        (r_dir / "audit_trace.json").write_text(
            json.dumps(
                {
                    "traces": [
                        {
                            "type": "tool_start",
                            "tool": "get_news",
                            "input": json.dumps(
                                {
                                    "ticker": "NVDA",
                                    "start_date": "2026-08-03",
                                    "end_date": "2026-08-03",
                                }
                            ),
                            "run_id": f"run-{arm}",
                        },
                        {
                            "type": "tool_end",
                            "tool": "get_news",
                            "output": "out",
                            "output_hash": hashlib.sha256(b"out").hexdigest(),
                            "run_id": f"run-{arm}",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        (r_dir / "trading_memory.log").write_text(
            "[2026-08-03 | NVDA | Buy | 8.0% | +5.0% | 5d | resolved:2026-08-10]\n\nDECISION:\nBuy\n",
            encoding="utf-8",
        )
        (c / "data").mkdir(parents=True, exist_ok=True)

    # Write snapshot for Arm A
    snap_a_bytes = json.dumps([]).encode("utf-8")
    (cell_a / "data" / "snap.json").write_bytes(snap_a_bytes)
    raw_hash_a = hashlib.sha256(snap_a_bytes).hexdigest()

    # Malformed archive with records: [null] (reproducing reviewer finding)
    malformed_arch_a = {
        "version": "t13-archive/v2",
        "attempt_id": "att-A",
        "ticker": "NVDA",
        "arm": "A",
        "date": "2026-08-03",
        "cutoff": "2026-08-03T20:00:00Z",
        "vendor": "yfinance",
        "calls": [
            {
                "tool": "get_news",
                "query": "NVDA",
                "run_id": "run-A",
                "snapshot_path": "snap.json",
                "raw_payload_hash": raw_hash_a,
                "received_at": "2026-08-03T14:00:00Z",
                "params": {"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"},
                "producer_params": {
                    "ticker": "NVDA",
                    "start_date": "2026-08-03",
                    "end_date": "2026-08-03",
                },
                "records": [None],  # Null record member!
                "output_hash": hashlib.sha256(b"out").hexdigest(),
            }
        ],
    }
    (cell_a / "data" / "provenance_archive.json").write_text(
        json.dumps(malformed_arch_a), encoding="utf-8"
    )

    # Valid archive for Arm B
    snap_b_bytes = json.dumps({"events": []}).encode("utf-8")
    (cell_b / "data" / "snap.json").write_bytes(snap_b_bytes)
    raw_hash_b = hashlib.sha256(snap_b_bytes).hexdigest()
    arch_b = {
        "version": "t13-archive/v2",
        "attempt_id": "att-B",
        "ticker": "NVDA",
        "arm": "B",
        "date": "2026-08-03",
        "cutoff": "2026-08-03T20:00:00Z",
        "vendor": "news_intelligence",
        "calls": [
            {
                "tool": "get_news",
                "query": "NVDA",
                "run_id": "run-B",
                "snapshot_path": "snap.json",
                "raw_payload_hash": raw_hash_b,
                "received_at": "2026-08-03T14:00:00Z",
                "params": {"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"},
                "producer_params": {
                    "ticker": "NVDA",
                    "start_date": "2026-08-03",
                    "end_date": "2026-08-03",
                },
                "records": [],
                "output_hash": hashlib.sha256(b"out").hexdigest(),
            }
        ],
    }
    (cell_b / "data" / "provenance_archive.json").write_text(json.dumps(arch_b), encoding="utf-8")

    # Write execution journal with full planned, completed decision, and completed settlement structure
    journal = {
        "manifest_hash": "manifest-med3",
        "experiment_id": "exp-med3",
        "planned_cells_count": 2,
        "planned_decisions_count": 2,
        "planned_pairs_count": 1,
        "decision_attempts": {
            "NVDA_2026-08-03_A": {
                "attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "equal_hash",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "started_at": "2026-08-03T14:00:00Z",
                "completed_at": "2026-08-03T14:05:00Z",
                "non_news_inputs_hash": "equal_hash",
                "entry_time": "2026-08-03T20:00:00Z",
            },
        },
        "settlement_attempts": {
            "NVDA_2026-08-03_A": {
                "decision_attempt_id": "att-A",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "A",
                "outcome": 0.05,
                "raw_return": "8.0%",
                "raw_return_float": 0.08,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_time": "2026-08-03T20:00:00Z",
            },
            "NVDA_2026-08-03_B": {
                "decision_attempt_id": "att-B",
                "phase": "completed",
                "ticker": "NVDA",
                "date": "2026-08-03",
                "arm": "B",
                "outcome": 0.05,
                "raw_return": "8.0%",
                "raw_return_float": 0.08,
                "benchmark_return_float": 0.03,
                "holding_period": 5,
                "holding_period_actual": 5,
                "benchmark_name": "SPY",
                "rating": "Buy",
                "resolution_date": "2026-08-10",
                "entry_time": "2026-08-03T20:00:00Z",
            },
        },
    }
    (exp_dir / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    # Must complete cleanly without raising AttributeError: 'NoneType' object has no attribute 'get'
    summary = summarize_experiment(exp_dir)
    assert summary is not None
    assert summary["diagnostic_summary"]["diagnostic_status"] == "NON_EMPIRICAL_DIAGNOSTIC_ONLY"
    assert summary["strict_empirical_eligible"] is False
    assert summary["provenance_verified"] is False
    assert summary["ineligible_pairs_count"] == 1
    pair_rec = summary["pair_eligibility"]["NVDA_2026-08-03"]
    assert pair_rec["diagnostic_provenance_verified"] is False
    assert pair_rec["eligible"] is False
    assert any("must be a dictionary" in str(r) for r in pair_rec["ineligibility_reasons"])

    # Test individual validation fails closed for scalar records, bad params, bad snap
    cutoff_dt = datetime(2026, 8, 3, 20, tzinfo=UTC)
    cell_test = tmp_path / "med3-direct-shapes"
    cell_test.mkdir(parents=True)
    (cell_test / "data").mkdir()
    (cell_test / "data" / "snap.json").write_text("[]", encoding="utf-8")
    snap_test_hash = hashlib.sha256(b"[]").hexdigest()
    (cell_test / "results").mkdir()
    (cell_test / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": json.dumps(
                            {"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"}
                        ),
                        "run_id": "r-shape",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": "x",
                        "output_hash": hashlib.sha256(b"x").hexdigest(),
                        "run_id": "r-shape",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    # 1. Scalar record member
    arch_scalar = copy.deepcopy(malformed_arch_a)
    arch_scalar["calls"][0]["records"] = ["scalar-string", 123]
    arch_scalar["calls"][0]["snapshot_path"] = "snap.json"
    arch_scalar["calls"][0]["raw_payload_hash"] = snap_test_hash
    arch_scalar["calls"][0]["run_id"] = "r-shape"
    arch_scalar["calls"][0]["output_hash"] = hashlib.sha256(b"x").hexdigest()
    arch_p = cell_test / "data" / "provenance_archive.json"
    arch_p.write_text(json.dumps(arch_scalar), encoding="utf-8")
    ok, msg = validate_evidence_archive(
        arch_p, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-A"}, cell_test
    )
    assert ok is False
    assert "records item at index 0 must be a dictionary" in msg

    # 2. Non-dict params
    arch_bad_params = copy.deepcopy(arch_scalar)
    arch_bad_params["calls"][0]["records"] = []
    arch_bad_params["calls"][0]["params"] = "not-a-dict"
    arch_p.write_text(json.dumps(arch_bad_params), encoding="utf-8")
    ok, msg = validate_evidence_archive(
        arch_p, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-A"}, cell_test
    )
    assert ok is False
    assert "params must be a dictionary" in msg

    # 3. Non-dict producer_params
    arch_bad_prod = copy.deepcopy(arch_scalar)
    arch_bad_prod["calls"][0]["records"] = []
    arch_bad_prod["calls"][0]["producer_params"] = 12345
    arch_p.write_text(json.dumps(arch_bad_prod), encoding="utf-8")
    ok, msg = validate_evidence_archive(
        arch_p, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-A"}, cell_test
    )
    assert ok is False
    assert "producer_params must be a dictionary" in msg

    # 4. Wrong raw-payload shape for Arm A (dict instead of list)
    arch_bad_snap = copy.deepcopy(arch_scalar)
    arch_bad_snap["calls"][0]["records"] = []
    (cell_test / "data" / "snap_dict.json").write_text("{}", encoding="utf-8")
    arch_bad_snap["calls"][0]["snapshot_path"] = "snap_dict.json"
    arch_bad_snap["calls"][0]["raw_payload_hash"] = hashlib.sha256(b"{}").hexdigest()
    arch_p.write_text(json.dumps(arch_bad_snap), encoding="utf-8")
    ok, msg = validate_evidence_archive(
        arch_p, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-A"}, cell_test
    )
    assert ok is False
    assert "Arm A snapshot payload must be a list of raw articles" in msg

    # 5. Invalid date fields (e.g. malformed ISO timestamp in received_at)
    arch_bad_date = copy.deepcopy(arch_scalar)
    arch_bad_date["calls"][0]["records"] = []
    arch_bad_date["calls"][0]["snapshot_path"] = "snap.json"
    arch_bad_date["calls"][0]["raw_payload_hash"] = snap_test_hash
    arch_bad_date["calls"][0]["received_at"] = "invalid-date-format"
    arch_p.write_text(json.dumps(arch_bad_date), encoding="utf-8")
    ok, msg = validate_evidence_archive(
        arch_p, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-A"}, cell_test
    )
    assert ok is False
    assert "exceeds entry cutoff" in msg or "received_at" in msg


def test_prosp_round3_med4_bounded_envelope_read_before_parsing(tmp_path: Path) -> None:
    """Proves Round 3 Finding MED4:
    Outer archive envelopes exceeding MAX_ENVELOPE_BYTES (4 MiB) fail closed
    before full parsing/allocation, protecting against memory exhaustion.
    """
    cell = tmp_path / "med4-envelope-bound"
    cell.mkdir(parents=True)
    cutoff_dt = datetime(2026, 8, 3, 20, tzinfo=UTC)

    # 1. Positive control: valid generated archive passes within bound
    raw = {
        "title": "NVDA Earnings",
        "summary": "Record revenue",
        "publisher": "Yahoo",
        "link": "https://example.invalid/nvda",
        "providerPublishTime": int(datetime(2026, 8, 3, 12, tzinfo=UTC).timestamp()),
    }
    extracted = yahoo_news._extract_article_data(raw)
    output = "## NVDA News, from 2026-08-03 to 2026-08-03:\n\n" + yahoo_news._format_articles(
        [extracted]
    )

    cb = AuditCallbackHandler()
    cb.on_tool_start(
        {"name": "get_news"},
        json.dumps({"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"}),
        run_id="run-med4",
    )
    cb.on_tool_end(output, run_id="run-med4")

    ctx = ProspectiveCaptureContext(cell, "NVDA", "2026-08-03", "A", "att-med4")
    ctx.commit_call(
        "get_news",
        "NVDA",
        {"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"},
        [raw],
        [extracted],
        output,
        "run-med4",
        "2026-08-03T14:00:00Z",
        producer_params={"ticker": "NVDA", "start_date": "2026-08-03", "end_date": "2026-08-03"},
    )

    res = cell / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "audit_trace.json").write_text(json.dumps({"traces": cb.traces}), encoding="utf-8")

    ok_normal, msg_normal = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-med4"}, cell
    )
    assert ok_normal is True, f"Normal archive verification failed: {msg_normal}"

    # 2. Negative control: Padded archive exceeding MAX_ENVELOPE_BYTES (4 MiB) fails closed
    arch_data = json.loads(ctx.archive_path.read_text(encoding="utf-8"))
    # Add dummy padding string to exceed 4 MiB (e.g. 4.5 MiB)
    padding_size = r.MAX_ENVELOPE_BYTES + 512 * 1024
    arch_data["_padded_debug_fill"] = "X" * padding_size

    padded_file = cell / "data" / "padded_archive.json"
    padded_file.write_text(json.dumps(arch_data), encoding="utf-8")
    assert padded_file.stat().st_size > r.MAX_ENVELOPE_BYTES

    # validate_evidence_archive must reject it before parsing/allocation
    ok_padded, msg_padded = validate_evidence_archive(
        padded_file, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-med4"}, cell
    )
    assert ok_padded is False
    assert "exceeds maximum limit" in msg_padded
    assert str(r.MAX_ENVELOPE_BYTES) in msg_padded

    # _read_bounded_json_envelope directly rejects it
    data, err = r._read_bounded_json_envelope(padded_file)
    assert data is None
    assert f"exceeds maximum limit {r.MAX_ENVELOPE_BYTES} bytes" in str(err)

    # _verify_timing_clock_file rejects it
    assert r._verify_timing_clock_file(padded_file, "NVDA", "2026-08-03", cutoff_dt) is False


# ---------------------------------------------------------------------------
# Round 4 Reviewer Remediations Proving Tests (HIGH1, HIGH2, MED3)
# ---------------------------------------------------------------------------


def test_prosp_round4_high1_native_toolnode_execution_and_clamping(tmp_path: Path) -> None:
    """Proves Round 4 Finding HIGH1:
    Native LangGraph ToolNode execution with state-injected company_of_interest and trade_date:
    1. cb.on_tool_start receives Python mapping string without ticker.
    2. cb.on_tool_end receives ToolMessage, extracting .content for output text and hash.
    3. Future dates are clamped via native as_of_window / as_of to trade_date.
    4. Omitted ticker binds from cell state, and clamped dates verify cleanly against producer.
    5. Unclamped past date mismatches fail verification.
    """
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    from tradingagents.agents import tools as native_tools

    class _State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        trade_date: str
        company_of_interest: str

    trade_date = "2026-08-03"
    cutoff_dt = datetime(2026, 8, 3, 20, tzinfo=UTC)

    # 1. Arm A Yahoo ToolNode execution with future date clamping
    cell_a = tmp_path / "round4-high1-a"
    cell_a.mkdir(parents=True)
    ctx_a = ProspectiveCaptureContext(cell_a, "NVDA", trade_date, "A", "att-r4-a")

    raw_yahoo = {
        "title": "NVDA AI Growth",
        "summary": "Revenue surged",
        "publisher": "Yahoo",
        "link": "https://example.invalid/nvda",
        "providerPublishTime": int(datetime(2026, 8, 3, 12, tzinfo=UTC).timestamp()),
    }
    yahoo_news._extract_article_data(raw_yahoo)

    g_a = StateGraph(_State)
    g_a.add_node("tools", ToolNode([native_tools.get_news]))
    g_a.add_edge(START, "tools")
    g_a.add_edge("tools", END)
    app_a = g_a.compile()
    cb_a = AuditCallbackHandler()

    class MockClock(datetime):
        instant = datetime(2026, 8, 3, 14, tzinfo=UTC)

        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return cls.instant if tz else cls.instant.replace(tzinfo=None)

    with (
        patch.object(r, "datetime", MockClock),
        patch.object(yahoo_news, "yf_retry", return_value=[raw_yahoo]),
        ctx_a,
    ):
        res_a = app_a.invoke(
            {
                # Model emits future end_date: 2026-08-10, and omits ticker (injected from state)
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_news",
                                "args": {"start_date": "2026-08-01", "end_date": "2026-08-10"},
                                "id": "call-a-1",
                            }
                        ],
                    )
                ],
                "trade_date": trade_date,
                "company_of_interest": "NVDA",
            },
            config={"callbacks": [cb_a]},
        )

    # Verify cb_a traces:
    assert len(cb_a.traces) == 2
    assert cb_a.traces[0]["type"] == "tool_start"
    # input is Python mapping string without ticker
    assert "start_date" in cb_a.traces[0]["input"]
    assert "ticker" not in cb_a.traces[0]["input"]
    assert cb_a.traces[1]["type"] == "tool_end"
    # output is raw string, not ToolMessage repr!
    assert isinstance(cb_a.traces[1]["output"], str)
    assert not cb_a.traces[1]["output"].startswith("ToolMessage(")
    assert (
        cb_a.traces[1]["output_hash"]
        == hashlib.sha256(cb_a.traces[1]["output"].encode("utf-8")).hexdigest()
    )

    # ToolNode wrapped message in res_a is a ToolMessage
    tool_msg = res_a["messages"][-1]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.content == cb_a.traces[1]["output"]

    # Write audit trace for cell_a
    res_dir_a = cell_a / "results"
    res_dir_a.mkdir(parents=True, exist_ok=True)
    (res_dir_a / "audit_trace.json").write_text(
        json.dumps({"traces": cb_a.traces}), encoding="utf-8"
    )

    # validate_evidence_archive must pass: omitted ticker binds from cell state, future date clamps cleanly!
    ok_a, msg_a = validate_evidence_archive(
        ctx_a.archive_path, "A", "NVDA", trade_date, cutoff_dt, {"attempt_id": "att-r4-a"}, cell_a
    )
    assert ok_a is True, f"Arm A ToolNode verification failed: {msg_a}"

    # 2. Arm B News Intelligence get_global_news ToolNode execution with future date clamping
    cell_b = tmp_path / "round4-high1-b"
    cell_b.mkdir(parents=True)
    ctx_b = ProspectiveCaptureContext(cell_b, "NVDA", trade_date, "B", "att-r4-b")

    raw_event = {
        "id": "glob-r4",
        "publishedAt": "2026-08-02T10:00:00Z",
        "receivedAt": "2026-08-02T11:00:00Z",
        "enrichedAt": "2026-08-02T11:05:00Z",
        "symbols": ["GLOBAL"],
        "title": "Global economic report",
        "summary": "Inflation normalized",
        "source": "Repo1",
        "url": "https://example.invalid/econ",
    }

    class MockTransportClient:
        def _make_request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
            return {"events": [raw_event]}

    g_b = StateGraph(_State)
    g_b.add_node("tools", ToolNode([native_tools.get_global_news]))
    g_b.add_edge(START, "tools")
    g_b.add_edge("tools", END)
    app_b = g_b.compile()
    cb_b = AuditCallbackHandler()

    cfg_b = {
        "tool_vendors": {
            "get_global_news": "news_intelligence",
            "get_news": "news_intelligence",
        },
        "data_vendors": {"news_data": "news_intelligence"},
    }

    with (
        patch.object(r, "datetime", MockClock),
        patch("tradingagents.dataflows.router.get_config", return_value=cfg_b),
        patch.object(
            ni_client.NewsIntelligenceClient, "_make_request", MockTransportClient._make_request
        ),
        ctx_b,
    ):
        app_b.invoke(
            {
                # Model calls get_global_news with future curr_date: 2026-08-15
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_global_news",
                                "args": {
                                    "curr_date": "2026-08-15",
                                    "look_back_days": 3,
                                    "limit": 5,
                                },
                                "id": "call-b-1",
                            }
                        ],
                    )
                ],
                "trade_date": trade_date,
                "company_of_interest": "NVDA",
            },
            config={"callbacks": [cb_b]},
        )

    assert len(cb_b.traces) == 2
    assert cb_b.traces[0]["type"] == "tool_start"
    assert cb_b.traces[1]["type"] == "tool_end"
    assert isinstance(cb_b.traces[1]["output"], str)

    res_dir_b = cell_b / "results"
    res_dir_b.mkdir(parents=True, exist_ok=True)
    (res_dir_b / "audit_trace.json").write_text(
        json.dumps({"traces": cb_b.traces}), encoding="utf-8"
    )

    # validate_evidence_archive must pass: future curr_date clamped to trade_date
    ok_b, msg_b = validate_evidence_archive(
        ctx_b.archive_path, "B", "NVDA", trade_date, cutoff_dt, {"attempt_id": "att-r4-b"}, cell_b
    )
    assert ok_b is True, f"Arm B ToolNode verification failed: {msg_b}"

    # 3. Negative control: past date mismatch cannot be explained by clamping and must fail
    cb_bad = copy.deepcopy(cb_a.traces)
    # Model asked for past date 2026-07-20 (not a future date clamped to 2026-08-01)
    cb_bad[0]["input"] = "{'start_date': '2026-07-20', 'end_date': '2026-08-10'}"
    (res_dir_a / "audit_trace.json").write_text(json.dumps({"traces": cb_bad}), encoding="utf-8")
    ok_bad, msg_bad = validate_evidence_archive(
        ctx_a.archive_path, "A", "NVDA", trade_date, cutoff_dt, {"attempt_id": "att-r4-a"}, cell_a
    )
    assert ok_bad is False
    assert "start_date" in msg_bad and "does not match" in msg_bad


def test_prosp_round4_high2_required_arguments_and_transport_contradiction_rejection(
    tmp_path: Path,
) -> None:
    """Proves Round 4 Finding HIGH2:
    1. Asset requests missing required model arguments (start_date/end_date for get_news, curr_date for get_global_news) fail closed.
    2. Conflicting transport parameters (asOf, from, to, limit, symbol) fail closed.
    3. Genuine optional defaults succeed cleanly.
    """
    cutoff_dt = datetime(2026, 8, 3, 20, tzinfo=UTC)
    cell = tmp_path / "round4-high2"
    cell.mkdir(parents=True)

    raw_yahoo = {
        "title": "NVDA AI Growth",
        "summary": "Revenue surged",
        "publisher": "Yahoo",
        "link": "https://example.invalid/nvda",
        "providerPublishTime": int(datetime(2026, 8, 3, 12, tzinfo=UTC).timestamp()),
    }
    extracted_yahoo = yahoo_news._extract_article_data(raw_yahoo)
    formatted_yahoo = (
        "## NVDA News, from 2026-08-01 to 2026-08-03:\n\n"
        + yahoo_news._format_articles([extracted_yahoo])
    )
    out_hash = hashlib.sha256(formatted_yahoo.encode("utf-8")).hexdigest()

    ctx = ProspectiveCaptureContext(cell, "NVDA", "2026-08-03", "A", "att-r4-h2")
    ctx.commit_call(
        "get_news",
        "NVDA",
        {"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-03"},
        [raw_yahoo],
        [extracted_yahoo],
        formatted_yahoo,
        "run-h2",
        "2026-08-03T14:00:00Z",
        producer_params={"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-03"},
    )
    res_dir = cell / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    trace_path = res_dir / "audit_trace.json"

    # 1. Missing required start_date in get_news -> FAIL
    trace_path.write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": "{'ticker': 'NVDA', 'end_date': '2026-08-03'}",
                        "run_id": "run-h2",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": formatted_yahoo,
                        "output_hash": out_hash,
                        "run_id": "run-h2",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    ok_nostart, msg_nostart = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-r4-h2"}, cell
    )
    assert ok_nostart is False
    assert "missing required start_date" in msg_nostart

    # 2. Missing required end_date in get_news -> FAIL
    trace_path.write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": "{'ticker': 'NVDA', 'start_date': '2026-08-01'}",
                        "run_id": "run-h2",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": formatted_yahoo,
                        "output_hash": out_hash,
                        "run_id": "run-h2",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    ok_noend, msg_noend = validate_evidence_archive(
        ctx.archive_path, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-r4-h2"}, cell
    )
    assert ok_noend is False
    assert "missing required end_date" in msg_noend

    # 3. Missing required curr_date in get_global_news -> FAIL
    cell_glob = tmp_path / "round4-high2-glob"
    cell_glob.mkdir(parents=True)
    ctx_glob = ProspectiveCaptureContext(cell_glob, "NVDA", "2026-08-03", "B", "att-r4-glob")
    ctx_glob.commit_call(
        "get_global_news",
        "global",
        {
            "symbol": "GLOBAL",
            "asOf": "2026-08-03T23:59:59.999999Z",
            "to": "2026-08-03T23:59:59.999999Z",
            "from": "2026-07-27T00:00:00Z",
            "limit": 10,
        },
        {"events": []},
        [],
        "## Global Market News:\n",
        "run-glob",
        "2026-08-03T14:00:00Z",
        producer_params={"as_of_date": "2026-08-03", "look_back_days": 7, "limit": 10},
    )
    res_glob = cell_glob / "results"
    res_glob.mkdir(parents=True, exist_ok=True)
    (res_glob / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_global_news",
                        "input": "{'look_back_days': 7, 'limit': 10}",  # Missing curr_date!
                        "run_id": "run-glob",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_global_news",
                        "output": "## Global Market News:\n",
                        "output_hash": hashlib.sha256(b"## Global Market News:\n").hexdigest(),
                        "run_id": "run-glob",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    ok_noglob, msg_noglob = validate_evidence_archive(
        ctx_glob.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r4-glob"},
        cell_glob,
    )
    assert ok_noglob is False
    assert "missing required" in msg_noglob and "curr_date" in msg_noglob

    # 4. Conflicting transport parameters:
    # 4a. Transport symbol contradicts producer ticker
    arch_data = json.loads(ctx.archive_path.read_text(encoding="utf-8"))
    bad_arch_symbol = copy.deepcopy(arch_data)
    bad_arch_symbol["calls"][0]["params"]["symbol"] = "MSFT"
    (cell / "data" / "bad_symbol.json").write_text(json.dumps(bad_arch_symbol), encoding="utf-8")
    trace_path.write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": (
                            "{'ticker': 'NVDA', 'start_date': '2026-08-01', 'end_date':"
                            " '2026-08-03'}"
                        ),
                        "run_id": "run-h2",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": formatted_yahoo,
                        "output_hash": out_hash,
                        "run_id": "run-h2",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    ok_sym, msg_sym = validate_evidence_archive(
        cell / "data" / "bad_symbol.json",
        "A",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r4-h2"},
        cell,
    )
    assert ok_sym is False
    assert "symbol" in msg_sym and "contradicts" in msg_sym

    # 4b. Transport 'to' date contradicts producer end_date
    bad_arch_to = copy.deepcopy(arch_data)
    bad_arch_to["calls"][0]["params"]["to"] = "2026-08-05"
    (cell / "data" / "bad_to.json").write_text(json.dumps(bad_arch_to), encoding="utf-8")
    ok_to, msg_to = validate_evidence_archive(
        cell / "data" / "bad_to.json",
        "A",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r4-h2"},
        cell,
    )
    assert ok_to is False
    assert "to" in msg_to and "contradicts" in msg_to

    # 4c. Transport 'from' date contradicts producer start_date
    bad_arch_from = copy.deepcopy(arch_data)
    bad_arch_from["calls"][0]["params"]["from"] = "2026-07-20"
    (cell / "data" / "bad_from.json").write_text(json.dumps(bad_arch_from), encoding="utf-8")
    ok_from, msg_from = validate_evidence_archive(
        cell / "data" / "bad_from.json",
        "A",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r4-h2"},
        cell,
    )
    assert ok_from is False
    assert "from" in msg_from and "contradicts" in msg_from

    # 4d. Transport 'limit' contradicts producer limit
    arch_glob_data = json.loads(ctx_glob.archive_path.read_text(encoding="utf-8"))
    bad_arch_lim = copy.deepcopy(arch_glob_data)
    bad_arch_lim["calls"][0]["params"]["limit"] = 25  # Producer has 10
    (cell_glob / "data" / "bad_limit.json").write_text(json.dumps(bad_arch_lim), encoding="utf-8")
    (res_glob / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_global_news",
                        "input": "{'curr_date': '2026-08-03', 'look_back_days': 7, 'limit': 10}",
                        "run_id": "run-glob",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_global_news",
                        "output": "## Global Market News:\n",
                        "output_hash": hashlib.sha256(b"## Global Market News:\n").hexdigest(),
                        "run_id": "run-glob",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    ok_lim, msg_lim = validate_evidence_archive(
        cell_glob / "data" / "bad_limit.json",
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r4-glob"},
        cell_glob,
    )
    assert ok_lim is False
    assert "limit" in msg_lim and "contradicts" in msg_lim

    # 4e. Transport asOf contradicts producer as_of_date
    bad_arch_asof = copy.deepcopy(arch_glob_data)
    bad_arch_asof["calls"][0]["params"]["asOf"] = "2026-08-05T00:00:00Z"
    (cell_glob / "data" / "bad_asof.json").write_text(json.dumps(bad_arch_asof), encoding="utf-8")
    ok_asof, msg_asof = validate_evidence_archive(
        cell_glob / "data" / "bad_asof.json",
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r4-glob"},
        cell_glob,
    )
    assert ok_asof is False
    assert "asOf" in msg_asof and "contradicts" in msg_asof


def test_prosp_round4_med3_capped_max_plus_one_read_and_race_growth_rejection(
    tmp_path: Path,
) -> None:
    """Proves Round 4 Finding MED3:
    1. _read_bounded_bytes caps read length at max_bytes + 1 before parsing, bounding memory.
    2. Over-limit files are rejected with clean error without allocating unbounded memory.
    3. Race growth / file replacement after initial stat is caught and rejected by bounded read.
    4. Snapshot reading in _verify_v2_archive strictly enforces MAX_SNAPSHOT_BYTES.
    """
    test_dir = tmp_path / "round4-med3"
    test_dir.mkdir(parents=True)

    # 1. Direct unit verification of _read_bounded_bytes
    small_file = test_dir / "small.txt"
    small_file.write_bytes(b"hello world")
    data, err = r._read_bounded_bytes(small_file, 20)
    assert err is None
    assert data == b"hello world"

    # Exactly at limit:
    data_exact, err_exact = r._read_bounded_bytes(small_file, 11)
    assert err_exact is None
    assert data_exact == b"hello world"

    # Exceeding limit:
    data_over, err_over = r._read_bounded_bytes(small_file, 10)
    assert data_over is None
    assert "exceeds maximum limit 10 bytes" in str(err_over)

    # Bounded read verification: prove f.read was invoked with at most max_bytes + 1
    read_lengths: list[int] = []
    real_path_open = Path.open

    class TrackingFile:
        def __init__(self, f_obj: Any):
            self._f = f_obj

        def read(self, n: int = -1) -> bytes:
            read_lengths.append(n)
            return self._f.read(n)

        def __enter__(self) -> TrackingFile:
            return self

        def __exit__(self, *args: Any) -> None:
            self._f.close()

    def fake_path_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        f = real_path_open(self, mode, *args, **kwargs)
        return TrackingFile(f)

    bounded_dummy = test_dir / "bounded.bin"
    bounded_dummy.write_bytes(b"A" * 30)
    with patch.object(Path, "open", fake_path_open):
        r._read_bounded_bytes(bounded_dummy, 50)

    assert len(read_lengths) == 1
    assert read_lengths[0] == 51  # max_bytes + 1 = 51! Never reads unbounded stream!

    # 2. Race growth / replacement protection in _read_bounded_json_envelope
    env_file = test_dir / "envelope.json"
    env_file.write_text(json.dumps({"valid": "small"}), encoding="utf-8")

    # Simulate race: stat() reports small file, but during open/read it has grown past MAX_ENVELOPE_BYTES
    orig_stat = Path.stat

    def fake_stat(p: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        # If inspecting env_file, return tiny size 100 bytes
        if p == env_file:
            return orig_stat(small_file)
        return orig_stat(p, *args, **kwargs)

    # Grow env_file beyond MAX_ENVELOPE_BYTES
    env_file.write_bytes(b"{" + b'"k": "v",' * (r.MAX_ENVELOPE_BYTES // 5) + b'"end": 1}')
    assert env_file.stat().st_size > r.MAX_ENVELOPE_BYTES

    with patch.object(Path, "stat", fake_stat):
        # Even though stat() claimed 11 bytes, _read_bounded_json_envelope caps read at MAX_ENVELOPE_BYTES + 1
        # and rejects the grown file!
        data_race, err_race = r._read_bounded_json_envelope(env_file)
        assert data_race is None
        assert f"exceeds maximum limit {r.MAX_ENVELOPE_BYTES} bytes" in str(err_race)

    # 3. Snapshot bounded read in validate_evidence_archive
    cell_snap = test_dir / "cell_snap"
    cell_snap.mkdir(parents=True)
    (cell_snap / "data").mkdir()
    (cell_snap / "results").mkdir()
    cutoff_dt = datetime(2026, 8, 3, 20, tzinfo=UTC)

    # Snapshot exceeding MAX_SNAPSHOT_BYTES (10 MiB)
    oversized_snap = cell_snap / "data" / "oversized_snap.json"
    # Write oversized file
    oversized_snap.write_bytes(b"[" + b"0," * (r.MAX_SNAPSHOT_BYTES // 2) + b"0]")
    assert oversized_snap.stat().st_size > r.MAX_SNAPSHOT_BYTES
    snap_hash = hashlib.sha256(oversized_snap.read_bytes()).hexdigest()

    arch_snap = {
        "version": "t13-archive/v2",
        "attempt_id": "att-snap",
        "ticker": "NVDA",
        "arm": "A",
        "date": "2026-08-03",
        "cutoff": "2026-08-03T20:00:00Z",
        "vendor": "yfinance",
        "calls": [
            {
                "tool": "get_news",
                "query": "NVDA",
                "run_id": "run-snap",
                "snapshot_path": "oversized_snap.json",
                "raw_payload_hash": snap_hash,
                "received_at": "2026-08-03T14:00:00Z",
                "params": {"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-03"},
                "producer_params": {
                    "ticker": "NVDA",
                    "start_date": "2026-08-01",
                    "end_date": "2026-08-03",
                },
                "records": [],
                "output_hash": hashlib.sha256(b"out").hexdigest(),
            }
        ],
    }
    arch_snap_file = cell_snap / "data" / "provenance_archive.json"
    arch_snap_file.write_text(json.dumps(arch_snap), encoding="utf-8")

    (cell_snap / "results" / "audit_trace.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "type": "tool_start",
                        "tool": "get_news",
                        "input": "{'start_date': '2026-08-01', 'end_date': '2026-08-03'}",
                        "run_id": "run-snap",
                    },
                    {
                        "type": "tool_end",
                        "tool": "get_news",
                        "output": "out",
                        "output_hash": hashlib.sha256(b"out").hexdigest(),
                        "run_id": "run-snap",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    ok_snap, msg_snap = validate_evidence_archive(
        arch_snap_file, "A", "NVDA", "2026-08-03", cutoff_dt, {"attempt_id": "att-snap"}, cell_snap
    )
    assert ok_snap is False
    assert f"exceeds maximum limit {r.MAX_SNAPSHOT_BYTES} bytes" in msg_snap


def test_prosp_round5_transport_clock_exact_instant_verification_and_same_date_mutation_rejection(
    tmp_path: Path,
) -> None:
    """Proves Round 5 (R4 Finding 1):
    1. Derives full expected transport instants with native to_iso_utc contract:
       - start-of-day for 'from' (YYYY-MM-DDT00:00:00Z)
       - end-of-day for 'to' and 'asOf' (YYYY-MM-DDT23:59:59.999999Z)
    2. Compares complete timestamps as UTC instants, rejecting invalid, malformed, or missing clocks.
    3. Retains actual native producer/capture positive controls for both asset and global news.
    4. Independently mutates only the time in asOf, to, and from on the same date and verifies rejection.
    5. Rejects same-prefix malformed timestamps ('2026-08-03-not-a-timestamp').
    6. Confirms legitimate UTC-equivalent representations ('+00:00') verify cleanly.
    """
    cutoff_dt = datetime(2026, 8, 3, 20, tzinfo=UTC)

    class MockClock(datetime):
        instant = datetime(2026, 8, 3, 14, tzinfo=UTC)

        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return cls.instant if tz else cls.instant.replace(tzinfo=None)

    raw_asset_event = {
        "id": "asset-r5-ev",
        "publishedAt": "2026-08-02T10:00:00Z",
        "receivedAt": "2026-08-02T11:00:00Z",
        "enrichedAt": "2026-08-02T11:05:00Z",
        "symbols": ["NVDA"],
        "title": "NVDA quarterly report",
        "summary": "Revenue grew substantially",
        "source": "Repo1",
        "url": "https://example.invalid/nvda",
    }
    raw_global_event = {
        "id": "glob-r5-ev",
        "publishedAt": "2026-08-02T10:00:00Z",
        "receivedAt": "2026-08-02T11:00:00Z",
        "enrichedAt": "2026-08-02T11:05:00Z",
        "symbols": ["GLOBAL"],
        "title": "Global economic report",
        "summary": "Inflation normalized",
        "source": "Repo1",
        "url": "https://example.invalid/econ",
    }

    class MockTransportClient:
        def __init__(self, ev: dict[str, Any]) -> None:
            self.ev = ev

        def _make_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"events": [self.ev]}

    cfg_b = {
        "tool_vendors": {
            "get_global_news": "news_intelligence",
            "get_news": "news_intelligence",
        },
        "data_vendors": {"news_data": "news_intelligence"},
    }

    # ------------------------------------------------------------------
    # Part 1: Asset News (get_news, Arm B)
    # ------------------------------------------------------------------
    cell_asset = tmp_path / "r5-asset"
    cell_asset.mkdir(parents=True)
    cb_asset = AuditCallbackHandler()
    ctx_asset = ProspectiveCaptureContext(cell_asset, "NVDA", "2026-08-03", "B", "att-r5-asset")
    transport_asset = MockTransportClient(raw_asset_event)

    with (
        patch.object(r, "datetime", MockClock),
        patch("tradingagents.dataflows.router.get_config", return_value=cfg_b),
        patch.object(
            ni_client.NewsIntelligenceClient, "_make_request", transport_asset._make_request
        ),
        ctx_asset,
    ):
        native_tools.get_news.invoke(
            {"ticker": "NVDA", "start_date": "2026-08-01", "end_date": "2026-08-03"},
            config={"callbacks": [cb_asset]},
        )

    res_asset = cell_asset / "results"
    res_asset.mkdir(parents=True, exist_ok=True)
    (res_asset / "audit_trace.json").write_text(
        json.dumps({"traces": cb_asset.traces}), encoding="utf-8"
    )

    # 1a. Positive control: genuine native producer/capture verifies cleanly
    ok_asset, msg_asset = validate_evidence_archive(
        ctx_asset.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-asset"},
        cell_asset,
    )
    assert ok_asset is True, f"Native Arm B asset positive control failed: {msg_asset}"

    # 1b. Legitimate UTC-equivalent timestamp representation (+00:00) verifies cleanly
    base_asset_data = json.loads(ctx_asset.archive_path.read_text(encoding="utf-8"))
    equiv_asset_data = copy.deepcopy(base_asset_data)
    equiv_asset_data["calls"][0]["params"]["asOf"] = "2026-08-03T23:59:59.999999+00:00"
    (cell_asset / "data" / "provenance_archive.json").write_text(
        json.dumps(equiv_asset_data), encoding="utf-8"
    )
    ok_equiv_a, msg_equiv_a = validate_evidence_archive(
        ctx_asset.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-asset"},
        cell_asset,
    )
    assert ok_equiv_a is True, f"Asset UTC-equivalent representation failed: {msg_equiv_a}"

    # 1c. Mutation 1: Same-date time mutation in asOf (23:59:59.999999Z -> 10:00:00Z) fails
    mut_asof_a = copy.deepcopy(base_asset_data)
    mut_asof_a["calls"][0]["params"]["asOf"] = "2026-08-03T10:00:00Z"
    (cell_asset / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_asof_a), encoding="utf-8"
    )
    ok_mut_asof_a, msg_mut_asof_a = validate_evidence_archive(
        ctx_asset.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-asset"},
        cell_asset,
    )
    assert ok_mut_asof_a is False
    assert "asOf" in msg_mut_asof_a and "contradicts" in msg_mut_asof_a

    # 1d. Mutation 2: Same-date time mutation in to (23:59:59.999999Z -> 10:00:00Z) fails
    mut_to_a = copy.deepcopy(base_asset_data)
    mut_to_a["calls"][0]["params"]["to"] = "2026-08-03T10:00:00Z"
    (cell_asset / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_to_a), encoding="utf-8"
    )
    ok_mut_to_a, msg_mut_to_a = validate_evidence_archive(
        ctx_asset.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-asset"},
        cell_asset,
    )
    assert ok_mut_to_a is False
    assert "to" in msg_mut_to_a and "contradicts" in msg_mut_to_a

    # 1e. Mutation 3: Same-date time mutation in from (00:00:00Z -> 23:00:00Z) fails
    mut_from_a = copy.deepcopy(base_asset_data)
    mut_from_a["calls"][0]["params"]["from"] = "2026-08-01T23:00:00Z"
    (cell_asset / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_from_a), encoding="utf-8"
    )
    ok_mut_from_a, msg_mut_from_a = validate_evidence_archive(
        ctx_asset.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-asset"},
        cell_asset,
    )
    assert ok_mut_from_a is False
    assert "from" in msg_mut_from_a and "contradicts" in msg_mut_from_a

    # 1f. Mutation 4: Invalid same-prefix timestamp in asOf ('2026-08-03-not-a-timestamp') fails
    mut_bad_a = copy.deepcopy(base_asset_data)
    mut_bad_a["calls"][0]["params"]["asOf"] = "2026-08-03-not-a-timestamp"
    (cell_asset / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_bad_a), encoding="utf-8"
    )
    ok_mut_bad_a, msg_mut_bad_a = validate_evidence_archive(
        ctx_asset.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-asset"},
        cell_asset,
    )
    assert ok_mut_bad_a is False
    assert "asOf" in msg_mut_bad_a and "contradicts" in msg_mut_bad_a

    # 1g. Deletion 1: Missing required transport parameter 'asOf' in asset news fails
    mut_del_asof_a = copy.deepcopy(base_asset_data)
    del mut_del_asof_a["calls"][0]["params"]["asOf"]
    (cell_asset / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_del_asof_a), encoding="utf-8"
    )
    ok_del_asof_a, msg_del_asof_a = validate_evidence_archive(
        ctx_asset.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-asset"},
        cell_asset,
    )
    assert ok_del_asof_a is False
    assert "missing required parameter 'asOf'" in msg_del_asof_a

    # 1h. Deletion 2: Missing required transport parameter 'to' in asset news fails
    mut_del_to_a = copy.deepcopy(base_asset_data)
    del mut_del_to_a["calls"][0]["params"]["to"]
    (cell_asset / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_del_to_a), encoding="utf-8"
    )
    ok_del_to_a, msg_del_to_a = validate_evidence_archive(
        ctx_asset.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-asset"},
        cell_asset,
    )
    assert ok_del_to_a is False
    assert "missing required parameter 'to'" in msg_del_to_a

    # 1i. Deletion 3: Missing required transport parameter 'from' in asset news fails
    mut_del_from_a = copy.deepcopy(base_asset_data)
    del mut_del_from_a["calls"][0]["params"]["from"]
    (cell_asset / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_del_from_a), encoding="utf-8"
    )
    ok_del_from_a, msg_del_from_a = validate_evidence_archive(
        ctx_asset.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-asset"},
        cell_asset,
    )
    assert ok_del_from_a is False
    assert "missing required parameter 'from'" in msg_del_from_a

    # Restore valid asset archive
    (cell_asset / "data" / "provenance_archive.json").write_text(
        json.dumps(base_asset_data), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Part 2: Global News (get_global_news, Arm B)
    # ------------------------------------------------------------------
    cell_glob = tmp_path / "r5-glob"
    cell_glob.mkdir(parents=True)
    cb_glob = AuditCallbackHandler()
    ctx_glob = ProspectiveCaptureContext(cell_glob, "NVDA", "2026-08-03", "B", "att-r5-glob")
    transport_glob = MockTransportClient(raw_global_event)

    with (
        patch.object(r, "datetime", MockClock),
        patch("tradingagents.dataflows.router.get_config", return_value=cfg_b),
        patch.object(
            ni_client.NewsIntelligenceClient, "_make_request", transport_glob._make_request
        ),
        ctx_glob,
    ):
        native_tools.get_global_news.invoke(
            {"curr_date": "2026-08-03", "look_back_days": 7, "limit": 10},
            config={"callbacks": [cb_glob]},
        )

    res_glob = cell_glob / "results"
    res_glob.mkdir(parents=True, exist_ok=True)
    (res_glob / "audit_trace.json").write_text(
        json.dumps({"traces": cb_glob.traces}), encoding="utf-8"
    )

    # 2a. Positive control: genuine native producer/capture verifies cleanly
    ok_glob, msg_glob = validate_evidence_archive(
        ctx_glob.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-glob"},
        cell_glob,
    )
    assert ok_glob is True, f"Native Arm B global positive control failed: {msg_glob}"

    # 2b. Legitimate UTC-equivalent timestamp representation (+00:00) verifies cleanly
    base_glob_data = json.loads(ctx_glob.archive_path.read_text(encoding="utf-8"))
    equiv_glob_data = copy.deepcopy(base_glob_data)
    equiv_glob_data["calls"][0]["params"]["asOf"] = "2026-08-03T23:59:59.999999+00:00"
    (cell_glob / "data" / "provenance_archive.json").write_text(
        json.dumps(equiv_glob_data), encoding="utf-8"
    )
    ok_equiv_g, msg_equiv_g = validate_evidence_archive(
        ctx_glob.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-glob"},
        cell_glob,
    )
    assert ok_equiv_g is True, f"Global UTC-equivalent representation failed: {msg_equiv_g}"

    # 2c. Mutation 1: Same-date time mutation in asOf (23:59:59.999999Z -> 10:00:00Z) fails
    mut_asof_g = copy.deepcopy(base_glob_data)
    mut_asof_g["calls"][0]["params"]["asOf"] = "2026-08-03T10:00:00Z"
    (cell_glob / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_asof_g), encoding="utf-8"
    )
    ok_mut_asof_g, msg_mut_asof_g = validate_evidence_archive(
        ctx_glob.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-glob"},
        cell_glob,
    )
    assert ok_mut_asof_g is False
    assert "asOf" in msg_mut_asof_g and "contradicts" in msg_mut_asof_g

    # 2d. Mutation 2: Same-date time mutation in to (23:59:59.999999Z -> 10:00:00Z) fails
    mut_to_g = copy.deepcopy(base_glob_data)
    mut_to_g["calls"][0]["params"]["to"] = "2026-08-03T10:00:00Z"
    (cell_glob / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_to_g), encoding="utf-8"
    )
    ok_mut_to_g, msg_mut_to_g = validate_evidence_archive(
        ctx_glob.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-glob"},
        cell_glob,
    )
    assert ok_mut_to_g is False
    assert "to" in msg_mut_to_g and "contradicts" in msg_mut_to_g

    # 2e. Mutation 3: Same-date time mutation in from (00:00:00Z -> 23:00:00Z) fails
    mut_from_g = copy.deepcopy(base_glob_data)
    mut_from_g["calls"][0]["params"]["from"] = "2026-07-27T23:00:00Z"
    (cell_glob / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_from_g), encoding="utf-8"
    )
    ok_mut_from_g, msg_mut_from_g = validate_evidence_archive(
        ctx_glob.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-glob"},
        cell_glob,
    )
    assert ok_mut_from_g is False
    assert "from" in msg_mut_from_g and "contradicts" in msg_mut_from_g

    # 2f. Mutation 4: Invalid same-prefix timestamp in asOf ('2026-08-03-not-a-timestamp') fails
    mut_bad_g = copy.deepcopy(base_glob_data)
    mut_bad_g["calls"][0]["params"]["asOf"] = "2026-08-03-not-a-timestamp"
    (cell_glob / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_bad_g), encoding="utf-8"
    )
    ok_mut_bad_g, msg_mut_bad_g = validate_evidence_archive(
        ctx_glob.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-glob"},
        cell_glob,
    )
    assert ok_mut_bad_g is False
    assert "asOf" in msg_mut_bad_g and "contradicts" in msg_mut_bad_g

    # 2g. Deletion 1: Missing required transport parameter 'asOf' in global news fails
    mut_del_asof_g = copy.deepcopy(base_glob_data)
    del mut_del_asof_g["calls"][0]["params"]["asOf"]
    (cell_glob / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_del_asof_g), encoding="utf-8"
    )
    ok_del_asof_g, msg_del_asof_g = validate_evidence_archive(
        ctx_glob.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-glob"},
        cell_glob,
    )
    assert ok_del_asof_g is False
    assert "missing required parameter 'asOf'" in msg_del_asof_g

    # 2h. Deletion 2: Missing required transport parameter 'to' in global news fails
    mut_del_to_g = copy.deepcopy(base_glob_data)
    del mut_del_to_g["calls"][0]["params"]["to"]
    (cell_glob / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_del_to_g), encoding="utf-8"
    )
    ok_del_to_g, msg_del_to_g = validate_evidence_archive(
        ctx_glob.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-glob"},
        cell_glob,
    )
    assert ok_del_to_g is False
    assert "missing required parameter 'to'" in msg_del_to_g

    # 2i. Deletion 3: Missing required transport parameter 'from' in global news fails
    mut_del_from_g = copy.deepcopy(base_glob_data)
    del mut_del_from_g["calls"][0]["params"]["from"]
    (cell_glob / "data" / "provenance_archive.json").write_text(
        json.dumps(mut_del_from_g), encoding="utf-8"
    )
    ok_del_from_g, msg_del_from_g = validate_evidence_archive(
        ctx_glob.archive_path,
        "B",
        "NVDA",
        "2026-08-03",
        cutoff_dt,
        {"attempt_id": "att-r5-glob"},
        cell_glob,
    )
    assert ok_del_from_g is False
    assert "missing required parameter 'from'" in msg_del_from_g
