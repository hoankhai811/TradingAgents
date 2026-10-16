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
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel

from experiments.t13_native_ab import (
    AuditCallbackHandler,
    JournalPersistenceError,
    JournalVerificationError,
    ManifestMismatchError,
    OwnershipError,
    _build_arm_config,
    _persist_journal,
    acquire_experiment_owner,
    compute_manifest_hash,
    main,
    run_experiment,
    sanitize_text,
    settle_experiment,
    summarize_experiment,
    validate_manifest,
)
from tradingagents.agents.rating import RATING_REVIEW
from tradingagents.dataflows.router import get_vendor
from tradingagents.llm_clients.factory import create_tier_client, tier_provider
from tradingagents.memory.log import TradingMemoryLog
from tradingagents.memory.settlement import Settlement
from tradingagents.portfolio import PortfolioContext


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

    # Only Pair 1 is eligible!
    assert summary["settled_pairs_count"] == 1
    assert (
        summary["unmatched_vintages_count"] == 1
    )  # AAPL rejected due to different resolution dates
    assert summary["orphaned_settlements_count"] >= 1  # ORPHAN rejected
    assert summary["mean_alpha_A"] == 0.030
    assert summary["mean_alpha_B"] == 0.030
    assert summary["alpha_difference"] == 0.000


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

    assert summary["settled_pairs_count"] == 2
    assert (
        summary["scored_pairs_count"] == 1
    )  # Only Pair 2 is scored; Pair 1 is unscored due to REVIEW
    assert summary["unscored_pairs_count"] == 1

    # Mean alphas only include the scored pair (BTC-USD: -0.010)
    assert summary["mean_alpha_A"] == -0.010
    assert summary["mean_alpha_B"] == -0.010
    assert summary["alpha_difference"] == 0.0

    # Hold has direction 0 -> zero directional count and zero hits
    assert summary["directional_count_A"] == 0
    assert summary["directional_hits_A"] == 0
    assert summary["directional_hit_rate_A"] is None

    assert summary["directional_count_B"] == 0
    assert summary["directional_hits_B"] == 0
    assert summary["directional_hit_rate_B"] is None


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
    assert summary["scored_pairs_count"] == 1
    assert summary["provenance_verified"] is True
    assert summary["timing_eligible"] is True


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
    assert summary["timing_eligible"] is True
    assert summary["provenance_verified"] is True
    assert summary["scored_pairs_count"] == 1


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
    assert s["scored_pairs_count"] == 1
    assert s["timing_eligible"] is True
    assert s["provenance_verified"] is True

    # 2. Both supported JSON shapes: list shape for both arms also passes
    list_A = [copy.deepcopy(base_A)]
    list_B = [copy.deepcopy(base_B)]
    write_archives(list_A, list_B)
    s = summarize_experiment(out)
    assert s["scored_pairs_count"] == 1
    assert s["provenance_verified"] is True

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
    assert s["scored_pairs_count"] == 1
    assert s["provenance_verified"] is True


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
    assert s1["scored_pairs_count"] == 1
    assert s1["provenance_verified"] is True
    assert s1["timing_eligible"] is True

    # 2. List archive shape: Fully linked matching capture passes
    list_A = [dict(base_A)]
    list_B = [dict(base_B)]
    write_archives(list_A, list_B)
    s2 = summarize_experiment(out)
    assert s2["scored_pairs_count"] == 1
    assert s2["provenance_verified"] is True
    assert s2["timing_eligible"] is True

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
    assert s_pos["timing_eligible"] is True
    assert s_pos["scored_pairs_count"] == 1

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
    assert s_pos["scored_pairs_count"] == 1
    assert s_pos["provenance_verified"] is True
    assert s_pos["timing_eligible"] is True

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
    assert s_utc["timing_eligible"] is True
    assert s_offset["timing_eligible"] is True
    assert s_utc["scored_pairs_count"] == 1
    assert s_offset["scored_pairs_count"] == 1
    assert s_utc["provenance_verified"] is True
    assert s_offset["provenance_verified"] is True

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
    assert s_early["scored_pairs_count"] == 1
    assert s_early["provenance_verified"] is True
    assert s_early["timing_eligible"] is True

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
    assert s_plain["scored_pairs_count"] == 1
    assert s_plain["provenance_verified"] is True
    assert s_plain["timing_eligible"] is True


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
    assert s_rev["scored_pairs_count"] == 1
    assert s_rev["provenance_verified"] is True
    assert s_rev["timing_eligible"] is True

    # 1b. Publisher 'Reuters' with routing vendor 'repo1'
    write_experiment("Reuters", arm_b_vendor="repo1")
    s_reuters = summarize_experiment(out)
    assert s_reuters["scored_pairs_count"] == 1
    assert s_reuters["provenance_verified"] is True
    assert s_reuters["timing_eligible"] is True

    # 1c. Publisher 'Bloomberg' with routing vendor 'news_intelligence'
    write_experiment("Bloomberg", arm_b_vendor="news_intelligence")
    s_bloomberg = summarize_experiment(out)
    assert s_bloomberg["scored_pairs_count"] == 1
    assert s_bloomberg["provenance_verified"] is True
    assert s_bloomberg["timing_eligible"] is True

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
    assert s_pos["scored_pairs_count"] == 1
    assert s_pos["provenance_verified"] is True
    assert s_pos["timing_eligible"] is True


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
    assert s_pure["scored_pairs_count"] == 1
    assert s_pure["provenance_verified"] is True
    assert s_pure["timing_eligible"] is True

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
    assert s_multi["scored_pairs_count"] == 1
    assert s_multi["provenance_verified"] is True
    assert s_multi["timing_eligible"] is True

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
        assert s_var["scored_pairs_count"] == 1, f"Failed for variant {label}"
        assert s_var["provenance_verified"] is True, f"Failed provenance for variant {label}"
        assert s_var["timing_eligible"] is True, f"Failed timing for variant {label}"

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
    assert s_mixed["scored_pairs_count"] == 1
    assert s_mixed["provenance_verified"] is True
    assert s_mixed["timing_eligible"] is True

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
