"""T13 Genuine Native A/B Experiment Runner.

Produces authentic paired news-vendor decisions and native settled decision-quality
metrics under identical controls, preserving every incomplete or failed pair.
Reuses native TradingAgents graph, memory log, and settlement facilities.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import sys
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dateutil.relativedelta import relativedelta
from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.agents.rating import RATING_REVIEW
from tradingagents.backtest import _DIRECTION, BacktestSummary, _alpha, summarize
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.files import locked
from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component
from tradingagents.default_config import build_default_config
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS
from tradingagents.graph.trading_graph import TradingAgentsGraph as _NativeTradingAgentsGraph
from tradingagents.llm_clients.errors import is_provider_limit_error
from tradingagents.memory import settlement
from tradingagents.memory.log import TradingMemoryLog
from tradingagents.portfolio import PortfolioContext
from tradingagents.service.worker import ensure_provider_configured

logger = logging.getLogger(__name__)


class TradingAgentsGraph(_NativeTradingAgentsGraph):
    """Native graph wrapper ensuring tool execution callbacks are wired through propagator."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._saved_callbacks = kwargs.get("callbacks")
        self._propagator: Any = None
        super().__init__(*args, **kwargs)

    @property
    def propagator(self) -> Any:
        return getattr(self, "_propagator", None)

    @propagator.setter
    def propagator(self, value: Any) -> None:
        self._propagator = value
        saved_cbs = getattr(self, "_saved_callbacks", None)
        if value and hasattr(value, "get_graph_args") and saved_cbs:
            orig = value.get_graph_args

            def _get_graph_args_with_cb(*a: Any, **kw: Any) -> dict[str, Any]:
                kw["callbacks"] = kw.get("callbacks") or saved_cbs
                return orig(*a, **kw)

            value.get_graph_args = _get_graph_args_with_cb


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OwnershipError(RuntimeError):
    """Raised when exclusive experiment lock cannot be acquired."""


class JournalPersistenceError(RuntimeError):
    """Raised when the journal cannot be written/replaced atomically."""


class JournalVerificationError(RuntimeError):
    """Raised when journal read-back verification fails."""


class ManifestMismatchError(ValueError):
    """Raised when the active manifest hash does not match the journal."""


class EvidenceStorageError(RuntimeError):
    """Raised when prospective evidence capture fails to persist snapshot or envelope."""


# ---------------------------------------------------------------------------
# Secret Scrubbing & Sanitization
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(
        r"(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]+['\"]?", re.IGNORECASE
    ),
]


def sanitize_text(text: Any) -> str:
    """Sanitize strings by redacting secrets, API keys, and sensitive tokens."""
    if not isinstance(text, str):
        text = str(text)
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


# ---------------------------------------------------------------------------
# Callback for Request Accounting & Tool/Model Trace Auditing (R6)
# ---------------------------------------------------------------------------

_current_tool_run_id = threading.local()
_thread_capture_state = threading.local()


class AuditCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler capturing model calls, tool messages, and usage.

    Excludes secrets, credentials, and sensitive headers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_calls: int = 0
        self.tool_calls: int = 0
        self.has_unknown_usage: bool = False
        self.token_usage: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "has_unknown_usage": False,
        }
        self.pending_calls: int = 0
        self.traces: list[dict[str, Any]] = []
        self.non_news_tool_hashes: list[str] = []
        self._pending_tool_calls: list[dict[str, Any]] = []
        self._pending_tool_calls_by_run_id: dict[str, dict[str, Any]] = {}
        self._active_run_ids: set[str] = set()

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        if run_id:
            if run_id in self._active_run_ids:
                return
            self._active_run_ids.add(run_id)

        self.model_calls += 1
        self.pending_calls += 1
        model_name = serialized.get("name") or serialized.get("id", ["unknown"])[-1]
        cleaned_prompts = [sanitize_text(p) for p in prompts]
        hashes = [hashlib.sha256(p.encode("utf-8")).hexdigest() for p in prompts]
        self.traces.append(
            {
                "type": "llm_start",
                "model": str(model_name),
                "prompt_count": len(prompts),
                "prompt_hashes": hashes,
                "prompts": cleaned_prompts,
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": run_id or None,
            }
        )

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id") or "")
        if run_id:
            if run_id in self._active_run_ids:
                self._record_tool_messages(messages)
                return
            self._active_run_ids.add(run_id)

        self.model_calls += 1
        self.pending_calls += 1
        model_name = serialized.get("name") or serialized.get("id", ["unknown"])[-1]

        cleaned_prompts: list[str] = []
        hashes: list[str] = []
        for msg_list in messages:
            for msg in msg_list:
                msg_type = getattr(msg, "type", None) or type(msg).__name__
                content = getattr(msg, "content", "")
                c_str = f"[{msg_type}] {content}"
                cleaned = sanitize_text(c_str)
                cleaned_prompts.append(cleaned)
                hashes.append(hashlib.sha256(cleaned.encode("utf-8")).hexdigest())

        self.traces.append(
            {
                "type": "llm_start",
                "model": str(model_name),
                "prompt_count": len(cleaned_prompts),
                "prompt_hashes": hashes,
                "prompts": cleaned_prompts,
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": run_id or None,
            }
        )
        self._record_tool_messages(messages)

    def _record_tool_messages(self, messages: list[list[Any]]) -> None:
        for msg_list in messages:
            for msg in msg_list:
                msg_type = getattr(msg, "type", None) or type(msg).__name__
                if msg_type in ("tool", "ToolMessage"):
                    tool_name = str(getattr(msg, "name", "tool") or "tool")
                    content = getattr(msg, "content", "")
                    c_str = str(content)
                    if not tool_name.startswith("get_news") and not tool_name.startswith(
                        "get_global_news"
                    ):
                        c_hash = hashlib.sha256(
                            f"consumed_tool_message:{tool_name}:{c_str}".encode()
                        ).hexdigest()
                        if c_hash not in self.non_news_tool_hashes:
                            self.non_news_tool_hashes.append(c_hash)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        if run_id:
            self._active_run_ids.discard(run_id)
        self.pending_calls = max(0, self.pending_calls - 1)
        usage: dict[str, Any] = {}
        found_usage = False

        # 1. Check response.llm_output
        if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
            raw_usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
                found_usage = True

        # 2. Check ChatGeneration message usage_metadata
        if (
            not found_usage
            and hasattr(response, "generations")
            and isinstance(response.generations, list)
        ):
            for gen_list in response.generations:
                if not isinstance(gen_list, list):
                    continue
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg is not None:
                        meta = getattr(msg, "usage_metadata", None)
                        if isinstance(meta, dict) and meta:
                            usage = meta
                            found_usage = True
                            break
                    gen_info = getattr(gen, "generation_info", None)
                    if isinstance(gen_info, dict) and "usage" in gen_info:
                        usage = gen_info["usage"]
                        found_usage = True
                        break
                if found_usage:
                    break

        if found_usage:
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            total_tokens = usage.get("total_tokens") or (
                int(prompt_tokens) + int(completion_tokens)
            )

            self.token_usage["input_tokens"] += int(prompt_tokens)
            self.token_usage["output_tokens"] += int(completion_tokens)
            self.token_usage["total_tokens"] += int(total_tokens)
        else:
            self.has_unknown_usage = True
            self.token_usage["has_unknown_usage"] = True

        # Capture generation text hashes and full sanitized outputs (R3, R6)
        gen_hashes: list[str] = []
        gen_texts: list[str] = []
        if hasattr(response, "generations") and isinstance(response.generations, list):
            for gen_list in response.generations:
                if isinstance(gen_list, list):
                    for gen in gen_list:
                        text = getattr(gen, "text", None) or (
                            getattr(getattr(gen, "message", None), "content", "")
                        )
                        if text:
                            gen_hashes.append(hashlib.sha256(str(text).encode("utf-8")).hexdigest())
                            gen_texts.append(sanitize_text(str(text)))

        self.traces.append(
            {
                "type": "llm_end",
                "usage": usage if found_usage else None,
                "has_unknown_usage": not found_usage,
                "generation_hashes": gen_hashes,
                "generations": gen_texts,
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": run_id or None,
            }
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        if run_id:
            self._active_run_ids.discard(run_id)
        self.pending_calls = max(0, self.pending_calls - 1)
        # R7: Failed request must report unknown usage
        self.has_unknown_usage = True
        self.token_usage["has_unknown_usage"] = True
        self.traces.append(
            {
                "type": "llm_error",
                "error": sanitize_text(str(error)),
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": run_id or None,
            }
        )

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self.tool_calls += 1
        tool_name = str(serialized.get("name", "tool"))
        clean_input = sanitize_text(input_str)
        inp_hash = hashlib.sha256(input_str.encode("utf-8")).hexdigest()
        run_id = str(kwargs.get("run_id") or "")
        _current_tool_run_id.value = run_id or None
        call_rec = {
            "name": tool_name,
            "input": input_str,
            "input_hash": inp_hash,
            "run_id": run_id,
        }
        if run_id:
            self._pending_tool_calls_by_run_id[run_id] = call_rec
        else:
            self._pending_tool_calls.append(call_rec)

        self.traces.append(
            {
                "type": "tool_start",
                "tool": tool_name,
                "input_hash": inp_hash,
                "input": clean_input,
                "run_id": run_id or None,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        _current_tool_run_id.value = None
        if hasattr(output, "content"):
            out_str = str(output.content)
        elif isinstance(output, dict) and "content" in output:
            out_str = str(output["content"])
        else:
            out_str = str(output)
        clean_out = sanitize_text(out_str)
        out_hash = hashlib.sha256(out_str.encode("utf-8")).hexdigest()
        run_id = str(kwargs.get("run_id") or "")
        if run_id and run_id in self._pending_tool_calls_by_run_id:
            pending = self._pending_tool_calls_by_run_id.pop(run_id)
        elif self._pending_tool_calls:
            pending = self._pending_tool_calls.pop()
        else:
            pending = None

        tool_name = pending["name"] if pending else "tool"
        inp_str = pending["input"] if pending else ""
        inp_hash = pending["input_hash"] if pending else ""

        if not tool_name.startswith("get_news") and not tool_name.startswith("get_global_news"):
            composite = f"{tool_name}:{inp_hash}:{out_hash}:{inp_str}:{out_str}"
            item_hash = hashlib.sha256(composite.encode("utf-8")).hexdigest()
            self.non_news_tool_hashes.append(item_hash)

        self.traces.append(
            {
                "type": "tool_end",
                "tool": tool_name,
                "output_hash": out_hash,
                "output_length": len(out_str),
                "output": clean_out,
                "output_preview": clean_out[:500] if len(clean_out) > 500 else clean_out,
                "run_id": run_id or None,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        _current_tool_run_id.value = None
        run_id = str(kwargs.get("run_id") or "")
        if run_id and run_id in self._pending_tool_calls_by_run_id:
            pending = self._pending_tool_calls_by_run_id.pop(run_id)
        elif self._pending_tool_calls:
            pending = self._pending_tool_calls.pop()
        else:
            pending = None

        tool_name = pending["name"] if pending else "tool"
        err_str = str(error)
        clean_err = sanitize_text(err_str)
        err_hash = hashlib.sha256(err_str.encode("utf-8")).hexdigest()
        if not tool_name.startswith("get_news") and not tool_name.startswith("get_global_news"):
            composite = f"{tool_name}:ERROR:{err_hash}:{clean_err}"
            item_hash = hashlib.sha256(composite.encode("utf-8")).hexdigest()
            self.non_news_tool_hashes.append(item_hash)
        self.traces.append(
            {
                "type": "tool_error",
                "tool": tool_name,
                "error": clean_err,
                "run_id": run_id or None,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )


# ---------------------------------------------------------------------------
# Locking & Atomic Journal Persistence
# ---------------------------------------------------------------------------


@contextmanager
def acquire_experiment_owner(output_root: Path) -> Iterator[None]:
    """Hold an exclusive owner lock on the output directory across processes.

    Does not remove, rotate, or replace the lock file with the journal.
    """
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_base = output_root / ".experiment-owner"
    with locked(lock_base, wait=False) as held:
        if not held:
            raise OwnershipError(
                f"Exclusive experiment lock could not be acquired for {output_root}"
            )
        yield


def _persist_journal(output_root: Path, journal: dict[str, Any]) -> None:
    """Atomically write and fsync journal.json, then read back to verify."""
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    journal_path = output_root / "journal.json"
    temp_path = output_root / f"journal.json.tmp.{uuid.uuid4().hex}"
    serialized = json.dumps(journal, indent=2, sort_keys=True)

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, journal_path)
    except Exception as exc:
        if temp_path.exists():
            with suppress(OSError):
                temp_path.unlink()
        raise JournalPersistenceError(f"Failed to persist journal atomically: {exc}") from exc

    # Fail-closed read-back verification
    try:
        read_back_text = journal_path.read_text(encoding="utf-8")
        read_back_json = json.loads(read_back_text)
        if read_back_json != journal:
            raise JournalVerificationError("Journal read-back mismatch: persisted content differs")
    except Exception as exc:
        raise JournalVerificationError(f"Journal read-back verification failed: {exc}") from exc


def compute_manifest_hash(manifest: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of the canonical manifest payload."""
    canonical_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest Validation (R3)
# ---------------------------------------------------------------------------


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize immutable experiment manifest payload.

    Enforces canonical dates/tickers, holding period, pinned Google model controls,
    and rejects arbitrary endpoints, secrets, and future dates.
    """
    if not isinstance(payload, dict):
        raise ValueError("Manifest payload must be a JSON object / dictionary")

    experiment_id = payload.get("experiment_id")
    if not experiment_id or not isinstance(experiment_id, str):
        raise ValueError("Manifest must contain a non-empty string 'experiment_id'")

    pairs = payload.get("pairs")
    if not pairs or not isinstance(pairs, list):
        raise ValueError("Manifest must contain a non-empty list of 'pairs'")

    holding_period = payload.get("holding_period", 5)
    if (
        isinstance(holding_period, bool)
        or not isinstance(holding_period, int)
        or holding_period <= 0
    ):
        raise ValueError("holding_period must be a positive integer")

    benchmark_ticker = payload.get("benchmark_ticker", "SPY")
    if not benchmark_ticker or not isinstance(benchmark_ticker, str):
        raise ValueError("benchmark_ticker must be a non-empty string")
    safe_benchmark = safe_ticker_component(benchmark_ticker.strip().upper())

    model_controls = payload.get("model_controls")
    if not model_controls or not isinstance(model_controls, dict):
        raise ValueError("Manifest must contain 'model_controls' dictionary")

    # Enforce pinned-Google plan strictly (R3)
    provider = model_controls.get("provider")
    if provider != "google":
        raise ValueError(f"model_controls.provider must be strictly 'google', got {provider!r}")

    quick_model = model_controls.get("quick_model")
    deep_model = model_controls.get("deep_model")
    if not quick_model or not isinstance(quick_model, str) or quick_model.strip() == "":
        raise ValueError("model_controls must specify non-empty string 'quick_model'")
    if not deep_model or not isinstance(deep_model, str) or deep_model.strip() == "":
        raise ValueError("model_controls must specify non-empty string 'deep_model'")

    # Validate analyst list against native ANALYST_NODE_SPECS
    analysts = model_controls.get("analysts", ["market", "news"])
    if not isinstance(analysts, list) or not analysts:
        raise ValueError("model_controls.analysts must be a non-empty list of analyst names")
    allowed_analysts = set(ANALYST_NODE_SPECS.keys())
    for a in analysts:
        if a not in allowed_analysts:
            raise ValueError(
                f"Invalid analyst '{a}' in model_controls.analysts. Allowed: {sorted(allowed_analysts)}"
            )

    # Strict numeric control types and ranges
    if "temperature" in model_controls:
        temp = model_controls["temperature"]
        if (
            isinstance(temp, bool)
            or not isinstance(temp, (int, float))
            or not (0.0 <= float(temp) <= 2.0)
        ):
            raise ValueError(
                f"model_controls.temperature must be a float between 0.0 and 2.0, got {temp!r}"
            )

    if "max_tokens" in model_controls:
        mt = model_controls["max_tokens"]
        if isinstance(mt, bool) or not isinstance(mt, int) or mt <= 0:
            raise ValueError(f"model_controls.max_tokens must be a positive integer, got {mt!r}")

    for k in ("max_retries", "llm_max_retries"):
        if k in model_controls:
            mr = model_controls[k]
            if isinstance(mr, bool) or not isinstance(mr, int) or mr < 0:
                raise ValueError(f"model_controls.{k} must be a non-negative integer, got {mr!r}")

    for k in (
        "max_debate_rounds",
        "max_risk_discuss_rounds",
        "max_tool_rounds",
        "recursion_limit",
        "max_recur_limit",
    ):
        if k in model_controls:
            val = model_controls[k]
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise ValueError(f"model_controls.{k} must be a positive integer, got {val!r}")

    # Validate optional portfolio with native PortfolioContext (R2)
    portfolio_data = payload.get("portfolio")
    if portfolio_data is not None:
        try:
            PortfolioContext.model_validate(portfolio_data)
        except Exception as exc:
            raise ValueError(f"Invalid portfolio in manifest: {exc}") from exc

    # Reject secrets and arbitrary provider endpoint URLs
    forbidden_substrings = ("api_key", "password", "secret", "credentials", "private_key")
    for k in list(payload.keys()) + list(model_controls.keys()):
        k_lower = k.lower()
        if (
            any(f in k_lower for f in forbidden_substrings)
            or k_lower in ("token", "auth_token", "access_token")
            or (k_lower.endswith("_token") and k_lower != "max_tokens")
        ):
            raise ValueError(f"Manifest must not contain secret or credential key '{k}'")

    for k, v in model_controls.items():
        if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"Arbitrary endpoint URL in model_controls is forbidden: '{k}'='{v}'")

    # Validate each pair
    seen_pair_keys: set[str] = set()
    validated_pairs: list[dict[str, str]] = []
    today_utc = datetime.now(UTC).strftime("%Y-%m-%d")

    for idx, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"Pair at index {idx} must be a dictionary")
        raw_ticker = pair.get("ticker")
        date_str = pair.get("date")
        asset_type = pair.get("asset_type", "stock")

        if not raw_ticker or not isinstance(raw_ticker, str) or not raw_ticker.strip():
            raise ValueError(f"Pair at index {idx} missing valid non-empty 'ticker'")
        if not date_str or not isinstance(date_str, str):
            raise ValueError(f"Pair at index {idx} missing valid 'date'")
        if asset_type not in ("stock", "crypto"):
            raise ValueError(f"Pair at index {idx} asset_type must be 'stock' or 'crypto'")

        # Safe ticker component check (reject slashes, backslashes, dots, path traversal)
        ticker = raw_ticker.strip().upper()
        if "/" in ticker or "\\" in ticker or ".." in ticker or ticker.startswith("."):
            raise ValueError(f"Pair ticker '{raw_ticker}' contains path traversal characters")
        try:
            safe_ticker = safe_ticker_component(ticker)
        except ValueError as exc:
            raise ValueError(
                f"Pair ticker '{raw_ticker}' failed safe ticker component check: {exc}"
            ) from exc

        # Strict round-trip canonical date validation (rejects 2026-08-3)
        try:
            parsed_dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Pair date '{date_str}' is not in canonical YYYY-MM-DD format"
            ) from None

        if parsed_dt.strftime("%Y-%m-%d") != date_str:
            raise ValueError(
                f"Pair date '{date_str}' is not in canonical YYYY-MM-DD format (must be 0-padded)"
            )

        if date_str > today_utc:
            raise ValueError(
                f"Pair date '{date_str}' is in the future relative to today ({today_utc})"
            )

        pair_key = f"{safe_ticker}_{date_str}"
        if pair_key in seen_pair_keys:
            raise ValueError(f"Duplicate pair detected in manifest: {pair_key}")
        seen_pair_keys.add(pair_key)

        validated_pairs.append(
            {
                "ticker": safe_ticker,
                "date": date_str,
                "asset_type": asset_type,
            }
        )

    normalized = copy.deepcopy(payload)
    normalized["pairs"] = validated_pairs
    normalized["holding_period"] = holding_period
    normalized["benchmark_ticker"] = safe_benchmark
    return normalized


# ---------------------------------------------------------------------------
# Config Builder with Exact Native Keys (R1)
# ---------------------------------------------------------------------------


def _build_base_config(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build and validate frozen effective base configuration with native keys."""
    config = build_default_config()
    mc = manifest["model_controls"]

    config["llm_provider"] = "google"
    config["quick_think_provider"] = "google"
    config["deep_think_provider"] = "google"
    config["quick_think_llm"] = str(mc["quick_model"])
    config["deep_think_llm"] = str(mc["deep_model"])

    thinking_level = mc.get("thinking_level") or mc.get("google_thinking_level")
    if thinking_level is not None:
        config["google_thinking_level"] = str(thinking_level)

    if "temperature" in mc:
        config["temperature"] = float(mc["temperature"])
    if "max_tokens" in mc:
        config["max_tokens"] = int(mc["max_tokens"])

    retries = mc.get("max_retries") if "max_retries" in mc else mc.get("llm_max_retries")
    if retries is not None:
        config["llm_max_retries"] = int(retries)

    if "max_debate_rounds" in mc:
        config["max_debate_rounds"] = int(mc["max_debate_rounds"])
    if "max_risk_discuss_rounds" in mc:
        config["max_risk_discuss_rounds"] = int(mc["max_risk_discuss_rounds"])
    if "max_tool_rounds" in mc:
        config["max_tool_rounds"] = int(mc["max_tool_rounds"])

    recur = mc.get("recursion_limit") if "recursion_limit" in mc else mc.get("max_recur_limit")
    if recur is not None:
        config["max_recur_limit"] = int(recur)

    lang = mc.get("language") if "language" in mc else mc.get("output_language")
    if lang is not None:
        config["output_language"] = str(lang)

    config["benchmark_ticker"] = manifest["benchmark_ticker"]
    config["holding_period_days"] = manifest["holding_period"]

    return config


def _build_arm_config(
    manifest: dict[str, Any],
    arm: str,
    results_dir: Path,
    data_dir: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    """Build concrete TradingAgents config for Arm A or Arm B using exact native keys."""
    base_config = _build_base_config(manifest)
    config = copy.deepcopy(base_config)

    # Isolated directory paths per cell (R1)
    config["results_dir"] = str(results_dir)
    config["data_dir"] = str(data_dir)
    config["data_cache_dir"] = str(cache_dir)
    config["memory_log_path"] = str(results_dir / "trading_memory.log")

    # Native data vendor routing (R1)
    data_vendors = dict(config.get("data_vendors", {}))
    tool_vendors = dict(config.get("tool_vendors", {}))

    if arm == "A":
        tool_vendors["get_news"] = "yfinance"
        tool_vendors["get_global_news"] = "yfinance"
        data_vendors["news_data"] = "yfinance"
    elif arm == "B":
        tool_vendors["get_news"] = "news_intelligence"
        tool_vendors["get_global_news"] = "news_intelligence"
        data_vendors["news_data"] = "news_intelligence"
    else:
        raise ValueError(f"Unknown arm '{arm}', must be 'A' or 'B'")

    config["tool_vendors"] = tool_vendors
    config["data_vendors"] = data_vendors
    return config


def _is_fatal_provider_error(exc_or_reason: Any) -> bool:
    """Return True if exception or reason indicates rate limit, quota exhaustion, or auth failure (R5)."""
    if isinstance(exc_or_reason, BaseException):
        if is_provider_limit_error(exc_or_reason):
            return True
        msg = str(exc_or_reason).lower()
    else:
        msg = str(exc_or_reason).lower()

    fatal_keywords = (
        "429",
        "quota",
        "ratelimit",
        "rate limit",
        "unauthorized",
        "authentication",
        "forbidden",
        "resourceexhausted",
        "resource_exhausted",
        "401",
        "403",
    )
    return any(k in msg for k in fatal_keywords)


# ---------------------------------------------------------------------------
# Prospective Evidence Capture Context & Native Hooks (TIP-14 Revision 5)
# ---------------------------------------------------------------------------

MAX_ENVELOPE_BYTES: int = 4 * 1024 * 1024  # 4 MiB finite bound on archive envelope size
MAX_SNAPSHOT_BYTES: int = 10 * 1024 * 1024  # 10 MB limit on raw snapshot size


def _read_bounded_bytes(fpath: Path, max_bytes: int) -> tuple[bytes | None, str | None]:
    """Read at most max_bytes + 1 from fpath and verify returned bytes <= max_bytes.

    Prevents unbounded allocation even if file grows or is replaced after stat (MED3).
    """
    if not fpath.exists():
        return None, f"File does not exist: {fpath}"
    if not fpath.is_file():
        return None, f"Path is not a regular file: {fpath}"
    try:
        size = fpath.stat().st_size
        if size > max_bytes:
            return None, f"File size ({size} bytes) exceeds maximum limit {max_bytes} bytes"
    except OSError as exc:
        return None, f"Failed to stat file: {exc}"

    try:
        with fpath.open("rb") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            return None, f"File read ({len(data)} bytes) exceeds maximum limit {max_bytes} bytes"
        return data, None
    except Exception as exc:
        return None, f"Failed to read file: {exc}"


def _read_bounded_json_envelope(fpath: Path) -> tuple[Any | None, str | None]:
    """Safely read and parse a JSON envelope within MAX_ENVELOPE_BYTES before allocation.

    Enforces MAX_ENVELOPE_BYTES before reading and bounds read length at MAX + 1.
    Returns (data, None) on success or (None, error_reason) on failure.
    """
    data, err = _read_bounded_bytes(fpath, MAX_ENVELOPE_BYTES)
    if err is not None:
        return None, err
    assert data is not None
    try:
        content = data.decode("utf-8")
        return json.loads(content), None
    except Exception as exc:
        return None, f"Invalid JSON ({exc})"


class ProspectiveCaptureContext:
    """Thread-safe context manager capturing raw news response snapshots and cumulative archive envelopes.

    Serializes sequence allocation, snapshot write, envelope update, temp write, fsync, os.replace,
    and read-back verification under threading.RLock().
    Installs scoped pre-format hooks on router.VENDOR_METHODS and direct vendor functions.
    Restores original implementations cleanly on exit.
    """

    def __init__(
        self,
        cell_dir: Path | str,
        ticker: str,
        date_str: str,
        arm: str,
        attempt_id: str,
        cutoff_iso: str | None = None,
        audit_handler: AuditCallbackHandler | None = None,
    ) -> None:
        self.cell_dir = Path(cell_dir).resolve()
        self.ticker = ticker
        self.date_str = date_str
        self.arm = arm
        self.attempt_id = attempt_id
        self.cutoff_iso = cutoff_iso
        self.audit_handler = audit_handler
        self._commit_lock = threading.RLock()
        self._call_seq: int = 0
        self.responses_dir = self.cell_dir / "data" / "responses"
        if self.arm == "A":
            self.archive_path = self.cell_dir / "data" / "baseline_archive.json"
        else:
            self.archive_path = self.cell_dir / "data" / "provenance_archive.json"
        self._orig_hooks: dict[str, Any] = {}
        self._active = False

    def commit_call(
        self,
        tool_name: str,
        query: str,
        params: dict[str, Any],
        raw_payload: Any,
        extracted_records: list[dict[str, Any]],
        output_text: str,
        run_id: str | None = None,
        received_at: str | None = None,
        producer_params: dict[str, Any] | None = None,
        feed_source: str | None = None,
    ) -> dict[str, Any]:
        """Atomically persist raw snapshot and update cumulative envelope under commit lock."""
        with self._commit_lock:
            try:
                self.responses_dir.mkdir(parents=True, exist_ok=True)
                self.archive_path.parent.mkdir(parents=True, exist_ok=True)

                self._call_seq += 1
                seq = self._call_seq
                call_id = str(uuid.uuid4())
                snapshot_filename = f"{seq:03d}_{tool_name}_{call_id}.json"
                snapshot_path = self.responses_dir / snapshot_filename
                rel_snapshot_path = f"responses/{snapshot_filename}"

                # Compute raw payload hash
                serialized_payload = json.dumps(raw_payload, sort_keys=True)
                raw_payload_hash = hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()

                # Atomic write snapshot with fsync and read-back
                temp_snap = self.responses_dir / f".tmp_{seq}_{uuid.uuid4().hex[:8]}"
                with open(temp_snap, "w", encoding="utf-8") as f:
                    f.write(json.dumps(raw_payload, indent=2, sort_keys=True))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_snap, snapshot_path)

                # Read-back verification of snapshot
                read_snap_bytes, snap_read_err = _read_bounded_bytes(
                    snapshot_path, MAX_SNAPSHOT_BYTES
                )
                if snap_read_err is not None or read_snap_bytes is None:
                    raise EvidenceStorageError(f"Snapshot read-back failed: {snap_read_err}")
                read_snap_obj = json.loads(read_snap_bytes.decode("utf-8"))
                re_hash = hashlib.sha256(
                    json.dumps(read_snap_obj, sort_keys=True).encode("utf-8")
                ).hexdigest()
                if re_hash != raw_payload_hash:
                    raise EvidenceStorageError(
                        f"Snapshot read-back hash mismatch: expected {raw_payload_hash}, got {re_hash}"
                    )

                output_hash = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
                rec_at = received_at or datetime.now(UTC).isoformat()

                # Clean extracted records: convert any datetime objects to ISO strings
                cleaned_records: list[dict[str, Any]] = []
                for rec_item in extracted_records:
                    r_copy = dict(rec_item)
                    for k_dt in ("pub_date", "publishedAt", "receivedAt", "enrichedAt"):
                        val_dt = r_copy.get(k_dt)
                        if hasattr(val_dt, "isoformat"):
                            r_copy[k_dt] = val_dt.isoformat()
                    cleaned_records.append(r_copy)

                call_record = {
                    "call_seq": seq,
                    "call_id": call_id,
                    "tool": tool_name,
                    "run_id": run_id,
                    "query": query,
                    "received_at": rec_at,
                    "params": params,
                    "snapshot_path": rel_snapshot_path,
                    "raw_payload_hash": raw_payload_hash,
                    "records": cleaned_records,
                    "output_hash": output_hash,
                }
                if producer_params is not None:
                    call_record["producer_params"] = producer_params
                if feed_source is not None:
                    call_record["feed_source"] = feed_source

                # Load or initialize cumulative envelope
                if self.archive_path.exists():
                    try:
                        envelope = json.loads(self.archive_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        raise EvidenceStorageError(
                            f"Failed to read cumulative envelope: {exc}"
                        ) from exc
                else:
                    envelope = {
                        "version": "t13-archive/v2",
                        "attempt_id": self.attempt_id,
                        "ticker": self.ticker,
                        "arm": self.arm,
                        "date": self.date_str,
                        "cutoff": self.cutoff_iso,
                        "vendor": "yfinance" if self.arm == "A" else "news_intelligence",
                        "created_at": datetime.now(UTC).isoformat(),
                        "calls": [],
                        "articles" if self.arm == "A" else "events": [],
                    }

                envelope.setdefault("calls", []).append(call_record)
                items_key = "articles" if self.arm == "A" else "events"
                envelope.setdefault(items_key, []).extend(cleaned_records)

                # Atomic write cumulative envelope with fsync and read-back
                temp_env = self.archive_path.parent / f".tmp_env_{uuid.uuid4().hex[:8]}"
                with open(temp_env, "w", encoding="utf-8") as f:
                    f.write(json.dumps(envelope, indent=2, sort_keys=True))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_env, self.archive_path)

                # Read-back verification (R7: full content equality check)
                read_env_bytes, env_read_err = _read_bounded_bytes(
                    self.archive_path, MAX_ENVELOPE_BYTES
                )
                if env_read_err is not None or read_env_bytes is None:
                    raise EvidenceStorageError(f"Envelope read-back failed: {env_read_err}")
                read_env_obj = json.loads(read_env_bytes.decode("utf-8"))
                if read_env_obj != envelope:
                    raise EvidenceStorageError("Envelope read-back content verification failed")

                return call_record

            except EvidenceStorageError:
                raise
            except Exception as exc:
                raise EvidenceStorageError(f"Evidence persistence failed: {exc}") from exc

    def __enter__(self) -> ProspectiveCaptureContext:
        self._active = True
        self._orig_hooks = {}
        import tradingagents.dataflows.router as router
        import tradingagents.dataflows.vendors.news_intelligence.client as ni_client
        import tradingagents.dataflows.vendors.yahoo.news as yahoo_news
        from tradingagents.dataflows.date_window import in_window

        context_self = self

        if self.arm == "A":
            orig_yf_retry = getattr(yahoo_news, "yf_retry", None)
            orig_search = getattr(yahoo_news, "_search_news", None)
            orig_get_news_yf = yahoo_news.get_news_yfinance
            orig_get_global_yf = yahoo_news.get_global_news_yfinance

            self._orig_hooks["yahoo_news.yf_retry"] = orig_yf_retry
            self._orig_hooks["yahoo_news._search_news"] = orig_search
            self._orig_hooks["yahoo_news.get_news_yfinance"] = orig_get_news_yf
            self._orig_hooks["yahoo_news.get_global_news_yfinance"] = orig_get_global_yf
            self._orig_hooks["router.get_news.yfinance"] = router.VENDOR_METHODS.get(
                "get_news", {}
            ).get("yfinance")
            self._orig_hooks["router.get_global_news.yfinance"] = router.VENDOR_METHODS.get(
                "get_global_news", {}
            ).get("yfinance")

            # Context-wide thread-isolated collectors (R2)
            def capturing_yf_retry(fn: Any) -> Any:
                res = orig_yf_retry(fn)
                collector = getattr(_thread_capture_state, "yahoo_raw", None)
                if collector is not None and isinstance(res, list):
                    collector.extend(copy.deepcopy(res))
                return res

            def capturing_search(canonical: str, limit: int) -> tuple[list, bool]:
                tagged, knows = orig_search(canonical, limit)
                collector = getattr(_thread_capture_state, "yahoo_raw", None)
                if collector is not None:
                    _thread_capture_state.yahoo_feed_source = "search"
                    if isinstance(tagged, list):
                        collector.extend(copy.deepcopy(tagged))
                return tagged, knows

            yahoo_news.yf_retry = capturing_yf_retry
            yahoo_news._search_news = capturing_search

            def hooked_get_news_yfinance(ticker: str, start_date: str, end_date: str) -> str:
                _thread_capture_state.yahoo_raw = []
                _thread_capture_state.yahoo_feed_source = "feed"
                try:
                    out_str = orig_get_news_yf(ticker, start_date, end_date)
                    rec_at = datetime.now(UTC).isoformat()  # Record immediately AFTER return (R4)
                    captured_raw = list(getattr(_thread_capture_state, "yahoo_raw", []) or [])
                    feed_source = getattr(_thread_capture_state, "yahoo_feed_source", "feed")
                finally:
                    _thread_capture_state.yahoo_raw = None
                    _thread_capture_state.yahoo_feed_source = None

                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                extracted = [yahoo_news._extract_article_data(a) for a in captured_raw]
                in_range = [a for a in extracted if in_window(a["pub_date"], start_dt, end_dt)]
                norm_records = []
                for a in in_range:
                    rc = dict(a)
                    val_dt = rc.get("pub_date")
                    if hasattr(val_dt, "isoformat"):
                        rc["pub_date"] = val_dt.isoformat()
                    norm_records.append(rc)

                active_run_id = getattr(_current_tool_run_id, "value", None)
                producer_p = {"ticker": ticker, "start_date": start_date, "end_date": end_date}
                context_self.commit_call(
                    tool_name="get_news",
                    query=ticker,
                    params={"ticker": ticker, "start_date": start_date, "end_date": end_date},
                    raw_payload=captured_raw,
                    extracted_records=norm_records,
                    output_text=out_str,
                    run_id=active_run_id,
                    received_at=rec_at,
                    producer_params=producer_p,
                    feed_source=feed_source,
                )
                return out_str

            def hooked_get_global_news_yfinance(
                as_of_date: str, look_back_days: int | None = None, limit: int | None = None
            ) -> str:
                _thread_capture_state.yahoo_raw = []
                _thread_capture_state.yahoo_feed_source = "feed"
                try:
                    out_str = orig_get_global_yf(
                        as_of_date, look_back_days=look_back_days, limit=limit
                    )
                    rec_at = datetime.now(UTC).isoformat()  # Record immediately AFTER return (R4)
                    captured_raw = list(getattr(_thread_capture_state, "yahoo_raw", []) or [])
                    feed_source = getattr(_thread_capture_state, "yahoo_feed_source", "feed")
                finally:
                    _thread_capture_state.yahoo_raw = None
                    _thread_capture_state.yahoo_feed_source = None

                cfg = get_config()
                eff_lb = (
                    look_back_days
                    if look_back_days is not None
                    else cfg.get("global_news_lookback_days", 7)
                )
                eff_lim = limit if limit is not None else cfg.get("global_news_article_limit", 10)
                curr_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
                start_dt = curr_dt - relativedelta(days=eff_lb)
                extracted = [yahoo_news._extract_article_data(a) for a in captured_raw]
                seen_titles = set()
                in_win = []
                for a in extracted:
                    if (
                        in_window(a["pub_date"], start_dt, curr_dt)
                        and a["title"]
                        and a["title"] not in seen_titles
                    ):
                        seen_titles.add(a["title"])
                        in_win.append(a)
                sub = in_win[:eff_lim]
                norm_records = []
                for a in sub:
                    rc = dict(a)
                    val_dt = rc.get("pub_date")
                    if hasattr(val_dt, "isoformat"):
                        rc["pub_date"] = val_dt.isoformat()
                    norm_records.append(rc)

                active_run_id = getattr(_current_tool_run_id, "value", None)
                producer_p = {
                    "as_of_date": as_of_date,
                    "look_back_days": look_back_days,
                    "limit": limit,
                }
                context_self.commit_call(
                    tool_name="get_global_news",
                    query="global",
                    params={
                        "as_of_date": as_of_date,
                        "look_back_days": look_back_days,
                        "limit": limit,
                    },
                    raw_payload=captured_raw,
                    extracted_records=norm_records,
                    output_text=out_str,
                    run_id=active_run_id,
                    received_at=rec_at,
                    producer_params=producer_p,
                    feed_source=feed_source,
                )
                return out_str

            yahoo_news.get_news_yfinance = hooked_get_news_yfinance
            yahoo_news.get_global_news_yfinance = hooked_get_global_news_yfinance
            if "get_news" in router.VENDOR_METHODS:
                router.VENDOR_METHODS["get_news"]["yfinance"] = hooked_get_news_yfinance
            if "get_global_news" in router.VENDOR_METHODS:
                router.VENDOR_METHODS["get_global_news"]["yfinance"] = (
                    hooked_get_global_news_yfinance
                )

        elif self.arm == "B":
            orig_make_request = ni_client.NewsIntelligenceClient._make_request
            orig_client_get_news = ni_client.NewsIntelligenceClient.get_news
            orig_client_get_global = ni_client.NewsIntelligenceClient.get_global_news
            orig_mod_get_news = getattr(ni_client, "get_news_news_intelligence", None)
            orig_mod_get_global = getattr(ni_client, "get_global_news_news_intelligence", None)

            self._orig_hooks["ni_client.NewsIntelligenceClient._make_request"] = orig_make_request
            self._orig_hooks["ni_client.NewsIntelligenceClient.get_news"] = orig_client_get_news
            self._orig_hooks["ni_client.NewsIntelligenceClient.get_global_news"] = (
                orig_client_get_global
            )
            self._orig_hooks["ni_client.get_news_news_intelligence"] = orig_mod_get_news
            self._orig_hooks["ni_client.get_global_news_news_intelligence"] = orig_mod_get_global
            self._orig_hooks["router.get_news.news_intelligence"] = router.VENDOR_METHODS.get(
                "get_news", {}
            ).get("news_intelligence")
            self._orig_hooks["router.get_global_news.news_intelligence"] = (
                router.VENDOR_METHODS.get("get_global_news", {}).get("news_intelligence")
            )

            def hooked_make_request(
                client_self: Any, endpoint: str, params: dict[str, Any]
            ) -> dict[str, Any]:
                data = orig_make_request(client_self, endpoint, params)
                rec_at = datetime.now(UTC).isoformat()  # Record immediately AFTER return (R4)
                reqs = getattr(_thread_capture_state, "repo1_requests", None)
                if reqs is not None:
                    reqs.append(
                        {
                            "endpoint": endpoint,
                            "params": copy.deepcopy(params),
                            "data": copy.deepcopy(data),
                            "received_at": rec_at,
                        }
                    )
                return data

            def hooked_get_news(
                client_self: Any, ticker: str, start_date: str, end_date: str
            ) -> str:
                _thread_capture_state.repo1_requests = []
                try:
                    out_str = orig_client_get_news(client_self, ticker, start_date, end_date)
                    rec_at = datetime.now(UTC).isoformat()
                    captured_reqs = list(getattr(_thread_capture_state, "repo1_requests", []) or [])
                finally:
                    _thread_capture_state.repo1_requests = None

                matching_reqs = [
                    r
                    for r in captured_reqs
                    if r.get("endpoint") in ("/api/v1/news", "/api/v1/news/global")
                ]
                raw_payload = matching_reqs[-1]["data"] if matching_reqs else {"events": []}
                rec_at = matching_reqs[-1]["received_at"] if matching_reqs else rec_at
                raw_params = (
                    matching_reqs[-1]["params"]
                    if matching_reqs
                    else {"symbol": ticker, "from": start_date, "to": end_date}
                )

                query_sym, resolved = ni_client.map_ticker_for_news_intelligence(ticker)
                as_of = ni_client.to_iso_utc(end_date, end_of_day=True)
                min_pub = ni_client.to_iso_utc(start_date, end_of_day=False)
                max_pub = ni_client.to_iso_utc(end_date, end_of_day=True)
                cutoff_dt = ni_client.parse_utc_timestamp(as_of)
                min_pub_dt = ni_client.parse_utc_timestamp(min_pub)
                max_pub_dt = ni_client.parse_utc_timestamp(max_pub)

                raw_events = raw_payload.get("events") or []
                sanitized_events = []
                for raw_event in raw_events:
                    if not ni_client.validate_canonical_event(raw_event):
                        continue
                    ev = ni_client.filter_and_sanitize_event(
                        raw_event,
                        cutoff_dt=cutoff_dt,
                        min_pub_dt=min_pub_dt,
                        max_pub_dt=max_pub_dt,
                    )
                    if ev is not None and query_sym in ev.get("symbols", []):
                        sanitized_events.append(ev)

                limit = get_config().get("news_article_limit", 20)
                active_run_id = getattr(_current_tool_run_id, "value", None)
                producer_p = {
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit,
                }
                context_self.commit_call(
                    tool_name="get_news",
                    query=ticker,
                    params=raw_params,
                    raw_payload=raw_payload,
                    extracted_records=sanitized_events[:limit],
                    output_text=out_str,
                    run_id=active_run_id,
                    received_at=rec_at,
                    producer_params=producer_p,
                )
                return out_str

            def hooked_get_global_news(
                client_self: Any,
                as_of_date: str,
                look_back_days: int | None = None,
                limit: int | None = None,
            ) -> str:
                _thread_capture_state.repo1_requests = []
                try:
                    out_str = orig_client_get_global(
                        client_self, as_of_date, look_back_days=look_back_days, limit=limit
                    )
                    rec_at = datetime.now(UTC).isoformat()
                    captured_reqs = list(getattr(_thread_capture_state, "repo1_requests", []) or [])
                finally:
                    _thread_capture_state.repo1_requests = None

                matching_reqs = [
                    r
                    for r in captured_reqs
                    if r.get("endpoint") in ("/api/v1/news/global", "/api/v1/news")
                ]
                raw_payload = matching_reqs[-1]["data"] if matching_reqs else {"events": []}
                rec_at = matching_reqs[-1]["received_at"] if matching_reqs else rec_at
                raw_params = (
                    matching_reqs[-1]["params"]
                    if matching_reqs
                    else {
                        "symbol": "GLOBAL",
                        "asOf": as_of_date,
                    }
                )

                cfg = get_config()
                eff_lb = (
                    look_back_days
                    if look_back_days is not None
                    else cfg.get("global_news_lookback_days", 7)
                )
                eff_lim = limit if limit is not None else cfg.get("global_news_article_limit", 10)
                curr_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
                start_dt = curr_dt - timedelta(days=eff_lb)
                start_date = start_dt.strftime("%Y-%m-%d")

                as_of = ni_client.to_iso_utc(as_of_date, end_of_day=True)
                min_pub = ni_client.to_iso_utc(start_date, end_of_day=False)
                max_pub = ni_client.to_iso_utc(as_of_date, end_of_day=True)
                cutoff_dt = ni_client.parse_utc_timestamp(as_of)
                min_pub_dt = ni_client.parse_utc_timestamp(min_pub)
                max_pub_dt = ni_client.parse_utc_timestamp(max_pub)

                raw_events = raw_payload.get("events") or []
                sanitized_events = []
                for raw_event in raw_events:
                    if not ni_client.validate_canonical_event(raw_event):
                        continue
                    ev = ni_client.filter_and_sanitize_event(
                        raw_event,
                        cutoff_dt=cutoff_dt,
                        min_pub_dt=min_pub_dt,
                        max_pub_dt=max_pub_dt,
                    )
                    if ev is not None:
                        sanitized_events.append(ev)

                active_run_id = getattr(_current_tool_run_id, "value", None)
                producer_p = {
                    "as_of_date": as_of_date,
                    "look_back_days": look_back_days,
                    "limit": limit,
                }
                context_self.commit_call(
                    tool_name="get_global_news",
                    query="global",
                    params=raw_params,
                    raw_payload=raw_payload,
                    extracted_records=sanitized_events[:eff_lim],
                    output_text=out_str,
                    run_id=active_run_id,
                    received_at=rec_at,
                    producer_params=producer_p,
                )
                return out_str

            def hooked_mod_get_news(ticker: str, start_date: str, end_date: str) -> str:
                cl = ni_client.NewsIntelligenceClient()
                return hooked_get_news(cl, ticker, start_date, end_date)

            def hooked_mod_get_global(
                as_of_date: str, look_back_days: int | None = None, limit: int | None = None
            ) -> str:
                cl = ni_client.NewsIntelligenceClient()
                return hooked_get_global_news(
                    cl, as_of_date, look_back_days=look_back_days, limit=limit
                )

            ni_client.NewsIntelligenceClient._make_request = hooked_make_request
            ni_client.NewsIntelligenceClient.get_news = hooked_get_news
            ni_client.NewsIntelligenceClient.get_global_news = hooked_get_global_news
            ni_client.get_news_news_intelligence = hooked_mod_get_news
            ni_client.get_global_news_news_intelligence = hooked_mod_get_global
            if "get_news" in router.VENDOR_METHODS:
                router.VENDOR_METHODS["get_news"]["news_intelligence"] = hooked_mod_get_news
            if "get_global_news" in router.VENDOR_METHODS:
                router.VENDOR_METHODS["get_global_news"]["news_intelligence"] = (
                    hooked_mod_get_global
                )

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._active = False
        import tradingagents.dataflows.router as router
        import tradingagents.dataflows.vendors.news_intelligence.client as ni_client
        import tradingagents.dataflows.vendors.yahoo.news as yahoo_news

        if "yahoo_news.yf_retry" in self._orig_hooks:
            yahoo_news.yf_retry = self._orig_hooks["yahoo_news.yf_retry"]
        if "yahoo_news._search_news" in self._orig_hooks:
            yahoo_news._search_news = self._orig_hooks["yahoo_news._search_news"]
        if "yahoo_news.get_news_yfinance" in self._orig_hooks:
            yahoo_news.get_news_yfinance = self._orig_hooks["yahoo_news.get_news_yfinance"]
        if "yahoo_news.get_global_news_yfinance" in self._orig_hooks:
            yahoo_news.get_global_news_yfinance = self._orig_hooks[
                "yahoo_news.get_global_news_yfinance"
            ]
        if "router.get_news.yfinance" in self._orig_hooks and "get_news" in router.VENDOR_METHODS:
            router.VENDOR_METHODS["get_news"]["yfinance"] = self._orig_hooks[
                "router.get_news.yfinance"
            ]
        if (
            "router.get_global_news.yfinance" in self._orig_hooks
            and "get_global_news" in router.VENDOR_METHODS
        ):
            router.VENDOR_METHODS["get_global_news"]["yfinance"] = self._orig_hooks[
                "router.get_global_news.yfinance"
            ]

        if "ni_client.NewsIntelligenceClient._make_request" in self._orig_hooks:
            ni_client.NewsIntelligenceClient._make_request = self._orig_hooks[
                "ni_client.NewsIntelligenceClient._make_request"
            ]
        if "ni_client.NewsIntelligenceClient.get_news" in self._orig_hooks:
            ni_client.NewsIntelligenceClient.get_news = self._orig_hooks[
                "ni_client.NewsIntelligenceClient.get_news"
            ]
        if "ni_client.NewsIntelligenceClient.get_global_news" in self._orig_hooks:
            ni_client.NewsIntelligenceClient.get_global_news = self._orig_hooks[
                "ni_client.NewsIntelligenceClient.get_global_news"
            ]
        if "ni_client.get_news_news_intelligence" in self._orig_hooks:
            ni_client.get_news_news_intelligence = self._orig_hooks[
                "ni_client.get_news_news_intelligence"
            ]
        if "ni_client.get_global_news_news_intelligence" in self._orig_hooks:
            ni_client.get_global_news_news_intelligence = self._orig_hooks[
                "ni_client.get_global_news_news_intelligence"
            ]
        if (
            "router.get_news.news_intelligence" in self._orig_hooks
            and "get_news" in router.VENDOR_METHODS
        ):
            router.VENDOR_METHODS["get_news"]["news_intelligence"] = self._orig_hooks[
                "router.get_news.news_intelligence"
            ]
        if (
            "router.get_global_news.news_intelligence" in self._orig_hooks
            and "get_global_news" in router.VENDOR_METHODS
        ):
            router.VENDOR_METHODS["get_global_news"]["news_intelligence"] = self._orig_hooks[
                "router.get_global_news.news_intelligence"
            ]

        self._orig_hooks.clear()


# ---------------------------------------------------------------------------
# Runner Execution: Decision Generation (R2, R3, R4, R5, R6, R8)
# ---------------------------------------------------------------------------


def run_experiment(manifest: dict[str, Any], output_dir: Path | str) -> dict[str, Any]:
    """Execute paired decision generation under exclusive owner lock.

    Preserves partial progress, never replays started/interrupted paid cells automatically.
    """
    output_root = Path(output_dir).resolve()
    validated_manifest = validate_manifest(manifest)
    manifest_hash = compute_manifest_hash(validated_manifest)

    with acquire_experiment_owner(output_root):
        cells_dir = output_root / "cells"
        journal_path = output_root / "journal.json"

        pairs = validated_manifest["pairs"]
        analysts = validated_manifest["model_controls"].get("analysts", ["market", "news"])
        portfolio_raw = validated_manifest.get("portfolio")
        portfolio = (
            PortfolioContext.model_validate(portfolio_raw) if portfolio_raw is not None else None
        )

        # Planned cells with alternating arm order across pairs
        planned_cells: list[tuple[dict[str, str], str]] = []
        for p_idx, pair in enumerate(pairs):
            arm_order = ["A", "B"] if (p_idx % 2 == 0) else ["B", "A"]
            for arm in arm_order:
                planned_cells.append((pair, arm))

        planned_cells_meta = [
            {
                "ticker": pair["ticker"],
                "date": pair["date"],
                "arm": arm,
                "asset_type": pair["asset_type"],
            }
            for pair, arm in planned_cells
        ]

        if journal_path.exists():
            try:
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise JournalPersistenceError(f"Failed to read existing journal: {exc}") from exc
            if journal.get("manifest_hash") != manifest_hash:
                raise ManifestMismatchError(
                    f"Journal manifest hash {journal.get('manifest_hash')} does not match active manifest {manifest_hash}"
                )
        else:
            # R4: If journal is absent but cells dir exists with contents, fail closed!
            if cells_dir.exists() and any(cells_dir.iterdir()):
                raise JournalPersistenceError(
                    f"Missing journal.json but existing cell artifacts found at {cells_dir}. "
                    "Cannot reinitialize empty experiment without reconciliation."
                )
            journal = {
                "manifest_hash": manifest_hash,
                "experiment_id": validated_manifest["experiment_id"],
                "planned_cells": planned_cells_meta,
                "planned_cells_count": len(planned_cells),
                "planned_decisions_count": len(planned_cells),
                "planned_pairs_count": len(pairs),
                "decision_attempts": {},
                "settlement_attempts": {},
            }
            _persist_journal(output_root, journal)

        # R1: Check that no planned cell directory exists with artifacts if absent from journal
        for p, a in planned_cells:
            ck = f"{p['ticker']}_{p['date']}_{a}"
            cd = output_root / "cells" / ck
            if ck not in journal.get("decision_attempts", {}) and cd.exists() and any(cd.iterdir()):
                raise JournalPersistenceError(
                    f"Cell {ck} has unjournaled artifacts on disk at {cd}; refusing fresh paid execution"
                )

        # Loop through planned cells
        for pair, arm in planned_cells:
            ticker = pair["ticker"]
            date_str = pair["date"]
            asset_type = pair["asset_type"]
            cell_key = f"{ticker}_{date_str}_{arm}"

            # Check directory path bounds (R3)
            cell_dir = (output_root / "cells" / cell_key).resolve()
            if not cell_dir.is_relative_to(output_root / "cells"):
                raise ValueError(f"Cell directory escapes output root: {cell_dir}")

            results_dir = cell_dir / "results"
            data_dir = cell_dir / "data"
            cache_dir = cell_dir / "cache"
            mem_log_path = results_dir / "trading_memory.log"

            existing_att = journal["decision_attempts"].get(cell_key)
            if existing_att:
                phase = existing_att.get("phase")
                expected_rep = results_dir / "complete_report.md"
                rep_path = (
                    output_root / existing_att["report_path"]
                    if existing_att.get("report_path")
                    else (expected_rep if expected_rep.exists() else None)
                )
                if not rep_path and (results_dir / "complete_report.html").exists():
                    rep_path = results_dir / "complete_report.html"

                matching_native: list[dict[str, Any]] = []
                if mem_log_path.exists():
                    entries = TradingMemoryLog(
                        {"memory_log_path": str(mem_log_path)}
                    ).load_entries()
                    matching_native = [
                        e
                        for e in entries
                        if e.get("ticker") == ticker and e.get("date") == date_str
                    ]

                if phase == "completed":
                    # Must have existing report AND matching native memory decision (R1, R2)
                    if rep_path and rep_path.exists() and matching_native:
                        continue
                    # Missing report or memory on disk: DO NOT re-admit fresh paid execution! Mark failed (R1)
                    existing_att["phase"] = "failed"
                    existing_att["error"] = (
                        "Corrupt cell: native memory log or complete report missing on disk for completed attempt"
                    )
                    _persist_journal(output_root, journal)
                    continue

                # R4/R5: Recoverable started/interrupted/unknown cell: check if artifacts already exist on disk!
                if rep_path and rep_path.exists() and matching_native:
                    existing_att["phase"] = "completed"
                    existing_att["decision"] = matching_native[0].get("rating")
                    existing_att["completed_at"] = datetime.now(UTC).isoformat()
                    existing_att["report_path"] = str(rep_path.relative_to(output_root))
                    trace_file = results_dir / "audit_trace.json"
                    if trace_file.exists():
                        try:
                            trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
                            existing_att["model_calls"] = trace_data.get("model_calls", 0)
                            existing_att["model_usage"] = trace_data.get("token_usage", {})
                            existing_att["has_unknown_usage"] = trace_data.get(
                                "has_unknown_usage", False
                            )
                            existing_att["non_news_inputs_hash"] = trace_data.get(
                                "non_news_inputs_hash"
                            )
                            existing_att["audit_trace_path"] = str(
                                trace_file.relative_to(output_root)
                            )
                        except Exception:
                            existing_att["has_unknown_usage"] = True
                    else:
                        existing_att["has_unknown_usage"] = True
                    _persist_journal(output_root, journal)
                    continue

                # Any existing attempt without complete artifacts: do NOT re-invoke model fresh! (R1)
                continue

            # Fresh cell: create directories and record started
            results_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

            attempt_id = str(uuid.uuid4())
            expected_rep_rel = str((results_dir / "complete_report.md").relative_to(output_root))
            attempt_rec: dict[str, Any] = {
                "attempt_id": attempt_id,
                "phase": "started",
                "arm": arm,
                "ticker": ticker,
                "date": date_str,
                "asset_type": asset_type,
                "started_at": datetime.now(UTC).isoformat(),
                "completed_at": None,
                "decision": None,
                "report_path": expected_rep_rel,  # R5: pre-populate deterministic expected report path
                "audit_trace_path": None,
                "non_news_inputs_hash": None,
                "model_calls": 0,
                "model_usage": {},
                "has_unknown_usage": False,
                "error": None,
            }
            journal["decision_attempts"][cell_key] = attempt_rec
            _persist_journal(output_root, journal)

            cell_config = _build_arm_config(
                validated_manifest, arm, results_dir, data_dir, cache_dir
            )

            try:
                ensure_provider_configured(cell_config)
            except Exception as exc:
                attempt_rec["phase"] = "failed"
                attempt_rec["completed_at"] = datetime.now(UTC).isoformat()
                attempt_rec["error"] = sanitize_text(f"Provider not configured: {exc}")
                _persist_journal(output_root, journal)
                # Provider config failure halts whole command (R5)
                break

            audit_cb = AuditCallbackHandler()
            graph = TradingAgentsGraph(
                selected_analysts=analysts, config=cell_config, callbacks=[audit_cb]
            )

            # Wire callbacks to propagator so LangGraph tool execution receives callbacks (R2)
            if hasattr(graph, "propagator") and hasattr(graph.propagator, "get_graph_args"):
                orig_get_graph_args = graph.propagator.get_graph_args

                def _get_graph_args_with_cb(
                    *a: Any,
                    _cb: Any = audit_cb,
                    _orig: Any = orig_get_graph_args,
                    **kw: Any,
                ) -> dict[str, Any]:
                    cbs = kw.get("callbacks") or [_cb]
                    kw["callbacks"] = cbs
                    return _orig(*a, **kw)

                graph.propagator.get_graph_args = _get_graph_args_with_cb

            try:
                # Prospective evidence capture context wraps graph execution (R1, R2, R4)
                with ProspectiveCaptureContext(
                    cell_dir=cell_dir,
                    ticker=ticker,
                    date_str=date_str,
                    arm=arm,
                    attempt_id=attempt_id,
                    audit_handler=audit_cb,
                ):
                    # Call propagate returning (final_state, signal) (R2)
                    final_state, signal = graph.propagate(
                        ticker, date_str, asset_type=asset_type, portfolio=portfolio
                    )
                report_file = graph.save_reports(final_state, ticker, save_path=results_dir)

                mem_log = TradingMemoryLog({"memory_log_path": str(mem_log_path)})
                entries = mem_log.load_entries()
                matching = [e for e in entries if e["ticker"] == ticker and e["date"] == date_str]
                if not matching:
                    raise RuntimeError("Native memory entry was not persisted after propagate")

                decision = signal or matching[0].get("rating")
                attempt_rec["phase"] = "completed"
                attempt_rec["completed_at"] = datetime.now(UTC).isoformat()
                attempt_rec["decision"] = decision
                attempt_rec["report_path"] = str(report_file.relative_to(output_root))
                attempt_rec["model_calls"] = audit_cb.model_calls
                attempt_rec["model_usage"] = audit_cb.token_usage
                attempt_rec["has_unknown_usage"] = audit_cb.has_unknown_usage

                # R2, R3: Non-news tool inputs and consumed outputs hash
                if audit_cb.non_news_tool_hashes:
                    non_news_hash = hashlib.sha256(
                        "".join(audit_cb.non_news_tool_hashes).encode("utf-8")
                    ).hexdigest()
                else:
                    non_news_hash = None
                attempt_rec["non_news_inputs_hash"] = non_news_hash

                # Persist audit trace (R3, R4, R6)
                trace_file = results_dir / "audit_trace.json"
                trace_payload = {
                    "cell_key": cell_key,
                    "attempt_id": attempt_id,
                    "model_calls": audit_cb.model_calls,
                    "tool_calls": audit_cb.tool_calls,
                    "token_usage": audit_cb.token_usage,
                    "has_unknown_usage": audit_cb.has_unknown_usage,
                    "non_news_inputs_hash": non_news_hash,
                    "traces": audit_cb.traces,
                }
                trace_file.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
                attempt_rec["audit_trace_path"] = str(trace_file.relative_to(output_root))

                _persist_journal(output_root, journal)

            except (JournalPersistenceError, JournalVerificationError):
                # R4: Fail-closed journal errors must immediately re-raise!
                raise
            except EvidenceStorageError as exc:
                # Fatal storage error: record in journal, mark FAILED_INCOMPLETE, zero paid replay
                attempt_rec["phase"] = "failed"
                attempt_rec["status"] = "FAILED_INCOMPLETE"
                attempt_rec["completed_at"] = datetime.now(UTC).isoformat()
                attempt_rec["error"] = sanitize_text(f"EvidenceStorageError: {exc}")
                attempt_rec["model_calls"] = audit_cb.model_calls
                attempt_rec["model_usage"] = audit_cb.token_usage
                attempt_rec["has_unknown_usage"] = True

                trace_file = results_dir / "audit_trace.json"
                with suppress(Exception):
                    trace_payload = {
                        "cell_key": cell_key,
                        "attempt_id": attempt_id,
                        "model_calls": audit_cb.model_calls,
                        "tool_calls": audit_cb.tool_calls,
                        "token_usage": audit_cb.token_usage,
                        "has_unknown_usage": True,
                        "traces": audit_cb.traces,
                        "error": sanitize_text(f"EvidenceStorageError: {exc}"),
                    }
                    trace_file.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
                    attempt_rec["audit_trace_path"] = str(trace_file.relative_to(output_root))

                _persist_journal(output_root, journal)
                continue
            except Exception as exc:
                attempt_rec["phase"] = "failed"
                attempt_rec["completed_at"] = datetime.now(UTC).isoformat()
                attempt_rec["error"] = sanitize_text(str(exc))
                attempt_rec["model_calls"] = audit_cb.model_calls
                attempt_rec["model_usage"] = audit_cb.token_usage
                # R7: Any error means usage is unknown
                attempt_rec["has_unknown_usage"] = True

                trace_file = results_dir / "audit_trace.json"
                with suppress(Exception):
                    trace_payload = {
                        "cell_key": cell_key,
                        "attempt_id": attempt_id,
                        "model_calls": audit_cb.model_calls,
                        "tool_calls": audit_cb.tool_calls,
                        "token_usage": audit_cb.token_usage,
                        "has_unknown_usage": True,
                        "traces": audit_cb.traces,
                        "error": sanitize_text(str(exc)),
                    }
                    trace_file.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
                    attempt_rec["audit_trace_path"] = str(trace_file.relative_to(output_root))

                _persist_journal(output_root, journal)

                # R5: Quota / auth error breaks out of whole command loop!
                if _is_fatal_provider_error(exc):
                    break

        return journal


# ---------------------------------------------------------------------------
# Settlement Execution: Native Outcomes & Honest Recovery (R4, R5, R6)
# ---------------------------------------------------------------------------


def _is_matching_settlement_audit(
    ad: Any,
    expected_settlement_id: str | None,
    expected_cell_key: str,
    expected_decision_attempt_id: str | None,
    expected_manifest_hash: str | None,
) -> bool:
    """Validate that audit trace on disk has exact matching non-missing identity fields (R3)."""
    if not isinstance(ad, dict):
        return False
    op = ad.get("operation")
    s_id = ad.get("settlement_id")
    c_key = ad.get("cell_key")
    d_att = ad.get("decision_attempt_id")
    m_hash = ad.get("manifest_hash")

    if op != "settlement":
        return False
    if not expected_settlement_id or s_id != expected_settlement_id:
        return False
    if not expected_cell_key or c_key != expected_cell_key:
        return False
    if not expected_decision_attempt_id or d_att != expected_decision_attempt_id:
        return False
    return bool(expected_manifest_hash and m_hash == expected_manifest_hash)


def settle_experiment(manifest: dict[str, Any], output_dir: Path | str) -> dict[str, Any]:
    """Settle past decisions whose holding windows have traded, preserving failed/pending state.

    Reconciles existing durable outcomes first; never replays started/interrupted reflections.
    """
    output_root = Path(output_dir).resolve()
    validated_manifest = validate_manifest(manifest)
    manifest_hash = compute_manifest_hash(validated_manifest)

    with acquire_experiment_owner(output_root):
        journal_path = output_root / "journal.json"
        if not journal_path.exists():
            raise FileNotFoundError(f"Cannot settle: journal.json does not exist at {output_root}")

        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise JournalPersistenceError(f"Failed to read existing journal: {exc}") from exc

        if journal.get("manifest_hash") != manifest_hash:
            raise ManifestMismatchError("Active manifest hash differs from journal")

        pairs = validated_manifest["pairs"]
        analysts = validated_manifest["model_controls"].get("analysts", ["market", "news"])

        # Flattened settlement cells (R5)
        planned_settle_cells: list[tuple[dict[str, str], str]] = [
            (pair, arm) for pair in pairs for arm in ("A", "B")
        ]

        for pair, arm in planned_settle_cells:
            ticker = pair["ticker"]
            date_str = pair["date"]
            cell_key = f"{ticker}_{date_str}_{arm}"

            decision_att = journal.get("decision_attempts", {}).get(cell_key)
            if not decision_att or decision_att.get("phase") != "completed":
                continue  # Only completed decisions can be settled

            cell_dir = (output_root / "cells" / cell_key).resolve()
            if not cell_dir.is_relative_to(output_root / "cells"):
                raise ValueError(f"Cell directory escapes output root: {cell_dir}")

            results_dir = cell_dir / "results"
            mem_log_path = results_dir / "trading_memory.log"

            settle_att = journal.get("settlement_attempts", {}).get(cell_key)

            # R2, R4: Verify memory log exists on disk; missing or empty log is failed, never pending_window
            # Preserve prior uncertain paid usage and attempt ID if attempt was already started
            if not mem_log_path.exists():
                if settle_att:
                    settle_att["phase"] = "failed"
                    settle_att["completed_at"] = datetime.now(UTC).isoformat()
                    settle_att["error"] = "Native memory log is missing on disk"
                    settle_att["settled"] = False
                else:
                    settle_rec: dict[str, Any] = {
                        "settlement_id": str(uuid.uuid4()),
                        "decision_attempt_id": decision_att["attempt_id"],
                        "arm": arm,
                        "ticker": ticker,
                        "date": date_str,
                        "phase": "failed",
                        "error": "Native memory log is missing on disk",
                        "started_at": datetime.now(UTC).isoformat(),
                        "completed_at": datetime.now(UTC).isoformat(),
                        "settled": False,
                        "model_calls": 0,
                        "model_usage": {},
                        "has_unknown_usage": False,
                    }
                    journal.setdefault("settlement_attempts", {})[cell_key] = settle_rec
                _persist_journal(output_root, journal)
                continue

            entries = TradingMemoryLog({"memory_log_path": str(mem_log_path)}).load_entries()
            if not entries:
                if settle_att:
                    settle_att["phase"] = "failed"
                    settle_att["completed_at"] = datetime.now(UTC).isoformat()
                    settle_att["error"] = "Native memory log is empty"
                    settle_att["settled"] = False
                else:
                    settle_rec = {
                        "settlement_id": str(uuid.uuid4()),
                        "decision_attempt_id": decision_att["attempt_id"],
                        "arm": arm,
                        "ticker": ticker,
                        "date": date_str,
                        "phase": "failed",
                        "error": "Native memory log is empty",
                        "started_at": datetime.now(UTC).isoformat(),
                        "completed_at": datetime.now(UTC).isoformat(),
                        "settled": False,
                        "model_calls": 0,
                        "model_usage": {},
                        "has_unknown_usage": False,
                    }
                    journal.setdefault("settlement_attempts", {})[cell_key] = settle_rec
                _persist_journal(output_root, journal)
                continue

            # R2: Enforce strict single-decision per log boundary; reject extra unjournaled entries
            extra_entries = [
                e for e in entries if e.get("ticker") != ticker or e.get("date") != date_str
            ]
            if extra_entries or len(entries) != 1:
                if settle_att:
                    settle_att["phase"] = "failed"
                    settle_att["completed_at"] = datetime.now(UTC).isoformat()
                    settle_att["error"] = (
                        f"Native memory log invariant violated: expected 1 entry for {ticker} on {date_str}, found {len(entries)}"
                    )
                    settle_att["settled"] = False
                else:
                    settle_rec = {
                        "settlement_id": str(uuid.uuid4()),
                        "decision_attempt_id": decision_att["attempt_id"],
                        "arm": arm,
                        "ticker": ticker,
                        "date": date_str,
                        "phase": "failed",
                        "error": f"Native memory log invariant violated: expected 1 entry for {ticker} on {date_str}, found {len(entries)}",
                        "started_at": datetime.now(UTC).isoformat(),
                        "completed_at": datetime.now(UTC).isoformat(),
                        "settled": False,
                        "model_calls": 0,
                        "model_usage": {},
                        "has_unknown_usage": False,
                    }
                    journal.setdefault("settlement_attempts", {})[cell_key] = settle_rec
                _persist_journal(output_root, journal)
                continue

            if settle_att:
                s_phase = settle_att.get("phase")
                if s_phase == "completed":
                    continue
                # If already settled on disk: reconcile
                if not entries[0]["pending"]:
                    e = entries[0]
                    settle_att["phase"] = "completed"
                    settle_att["completed_at"] = (
                        settle_att.get("completed_at") or datetime.now(UTC).isoformat()
                    )
                    settle_att["settled"] = True
                    settle_att["outcome"] = _alpha(e)
                    settle_att["raw_return"] = e.get("raw")
                    settle_att["rating"] = e.get("rating")
                    settle_att["resolution_date"] = e.get("resolved")
                    # R3: Clears unrecovered float evidence so missing evidence never bypasses vintage matching
                    settle_att["raw_return_float"] = None
                    settle_att["alpha_float"] = None
                    settle_att["benchmark_return_float"] = None

                    # R3: Restore usage only from exact matching settlement audit trace on disk; never borrow decision audit
                    s_id = settle_att.get("settlement_id")
                    d_att_id = decision_att.get("attempt_id")
                    audit_file = None
                    if s_id and (results_dir / f"settlement_audit_{s_id}.json").exists():
                        audit_file = results_dir / f"settlement_audit_{s_id}.json"

                    is_valid_audit = False
                    if audit_file:
                        try:
                            ad = json.loads(audit_file.read_text(encoding="utf-8"))
                            if _is_matching_settlement_audit(
                                ad, s_id, cell_key, d_att_id, manifest_hash
                            ):
                                settle_att["model_calls"] = ad.get("model_calls", 0)
                                settle_att["model_usage"] = (
                                    ad.get("token_usage") or ad.get("model_usage") or {}
                                )
                                settle_att["has_unknown_usage"] = bool(
                                    ad.get("has_unknown_usage", False)
                                )
                                settle_att["audit_trace_path"] = str(
                                    audit_file.relative_to(output_root)
                                )
                                is_valid_audit = True
                        except Exception:
                            pass

                    if not is_valid_audit:
                        settle_att["model_calls"] = 0
                        settle_att["model_usage"] = {}
                        settle_att["has_unknown_usage"] = True
                        settle_att["audit_trace_path"] = None

                    _persist_journal(output_root, journal)
                    continue

                if (
                    s_phase in ("started", "failed", "interrupted", "unknown")
                    or s_phase != "pending_window"
                ):
                    # R1, R3: Incomplete/failed/unknown without verified on-disk outcome: do NOT reflect again automatically
                    # Restore verified matching audit evidence if available, otherwise retain unknown and do not replay
                    s_id = settle_att.get("settlement_id")
                    d_att_id = decision_att.get("attempt_id")
                    audit_file = None
                    if s_id and (results_dir / f"settlement_audit_{s_id}.json").exists():
                        audit_file = results_dir / f"settlement_audit_{s_id}.json"

                    is_valid_audit = False
                    if audit_file:
                        try:
                            ad = json.loads(audit_file.read_text(encoding="utf-8"))
                            if _is_matching_settlement_audit(
                                ad, s_id, cell_key, d_att_id, manifest_hash
                            ):
                                settle_att["model_calls"] = ad.get(
                                    "model_calls", settle_att.get("model_calls", 0)
                                )
                                settle_att["model_usage"] = (
                                    ad.get("token_usage")
                                    or ad.get("model_usage")
                                    or settle_att.get("model_usage", {})
                                )
                                settle_att["audit_trace_path"] = str(
                                    audit_file.relative_to(output_root)
                                )
                                is_valid_audit = True
                        except Exception:
                            pass

                    if not is_valid_audit:
                        settle_att["model_calls"] = 0
                        settle_att["model_usage"] = {}
                        settle_att["audit_trace_path"] = None

                    settle_att["has_unknown_usage"] = True

                    _persist_journal(output_root, journal)
                    continue

            # If already settled on disk (even if settle_att is absent):
            if not entries[0]["pending"]:
                e = entries[0]
                settle_id = str(uuid.uuid4())
                m_calls = 0
                m_usage: dict[str, Any] = {}
                unk_usage = True
                tr_path = None

                settle_rec = {
                    "settlement_id": settle_id,
                    "decision_attempt_id": decision_att["attempt_id"],
                    "arm": arm,
                    "ticker": ticker,
                    "date": date_str,
                    "phase": "completed",
                    "started_at": datetime.now(UTC).isoformat(),
                    "completed_at": datetime.now(UTC).isoformat(),
                    "settled": True,
                    "outcome": _alpha(e),
                    "raw_return": e.get("raw"),
                    "raw_return_float": None,
                    "alpha_float": None,
                    "benchmark_return_float": None,
                    "rating": e.get("rating"),
                    "holding_period": validated_manifest.get("holding_period", 5),
                    "resolution_date": e.get("resolved"),
                    "benchmark": validated_manifest.get("benchmark_ticker", "SPY"),
                    "model_calls": m_calls,
                    "model_usage": m_usage,
                    "has_unknown_usage": unk_usage,
                    "audit_trace_path": tr_path,
                    "error": None,
                }
                journal.setdefault("settlement_attempts", {})[cell_key] = settle_rec
                _persist_journal(output_root, journal)
                continue

            # R5, R6: Preserving recheck history in a flat non-recursive structure
            prior_attempt_id = settle_att.get("settlement_id") if settle_att else None
            if settle_att:
                settle_att_clean = {k: v for k, v in settle_att.items() if k != "history"}
                settle_history = journal.setdefault("settlement_history", [])
                settle_history.append(settle_att_clean)

            settle_id = str(uuid.uuid4())
            settle_rec = {
                "settlement_id": settle_id,
                "prior_attempt_id": prior_attempt_id,
                "decision_attempt_id": decision_att["attempt_id"],
                "arm": arm,
                "ticker": ticker,
                "date": date_str,
                "phase": "started",
                "started_at": datetime.now(UTC).isoformat(),
                "completed_at": None,
                "settled": False,
                "outcome": None,
                "raw_return": None,
                "raw_return_float": None,
                "alpha_float": None,
                "benchmark_return_float": None,
                "holding_period_actual": None,
                "benchmark_name": validated_manifest.get("benchmark_ticker", "SPY"),
                "rating": decision_att.get("decision"),
                "holding_period": validated_manifest.get("holding_period", 5),
                "resolution_date": None,
                "benchmark": validated_manifest.get("benchmark_ticker", "SPY"),
                "model_calls": 0,
                "model_usage": {},
                "has_unknown_usage": True,  # R3: In-flight usage is explicitly unknown until verified
                "audit_trace_path": None,
                "error": None,
            }
            if settle_att:
                existing_hist = settle_att.get("history", [])
                settle_att_clean = {k: v for k, v in settle_att.items() if k != "history"}
                settle_rec["history"] = [
                    {k: v for k, v in h.items() if k != "history"} for h in existing_hist
                ] + [settle_att_clean]
            else:
                settle_rec["history"] = []

            journal.setdefault("settlement_attempts", {})[cell_key] = settle_rec
            _persist_journal(output_root, journal)

            cell_config = _build_arm_config(
                validated_manifest, arm, results_dir, cell_dir / "data", cell_dir / "cache"
            )

            try:
                ensure_provider_configured(cell_config)
            except Exception as exc:
                settle_rec["phase"] = "failed"
                settle_rec["completed_at"] = datetime.now(UTC).isoformat()
                settle_rec["error"] = sanitize_text(f"Provider not configured: {exc}")
                _persist_journal(output_root, journal)
                # Provider failure breaks whole command loop (R5)
                break

            audit_cb = AuditCallbackHandler()
            graph = TradingAgentsGraph(
                selected_analysts=analysts, config=cell_config, callbacks=[audit_cb]
            )

            # R4: Capture exact returns from reflector invocation and native first close (R2)
            captured_returns: dict[str, Any] = {}
            orig_reflect = getattr(graph.reflector, "reflect_on_final_decision", None)
            if orig_reflect is not None:

                def capturing_reflect(
                    *,
                    _captured: dict[str, Any] = captured_returns,
                    _orig: Any = orig_reflect,
                    **kwargs: Any,
                ) -> Any:
                    raw_r = kwargs.get("raw_return")
                    alpha_r = kwargs.get("alpha_return")
                    bench_n = kwargs.get("benchmark_name")
                    h_days = kwargs.get("holding_days")
                    if raw_r is not None:
                        _captured["raw_return_float"] = float(raw_r)
                    if alpha_r is not None:
                        _captured["alpha_float"] = float(alpha_r)
                    if raw_r is not None and alpha_r is not None:
                        _captured["benchmark_return_float"] = float(raw_r) - float(alpha_r)
                    if bench_n is not None:
                        _captured["benchmark_name"] = str(bench_n)
                    if h_days is not None:
                        _captured["holding_days"] = int(h_days)
                    return _orig(**kwargs)

                graph.reflector.reflect_on_final_decision = capturing_reflect

            orig_get_closes = getattr(settlement, "get_closes", None)
            if orig_get_closes is not None:

                def capturing_get_closes(
                    sym: str,
                    *args: Any,
                    _orig: Any = orig_get_closes,
                    _target_ticker: str = ticker,
                    _captured: dict[str, Any] = captured_returns,
                    **kwargs: Any,
                ) -> Any:
                    res = _orig(sym, *args, **kwargs)
                    if sym == _target_ticker and res is not None:
                        try:
                            # R2: Apply native _by_day filter to drop missing/non-positive closes
                            valid_res = settlement._by_day(res)
                            if hasattr(valid_res, "index") and len(valid_res.index) > 0:
                                first_idx = valid_res.index[0]
                                dt_entry, err = _validate_entry_timestamp_candidate(
                                    first_idx, "first_close", ticker=_target_ticker
                                )
                                if dt_entry is not None and not err:
                                    _captured["entry_time"] = dt_entry.isoformat()
                        except Exception:
                            pass
                    return res

                settlement.get_closes = capturing_get_closes

            try:
                try:
                    res = graph.settle_pending(ticker)
                finally:
                    if orig_get_closes is not None:
                        settlement.get_closes = orig_get_closes
                settle_rec["model_calls"] = audit_cb.model_calls
                settle_rec["model_usage"] = audit_cb.token_usage
                settle_rec["has_unknown_usage"] = audit_cb.has_unknown_usage

                # R2/R5: Check for fatal provider quota/auth errors anywhere in res.failed
                any_quota_err = None
                for _ft, _fd, freason in getattr(res, "failed", []):
                    if _is_fatal_provider_error(freason):
                        any_quota_err = freason
                        break

                if any_quota_err is not None:
                    settle_rec["phase"] = "failed"
                    settle_rec["completed_at"] = datetime.now(UTC).isoformat()
                    settle_rec["error"] = sanitize_text(
                        f"Quota error in settlement sweep: {any_quota_err}"
                    )
                    settle_rec["has_unknown_usage"] = True
                    trace_file = results_dir / f"settlement_audit_{settle_id}.json"
                    with suppress(Exception):
                        trace_payload = {
                            "operation": "settlement",
                            "cell_key": cell_key,
                            "settlement_id": settle_id,
                            "decision_attempt_id": decision_att["attempt_id"],
                            "manifest_hash": manifest_hash,
                            "model_calls": audit_cb.model_calls,
                            "tool_calls": audit_cb.tool_calls,
                            "token_usage": audit_cb.token_usage,
                            "has_unknown_usage": True,
                            "traces": audit_cb.traces,
                        }
                        trace_file.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
                        settle_rec["audit_trace_path"] = str(trace_file.relative_to(output_root))
                    _persist_journal(output_root, journal)
                    break

                is_settled = any(t == ticker and d == date_str for t, d in res.settled)
                is_failed = any(t == ticker and d == date_str for t, d, _ in res.failed)

                if is_settled:
                    ents = TradingMemoryLog({"memory_log_path": str(mem_log_path)}).load_entries()
                    settled_matches = [
                        x
                        for x in ents
                        if x["ticker"] == ticker and x["date"] == date_str and not x["pending"]
                    ]
                    if settled_matches:
                        e = settled_matches[0]
                        settle_rec["phase"] = "completed"
                        settle_rec["completed_at"] = datetime.now(UTC).isoformat()
                        settle_rec["settled"] = True
                        settle_rec["outcome"] = _alpha(e)
                        settle_rec["raw_return"] = e.get("raw")
                        settle_rec["raw_return_float"] = captured_returns.get("raw_return_float")
                        settle_rec["alpha_float"] = captured_returns.get("alpha_float")
                        settle_rec["benchmark_return_float"] = captured_returns.get(
                            "benchmark_return_float"
                        )
                        settle_rec["holding_period_actual"] = captured_returns.get("holding_days")
                        settle_rec["benchmark_name"] = captured_returns.get(
                            "benchmark_name", settle_rec["benchmark"]
                        )
                        settle_rec["rating"] = e.get("rating")
                        settle_rec["resolution_date"] = e.get("resolved")
                        settle_rec["has_unknown_usage"] = audit_cb.has_unknown_usage
                        if "entry_time" in captured_returns:
                            settle_rec["entry_time"] = captured_returns["entry_time"]
                            decision_att["entry_time"] = captured_returns["entry_time"]
                    else:
                        settle_rec["phase"] = "interrupted"
                        settle_rec["completed_at"] = datetime.now(UTC).isoformat()
                        settle_rec["error"] = (
                            "Settlement reported settled by graph but entry not found in log"
                        )
                        settle_rec["has_unknown_usage"] = True

                elif is_failed:
                    fail_reason = [r for t, d, r in res.failed if t == ticker and d == date_str][0]
                    settle_rec["phase"] = "failed"
                    settle_rec["completed_at"] = datetime.now(UTC).isoformat()
                    settle_rec["error"] = sanitize_text(f"Settlement failed: {fail_reason}")
                    settle_rec["has_unknown_usage"] = True

                else:
                    ents = TradingMemoryLog({"memory_log_path": str(mem_log_path)}).load_entries()
                    matching_p = [
                        x
                        for x in ents
                        if x["ticker"] == ticker and x["date"] == date_str and x.get("pending")
                    ]
                    if matching_p and audit_cb.model_calls == 0:
                        settle_rec["phase"] = "pending_window"
                        settle_rec["completed_at"] = datetime.now(UTC).isoformat()
                        settle_rec["has_unknown_usage"] = False
                    else:
                        settle_rec["phase"] = "interrupted"
                        settle_rec["completed_at"] = datetime.now(UTC).isoformat()
                        settle_rec["error"] = (
                            "Entry not settled and no valid immature pending window established"
                        )
                        settle_rec["has_unknown_usage"] = True

                # Save settlement audit trace with unique ID (R6) and accurate has_unknown_usage
                trace_file = results_dir / f"settlement_audit_{settle_id}.json"
                with suppress(Exception):
                    trace_payload = {
                        "operation": "settlement",
                        "cell_key": cell_key,
                        "settlement_id": settle_id,
                        "decision_attempt_id": decision_att["attempt_id"],
                        "manifest_hash": manifest_hash,
                        "model_calls": audit_cb.model_calls,
                        "tool_calls": audit_cb.tool_calls,
                        "token_usage": audit_cb.token_usage,
                        "has_unknown_usage": settle_rec["has_unknown_usage"],
                        "traces": audit_cb.traces,
                    }
                    trace_file.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
                    settle_rec["audit_trace_path"] = str(trace_file.relative_to(output_root))

                _persist_journal(output_root, journal)

                # R5: Check returned failure for quota / auth stop!
                if is_failed and _is_fatal_provider_error(fail_reason):
                    break

            except (JournalPersistenceError, JournalVerificationError):
                raise
            except Exception as exc:
                settle_rec["phase"] = "failed"
                settle_rec["completed_at"] = datetime.now(UTC).isoformat()
                settle_rec["error"] = sanitize_text(str(exc))
                settle_rec["model_calls"] = audit_cb.model_calls
                settle_rec["model_usage"] = audit_cb.token_usage
                # R7: Any error means usage is unknown
                settle_rec["has_unknown_usage"] = True

                trace_file = results_dir / f"settlement_audit_{settle_id}.json"
                with suppress(Exception):
                    trace_payload = {
                        "operation": "settlement",
                        "cell_key": cell_key,
                        "settlement_id": settle_id,
                        "decision_attempt_id": decision_att["attempt_id"],
                        "manifest_hash": manifest_hash,
                        "model_calls": audit_cb.model_calls,
                        "tool_calls": audit_cb.tool_calls,
                        "token_usage": audit_cb.token_usage,
                        "has_unknown_usage": True,
                        "traces": audit_cb.traces,
                    }
                    trace_file.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
                    settle_rec["audit_trace_path"] = str(trace_file.relative_to(output_root))

                _persist_journal(output_root, journal)

                # R5: Thrown fatal error breaks out of whole command loop!
                if _is_fatal_provider_error(exc):
                    break

        return journal


# ---------------------------------------------------------------------------
# Summarize & Paired Honest Reporting (R7, R8)
# ---------------------------------------------------------------------------


def _parse_utc_iso(val: Any) -> datetime | None:
    if not val or not isinstance(val, str):
        return None
    try:
        norm = val.replace("Z", "+00:00")
        dt = datetime.fromisoformat(norm)
        if dt.tzinfo is None:
            return None
        return dt.astimezone(UTC)
    except Exception:
        return None


def _find_valid_json_files(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    for p in paths:
        if not p.exists():
            continue
        if p.is_file():
            if p.stat().st_size > 0:
                result.append(p)
        elif p.is_dir():
            for f in p.glob("*"):
                if f.is_file() and f.stat().st_size > 0 and (f.suffix == ".json" or not f.suffix):
                    result.append(f)
    return result


def _verify_timing_clock_file(fpath: Path, ticker: str, date_val: str, cutoff: datetime) -> bool:
    data, err = _read_bounded_json_envelope(fpath)
    if err is not None or not data or not isinstance(data, (dict, list)):
        return False

    if not data or not isinstance(data, (dict, list)):
        return False

    top_sym = None
    top_dt = None
    if isinstance(data, dict):
        top_sym = data.get("ticker") or data.get("symbol") or data.get("query_symbol")
        top_dt = data.get("date") or data.get("trade_date")
        if top_sym and top_sym != ticker:
            return False
        if top_dt and top_dt != date_val:
            return False

    records = (
        data
        if isinstance(data, list)
        else (data.get("events") or data.get("records") or data.get("clocks") or [data])
    )
    if not records or not isinstance(records, list):
        return False

    found_matching = False
    for r in records:
        if not isinstance(r, dict):
            continue
        sym = r.get("ticker") or r.get("symbol") or r.get("query_symbol") or top_sym
        rec_date = r.get("date") or r.get("trade_date") or top_dt
        if not sym or sym != ticker:
            continue
        if not rec_date or rec_date != date_val:
            continue

        # Reject conflicting clock aliases in record (R1)
        for c_k, a_k in [
            ("publishedAt", "published_at"),
            ("receivedAt", "received_at"),
            ("enrichedAt", "enriched_at"),
        ]:
            if r.get(c_k) is not None and r.get(a_k) is not None:
                dt_c = _parse_utc_iso(str(r[c_k]))
                dt_a = _parse_utc_iso(str(r[a_k]))
                if dt_c and dt_a and dt_c != dt_a:
                    return False

        ts_candidates = [
            r.get("published_at"),
            r.get("received_at"),
            r.get("input_clock"),
            r.get("as_of"),
            r.get("fetch_time"),
            r.get("timestamp"),
            r.get("publishedAt"),
            r.get("receivedAt"),
            r.get("enriched_at"),
            r.get("enrichedAt"),
        ]
        valid_ts = [ts for ts in ts_candidates if ts is not None]
        if not valid_ts:
            continue

        for ts_str in valid_ts:
            dt = _parse_utc_iso(str(ts_str))
            if not dt or dt > cutoff:
                return False

        found_matching = True

    return found_matching


def _parse_strict_tool_start_input(raw_input: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Parse tool_start input strictly as JSON or Python-style mapping.

    Rejects None, empty/whitespace, scalars, lists, and unparseable text.
    Permissive regex fallback is not allowed for strict request binding (HIGH1).
    """
    if raw_input is None:
        return None, "tool_start missing input"
    if isinstance(raw_input, dict):
        return raw_input, None
    if isinstance(raw_input, str):
        s = raw_input.strip()
        if not s:
            return None, "tool_start has empty input"
        # 1. Try strict JSON
        try:
            val = json.loads(s)
            if isinstance(val, dict):
                return val, None
            return (
                None,
                f"tool_start input JSON must decode to a dictionary (object), got {type(val).__name__}",
            )
        except Exception:
            pass
        # 2. Try Python-style literal mapping (ast.literal_eval)
        try:
            import ast

            val = ast.literal_eval(s)
            if isinstance(val, dict):
                return val, None
            return (
                None,
                f"tool_start input mapping must decode to a dictionary (object), got {type(val).__name__}",
            )
        except Exception as exc:
            return None, f"tool_start input is not valid JSON or mapping: {exc}"
    return (
        None,
        f"tool_start input must be a dictionary or JSON/mapping string, got {type(raw_input).__name__}",
    )


def _extract_tool_start_params(input_val: Any) -> dict[str, Any]:
    """Extract tool arguments from tool_start trace input."""
    parsed, _ = _parse_strict_tool_start_input(input_val)
    return parsed if parsed is not None else {}


def _resolve_and_validate_clock(
    rec: dict[str, Any],
    canonical_key: str,
    alias_keys: list[str],
    top_data: dict[str, Any] | None,
    dt_cutoff: datetime,
    mandatory: bool = False,
    field_name: str = "clock",
) -> tuple[datetime | None, str | None]:
    """Resolve clock value, reject conflicting aliases, and verify <= dt_cutoff.

    Returns (dt_utc, error_message).
    """
    candidates: list[tuple[str, Any]] = []

    # 1. Canonical key explicitly present in rec
    if canonical_key in rec:
        if rec[canonical_key] is None:
            # Explicit canonical null: represents unknown availability.
            # Never backfill from top_data or aliases.
            for ak in alias_keys:
                if ak in rec and rec[ak] is not None:
                    return None, (
                        f"Conflicting {field_name} clock aliases: {canonical_key} is null "
                        f"but alias {ak}='{rec[ak]}' is present"
                    )
            if mandatory:
                return (
                    None,
                    f"Record has explicit null/unknown mandatory {field_name} timestamp ({canonical_key})",
                )
            rec[canonical_key] = None
            return None, None

        rec_val = rec[canonical_key]
        for ak in alias_keys:
            if ak in rec and rec[ak] is None:
                return None, (
                    f"Conflicting {field_name} clock aliases: {canonical_key}='{rec_val}' is present "
                    f"but alias {ak} is null"
                )
        candidates.append((canonical_key, rec_val))
        for ak in alias_keys:
            if ak in rec and rec[ak] is not None:
                candidates.append((ak, rec[ak]))

    else:
        # Canonical key is NOT in rec; check aliases in rec
        aliases_present = [ak for ak in alias_keys if ak in rec]
        if aliases_present:
            null_aliases = [ak for ak in aliases_present if rec[ak] is None]
            non_null_aliases = [ak for ak in aliases_present if rec[ak] is not None]
            if null_aliases and non_null_aliases:
                return None, (
                    f"Conflicting {field_name} clock aliases: {null_aliases[0]} is null "
                    f"but alias {non_null_aliases[0]}='{rec[non_null_aliases[0]]}' is present"
                )
            if null_aliases:
                if mandatory:
                    return (
                        None,
                        f"Record has explicit null/unknown mandatory {field_name} timestamp ({null_aliases[0]})",
                    )
                rec[canonical_key] = None
                return None, None
            for ak in non_null_aliases:
                candidates.append((ak, rec[ak]))

        else:
            # Genuinely absent from rec
            if field_name == "enrichment" or canonical_key == "enrichedAt":
                # Enrichment is an event-level NLP fact: never inherit from wrapper top_data
                rec[canonical_key] = None
                return None, None

            # Publication/receipt can inherit from wrapper top_data if present
            if top_data:
                if canonical_key in top_data:
                    if top_data[canonical_key] is None:
                        if mandatory:
                            return (
                                None,
                                f"Top-level has explicit null/unknown mandatory {field_name} timestamp ({canonical_key})",
                            )
                        return None, None
                    candidates.append((f"top-level {canonical_key}", top_data[canonical_key]))
                    for ak in alias_keys:
                        if ak in top_data:
                            if top_data[ak] is None:
                                return (
                                    None,
                                    f"Conflicting top-level {field_name} clock aliases: {canonical_key} is present but {ak} is null",
                                )
                            candidates.append((f"top-level {ak}", top_data[ak]))
                else:
                    top_aliases_present = [ak for ak in alias_keys if ak in top_data]
                    null_top = [ak for ak in top_aliases_present if top_data[ak] is None]
                    non_null_top = [ak for ak in top_aliases_present if top_data[ak] is not None]
                    if null_top and non_null_top:
                        return (
                            None,
                            f"Conflicting top-level {field_name} clock aliases: {null_top[0]} is null but {non_null_top[0]} is present",
                        )
                    if null_top:
                        if mandatory:
                            return (
                                None,
                                f"Top-level has explicit null/unknown mandatory {field_name} timestamp ({null_top[0]})",
                            )
                        return None, None
                    for ak in non_null_top:
                        candidates.append((f"top-level {ak}", top_data[ak]))

    if not candidates:
        if mandatory:
            return None, f"Record missing required {field_name} timestamp"
        return None, None

    parsed_instants: list[tuple[str, Any, datetime]] = []
    for k, v in candidates:
        dt = _parse_utc_iso(str(v))
        if dt is None:
            return None, f"Record {k} timestamp '{v}' is naive or unparseable ISO format"
        parsed_instants.append((k, v, dt))

    ref_k, ref_v, ref_dt = parsed_instants[0]
    for k, v, dt in parsed_instants[1:]:
        if dt != ref_dt:
            return (
                None,
                f"Conflicting {field_name} clock aliases: {ref_k}='{ref_v}' ({ref_dt.isoformat()}) "
                f"differs from {k}='{v}' ({dt.isoformat()})",
            )

    if ref_dt > dt_cutoff:
        return (
            None,
            f"Record {field_name} clock ({ref_dt.isoformat()}) is later than entry cutoff {dt_cutoff.isoformat()}",
        )

    # Normalize canonical key into rec so downstream native functions receive canonical field
    rec[canonical_key] = ref_v

    return ref_dt, None


def _copy_top_level_wrapper_keys(target_dict: dict[str, Any], source_dict: dict[str, Any]) -> None:
    """Copy wrapper payload keys to event records without contaminating existing clocks or aliases."""
    alias_families = (
        {"received_at", "receivedAt"},
        {"published_at", "publishedAt", "timestamp", "as_of", "pub_date"},
        {"enriched_at", "enrichedAt"},
        {"ticker", "symbol", "query_symbol"},
        {"query", "query_symbol"},
        {"date", "trade_date"},
        {"cutoff", "effective_cutoff", "cutoff_time"},
    )
    for pk in (
        "ticker",
        "symbol",
        "query_symbol",
        "date",
        "trade_date",
        "arm",
        "source",
        "vendor",
        "attempt_id",
        "query",
        "cutoff",
        "effective_cutoff",
        "received_at",
        "receivedAt",
        "published_at",
        "publishedAt",
        "output_hash",
        "output_hashes",
        "content_hash",
        "response_hash",
    ):
        if pk not in source_dict:
            continue
        if pk in target_dict:
            continue
        skip_alias = False
        for fam in alias_families:
            if pk in fam and any(m in target_dict for m in fam):
                skip_alias = True
                break
        if skip_alias:
            continue
        if pk in ("ticker", "symbol", "query_symbol", "query") and (
            target_dict.get("query") in ("global", "market", "macro", "all")
            or (isinstance(target_dict.get("symbols"), list) and not target_dict["symbols"])
        ):
            continue
        if pk in ("date", "trade_date") and any(
            k in target_dict for k in ("published_at", "publishedAt", "pub_date", "timestamp")
        ):
            continue
        target_dict[pk] = source_dict[pk]


MAX_SNAPSHOT_BYTES = 10 * 1024 * 1024  # 10 MB limit
MAX_ARCHIVE_CALLS = 100
MAX_CALL_RECORDS = 500


def _verify_v2_archive(
    data: dict[str, Any],
    fpath: Path,
    arm: str,
    ticker: str,
    date_val: str,
    dt_cutoff: datetime,
    decision_att: dict[str, Any],
    cell_dir: Path,
    news_ends: list[dict[str, Any]],
    tr_data: dict[str, Any],
) -> tuple[bool, str]:
    """Strict unbroken derivation verification for t13-archive/v2 envelopes.

    Enforces:
    1. Bounded snapshot read & raw SHA-256 digest verification with strict path containment (R6).
    2. Native normalization (Arm A) and point-in-time sanitization (Arm B).
    3. Arm-specific field-by-field record equality without fabricated clocks.
    4. Complete native producer response reconstruction (headers, windows, limit truncation, sentinels) (R5).
    5. Exact hash and text comparison with 1-to-1 bijection against observed tool executions in audit_trace.json (R3, R4).
    Fails closed immediately on any violation; never falls back to legacy candidate matching.
    """
    att_expected = decision_att.get("attempt_id")
    top_att = data.get("attempt_id")
    if not top_att or top_att != att_expected:
        return (
            False,
            f"Archive v2 verification failure: attempt_id '{top_att}' does not match expected '{att_expected}'",
        )

    top_arm = data.get("arm")
    if not top_arm or top_arm != arm:
        return (
            False,
            f"Archive v2 verification failure: arm '{top_arm}' does not match expected '{arm}'",
        )

    top_ticker = data.get("ticker")
    if not top_ticker or top_ticker != ticker:
        return (
            False,
            f"Archive v2 verification failure: ticker '{top_ticker}' does not match expected '{ticker}'",
        )

    top_date = data.get("date")
    if not top_date or top_date != date_val:
        return (
            False,
            f"Archive v2 verification failure: date '{top_date}' does not match expected '{date_val}'",
        )

    top_cutoff = data.get("cutoff")
    if top_cutoff is not None:
        dt_top_cut = _parse_utc_iso(str(top_cutoff))
        if not dt_top_cut or dt_top_cut > dt_cutoff:
            return (
                False,
                f"Archive v2 verification failure: cutoff '{top_cutoff}' exceeds entry cutoff {dt_cutoff.isoformat()}",
            )

    top_vendor = data.get("vendor")
    if arm == "A" and top_vendor not in ("yfinance", "yahoo"):
        return (
            False,
            f"Archive v2 verification failure: Arm A requires yfinance/yahoo vendor, got '{top_vendor}'",
        )
    if arm == "B" and top_vendor not in ("news_intelligence", "repo1"):
        return (
            False,
            f"Archive v2 verification failure: Arm B requires news_intelligence/repo1 vendor, got '{top_vendor}'",
        )

    calls = data.get("calls")
    if not calls or not isinstance(calls, list) or len(calls) == 0:
        return False, "Archive v2 verification failure: missing or empty calls list"

    if len(calls) > MAX_ARCHIVE_CALLS:
        return (
            False,
            f"Archive v2 verification failure: call count {len(calls)} exceeds maximum limit {MAX_ARCHIVE_CALLS}",
        )

    # 1-to-1 cardinality check (R3): every observed news tool execution must be covered by an archive call
    news_starts = [
        t
        for t in tr_data.get("traces", [])
        if isinstance(t, dict)
        and t.get("type") == "tool_start"
        and (
            str(t.get("tool", "")).startswith("get_news")
            or str(t.get("tool", "")).startswith("get_global_news")
        )
    ]
    news_ends = [
        t
        for t in tr_data.get("traces", [])
        if isinstance(t, dict)
        and t.get("type") == "tool_end"
        and (
            str(t.get("tool", "")).startswith("get_news")
            or str(t.get("tool", "")).startswith("get_global_news")
        )
    ]

    if len(calls) != len(news_ends):
        return (
            False,
            f"Archive v2 verification failure: call count ({len(calls)}) does not match observed news tool execution count ({len(news_ends)})",
        )

    if len(calls) != len(news_starts):
        return (
            False,
            f"Archive v2 verification failure: call count ({len(calls)}) does not match observed news tool start count ({len(news_starts)})",
        )

    bound_run_ids: set[str] = set()

    for call in calls:
        if not isinstance(call, dict):
            return False, "Archive v2 verification failure: call entry must be a dictionary"

        call_run_id = call.get("run_id")
        if not call_run_id or not isinstance(call_run_id, str):
            return False, "Archive v2 verification failure: call missing valid run_id"

        if call_run_id in bound_run_ids:
            return (
                False,
                f"Archive v2 verification failure: duplicate binding to run_id '{call_run_id}'",
            )
        bound_run_ids.add(call_run_id)

        snap_rel = call.get("snapshot_path")
        if not snap_rel or not isinstance(snap_rel, str):
            return False, "Archive v2 verification failure: call missing valid snapshot_path"

        # Containment & escape rejection (R6)
        if os.path.isabs(snap_rel) or Path(snap_rel).is_absolute() or ".." in Path(snap_rel).parts:
            return (
                False,
                f"Archive v2 verification failure: snapshot path must be relative without traversal: {snap_rel}",
            )

        candidates = [
            (cell_dir / "data" / snap_rel).resolve(),
            (cell_dir / snap_rel).resolve(),
            (fpath.parent / snap_rel).resolve(),
            (fpath.parent / "responses" / Path(snap_rel).name).resolve(),
            (cell_dir / "data" / "responses" / Path(snap_rel).name).resolve(),
        ]
        snap_path: Path | None = None
        cell_resolved = cell_dir.resolve()
        for c in candidates:
            if c.exists() and c.is_relative_to(cell_resolved):
                snap_path = c
                break

        if not snap_path:
            return (
                False,
                f"Archive v2 verification failure: snapshot file missing on disk or escapes cell: {snap_rel}",
            )

        if snap_path.is_symlink():
            return (
                False,
                f"Archive v2 verification failure: snapshot path cannot be a symlink: {snap_rel}",
            )

        if not snap_path.is_file():
            return (
                False,
                f"Archive v2 verification failure: snapshot path is not a regular file: {snap_rel}",
            )

        # Step 1: Bounded Snapshot Read & Hash Verification (MED3)
        snap_bytes, snap_err = _read_bounded_bytes(snap_path, MAX_SNAPSHOT_BYTES)
        if snap_err is not None or snap_bytes is None:
            return (
                False,
                f"Archive v2 verification failure: {snap_err}",
            )
        try:
            snap_payload = json.loads(snap_bytes.decode("utf-8"))
        except Exception as exc:
            return (
                False,
                f"Archive v2 verification failure: snapshot JSON decode failed: {exc}",
            )

        comp_raw_hash = hashlib.sha256(
            json.dumps(snap_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        decl_raw_hash = call.get("raw_payload_hash")
        if not decl_raw_hash or comp_raw_hash != decl_raw_hash:
            return (
                False,
                f"Archive v2 verification failure: snapshot hash mismatch ({comp_raw_hash} != {decl_raw_hash})",
            )

        # Validate snapshot payload shape (MED3)
        if arm == "A":
            if not isinstance(snap_payload, list):
                return (
                    False,
                    f"Archive v2 verification failure: Arm A snapshot payload must be a list of raw articles, got {type(snap_payload).__name__}",
                )
            for idx, a_raw in enumerate(snap_payload):
                if not isinstance(a_raw, dict):
                    return (
                        False,
                        f"Archive v2 verification failure: Arm A raw article at index {idx} must be a dictionary, got {type(a_raw).__name__}",
                    )
        elif arm == "B":
            if not isinstance(snap_payload, dict):
                return (
                    False,
                    f"Archive v2 verification failure: Arm B snapshot payload must be a dictionary containing 'events', got {type(snap_payload).__name__}",
                )
            raw_events_chk = snap_payload.get("events")
            if not isinstance(raw_events_chk, list):
                return (
                    False,
                    f"Archive v2 verification failure: Arm B snapshot 'events' must be a list, got {type(raw_events_chk).__name__}",
                )
            for idx, e_raw in enumerate(raw_events_chk):
                if not isinstance(e_raw, dict):
                    return (
                        False,
                        f"Archive v2 verification failure: Arm B raw event at index {idx} must be a dictionary, got {type(e_raw).__name__}",
                    )

        # Observation clock verification (R4)
        rec_at = call.get("received_at")
        if not rec_at:
            return False, "Archive v2 verification failure: call missing received_at"
        dt_rec = _parse_utc_iso(str(rec_at))
        if not dt_rec or dt_rec > dt_cutoff:
            return (
                False,
                f"Archive v2 verification failure: call received_at '{rec_at}' exceeds entry cutoff {dt_cutoff.isoformat()}",
            )

        tool_name = str(call.get("tool", ""))
        prod_p_raw = call.get("producer_params")
        if prod_p_raw is not None and not isinstance(prod_p_raw, dict):
            return (
                False,
                f"Archive v2 verification failure: call producer_params must be a dictionary if present, got {type(prod_p_raw).__name__}",
            )

        params_raw = call.get("params")
        if params_raw is not None and not isinstance(params_raw, dict):
            return (
                False,
                f"Archive v2 verification failure: call params must be a dictionary if present, got {type(params_raw).__name__}",
            )
        params = params_raw or {}
        prod_p = prod_p_raw if prod_p_raw is not None else params

        stored_records = call.get("records")
        if stored_records is None or not isinstance(stored_records, list):
            return False, "Archive v2 verification failure: call missing records list"
        if len(stored_records) > MAX_CALL_RECORDS:
            return (
                False,
                f"Archive v2 verification failure: record count {len(stored_records)} exceeds limit {MAX_CALL_RECORDS}",
            )

        for idx, r_item in enumerate(stored_records):
            if not isinstance(r_item, dict):
                return (
                    False,
                    f"Archive v2 verification failure: call records item at index {idx} must be a dictionary, got {type(r_item).__name__}",
                )

        # Consumed records timestamp cutoff check (R4)
        for r_item in stored_records:
            for k_ts in ("publishedAt", "receivedAt", "enrichedAt", "pub_date"):
                val = r_item.get(k_ts)
                if val is not None:
                    if not isinstance(val, (str, int, float)):
                        return (
                            False,
                            f"Archive v2 verification failure: record {k_ts} has invalid non-scalar type {type(val).__name__}",
                        )
                    dt_val = _parse_utc_iso(str(val))
                    if dt_val and dt_val > dt_cutoff:
                        return (
                            False,
                            f"Archive v2 verification failure: record {k_ts} '{val}' exceeds entry cutoff {dt_cutoff.isoformat()}",
                        )

        # Match exactly one tool_end with run_id == call_run_id (R3, HIGH1)
        matching_ne_list = [ne for ne in news_ends if ne.get("run_id") == call_run_id]
        if len(matching_ne_list) != 1:
            return (
                False,
                f"Archive v2 verification failure: expected exactly one observed tool_end for run_id '{call_run_id}', found {len(matching_ne_list)}",
            )
        matching_ne = matching_ne_list[0]

        # Match exactly one tool_start with run_id == call_run_id (R3, HIGH1)
        matching_ns_list = [ns for ns in news_starts if ns.get("run_id") == call_run_id]
        if len(matching_ns_list) != 1:
            return (
                False,
                f"Archive v2 verification failure: expected exactly one observed tool_start for run_id '{call_run_id}', found {len(matching_ns_list)}",
            )
        matching_ns = matching_ns_list[0]

        if not str(matching_ne.get("tool", "")).startswith(tool_name) or not str(
            matching_ns.get("tool", "")
        ).startswith(tool_name):
            return (
                False,
                f"Archive v2 verification failure: tool name mismatch for run_id '{call_run_id}'",
            )

        # Parse and validate tool_start input strictly as a valid argument object (HIGH1, R3)
        ns_input_raw = matching_ns.get("input")
        ns_input, ns_err = _parse_strict_tool_start_input(ns_input_raw)
        if ns_err is not None:
            return (
                False,
                f"Archive v2 verification failure: {ns_err} for run_id '{call_run_id}'",
            )
        assert ns_input is not None

        # Check query, ticker, date, window, and limit binding (R3, HIGH1, HIGH2)
        call_query = call.get("query")
        from tradingagents.dataflows.symbols import normalize_symbol

        if "get_global_news" in tool_name:
            if call_query not in ("global", "market", "macro", "all", "GLOBAL", "MARKET"):
                return (
                    False,
                    f"Archive v2 verification failure: global call query mismatch '{call_query}'",
                )

            # 1. Required model-visible argument: curr_date (or as_of_date/date/asOf)
            raw_ns_asof = (
                ns_input.get("curr_date")
                or ns_input.get("as_of_date")
                or ns_input.get("date")
                or ns_input.get("asOf")
            )
            if raw_ns_asof is None or str(raw_ns_asof).strip() == "":
                return (
                    False,
                    f"Archive v2 verification failure: tool_start missing required date/curr_date for global news for run_id '{call_run_id}'",
                )

            # Apply native as_of date clamping
            from tradingagents.dataflows.date_window import as_of

            eff_asof_dt_str = as_of(str(raw_ns_asof).strip(), date_val)
            if not eff_asof_dt_str:
                return (
                    False,
                    f"Archive v2 verification failure: invalid as_of date '{raw_ns_asof}' in tool_start",
                )
            eff_asof = str(eff_asof_dt_str)[:10]

            # 2. Derive effective look_back_days (model input or config default)
            cfg = get_config()
            ns_lb = (
                ns_input.get("look_back_days")
                if ns_input.get("look_back_days") is not None
                else ns_input.get("lookback_days")
            )
            if ns_lb is not None:
                eff_lb = int(ns_lb)
            else:
                eff_lb = int(cfg.get("global_news_lookback_days", 7))

            # 3. Derive effective limit (model input or config default, preserving explicit 0)
            ns_lim = (
                ns_input.get("limit")
                if ns_input.get("limit") is not None
                else ns_input.get("article_limit")
            )
            if ns_lim is not None:
                eff_lim = int(ns_lim)
            else:
                eff_lim = int(cfg.get("global_news_article_limit", 10))

            # 4. Validate producer_params matches effective values
            prod_asof = prod_p.get("as_of_date")
            if not prod_asof or str(prod_asof)[:10] != eff_asof:
                return (
                    False,
                    f"Archive v2 verification failure: producer_params as_of_date '{prod_asof}' does not match effective as_of_date '{eff_asof}'",
                )
            prod_lb = prod_p.get("look_back_days")
            if prod_lb is not None and int(prod_lb) != eff_lb:
                return (
                    False,
                    f"Archive v2 verification failure: producer_params look_back_days '{prod_lb}' does not match effective look_back_days '{eff_lb}'",
                )
            prod_lim = prod_p.get("limit")
            if prod_lim is not None and int(prod_lim) != eff_lim:
                return (
                    False,
                    f"Archive v2 verification failure: producer_params limit '{prod_lim}' does not match effective limit '{eff_lim}'",
                )

            # 5. Validate transport params and reject contradictions
            curr_dt = datetime.strptime(eff_asof, "%Y-%m-%d")
            expected_from_date = (curr_dt - timedelta(days=eff_lb)).strftime("%Y-%m-%d")

            if arm == "B":
                from tradingagents.dataflows.vendors.news_intelligence.client import to_iso_utc

                expected_asof_iso = to_iso_utc(eff_asof, end_of_day=True)
                expected_from_iso = to_iso_utc(expected_from_date, end_of_day=False)

                dt_exp_asof = _parse_utc_iso(expected_asof_iso)
                dt_exp_from = _parse_utc_iso(expected_from_iso)

                if "asOf" not in params or params["asOf"] is None:
                    return (
                        False,
                        "Archive v2 verification failure: transport missing required parameter 'asOf'",
                    )
                dt_asof = _parse_utc_iso(params["asOf"])
                if dt_asof is None or dt_exp_asof is None or dt_asof != dt_exp_asof:
                    return (
                        False,
                        f"Archive v2 verification failure: transport 'asOf' '{params['asOf']}' contradicts effective as_of_date '{eff_asof}'",
                    )
                if "to" not in params or params["to"] is None:
                    return (
                        False,
                        "Archive v2 verification failure: transport missing required parameter 'to'",
                    )
                dt_to = _parse_utc_iso(params["to"])
                if dt_to is None or dt_exp_asof is None or dt_to != dt_exp_asof:
                    return (
                        False,
                        f"Archive v2 verification failure: transport 'to' '{params['to']}' contradicts effective as_of_date '{eff_asof}'",
                    )
                if "from" not in params or params["from"] is None:
                    return (
                        False,
                        "Archive v2 verification failure: transport missing required parameter 'from'",
                    )
                dt_from = _parse_utc_iso(params["from"])
                if dt_from is None or dt_exp_from is None or dt_from != dt_exp_from:
                    return (
                        False,
                        f"Archive v2 verification failure: transport 'from' '{params['from']}' contradicts expected from date '{expected_from_date}'",
                    )
                if "limit" in params and int(params["limit"]) != eff_lim:
                    return (
                        False,
                        f"Archive v2 verification failure: transport limit '{params['limit']}' contradicts effective limit '{eff_lim}'",
                    )
            else:
                # Arm A (Yahoo)
                if "as_of_date" in params or "curr_date" in params:
                    p_asof = params.get("as_of_date") or params.get("curr_date")
                    if str(p_asof)[:10] != eff_asof:
                        return (
                            False,
                            f"Archive v2 verification failure: transport as_of_date '{p_asof}' contradicts '{eff_asof}'",
                        )
                if (
                    "look_back_days" in params
                    and params["look_back_days"] is not None
                    and int(params["look_back_days"]) != eff_lb
                ):
                    return (
                        False,
                        f"Archive v2 verification failure: transport look_back_days '{params['look_back_days']}' contradicts '{eff_lb}'",
                    )
                if (
                    "limit" in params
                    and params["limit"] is not None
                    and int(params["limit"]) != eff_lim
                ):
                    return (
                        False,
                        f"Archive v2 verification failure: transport limit '{params['limit']}' contradicts '{eff_lim}'",
                    )

        else:
            # get_news
            if not call_query or normalize_symbol(call_query) != normalize_symbol(ticker):
                return (
                    False,
                    f"Archive v2 verification failure: call query '{call_query}' does not match cell ticker '{ticker}'",
                )

            # 1. Required model-visible arguments: start_date, end_date
            raw_start = ns_input.get("start_date") or ns_input.get("from")
            if raw_start is None or str(raw_start).strip() == "":
                return (
                    False,
                    f"Archive v2 verification failure: tool_start missing required start_date for run_id '{call_run_id}'",
                )
            raw_end = ns_input.get("end_date") or ns_input.get("to")
            if raw_end is None or str(raw_end).strip() == "":
                return (
                    False,
                    f"Archive v2 verification failure: tool_start missing required end_date for run_id '{call_run_id}'",
                )

            # 2. Ticker handling: injected from graph state, but if model provided it, must match cell ticker
            ns_sym = ns_input.get("ticker") or ns_input.get("symbol")
            if (
                ns_sym is not None
                and str(ns_sym).strip() != ""
                and normalize_symbol(str(ns_sym)) != normalize_symbol(ticker)
            ):
                return (
                    False,
                    f"Archive v2 verification failure: tool_start requested ticker '{ns_sym}' does not match cell ticker '{ticker}'",
                )
            eff_ticker = ticker

            # 3. Apply native as_of_window clamping
            from tradingagents.dataflows.date_window import as_of_window

            try:
                eff_start, eff_end = as_of_window(
                    str(raw_start).strip(), str(raw_end).strip(), date_val
                )
            except ValueError as exc:
                return (
                    False,
                    f"Archive v2 verification failure: invalid tool_start date window: {exc}",
                )

            # 4. Validate producer_params matches effective values
            prod_sym = prod_p.get("ticker") or prod_p.get("symbol")
            if not prod_sym or normalize_symbol(str(prod_sym)) != normalize_symbol(eff_ticker):
                return (
                    False,
                    f"Archive v2 verification failure: producer_params ticker '{prod_sym}' does not match '{eff_ticker}'",
                )
            prod_start = prod_p.get("start_date")
            if not prod_start or str(prod_start)[:10] != eff_start[:10]:
                return (
                    False,
                    f"Archive v2 verification failure: producer_params start_date '{prod_start}' does not match effective start_date '{eff_start}'",
                )
            prod_end = prod_p.get("end_date")
            if not prod_end or str(prod_end)[:10] != eff_end[:10]:
                return (
                    False,
                    f"Archive v2 verification failure: producer_params end_date '{prod_end}' does not match effective end_date '{eff_end}'",
                )

            # Optional limit check (if present in tool_start)
            ns_lim = (
                ns_input.get("limit")
                if ns_input.get("limit") is not None
                else ns_input.get("article_limit")
            )
            eff_call_lim = (
                prod_p.get("limit") if prod_p.get("limit") is not None else params.get("limit")
            )
            if ns_lim is not None and eff_call_lim is not None and int(ns_lim) != int(eff_call_lim):
                return (
                    False,
                    f"Archive v2 verification failure: tool_start limit '{ns_lim}' does not match call limit '{eff_call_lim}'",
                )

            # 5. Validate transport params and reject contradictions
            if arm == "B":
                from tradingagents.dataflows.vendors.news_intelligence.client import (
                    map_ticker_for_news_intelligence,
                    to_iso_utc,
                )

                expected_sym, _ = map_ticker_for_news_intelligence(eff_ticker)
                if (
                    "symbol" in params
                    and str(params["symbol"]) != expected_sym
                    and normalize_symbol(str(params["symbol"])) != normalize_symbol(eff_ticker)
                ):
                    return (
                        False,
                        f"Archive v2 verification failure: transport symbol '{params['symbol']}' contradicts ticker '{eff_ticker}'",
                    )
                expected_from_iso = to_iso_utc(eff_start, end_of_day=False)
                expected_to_iso = to_iso_utc(eff_end, end_of_day=True)

                dt_exp_from = _parse_utc_iso(expected_from_iso)
                dt_exp_to = _parse_utc_iso(expected_to_iso)

                if "from" not in params or params["from"] is None:
                    return (
                        False,
                        "Archive v2 verification failure: transport missing required parameter 'from'",
                    )
                dt_from = _parse_utc_iso(params["from"])
                if dt_from is None or dt_exp_from is None or dt_from != dt_exp_from:
                    return (
                        False,
                        f"Archive v2 verification failure: transport 'from' '{params['from']}' contradicts effective start_date '{eff_start}'",
                    )
                if "to" not in params or params["to"] is None:
                    return (
                        False,
                        "Archive v2 verification failure: transport missing required parameter 'to'",
                    )
                dt_to = _parse_utc_iso(params["to"])
                if dt_to is None or dt_exp_to is None or dt_to != dt_exp_to:
                    return (
                        False,
                        f"Archive v2 verification failure: transport 'to' '{params['to']}' contradicts effective end_date '{eff_end}'",
                    )
                if "asOf" not in params or params["asOf"] is None:
                    return (
                        False,
                        "Archive v2 verification failure: transport missing required parameter 'asOf'",
                    )
                dt_asof = _parse_utc_iso(params["asOf"])
                if dt_asof is None or dt_exp_to is None or dt_asof != dt_exp_to:
                    return (
                        False,
                        f"Archive v2 verification failure: transport 'asOf' '{params['asOf']}' contradicts effective end_date '{eff_end}'",
                    )
                if (
                    "limit" in params
                    and prod_p.get("limit") is not None
                    and int(params["limit"]) != int(prod_p["limit"])
                ):
                    return (
                        False,
                        f"Archive v2 verification failure: transport limit '{params['limit']}' contradicts producer limit '{prod_p['limit']}'",
                    )
            else:
                # Arm A (Yahoo)
                if "ticker" in params:
                    p_sym = params["ticker"]
                    if normalize_symbol(str(p_sym)) != normalize_symbol(eff_ticker):
                        return (
                            False,
                            f"Archive v2 verification failure: transport ticker '{p_sym}' contradicts '{eff_ticker}'",
                        )
                if "symbol" in params:
                    p_sym = params["symbol"]
                    if normalize_symbol(str(p_sym)) != normalize_symbol(eff_ticker):
                        return (
                            False,
                            f"Archive v2 verification failure: transport symbol '{p_sym}' contradicts '{eff_ticker}'",
                        )
                if "start_date" in params:
                    p_s = params["start_date"]
                    if str(p_s)[:10] != eff_start[:10]:
                        return (
                            False,
                            f"Archive v2 verification failure: transport start_date '{p_s}' contradicts '{eff_start}'",
                        )
                if "from" in params:
                    p_s = params["from"]
                    if str(p_s)[:10] != eff_start[:10]:
                        return (
                            False,
                            f"Archive v2 verification failure: transport 'from' '{p_s}' contradicts '{eff_start}'",
                        )
                if "end_date" in params:
                    p_e = params["end_date"]
                    if str(p_e)[:10] != eff_end[:10]:
                        return (
                            False,
                            f"Archive v2 verification failure: transport end_date '{p_e}' contradicts '{eff_end}'",
                        )
                if "to" in params:
                    p_e = params["to"]
                    if str(p_e)[:10] != eff_end[:10]:
                        return (
                            False,
                            f"Archive v2 verification failure: transport 'to' '{p_e}' contradicts '{eff_end}'",
                        )
                if (
                    "limit" in params
                    and prod_p.get("limit") is not None
                    and int(params["limit"]) != int(prod_p["limit"])
                ):
                    return (
                        False,
                        f"Archive v2 verification failure: transport limit '{params['limit']}' contradicts producer limit '{prod_p['limit']}'",
                    )

        # Validate declared output_hash against tool_end output_hash (R3)
        decl_out_hash = call.get("output_hash")
        ne_out_hash = matching_ne.get("output_hash")
        if decl_out_hash != ne_out_hash:
            return (
                False,
                f"Archive v2 verification failure: call declared output_hash does not match observed tool_end output_hash ({decl_out_hash} != {ne_out_hash})",
            )

        # Step 2: Native Normalization & Sanitization
        if arm == "A":
            import tradingagents.dataflows.vendors.yahoo.news as yahoo_news
            from tradingagents.dataflows.date_window import coverage_gap, in_window

            raw_articles = snap_payload if isinstance(snap_payload, list) else []

            if "get_global_news" in tool_name:
                as_of_date = prod_p.get("as_of_date") or params.get("as_of_date") or date_val
                cfg = get_config()
                eff_lb = (
                    prod_p.get("look_back_days")
                    if prod_p.get("look_back_days") is not None
                    else params.get("look_back_days")
                )
                if eff_lb is None:
                    eff_lb = cfg.get("global_news_lookback_days", 7)
                eff_lim = (
                    prod_p.get("limit") if prod_p.get("limit") is not None else params.get("limit")
                )
                if eff_lim is None:
                    eff_lim = cfg.get("global_news_article_limit", 10)

                curr_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
                start_dt = curr_dt - relativedelta(days=eff_lb)
                start_date = start_dt.strftime("%Y-%m-%d")

                in_window_news = []
                seen_titles: set[str] = set()
                for article in raw_articles:
                    data_art = yahoo_news._extract_article_data(article)
                    if not in_window(data_art["pub_date"], start_dt, curr_dt):
                        continue
                    if data_art["title"] and data_art["title"] not in seen_titles:
                        seen_titles.add(data_art["title"])
                        in_window_news.append(data_art)

                derived_records = in_window_news[:eff_lim]

                # Step 3: Arm A Field-by-field equality check
                if len(stored_records) != len(derived_records):
                    return (
                        False,
                        f"Archive v2 verification failure: Arm A stored records count ({len(stored_records)}) != derived records count ({len(derived_records)})",
                    )

                for s_rec, d_rec in zip(stored_records, derived_records, strict=True):
                    for k_f in ("title", "summary", "publisher", "link"):
                        if s_rec.get(k_f) != d_rec.get(k_f):
                            return (
                                False,
                                f"Archive v2 verification failure: Arm A field mismatch on '{k_f}': {s_rec.get(k_f)!r} != {d_rec.get(k_f)!r}",
                            )
                    d_pd = d_rec.get("pub_date")
                    d_pd_str = d_pd.isoformat() if isinstance(d_pd, datetime) else d_pd
                    s_pd = s_rec.get("pub_date")
                    if s_pd != d_pd_str:
                        dt_s = _parse_utc_iso(str(s_pd)) if s_pd else None
                        dt_d = _parse_utc_iso(str(d_pd_str)) if d_pd_str else None
                        if dt_s != dt_d:
                            return (
                                False,
                                f"Archive v2 verification failure: Arm A pub_date mismatch: {s_pd!r} != {d_pd_str!r}",
                            )

                # Step 4: Producer Output Reconstruction
                news_str = yahoo_news._format_articles(derived_records)
                if not news_str:
                    gap = coverage_gap(
                        (), start_date, as_of_date, "Yahoo Finance global news", "market news"
                    )
                    candidate_text = (
                        gap or f"No global news found between {start_date} and {as_of_date}"
                    )
                else:
                    candidate_text = (
                        f"## Global Market News, from {start_date} to {as_of_date}:\n\n{news_str}"
                    )

            else:
                # get_news
                req_ticker = prod_p.get("ticker") or params.get("ticker") or ticker
                start_date = prod_p.get("start_date") or params.get("start_date") or date_val
                end_date = prod_p.get("end_date") or params.get("end_date") or date_val

                canonical = normalize_symbol(req_ticker)
                resolved = "" if canonical == req_ticker else f" (resolved to {canonical})"
                subject = f"news for {req_ticker}{resolved}"

                articles = [yahoo_news._extract_article_data(a) for a in raw_articles]
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                in_range = [a for a in articles if in_window(a["pub_date"], start_dt, end_dt)]
                derived_records = in_range

                # Step 3: Arm A Field-by-field equality check
                if len(stored_records) != len(derived_records):
                    return (
                        False,
                        f"Archive v2 verification failure: Arm A stored records count ({len(stored_records)}) != derived records count ({len(derived_records)})",
                    )

                for s_rec, d_rec in zip(stored_records, derived_records, strict=True):
                    for k_f in ("title", "summary", "publisher", "link"):
                        if s_rec.get(k_f) != d_rec.get(k_f):
                            return (
                                False,
                                f"Archive v2 verification failure: Arm A field mismatch on '{k_f}': {s_rec.get(k_f)!r} != {d_rec.get(k_f)!r}",
                            )
                    d_pd = d_rec.get("pub_date")
                    d_pd_str = d_pd.isoformat() if isinstance(d_pd, datetime) else d_pd
                    s_pd = s_rec.get("pub_date")
                    if s_pd != d_pd_str:
                        dt_s = _parse_utc_iso(str(s_pd)) if s_pd else None
                        dt_d = _parse_utc_iso(str(d_pd_str)) if d_pd_str else None
                        if dt_s != dt_d:
                            return (
                                False,
                                f"Archive v2 verification failure: Arm A pub_date mismatch: {s_pd!r} != {d_pd_str!r}",
                            )

                # Step 4: Producer Output Reconstruction (R5: Yahoo search fallback sentinel)
                if in_range:
                    candidate_text = f"## {req_ticker}{resolved} News, from {start_date} to {end_date}:\n\n{yahoo_news._format_articles(in_range)}"
                else:
                    feed_src = call.get("feed_source")
                    if feed_src == "search":
                        candidate_text = (
                            f"<Yahoo Finance news unavailable for {start_date}..{end_date}: Yahoo search "
                            f"returns only a sample of recent articles about {canonical}, so this is not "
                            f"an absence of {subject}>"
                        )
                    else:
                        gap = coverage_gap(
                            (a["pub_date"] for a in articles),
                            start_date,
                            end_date,
                            "Yahoo Finance news",
                            subject,
                        )
                        candidate_text = (
                            gap
                            or f"No news found for {req_ticker}{resolved} between {start_date} and {end_date}"
                        )

        elif arm == "B":
            import tradingagents.dataflows.vendors.news_intelligence.client as ni_client

            raw_events = snap_payload.get("events") if isinstance(snap_payload, dict) else []

            if "get_global_news" in tool_name:
                cfg = get_config()
                as_of_date = prod_p.get("as_of_date")
                if not as_of_date and "asOf" in params:
                    as_of_date = str(params["asOf"])[:10]
                if not as_of_date:
                    as_of_date = params.get("as_of_date") or date_val

                eff_lb = prod_p.get("look_back_days")
                if eff_lb is None and "from" in params and "asOf" in params:
                    dt_from = _parse_utc_iso(str(params["from"]))
                    dt_asof = _parse_utc_iso(str(params["asOf"]))
                    if dt_from and dt_asof:
                        eff_lb = (dt_asof.date() - dt_from.date()).days
                if eff_lb is None and "from" in params and as_of_date:
                    dt_from = _parse_utc_iso(str(params["from"]))
                    dt_curr = datetime.strptime(as_of_date, "%Y-%m-%d").replace(tzinfo=UTC)
                    if dt_from:
                        eff_lb = (dt_curr.date() - dt_from.date()).days
                if eff_lb is None:
                    eff_lb = params.get("look_back_days")
                if eff_lb is None:
                    eff_lb = cfg.get("global_news_lookback_days", 7)

                eff_lim = (
                    prod_p.get("limit") if prod_p.get("limit") is not None else params.get("limit")
                )
                if eff_lim is None:
                    eff_lim = cfg.get("global_news_article_limit", 10)

                curr_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
                start_dt = curr_dt - timedelta(days=eff_lb)
                start_date = start_dt.strftime("%Y-%m-%d")

                as_of = ni_client.to_iso_utc(as_of_date, end_of_day=True)
                min_pub = ni_client.to_iso_utc(start_date, end_of_day=False)
                max_pub = ni_client.to_iso_utc(as_of_date, end_of_day=True)
                cutoff_dt = ni_client.parse_utc_timestamp(as_of)
                min_pub_dt = ni_client.parse_utc_timestamp(min_pub)
                max_pub_dt = ni_client.parse_utc_timestamp(max_pub)

                sanitized_events = []
                for raw_event in raw_events:
                    if not ni_client.validate_canonical_event(raw_event):
                        continue
                    ev = ni_client.filter_and_sanitize_event(
                        raw_event,
                        cutoff_dt=cutoff_dt,
                        min_pub_dt=min_pub_dt,
                        max_pub_dt=max_pub_dt,
                    )
                    if ev is not None:
                        sanitized_events.append(ev)

                derived_records = sanitized_events[:eff_lim]

                # Step 3: Arm B Field-by-field equality check
                if len(stored_records) != len(derived_records):
                    return (
                        False,
                        f"Archive v2 verification failure: Arm B stored records count ({len(stored_records)}) != derived records count ({len(derived_records)})",
                    )

                for s_rec, d_rec in zip(stored_records, derived_records, strict=True):
                    for k_f in ("id", "publishedAt", "receivedAt", "enrichedAt", "symbols"):
                        if s_rec.get(k_f) != d_rec.get(k_f):
                            return (
                                False,
                                f"Archive v2 verification failure: Arm B canonical field mismatch on '{k_f}': {s_rec.get(k_f)!r} != {d_rec.get(k_f)!r}",
                            )

                # Step 4: Producer Output Reconstruction
                if sanitized_events:
                    formatted = "\n".join(
                        ni_client.format_article(e) for e in sanitized_events[:eff_lim]
                    )
                    candidate_text = (
                        f"## Global Market News, from {start_date} to {as_of_date}:\n\n{formatted}"
                    )
                else:
                    candidate_text = f"No global news found between {start_date} and {as_of_date}"

            else:
                # get_news
                cfg = get_config()
                req_ticker = (
                    prod_p.get("ticker") or params.get("ticker") or params.get("symbol") or ticker
                )
                start_date = prod_p.get("start_date")
                if not start_date and "from" in params:
                    start_date = str(params["from"])[:10]
                if not start_date:
                    start_date = params.get("start_date") or date_val

                end_date = prod_p.get("end_date")
                if not end_date and "to" in params:
                    end_date = str(params["to"])[:10]
                elif not end_date and "asOf" in params:
                    end_date = str(params["asOf"])[:10]
                if not end_date:
                    end_date = params.get("end_date") or date_val

                query_symbol, resolved = ni_client.map_ticker_for_news_intelligence(req_ticker)
                limit = (
                    prod_p.get("limit") if prod_p.get("limit") is not None else params.get("limit")
                )
                if limit is None:
                    limit = cfg.get("news_article_limit", 20)

                as_of = ni_client.to_iso_utc(end_date, end_of_day=True)
                min_pub = ni_client.to_iso_utc(start_date, end_of_day=False)
                max_pub = ni_client.to_iso_utc(end_date, end_of_day=True)
                cutoff_dt = ni_client.parse_utc_timestamp(as_of)
                min_pub_dt = ni_client.parse_utc_timestamp(min_pub)
                max_pub_dt = ni_client.parse_utc_timestamp(max_pub)

                sanitized_events = []
                for raw_event in raw_events:
                    if not ni_client.validate_canonical_event(raw_event):
                        continue
                    ev = ni_client.filter_and_sanitize_event(
                        raw_event,
                        cutoff_dt=cutoff_dt,
                        min_pub_dt=min_pub_dt,
                        max_pub_dt=max_pub_dt,
                    )
                    if ev is not None and query_symbol in ev.get("symbols", []):
                        sanitized_events.append(ev)

                derived_records = sanitized_events[:limit]

                # Step 3: Arm B Field-by-field equality check
                if len(stored_records) != len(derived_records):
                    return (
                        False,
                        f"Archive v2 verification failure: Arm B stored records count ({len(stored_records)}) != derived records count ({len(derived_records)})",
                    )

                for s_rec, d_rec in zip(stored_records, derived_records, strict=True):
                    for k_f in ("id", "publishedAt", "receivedAt", "enrichedAt", "symbols"):
                        if s_rec.get(k_f) != d_rec.get(k_f):
                            return (
                                False,
                                f"Archive v2 verification failure: Arm B canonical field mismatch on '{k_f}': {s_rec.get(k_f)!r} != {d_rec.get(k_f)!r}",
                            )

                # Step 4: Producer Output Reconstruction
                if sanitized_events:
                    formatted = "\n".join(
                        ni_client.format_article(e) for e in sanitized_events[:limit]
                    )
                    candidate_text = f"## {req_ticker}{resolved} News, from {start_date} to {end_date}:\n\n{formatted}"
                else:
                    candidate_text = f"No news found for {req_ticker}{resolved} between {start_date} and {end_date}"

        candidate_hash = hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()

        # Step 5: Exact Comparison Against Observed Consumed Tool Output in audit_trace
        if ne_out_hash != candidate_hash:
            return (
                False,
                f"Archive v2 verification failure: derived candidate hash mismatch ({candidate_hash} != {ne_out_hash})",
            )

        ne_out = matching_ne.get("output")
        if (
            ne_out is not None
            and str(ne_out) != candidate_text
            and sanitize_text(candidate_text) != str(ne_out)
        ):
            return (
                False,
                "Archive v2 verification failure: derived candidate output text does not match observed tool output",
            )

        bound_run_ids.add(call_run_id)

    if len(bound_run_ids) != len(news_ends):
        return (
            False,
            f"Archive v2 verification failure: only {len(bound_run_ids)} of {len(news_ends)} observed tool executions were bound",
        )

    return True, f"Verified authentic evidence for Arm {arm}"


def _verify_archive_for_arm(
    fpath: Path,
    arm: str,
    ticker: str,
    date_val: str,
    dt_cutoff: datetime | None,
    decision_att: dict[str, Any],
    cell_dir: Path,
) -> tuple[bool, str]:
    if dt_cutoff is None:
        return False, "Archive verification requires explicit, valid entry cutoff"

    data, err = _read_bounded_json_envelope(fpath)
    if err is not None:
        return False, f"Archive verification failure: {err}"

    if not data or not isinstance(data, (dict, list)):
        return False, "Archive JSON must be a non-empty dictionary or list"

    att_expected = decision_att.get("attempt_id")
    if not att_expected:
        return False, "Decision attempt missing required attempt_id"

    # Mandatory audit trace presence and non-empty news tool ends verification (checked upfront)
    tr_path = cell_dir / "results" / "audit_trace.json"
    if not tr_path.exists():
        return False, f"Audit trace missing on disk at {tr_path}"

    try:
        tr_data = json.loads(tr_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed to parse audit trace at {tr_path}: {exc}"

    news_ends = [
        t
        for t in tr_data.get("traces", [])
        if t.get("type") == "tool_end"
        and (
            str(t.get("tool", "")).startswith("get_news")
            or str(t.get("tool", "")).startswith("get_global_news")
        )
    ]
    if not news_ends:
        return False, f"Audit trace contains no observed news tool executions for Arm {arm}"

    # Version dispatch: t13-archive/v2 requires unbroken derivation verification; never falls back to legacy
    if isinstance(data, dict) and data.get("version") == "t13-archive/v2":
        try:
            return _verify_v2_archive(
                data=data,
                fpath=fpath,
                arm=arm,
                ticker=ticker,
                date_val=date_val,
                dt_cutoff=dt_cutoff,
                decision_att=decision_att,
                cell_dir=cell_dir,
                news_ends=news_ends,
                tr_data=tr_data,
            )
        except Exception as exc:
            return False, f"Archive v2 verification failure: {exc}"

    has_global_news = any("get_global_news" in str(t.get("tool", "")) for t in news_ends)
    has_asset_news = any(
        "get_news" in str(t.get("tool", "")) and "get_global_news" not in str(t.get("tool", ""))
        for t in news_ends
    )

    top_sym = None
    top_dt = None
    top_arm = None
    top_vendor = None
    top_att = None
    top_query = None
    top_cutoff = None

    if isinstance(data, dict):
        top_sym = data.get("ticker") or data.get("symbol") or data.get("query_symbol")
        if (
            top_sym
            and top_sym != ticker
            and not (
                has_global_news
                and top_sym in ("global", "market", "macro", "all", "GLOBAL", "MARKET")
            )
        ):
            return False, f"Archive top-level ticker '{top_sym}' does not match '{ticker}'"

        top_dt = data.get("date") or data.get("trade_date")
        if top_dt and top_dt != date_val:
            return False, f"Archive top-level date '{top_dt}' does not match '{date_val}'"

        top_arm = data.get("arm")
        if top_arm and top_arm != arm:
            return False, f"Archive top-level arm '{top_arm}' does not match '{arm}'"

        top_att = data.get("attempt_id")
        if top_att and top_att != att_expected:
            return (
                False,
                f"Archive top-level attempt_id '{top_att}' does not match '{att_expected}'",
            )

        top_query = data.get("query") or data.get("query_symbol")
        if (
            top_query
            and top_query != ticker
            and not (
                has_global_news
                and top_query
                in ("global", "market", "macro", "all", "global_news", "GLOBAL", "MARKET")
            )
        ):
            return False, f"Archive top-level query '{top_query}' does not match '{ticker}'"

        top_cutoff = data.get("cutoff") or data.get("effective_cutoff") or data.get("cutoff_time")
        if top_cutoff is not None:
            dt_top_cut = _parse_utc_iso(str(top_cutoff))
            if not dt_top_cut or dt_top_cut > dt_cutoff:
                return (
                    False,
                    f"Archive top-level cutoff '{top_cutoff}' is unparseable or later than entry cutoff {dt_cutoff.isoformat()}",
                )

        top_vendor = data.get("vendor")
        if top_vendor is None and data.get("source") in (
            "news_intelligence",
            "repo1",
            "yfinance",
            "yahoo",
        ):
            top_vendor = data.get("source")

        if top_vendor:
            if arm == "A" and top_vendor not in ("yfinance", "yahoo"):
                return False, f"Arm A requires Yahoo baseline; found routing vendor '{top_vendor}'"
            if arm == "B" and top_vendor not in ("news_intelligence", "repo1"):
                return (
                    False,
                    f"Arm B requires Repo 1 News Intelligence; found routing vendor '{top_vendor}'",
                )

        # Check top-level conflicting clock aliases (R1)
        for canon, aliases in [
            ("publishedAt", ["published_at"]),
            ("receivedAt", ["received_at"]),
            ("enrichedAt", ["enriched_at"]),
        ]:
            c_v = data.get(canon)
            for a_k in aliases:
                a_v = data.get(a_k)
                if (c_v is None and a_v is not None and canon in data) or (
                    c_v is not None and a_v is None and a_k in data
                ):
                    return (
                        False,
                        f"Top-level conflicting clock aliases for {canon}: null vs non-null",
                    )
                if c_v is not None and a_v is not None:
                    dt_c = _parse_utc_iso(str(c_v))
                    dt_a = _parse_utc_iso(str(a_v))
                    if dt_c and dt_a and dt_c != dt_a:
                        return (
                            False,
                            f"Top-level conflicting clock aliases for {canon}: '{c_v}' vs '{a_v}'",
                        )

        for clk_k in (
            "published_at",
            "publishedAt",
            "received_at",
            "receivedAt",
            "enriched_at",
            "enrichedAt",
        ):
            clk_v = data.get(clk_k)
            if clk_v is not None:
                dt_c = _parse_utc_iso(str(clk_v))
                if not dt_c or dt_c > dt_cutoff:
                    return (
                        False,
                        f"Top-level clock '{clk_k}' ({clk_v}) is naive, unparseable, or later than entry cutoff {dt_cutoff.isoformat()}",
                    )

    records: list[dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            nested = (
                item.get("events")
                or item.get("articles")
                or item.get("snapshots")
                or item.get("records")
                or item.get("data")
            )
            if isinstance(nested, list) and nested:
                for ne in nested:
                    if isinstance(ne, dict):
                        ne_copy = dict(ne)
                        _copy_top_level_wrapper_keys(ne_copy, item)
                        records.append(ne_copy)
            else:
                records.append(item)
    else:
        raw_recs = (
            data.get("events")
            or data.get("articles")
            or data.get("snapshots")
            or data.get("records")
            or data.get("data")
        )
        if not raw_recs:
            if any(
                k in data
                for k in ("id", "title", "content", "summary", "text", "response", "output")
            ):
                records = [data]
            else:
                return False, "Archive contains no events or records list"
        elif isinstance(raw_recs, list):
            for r in raw_recs:
                if isinstance(r, dict):
                    r_copy = dict(r)
                    _copy_top_level_wrapper_keys(r_copy, data)
                    records.append(r_copy)
        else:
            return False, "Archive records must be a list"

    if not records:
        return False, "Archive records must be a non-empty list"

    matching_records: list[dict[str, Any]] = []
    orig_matching: list[dict[str, Any]] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        orig_r = copy.deepcopy(r)
        is_global_event = (
            r.get("query") in ("global", "market", "macro", "all")
            or r.get("ticker") in ("global", "market", "macro", "all")
            or r.get("symbol") in ("global", "market", "macro", "all")
            or (isinstance(r.get("symbols"), list) and not r["symbols"])
            or (has_global_news and not has_asset_news)
        )
        if is_global_event:
            r_sym = r.get("ticker") or r.get("symbol") or "global"
            r_query = r.get("query") or "global"
        else:
            r_sym = r.get("ticker") or r.get("symbol") or r.get("query_symbol") or top_sym
            r_query = r.get("query") or r.get("query_symbol") or top_query

        r_dt = (
            r.get("date")
            or r.get("trade_date")
            or (
                str(r.get("publishedAt") or r.get("published_at") or r.get("pub_date")).split("T")[
                    0
                ]
                if (r.get("publishedAt") or r.get("published_at") or r.get("pub_date"))
                else None
            )
            or top_dt
        )
        r_arm = r.get("arm") or top_arm
        r_att = r.get("attempt_id") or top_att
        r_cut = r.get("cutoff") or r.get("effective_cutoff") or r.get("cutoff_time") or top_cutoff

        # Check symbol:
        sym_match = (r_sym and r_sym == ticker) or bool(
            has_global_news
            and (
                not r_sym
                or r_sym in ("global", "market", "macro", "all", "GLOBAL", "MARKET")
                or (
                    isinstance(r.get("symbols"), list)
                    and (ticker in r["symbols"] or not r["symbols"])
                )
            )
        )
        if not sym_match:
            continue

        # Check date:
        dt_match = False
        if r_dt and r_dt == date_val:
            dt_match = True
        elif (is_global_event or has_global_news) and r_dt:
            r_d_str = str(r_dt).split("T")[0]
            cell_d_str = str(date_val).split("T")[0]
            if r_d_str <= cell_d_str:
                dt_match = True
        if not dt_match:
            continue

        # Explicit arm identity is required (R1)
        if not r_arm or r_arm != arm:
            return False, f"Archive record missing or mismatched arm '{r_arm}' (expected '{arm}')"

        # Explicit routing vendor identity is required (R2)
        r_vendor = r.get("vendor") or top_vendor
        if not r_vendor and r.get("source") in ("news_intelligence", "repo1", "yfinance", "yahoo"):
            r_vendor = r.get("source")
        if not r_vendor:
            return False, f"Archive record missing routing vendor for Arm {arm}"
        if arm == "A" and r_vendor not in ("yfinance", "yahoo"):
            return False, f"Arm A requires Yahoo baseline; found routing vendor '{r_vendor}'"
        if arm == "B" and r_vendor not in ("news_intelligence", "repo1"):
            return (
                False,
                f"Arm B requires Repo 1 News Intelligence; found routing vendor '{r_vendor}'",
            )

        # Explicit canonical article publisher/source (R2)
        pub_source = r.get("source")
        if not pub_source or not isinstance(pub_source, str) or not pub_source.strip():
            if (
                isinstance(data, dict)
                and data.get("source")
                and data.get("source") not in ("news_intelligence", "repo1", "yfinance", "yahoo")
            ):
                pub_source = str(data.get("source"))
            elif arm == "B":
                pub_source = "News Intelligence"
            elif arm == "A":
                pub_source = "Yahoo"
        if not pub_source or not isinstance(pub_source, str) or not pub_source.strip():
            return False, f"Archive record missing required article publisher/source for Arm {arm}"
        r["source"] = pub_source

        # Mandatory attempt_id binding (R1)
        if not r_att:
            return False, f"Archive record missing required attempt_id for Arm {arm}"
        if r_att != att_expected:
            return (
                False,
                f"Archive attempt_id '{r_att}' does not match expected decision attempt '{att_expected}'",
            )

        # Mandatory query validation (R1)
        if not r_query:
            if has_global_news and not has_asset_news:
                r_query = "global"
            else:
                return False, f"Archive record missing required query for Arm {arm}"
        query_valid = (r_query == ticker) or (
            has_global_news
            and r_query in ("global", "market", "macro", "all", "global_news", "GLOBAL", "MARKET")
        )
        if not query_valid:
            return False, f"Archive record query '{r_query}' does not match ticker '{ticker}'"

        # Mandatory effective cutoff validation (R1)
        if r_cut is None:
            return False, f"Archive record missing required effective cutoff for Arm {arm}"
        dt_r_cut = _parse_utc_iso(str(r_cut))
        if not dt_r_cut or dt_r_cut > dt_cutoff:
            return (
                False,
                f"Archive record effective cutoff ({r_cut}) is invalid or later than entry cutoff {dt_cutoff.isoformat()}",
            )

        has_substance = bool(
            r.get("title")
            or r.get("content")
            or r.get("summary")
            or r.get("text")
            or r.get("articles")
            or r.get("headline")
        )
        if not has_substance:
            return False, "Archive record lacks substance"

        # Publication timestamp validation & alias conflict detection (R1)
        dt_pub, err_pub = _resolve_and_validate_clock(
            r,
            "publishedAt",
            ["published_at", "timestamp", "as_of"],
            data if isinstance(data, dict) else None,
            dt_cutoff,
            mandatory=True,
            field_name="publication",
        )
        if err_pub:
            return False, err_pub

        # Receipt timestamp validation & alias conflict detection (R1)
        dt_rec, err_rec = _resolve_and_validate_clock(
            r,
            "receivedAt",
            ["received_at"],
            data if isinstance(data, dict) else None,
            dt_cutoff,
            mandatory=True,
            field_name="receipt",
        )
        if err_rec:
            return False, err_rec

        # Enrichment timestamp validation & alias conflict detection (R1)
        dt_enr, err_enr = _resolve_and_validate_clock(
            r,
            "enrichedAt",
            ["enriched_at"],
            data if isinstance(data, dict) else None,
            dt_cutoff,
            mandatory=False,
            field_name="enrichment",
        )
        if err_enr:
            return False, err_enr

        # Native validation and point-in-time sanitization for Arm B (R1)
        if arm == "B":
            from tradingagents.dataflows.vendors.news_intelligence.client import (
                filter_and_sanitize_event,
                map_ticker_for_news_intelligence,
            )

            r_norm = dict(r)
            if "id" not in r_norm:
                r_norm["id"] = str(
                    r.get("event_id") or r.get("article_id") or f"event-{len(matching_records) + 1}"
                )
            if "source" not in r_norm or not r_norm["source"]:
                r_norm["source"] = pub_source
            if "title" not in r_norm:
                r_norm["title"] = str(r.get("headline") or "No title")
            query_sym, _ = map_ticker_for_news_intelligence(ticker)
            if "symbols" not in r_norm or not isinstance(r_norm["symbols"], list):
                if has_global_news and r_sym in ("global", "market", "macro"):
                    r_norm["symbols"] = []
                else:
                    r_norm["symbols"] = [query_sym] if query_sym else [ticker]

            # If enrichment clock was null/missing, ensure enrichedAt is None in r_norm
            if dt_enr is None:
                r_norm["enrichedAt"] = None

            san = filter_and_sanitize_event(r_norm, cutoff_dt=dt_cutoff)
            if san is None:
                return (
                    False,
                    f"Record failed native bi-temporal validation/sanitization for Arm B: {r.get('id') or r.get('title')}",
                )
            matching_records.append(san)
            orig_matching.append(orig_r)
        else:
            matching_records.append(r)
            orig_matching.append(orig_r)

    if not matching_records:
        return (
            False,
            f"No substantive, clock-valid records matching {ticker} on {date_val} for Arm {arm}",
        )

    # Verify each observed news tool execution and its linked tool_start request
    for ne in news_ends:
        r_id = ne.get("run_id")
        ns = None
        if r_id is not None:
            # Explicit run_id: must find EXACTLY ONE matching tool_start; never fall back
            matching_starts = [
                t
                for t in tr_data.get("traces", [])
                if t.get("type") == "tool_start" and t.get("run_id") == r_id
            ]
            if len(matching_starts) != 1:
                return (
                    False,
                    f"Observed news tool completion has explicit run_id '{r_id}' but matching tool_start is missing or ambiguous (found {len(matching_starts)})",
                )
            ns = matching_starts[0]
        else:
            # Genuinely absent run_id on tool_end: sequence fallback allowed ONLY when unambiguous
            try:
                idx_end = tr_data.get("traces", []).index(ne)
                preceding_starts = [
                    t
                    for t in tr_data.get("traces", [])[:idx_end]
                    if t.get("type") == "tool_start" and t.get("tool") == ne.get("tool")
                ]
                if len(preceding_starts) == 1:
                    ns = preceding_starts[0]
                else:
                    return (
                        False,
                        f"Observed news tool completion lacks run_id and preceding tool_start association is ambiguous (found {len(preceding_starts)})",
                    )
            except Exception as exc:
                return False, f"Failed resolving tool_start for news tool completion: {exc}"

        if ns is None:
            return False, "Observed news tool execution missing linked tool_start request"

        ts_inp = ns.get("input")
        if ts_inp is None or (isinstance(ts_inp, (str, dict, list)) and len(ts_inp) == 0):
            return False, "Observed news tool request input is absent or empty in tool_start"

        params = _extract_tool_start_params(ts_inp)
        if not params:
            return False, f"Observed news tool request input is unparseable: {ts_inp!r}"

        # Require all required arguments for the tool
        tool_name = str(ne.get("tool", ""))
        is_ticker_tool = "get_news" in tool_name and "get_global_news" not in tool_name

        if is_ticker_tool:
            req_ticker = params.get("ticker") or params.get("symbol")
            if not req_ticker:
                return (
                    False,
                    f"Observed news tool request missing required ticker/symbol parameter: {params}",
                )
            norm_req = normalize_symbol(str(req_ticker))
            norm_exp = normalize_symbol(ticker)
            if norm_req != norm_exp:
                return (
                    False,
                    f"Observed news tool request ticker '{req_ticker}' does not match cell ticker '{ticker}'",
                )

        req_date = (
            params.get("end_date")
            or params.get("trade_date")
            or params.get("curr_date")
            or params.get("as_of")
            or params.get("as_of_date")
            or params.get("date")
            or params.get("to")
        )
        if not req_date:
            return False, f"Observed news tool request missing required date parameter: {params}"

        req_d_str = str(req_date).split("T")[0]
        cell_d_str = str(date_val).split("T")[0]
        if req_d_str > cell_d_str:
            return (
                False,
                f"Observed news tool request date '{req_date}' is after cell date '{date_val}'",
            )

        dt_req = _parse_utc_iso(str(req_date))
        if dt_req and dt_req > dt_cutoff:
            return (
                False,
                f"Observed news tool request date/cutoff '{req_date}' exceeds entry cutoff {dt_cutoff.isoformat()}",
            )

        if "start_date" in params:
            s_d_str = str(params["start_date"]).split("T")[0]
            if s_d_str > cell_d_str:
                return (
                    False,
                    f"Observed news tool request start_date '{params['start_date']}' is after cell date '{date_val}'",
                )

        for cut_k in ("as_of", "as_of_date", "cutoff", "cutoff_date"):
            if cut_k in params:
                c_val = params[cut_k]
                dt_c = _parse_utc_iso(str(c_val))
                if dt_c and dt_c > dt_cutoff:
                    return (
                        False,
                        f"Observed news tool request explicit cutoff '{c_val}' exceeds entry cutoff {dt_cutoff.isoformat()}",
                    )

        for int_k in ("look_back_days", "lookback_days", "limit", "article_limit"):
            if int_k in params and params[int_k] is not None:
                try:
                    int_v = int(params[int_k])
                    if int_v < 0:
                        return (
                            False,
                            f"Observed news tool request {int_k} '{params[int_k]}' cannot be negative",
                        )
                except (ValueError, TypeError):
                    return (
                        False,
                        f"Observed news tool request {int_k} '{params[int_k]}' is not a valid integer",
                    )

    # Recompute payload digests exclusively from valid matching records and their native derivations
    candidate_payloads: list[str] = []

    # Whole-structure serializations of matching records
    with suppress(Exception):
        candidate_payloads.append(json.dumps(matching_records))
        candidate_payloads.append(json.dumps(matching_records, sort_keys=True))
        candidate_payloads.append(json.dumps({"events": matching_records}))
        candidate_payloads.append(json.dumps({"events": matching_records}, sort_keys=True))
        candidate_payloads.append(json.dumps({"articles": matching_records}))
        candidate_payloads.append(json.dumps({"articles": matching_records}, sort_keys=True))

    # Identify subsets of records for multi-tool runs
    asset_recs = [
        r
        for r in matching_records
        if (
            (
                r.get("query") == ticker
                or r.get("ticker") == ticker
                or (isinstance(r.get("symbols"), list) and ticker in r["symbols"])
            )
            and r.get("query") not in ("global", "market", "macro", "all")
            and r.get("ticker") not in ("global", "market", "macro", "all")
            and (not isinstance(r.get("symbols"), list) or r.get("symbols") != [])
        )
    ]
    if not asset_recs:
        asset_recs = [
            r
            for r in matching_records
            if r.get("query") not in ("global", "market", "macro", "all")
            and r.get("ticker") not in ("global", "market", "macro", "all")
            and (not isinstance(r.get("symbols"), list) or r.get("symbols") != [])
        ]
    if not asset_recs:
        asset_recs = matching_records

    global_recs = [
        r
        for r in matching_records
        if r.get("query") in ("global", "market", "macro", "all")
        or r.get("ticker") in ("global", "market", "macro", "all")
        or (
            isinstance(r.get("symbols"), list)
            and (not r["symbols"] or "global" in [str(s).lower() for s in r["symbols"]])
        )
    ]
    if not global_recs:
        global_recs = matching_records

    if arm == "B":
        try:
            from tradingagents.dataflows.vendors.news_intelligence.client import (
                format_article,
                map_ticker_for_news_intelligence,
            )

            # --- Arm B Asset News Candidates ---
            candidate_record_sets = [asset_recs]
            if matching_records != asset_recs:
                candidate_record_sets.append(matching_records)

            for cr_set in candidate_record_sets:
                records_variants: list[list[dict[str, Any]]] = [cr_set]
                variant_norm: list[dict[str, Any]] = []
                for r in cr_set:
                    r_v = copy.deepcopy(r)
                    if r_v.get("source") in ("news_intelligence", "repo1", None):
                        r_v["source"] = "News Intelligence"
                    variant_norm.append(r_v)
                if variant_norm != cr_set:
                    records_variants.append(variant_norm)

                if any("publishedAt" not in orig_r for orig_r in orig_matching):
                    variant_legacy = []
                    for r, orig_r in zip(cr_set, orig_matching[: len(cr_set)], strict=False):
                        r_l = copy.deepcopy(r)
                        if "publishedAt" not in orig_r:
                            r_l.pop("publishedAt", None)
                        variant_legacy.append(r_l)
                    records_variants.append(variant_legacy)
                    variant_legacy_src = []
                    for r in variant_legacy:
                        r_ls = copy.deepcopy(r)
                        if r_ls.get("source") in ("news_intelligence", "repo1", None):
                            r_ls["source"] = "News Intelligence"
                        variant_legacy_src.append(r_ls)
                    if variant_legacy_src != variant_legacy:
                        records_variants.append(variant_legacy_src)

                for recs in records_variants:
                    formatted_articles = "\n".join(format_article(e) for e in recs)
                    candidate_payloads.append(formatted_articles)

                    query_sym, resolved = map_ticker_for_news_intelligence(ticker)

                    tool_windows: list[tuple[str, str]] = []
                    for t in tr_data.get("traces", []):
                        if (
                            t.get("type") == "tool_start"
                            and str(t.get("tool", "")).startswith("get_news")
                            and not str(t.get("tool", "")).startswith("get_global_news")
                        ):
                            p = _extract_tool_start_params(t.get("input"))
                            s_d = p.get("start_date")
                            e_d = p.get("end_date") or p.get("trade_date")
                            if s_d and e_d:
                                tool_windows.append((str(s_d), str(e_d)))

                    d_end = datetime.strptime(date_val, "%Y-%m-%d")
                    d_start_7 = (d_end - timedelta(days=7)).strftime("%Y-%m-%d")
                    d_start_30 = (d_end - timedelta(days=30)).strftime("%Y-%m-%d")
                    tool_windows.extend(
                        [
                            (d_start_7, date_val),
                            (d_start_30, date_val),
                            (date_val, date_val),
                        ]
                    )

                    for s_d, e_d in tool_windows:
                        candidate_payloads.append(
                            f"## {ticker}{resolved} News, from {s_d} to {e_d}:\n\n{formatted_articles}"
                        )
                        if resolved:
                            candidate_payloads.append(
                                f"## {ticker} News, from {s_d} to {e_d}:\n\n{formatted_articles}"
                            )
                        recs_in_win = []
                        for r_ev in recs:
                            r_pub = r_ev.get("publishedAt") or r_ev.get("published_at")
                            if r_pub:
                                d_p = str(r_pub).split("T")[0]
                                if not (s_d <= d_p <= e_d):
                                    continue
                            recs_in_win.append(r_ev)
                        if recs_in_win != recs:
                            fmt_win = "\n".join(format_article(e) for e in recs_in_win)
                            candidate_payloads.append(
                                f"## {ticker}{resolved} News, from {s_d} to {e_d}:\n\n{fmt_win}"
                            )
                            if resolved:
                                candidate_payloads.append(
                                    f"## {ticker} News, from {s_d} to {e_d}:\n\n{fmt_win}"
                                )

            # --- Arm B Global Market News Candidates (R2) ---
            eff_cfg = get_config()
            default_lb = int(eff_cfg.get("global_news_lookback_days", 7))
            default_lim = int(eff_cfg.get("global_news_article_limit", 10))

            global_tool_configs: list[dict[str, Any]] = []
            for t in tr_data.get("traces", []):
                if t.get("type") == "tool_start" and "get_global_news" in str(t.get("tool", "")):
                    p = _extract_tool_start_params(t.get("input"))
                    as_of_p = (
                        p.get("as_of_date")
                        or p.get("curr_date")
                        or p.get("trade_date")
                        or p.get("date")
                        or date_val
                    )
                    raw_lb = (
                        p.get("look_back_days") if "look_back_days" in p else p.get("lookback_days")
                    )
                    if raw_lb is None:
                        lb_p = default_lb
                    else:
                        try:
                            lb_p = int(raw_lb)
                        except (ValueError, TypeError):
                            lb_p = default_lb

                    raw_lim = p.get("limit") if "limit" in p else p.get("article_limit")
                    if raw_lim is None:
                        lim_p = default_lim
                    else:
                        try:
                            lim_p = int(raw_lim)
                        except (ValueError, TypeError):
                            lim_p = default_lim

                    global_tool_configs.append(
                        {
                            "as_of_date": str(as_of_p).split("T")[0],
                            "look_back_days": lb_p,
                            "limit": lim_p,
                        }
                    )
            if has_global_news and not global_tool_configs:
                global_tool_configs.append(
                    {"as_of_date": date_val, "look_back_days": default_lb, "limit": default_lim}
                )
                global_tool_configs.append(
                    {"as_of_date": date_val, "look_back_days": 30, "limit": default_lim}
                )

            global_candidate_record_sets = [global_recs]
            if matching_records != global_recs:
                global_candidate_record_sets.append(matching_records)

            for cfg in global_tool_configs:
                cfg_as_of = cfg["as_of_date"]
                cfg_lb = cfg["look_back_days"]
                cfg_lim = cfg["limit"]
                c_dt = datetime.strptime(cfg_as_of, "%Y-%m-%d")
                s_dt = c_dt - timedelta(days=cfg_lb)
                s_date_str = s_dt.strftime("%Y-%m-%d")

                min_pub_utc = _parse_utc_iso(f"{s_date_str}T00:00:00Z")
                max_pub_utc = _parse_utc_iso(f"{cfg_as_of}T23:59:59Z")

                for g_set in global_candidate_record_sets:
                    in_window_events = []
                    for ev in g_set:
                        ev_pub = ev.get("publishedAt") or ev.get("published_at")
                        if ev_pub:
                            dt_p = _parse_utc_iso(str(ev_pub))
                            if min_pub_utc and dt_p and dt_p < min_pub_utc:
                                continue
                            if max_pub_utc and dt_p and dt_p > max_pub_utc:
                                continue
                        in_window_events.append(ev)

                    if in_window_events:
                        sub_events = in_window_events[:cfg_lim]
                        fmt_g = "\n".join(format_article(e) for e in sub_events)
                        candidate_payloads.append(
                            f"## Global Market News, from {s_date_str} to {cfg_as_of}:\n\n{fmt_g}"
                        )
                        sub_ni = []
                        for e in sub_events:
                            e_c = dict(e)
                            if e_c.get("source") in ("news_intelligence", "repo1", None):
                                e_c["source"] = "News Intelligence"
                            sub_ni.append(e_c)
                        if sub_ni != sub_events:
                            fmt_g_ni = "\n".join(format_article(e) for e in sub_ni)
                            candidate_payloads.append(
                                f"## Global Market News, from {s_date_str} to {cfg_as_of}:\n\n{fmt_g_ni}"
                            )
                    else:
                        candidate_payloads.append(
                            f"No global news found between {s_date_str} and {cfg_as_of}"
                        )

        except Exception as exc:
            logger.debug("Arm B candidate reconstruction exception: %s", exc)

    elif arm == "A":
        try:
            from tradingagents.dataflows.date_window import in_window
            from tradingagents.dataflows.vendors.yahoo.news import _format_articles

            # --- Arm A Asset News Candidates ---
            candidate_record_sets = [asset_recs]
            if matching_records != asset_recs:
                candidate_record_sets.append(matching_records)

            for cr_set in candidate_record_sets:
                norm_a_records = []
                for r in cr_set:
                    rc = dict(r)
                    if not rc.get("publisher"):
                        rc["publisher"] = rc.get("source") or "Yahoo"
                    if "summary" not in rc:
                        rc["summary"] = rc.get("content") or ""
                    if "link" not in rc:
                        rc["link"] = ""
                    norm_a_records.append(rc)

                formatted_a = _format_articles(norm_a_records)
                candidate_payloads.append(formatted_a)
                candidate_payloads.append(formatted_a.strip())

                d_end = datetime.strptime(date_val, "%Y-%m-%d")
                d_start_7 = (d_end - timedelta(days=7)).strftime("%Y-%m-%d")
                d_start_30 = (d_end - timedelta(days=30)).strftime("%Y-%m-%d")
                for s_d, e_d in [
                    (d_start_7, date_val),
                    (d_start_30, date_val),
                    (date_val, date_val),
                ]:
                    candidate_payloads.append(
                        f"## {ticker} News, from {s_d} to {e_d}:\n\n{formatted_a}"
                    )
                    candidate_payloads.append(
                        f"## {ticker} News, from {s_d} to {e_d}:\n\n{formatted_a.strip()}"
                    )
                    s_dt_obj = datetime.strptime(s_d, "%Y-%m-%d")
                    e_dt_obj = datetime.strptime(e_d, "%Y-%m-%d")
                    a_in_win = []
                    for r_ev in norm_a_records:
                        r_pub = (
                            r_ev.get("pub_date")
                            or r_ev.get("publishedAt")
                            or r_ev.get("published_at")
                        )
                        if r_pub:
                            dt_p = _parse_utc_iso(str(r_pub))
                            if dt_p and not in_window(dt_p, s_dt_obj, e_dt_obj):
                                continue
                        a_in_win.append(r_ev)
                    if a_in_win != norm_a_records:
                        fmt_a_win = _format_articles(a_in_win)
                        candidate_payloads.append(
                            f"## {ticker} News, from {s_d} to {e_d}:\n\n{fmt_a_win}"
                        )
                        candidate_payloads.append(
                            f"## {ticker} News, from {s_d} to {e_d}:\n\n{fmt_a_win.strip()}"
                        )

            # --- Arm A Global Market News Candidates (R2) ---
            eff_cfg_a = get_config()
            default_lb_a = int(eff_cfg_a.get("global_news_lookback_days", 7))
            default_lim_a = int(eff_cfg_a.get("global_news_article_limit", 10))

            global_tool_configs_a: list[dict[str, Any]] = []
            for t in tr_data.get("traces", []):
                if t.get("type") == "tool_start" and "get_global_news" in str(t.get("tool", "")):
                    p = _extract_tool_start_params(t.get("input"))
                    as_of_p = (
                        p.get("as_of_date")
                        or p.get("curr_date")
                        or p.get("trade_date")
                        or p.get("date")
                        or date_val
                    )
                    raw_lb = (
                        p.get("look_back_days") if "look_back_days" in p else p.get("lookback_days")
                    )
                    if raw_lb is None:
                        lb_p = default_lb_a
                    else:
                        try:
                            lb_p = int(raw_lb)
                        except (ValueError, TypeError):
                            lb_p = default_lb_a

                    raw_lim = p.get("limit") if "limit" in p else p.get("article_limit")
                    if raw_lim is None:
                        lim_p = default_lim_a
                    else:
                        try:
                            lim_p = int(raw_lim)
                        except (ValueError, TypeError):
                            lim_p = default_lim_a

                    global_tool_configs_a.append(
                        {
                            "as_of_date": str(as_of_p).split("T")[0],
                            "look_back_days": lb_p,
                            "limit": lim_p,
                        }
                    )
            if has_global_news and not global_tool_configs_a:
                global_tool_configs_a.append(
                    {"as_of_date": date_val, "look_back_days": default_lb_a, "limit": default_lim_a}
                )
                global_tool_configs_a.append(
                    {"as_of_date": date_val, "look_back_days": 30, "limit": default_lim_a}
                )

            global_candidate_record_sets_a = [global_recs]
            if matching_records != global_recs:
                global_candidate_record_sets_a.append(matching_records)

            for cfg in global_tool_configs_a:
                cfg_as_of = cfg["as_of_date"]
                cfg_lb = cfg["look_back_days"]
                cfg_lim = cfg["limit"]
                c_dt = datetime.strptime(cfg_as_of, "%Y-%m-%d")
                s_dt = c_dt - timedelta(days=cfg_lb)
                s_date_str = s_dt.strftime("%Y-%m-%d")

                for g_set in global_candidate_record_sets_a:
                    in_window_a = []
                    for ev in g_set:
                        ev_c = dict(ev)
                        if not ev_c.get("publisher"):
                            ev_c["publisher"] = ev_c.get("source") or "Yahoo"
                        if "summary" not in ev_c:
                            ev_c["summary"] = ev_c.get("content") or ""
                        if "link" not in ev_c:
                            ev_c["link"] = ""

                        pub_val = (
                            ev_c.get("pub_date")
                            or ev_c.get("publishedAt")
                            or ev_c.get("published_at")
                        )
                        if pub_val:
                            dt_p = _parse_utc_iso(str(pub_val))
                            if dt_p and not in_window(dt_p, s_dt, c_dt):
                                continue
                        in_window_a.append(ev_c)

                    if in_window_a:
                        sub_a = in_window_a[:cfg_lim]
                        news_str = _format_articles(sub_a)
                        candidate_payloads.append(
                            f"## Global Market News, from {s_date_str} to {cfg_as_of}:\n\n{news_str}"
                        )
                        candidate_payloads.append(
                            f"## Global Market News, from {s_date_str} to {cfg_as_of}:\n\n{news_str.strip()}"
                        )
                    else:
                        candidate_payloads.append(
                            f"No global news found between {s_date_str} and {cfg_as_of}"
                        )

        except Exception as exc:
            logger.debug("Arm A candidate reconstruction exception: %s", exc)

    if isinstance(data, list) and len(data) == len(matching_records):
        with suppress(Exception):
            candidate_payloads.append(json.dumps(data))
            candidate_payloads.append(json.dumps(data, sort_keys=True))
    elif isinstance(data, dict):
        if data.get("events") and len(data["events"]) == len(matching_records):
            with suppress(Exception):
                candidate_payloads.append(json.dumps(data["events"]))
                candidate_payloads.append(json.dumps(data["events"], sort_keys=True))
        if data.get("articles") and len(data["articles"]) == len(matching_records):
            with suppress(Exception):
                candidate_payloads.append(json.dumps(data["articles"]))
                candidate_payloads.append(json.dumps(data["articles"], sort_keys=True))

    for p in list(candidate_payloads):
        clean_p = sanitize_text(p)
        if clean_p not in candidate_payloads:
            candidate_payloads.append(clean_p)

    recomputed_hashes: dict[str, str] = {}
    for p in candidate_payloads:
        h = hashlib.sha256(p.encode("utf-8")).hexdigest()
        recomputed_hashes[h] = p

    # Collect declared hashes from archive
    declared_hashes: set[str] = set()
    if isinstance(data, dict):
        sym_ok = (
            not top_sym
            or top_sym == ticker
            or (has_global_news and top_sym in ("global", "market", "macro"))
        )
        arm_ok = not top_arm or top_arm == arm
        if sym_ok and arm_ok:
            for hk in (
                "output_hash",
                "output_hashes",
                "response_hash",
                "content_hash",
                "hash",
                "payload_hash",
            ):
                hv = data.get(hk)
                if isinstance(hv, list):
                    for h_item in hv:
                        if h_item:
                            declared_hashes.add(str(h_item))
                elif hv:
                    declared_hashes.add(str(hv))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                i_sym = item.get("ticker") or item.get("symbol") or item.get("query_symbol")
                i_arm = item.get("arm")
                i_dt = item.get("date") or item.get("trade_date")
                sym_ok = (
                    not i_sym
                    or i_sym == ticker
                    or (has_global_news and i_sym in ("global", "market", "macro"))
                )
                arm_ok = not i_arm or i_arm == arm
                dt_ok = (
                    not i_dt
                    or i_dt == date_val
                    or (has_global_news and str(i_dt).split("T")[0] <= str(date_val).split("T")[0])
                )
                if sym_ok and arm_ok and dt_ok:
                    for hk in (
                        "output_hash",
                        "output_hashes",
                        "response_hash",
                        "content_hash",
                        "hash",
                        "payload_hash",
                    ):
                        hv = item.get(hk)
                        if isinstance(hv, list):
                            for h_item in hv:
                                if h_item:
                                    declared_hashes.add(str(h_item))
                        elif hv:
                            declared_hashes.add(str(hv))

    for r in matching_records:
        for hk in (
            "output_hash",
            "output_hashes",
            "response_hash",
            "content_hash",
            "hash",
            "payload_hash",
        ):
            hv = r.get(hk)
            if isinstance(hv, list):
                for h_item in hv:
                    if h_item:
                        declared_hashes.add(str(h_item))
            elif hv:
                declared_hashes.add(str(hv))

    # If declared hashes exist, verify that they are backed by real recomputed payload digests
    if declared_hashes:
        for dh in declared_hashes:
            if dh not in recomputed_hashes:
                return (
                    False,
                    f"Archive declared hash '{dh}' does not match recomputed payload digest (tampered or unlinked content)",
                )

    for ne in news_ends:
        out_h = ne.get("output_hash")
        if not out_h:
            return False, "Observed news tool execution missing output_hash"

        # The observed tool output hash must match recomputed archive payload digest
        if out_h not in recomputed_hashes:
            return (
                False,
                f"Archive recomputed payload digests do not match consumed tool output hash '{out_h}'",
            )

        # If declared hashes exist, tool output hash must match declared hash
        if declared_hashes and out_h not in declared_hashes:
            return (
                False,
                f"Archive declared output hash {declared_hashes} does not match consumed tool output hash '{out_h}'",
            )

        # If trace retains output text, verify actual consumed response identity
        out_str = ne.get("output")
        if out_str is not None:
            actual_str = str(out_str)
            actual_h = hashlib.sha256(actual_str.encode("utf-8")).hexdigest()
            if actual_h != out_h:
                return (
                    False,
                    f"Observed news tool output text does not hash to declared tool hash '{out_h}'",
                )
            if (
                recomputed_hashes.get(out_h) != actual_str
                and sanitize_text(recomputed_hashes.get(out_h, "")) != actual_str
            ):
                return (
                    False,
                    f"Retained archive payload does not match observed news tool output text for '{out_h}'",
                )

    return True, f"Verified authentic evidence for Arm {arm}"


def _verify_provenance_archive_file(
    fpath: Path, ticker: str, date_val: str, dA: dict[str, Any], dB: dict[str, Any]
) -> bool:
    cutoff = datetime.now(UTC)
    ok_A, _ = _verify_archive_for_arm(fpath, "A", ticker, date_val, cutoff, dA, Path("."))
    ok_B, _ = _verify_archive_for_arm(fpath, "B", ticker, date_val, cutoff, dB, Path("."))
    return ok_A or ok_B


def validate_evidence_archive(
    fpath: Path | str,
    arm: str,
    ticker: str,
    date_val: str,
    dt_cutoff: datetime | None,
    decision_att: dict[str, Any],
    cell_dir: Path | str,
) -> tuple[bool, str]:
    """Public helper to validate an authentic evidence archive for an arm."""
    return _verify_archive_for_arm(
        Path(fpath),
        arm,
        ticker,
        date_val,
        dt_cutoff,
        decision_att,
        Path(cell_dir),
    )


def _get_market_tz(ticker: str | None) -> ZoneInfo:
    """Return exchange market timezone for an asset to distinguish daily-bar labels from availability clocks."""
    if not ticker:
        return ZoneInfo("America/New_York")
    sym = ticker.upper().strip()
    if sym.endswith(".T"):
        return ZoneInfo("Asia/Tokyo")
    if sym.endswith(".L"):
        return ZoneInfo("Europe/London")
    if sym.endswith(".DE") or sym.endswith(".PA"):
        return ZoneInfo("Europe/Berlin")
    if (
        sym.endswith("-USD")
        or sym.endswith("-EUR")
        or sym.endswith("-BTC")
        or sym in ("BTC", "ETH", "SOL", "USDT")
    ):
        return ZoneInfo("UTC")
    return ZoneInfo("America/New_York")


def _validate_entry_timestamp_candidate(
    val: Any, key_name: str = "entry", ticker: str | None = None
) -> tuple[datetime | None, str | None]:
    """Validate and parse a supplied entry timestamp candidate.

    Rejects:
    - unparseable, empty, or naive timestamps
    - market-local daily-bar labels (00:00:00) in the asset's market exchange timezone
    - UTC midnight daily-bar labels (00:00:00Z)

    Accepts:
    - proven availability instants regardless of offset representation (e.g. 20:00:00Z == 00:00:00+04:00)
    - legitimate availability clocks using any hour when not a market/UTC midnight bar label (e.g. 04:00:00Z for crypto, 05:00:00Z EDT for US)
    """
    if val is None:
        return None, None
    if not isinstance(val, (str, datetime)):
        if hasattr(val, "to_pydatetime"):
            with suppress(Exception):
                val = val.to_pydatetime()
        if not isinstance(val, (str, datetime)):
            return None, f"Candidate '{key_name}' ({val!r}) is not a valid string or datetime"

    if isinstance(val, datetime):
        if val.tzinfo is None:
            return None, f"Candidate '{key_name}' ({val!r}) is naive (missing timezone)"
        raw_dt = val
        raw_str = val.isoformat()
    else:
        raw_str = val.strip()
        if not raw_str:
            return None, f"Candidate '{key_name}' is empty string"
        try:
            norm = raw_str.replace("Z", "+00:00")
            raw_dt = datetime.fromisoformat(norm)
        except Exception as exc:
            return None, f"Candidate '{key_name}' ('{raw_str}') is unparseable: {exc}"
        if raw_dt.tzinfo is None:
            return None, f"Candidate '{key_name}' ('{raw_str}') is naive (missing timezone)"

    # Normalize to UTC
    utc_dt = raw_dt.astimezone(UTC)

    # 1. Market exchange daily-bar label check:
    # Convert the candidate instant to the asset's market timezone.
    # If the time components are zeroed (00:00:00), this instant represents a market daily-bar label.
    market_tz = _get_market_tz(ticker)
    market_dt = raw_dt.astimezone(market_tz)
    if (
        market_dt.hour == 0
        and market_dt.minute == 0
        and market_dt.second == 0
        and market_dt.microsecond == 0
    ):
        return (
            None,
            f"Candidate '{key_name}' ('{raw_str}') is a calendar daily-bar midnight label (00:00:00) in {market_tz.key}, not an availability clock",
        )

    # 2. UTC daily-bar label check:
    # If the UTC time components are zeroed (00:00:00Z), this instant represents a UTC daily-bar label.
    if utc_dt.hour == 0 and utc_dt.minute == 0 and utc_dt.second == 0 and utc_dt.microsecond == 0:
        return (
            None,
            f"Candidate '{key_name}' ('{raw_str}') is a calendar daily-bar midnight label (00:00:00) in UTC (00:00:00Z), not an availability clock",
        )

    return utc_dt, None


def _extract_arm_entry_candidate(
    d_rec: dict[str, Any],
    s_rec: dict[str, Any] | None = None,
    ticker: str | None = None,
) -> tuple[datetime | None, str | None]:
    """Extract and validate entry timestamp for an arm, ensuring internal consistency and rejecting invalid candidates or daily-bar labels (R2)."""
    d_candidates = {
        "entry_time": d_rec.get("entry_time"),
        "entry_close": d_rec.get("entry_close"),
        "first_close": d_rec.get("first_close"),
        "entry_deadline": d_rec.get("entry_deadline"),
    }
    s_candidates = (
        {
            "entry_time": s_rec.get("entry_time"),
            "entry_close": s_rec.get("entry_close"),
            "first_close": s_rec.get("first_close"),
            "entry_deadline": s_rec.get("entry_deadline"),
        }
        if s_rec
        else {}
    )

    d_dts: list[datetime] = []
    for k, v in d_candidates.items():
        if v is not None:
            dt, err = _validate_entry_timestamp_candidate(v, f"decision.{k}", ticker=ticker)
            if err:
                return None, err
            if dt is not None:
                d_dts.append(dt)

    s_dts: list[datetime] = []
    for k, v in s_candidates.items():
        if v is not None:
            dt, err = _validate_entry_timestamp_candidate(v, f"settlement.{k}", ticker=ticker)
            if err:
                return None, err
            if dt is not None:
                s_dts.append(dt)

    if not d_dts and not s_dts:
        return None, "Missing entry/availability evidence"

    # Verify consistency within decision candidates
    if len(set(d_dts)) > 1:
        return (
            None,
            f"Contradictory entry timestamps within decision attempt: {[dt.isoformat() for dt in d_dts]}",
        )

    # Verify consistency within settlement candidates
    if len(set(s_dts)) > 1:
        return (
            None,
            f"Contradictory entry timestamps within settlement attempt: {[dt.isoformat() for dt in s_dts]}",
        )

    # Verify decision vs settlement consistency if both present
    if d_dts and s_dts and d_dts[0] != s_dts[0]:
        return (
            None,
            f"Contradictory entry timestamps between decision ({d_dts[0].isoformat()}) and settlement ({s_dts[0].isoformat()})",
        )

    chosen_dt = d_dts[0] if d_dts else s_dts[0]
    return chosen_dt, None


def _validate_archive_evidence(
    output_root: Path,
    ticker: str,
    date_val: str,
    cell_A_dir: Path,
    cell_B_dir: Path,
    dA: dict[str, Any],
    dB: dict[str, Any],
    sA: dict[str, Any] | None = None,
    sB: dict[str, Any] | None = None,
) -> tuple[bool, bool, list[str]]:
    reasons: list[str] = []

    # 1. Timing Eligibility Validation
    timing_eligible = False
    startA = dA.get("started_at")
    startB = dB.get("started_at")

    dt_cutoff: datetime | None = None

    if not startA or not startB:
        reasons.append("Missing decision started_at timestamp in one or both arms")
    else:
        dt_startA = _parse_utc_iso(startA)
        dt_startB = _parse_utc_iso(startB)
        if not dt_startA or not dt_startB:
            reasons.append("Unparseable or naive decision started_at timestamp")
        else:
            # R2: Extract and validate entry evidence per arm; no guessed 20:00Z fallback
            dt_entry_A, err_A = _extract_arm_entry_candidate(dA, sA, ticker=ticker)
            dt_entry_B, err_B = _extract_arm_entry_candidate(dB, sB, ticker=ticker)

            if err_A:
                reasons.append(f"Arm A entry evidence invalid: {err_A}")
            if err_B:
                reasons.append(f"Arm B entry evidence invalid: {err_B}")

            if dt_entry_A and dt_entry_B:
                if dt_entry_A != dt_entry_B:
                    reasons.append(
                        f"Conflicting entry timestamps between Arm A ({dt_entry_A.isoformat()}) and Arm B ({dt_entry_B.isoformat()})"
                    )
                else:
                    dt_cutoff = dt_entry_A

            if dt_cutoff is not None:
                is_retro = (dt_startA.date() > dt_cutoff.date()) or (
                    dt_startB.date() > dt_cutoff.date()
                )

                lookahead_detected = False

                # In prospective mode, verify starting and completion happened on/before entry cutoff
                if not is_retro:
                    compA = dA.get("completed_at")
                    compB = dB.get("completed_at")
                    dt_compA = _parse_utc_iso(compA) if compA else None
                    dt_compB = _parse_utc_iso(compB) if compB else None
                    if dt_startA and dt_startA > dt_cutoff:
                        lookahead_detected = True
                        reasons.append(
                            f"Arm A started ({startA}) after entry cutoff {dt_cutoff.isoformat()}"
                        )
                    if dt_startB and dt_startB > dt_cutoff:
                        lookahead_detected = True
                        reasons.append(
                            f"Arm B started ({startB}) after entry cutoff {dt_cutoff.isoformat()}"
                        )
                    if dt_compA and dt_compA > dt_cutoff:
                        lookahead_detected = True
                        reasons.append(
                            f"Arm A completed ({compA}) after entry cutoff {dt_cutoff.isoformat()}"
                        )
                    if dt_compB and dt_compB > dt_cutoff:
                        lookahead_detected = True
                        reasons.append(
                            f"Arm B completed ({compB}) after entry cutoff {dt_cutoff.isoformat()}"
                        )

                # Check audit traces: execution events during run are normal in-run activity!
                # Reject only if audit execution event happened after cutoff in prospective mode
                for cell_d in [cell_A_dir, cell_B_dir]:
                    tr_path = cell_d / "results" / "audit_trace.json"
                    if tr_path.exists():
                        try:
                            tr_data = json.loads(tr_path.read_text(encoding="utf-8"))
                            for t in tr_data.get("traces", []):
                                t_ts = _parse_utc_iso(t.get("timestamp"))
                                if not is_retro and t_ts and t_ts > dt_cutoff:
                                    lookahead_detected = True
                                    reasons.append(
                                        f"Audit trace event timestamp ({t.get('timestamp')}) occurred after entry cutoff ({dt_cutoff.isoformat()})"
                                    )
                                    break
                        except Exception:
                            pass

                if not lookahead_detected:
                    if is_retro:
                        timing_provenance_files = _find_valid_json_files(
                            [
                                cell_A_dir / "data" / "input_clock_provenance.json",
                                cell_B_dir / "data" / "input_clock_provenance.json",
                                output_root / "timing_provenance.json",
                                output_root / "baseline_archive",
                            ]
                        )
                        valid_clock_proof = False
                        for t_file in timing_provenance_files:
                            if _verify_timing_clock_file(t_file, ticker, date_val, dt_cutoff):
                                valid_clock_proof = True
                                break
                        if valid_clock_proof:
                            timing_eligible = True
                        else:
                            reasons.append(
                                f"Retrospective decision clock ({startA or startB}) without verified contemporaneous input clock archive matching {ticker} on {date_val}"
                            )
                    else:
                        timing_eligible = True

    # 2. Immutable Baseline Archive / Provenance Validation for BOTH arms
    # R2: If entry cutoff is missing or invalid, archive evidence validation fails closed
    if dt_cutoff is None:
        reasons.append("Archive evidence validation blocked: missing or invalid entry cutoff")
        return False, False, reasons, False

    dt_cutoff_val = dt_cutoff

    candidate_sources_A = [
        output_root / "baseline_archive",
        output_root / "baseline_archive.json",
        cell_A_dir / "data" / "baseline_archive.json",
        cell_A_dir / "data" / "provenance_archive.json",
        cell_A_dir / "data" / "api_snapshots.json",
        output_root / "provenance_archive.json",
    ]
    candidate_sources_B = [
        cell_B_dir / "data" / "provenance_archive.json",
        cell_B_dir / "data" / "api_snapshots.json",
        cell_B_dir / "data" / "news_intelligence_archive.json",
        output_root / "news_intelligence_archive",
        output_root / "news_intelligence_archive.json",
        output_root / "provenance_archive.json",
    ]

    files_A = _find_valid_json_files(candidate_sources_A)
    verified_A = False
    reason_A = "No archive file found"
    for f in files_A:
        ok, msg = _verify_archive_for_arm(f, "A", ticker, date_val, dt_cutoff_val, dA, cell_A_dir)
        if ok:
            verified_A = True
            break
        else:
            reason_A = msg
    if not verified_A:
        reasons.append(f"Arm A (Yahoo baseline): {reason_A}")

    files_B = _find_valid_json_files(candidate_sources_B)
    verified_B = False
    reason_B = "No archive file found"
    for f in files_B:
        ok, msg = _verify_archive_for_arm(f, "B", ticker, date_val, dt_cutoff_val, dB, cell_B_dir)
        if ok:
            verified_B = True
            break
        else:
            reason_B = msg
    if not verified_B:
        reasons.append(f"Arm B (Repo 1 News Intelligence): {reason_B}")

    has_v2 = False
    for f in files_A + files_B:
        c_data, err = _read_bounded_json_envelope(f)
        if err is None and isinstance(c_data, dict) and c_data.get("version") == "t13-archive/v2":
            has_v2 = True
            break

    provenance_verified = bool(verified_A and verified_B)

    return provenance_verified, timing_eligible, reasons, has_v2


def _write_entries_to_log(path: Path, entries: list[dict[str, Any]]) -> None:
    """Helper to write native TradingMemoryLog format from parsed entries."""
    if not entries:
        if path.exists():
            path.unlink()
        return
    blocks: list[str] = []
    for e in entries:
        trade_date = e.get("date", "2026-08-03")
        ticker = e.get("ticker", "NVDA")
        rating = e.get("rating", "Hold")
        if e.get("pending"):
            tag = f"[{trade_date} | {ticker} | {rating} | pending]"
        else:
            raw_pct = e.get("raw") or "n/a"
            alpha_pct = e.get("alpha") or "n/a"
            holding_days = str(e.get("holding") or "5d")
            if not holding_days.endswith("d"):
                holding_days = f"{holding_days}d"
            res = e.get("resolved")
            tag = f"[{trade_date} | {ticker} | {rating} | {raw_pct} | {alpha_pct} | {holding_days}"
            if res:
                tag += f" | resolved:{res}"
            tag += "]"
        decision = e.get("decision", "")
        reflection = e.get("reflection", "")
        body = f"{tag}\n\nDECISION:\n{decision}"
        if reflection:
            body += f"\n\nREFLECTION:\n{reflection}"
        blocks.append(body)
    separator = "\n\n<!-- ENTRY_END -->\n\n"
    content = separator.join(blocks) + separator
    path.write_text(content, encoding="utf-8")


def summarize_experiment(output_dir: Path | str) -> dict[str, Any]:
    """Summarize settled paired cells using native metrics without graph execution."""
    output_root = Path(output_dir).resolve()
    journal_path = output_root / "journal.json"
    if not journal_path.exists():
        raise FileNotFoundError(f"journal.json not found at {output_root}")

    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    decisions = journal.get("decision_attempts", {})
    settlements = journal.get("settlement_attempts", {})

    # R8: Read immutable planned denominator
    planned_cells_count = journal.get("planned_cells_count") or (
        len(journal.get("planned_cells", [])) or (len(decisions) if decisions else 0)
    )
    planned_decisions_count = journal.get("planned_decisions_count", planned_cells_count)
    planned_pairs_count = journal.get(
        "planned_pairs_count", planned_cells_count // 2 if planned_cells_count else 0
    )

    # Decisions accounting
    completed_decisions_A = {
        k[:-2]: v
        for k, v in decisions.items()
        if k.endswith("_A") and v.get("phase") == "completed"
    }
    completed_decisions_B = {
        k[:-2]: v
        for k, v in decisions.items()
        if k.endswith("_B") and v.get("phase") == "completed"
    }
    failed_decisions = [v for v in decisions.values() if v.get("phase") == "failed"]
    attempted_decisions_count = len(decisions)
    unattempted_decisions_count = max(0, planned_decisions_count - attempted_decisions_count)

    # Matched decisions: both arms completed
    matched_decision_keys = sorted(
        set(completed_decisions_A.keys()) & set(completed_decisions_B.keys())
    )

    # Settlements accounting
    settled_A = {
        k[:-2]: v
        for k, v in settlements.items()
        if k.endswith("_A") and v.get("phase") == "completed" and v.get("outcome") is not None
    }
    settled_B = {
        k[:-2]: v
        for k, v in settlements.items()
        if k.endswith("_B") and v.get("phase") == "completed" and v.get("outcome") is not None
    }
    pending_window_settlements = [
        v for v in settlements.values() if v.get("phase") == "pending_window"
    ]
    failed_settlements = [v for v in settlements.values() if v.get("phase") == "failed"]
    attempted_settlements_count = len(settlements)
    unattempted_settlements_count = max(0, planned_cells_count - attempted_settlements_count)

    # Native per-arm summaries from compiled logs (R4, R7)
    arm_A_entries: list[dict[str, Any]] = []
    arm_B_entries: list[dict[str, Any]] = []
    for k, v in decisions.items():
        if v.get("phase") != "completed":
            continue
        c_dir = output_root / "cells" / k / "results" / "trading_memory.log"
        if c_dir.exists():
            ents = TradingMemoryLog({"memory_log_path": str(c_dir)}).load_entries()
            if k.endswith("_A"):
                arm_A_entries.extend(ents)
            elif k.endswith("_B"):
                arm_B_entries.extend(ents)

    log_A_path = output_root / "arm_A_memory.md"
    log_B_path = output_root / "arm_B_memory.md"
    _write_entries_to_log(log_A_path, arm_A_entries)
    _write_entries_to_log(log_B_path, arm_B_entries)

    native_summary_A: BacktestSummary | None = (
        summarize(log_A_path) if (log_A_path.exists() and arm_A_entries) else None
    )
    native_summary_B: BacktestSummary | None = (
        summarize(log_B_path) if (log_B_path.exists() and arm_B_entries) else None
    )
    native_evidence_available = bool(native_summary_A is not None and native_summary_B is not None)

    # R3, R4, R7: Join eligible matched cells with verified decision identity, resolution date, holding period, and exact price outcomes
    candidate_keys = sorted(set(settled_A.keys()) & set(settled_B.keys()))
    eligible_matched_keys: list[str] = []
    pair_eligibility: dict[str, dict[str, Any]] = {}
    unmatched_vintages_count = 0
    orphaned_settlements_count = 0

    for k in candidate_keys:
        elig_rec: dict[str, Any] = {
            "eligible": False,
            "decision_verified": False,
            "native_logs_present": False,
            "provenance_verified": False,
            "timing_eligible": False,
            "diagnostic_eligible": False,
            "diagnostic_provenance_verified": False,
            "diagnostic_timing_eligible": False,
            "non_news_inputs_equal": False,
            "price_vintage_equal": False,
            "benchmark_vintage_equal": False,
            "ineligibility_reasons": [],
        }
        pair_eligibility[k] = elig_rec

        # 1. Verify matching completed decision attempts exist
        if k not in matched_decision_keys:
            orphaned_settlements_count += 1
            elig_rec["ineligibility_reasons"].append("Missing matching completed decision attempt")
            continue

        sA = settled_A[k]
        sB = settled_B[k]
        dA = completed_decisions_A[k]
        dB = completed_decisions_B[k]

        # 2. Verify attempt ID linkage
        if sA.get("decision_attempt_id") != dA.get("attempt_id") or sB.get(
            "decision_attempt_id"
        ) != dB.get("attempt_id"):
            orphaned_settlements_count += 1
            elig_rec["ineligibility_reasons"].append(
                "Settlement decision_attempt_id does not match completed decision attempt_id"
            )
            continue
        elig_rec["decision_verified"] = True

        # 3. Verify native logs on disk exist and have matching resolved entries
        log_A_file = output_root / "cells" / f"{k}_A" / "results" / "trading_memory.log"
        log_B_file = output_root / "cells" / f"{k}_B" / "results" / "trading_memory.log"
        if not log_A_file.exists() or not log_B_file.exists():
            unmatched_vintages_count += 1
            elig_rec["ineligibility_reasons"].append("Native trading_memory.log missing on disk")
            continue
        elig_rec["native_logs_present"] = True

        # Check native log content
        ents_A = TradingMemoryLog({"memory_log_path": str(log_A_file)}).load_entries()
        ents_B = TradingMemoryLog({"memory_log_path": str(log_B_file)}).load_entries()
        ticker_val = dA.get("ticker")
        date_val = dA.get("date")
        m_A = [
            e
            for e in ents_A
            if e.get("ticker") == ticker_val and e.get("date") == date_val and not e.get("pending")
        ]
        m_B = [
            e
            for e in ents_B
            if e.get("ticker") == ticker_val and e.get("date") == date_val and not e.get("pending")
        ]
        if not m_A or not m_B:
            unmatched_vintages_count += 1
            elig_rec["ineligibility_reasons"].append(
                "Native trading_memory.log missing matching resolved decision entry"
            )
            continue

        entry_A = m_A[0]
        entry_B = m_B[0]

        # R3: Strict contradiction check between on-disk native entry and journal record
        if (
            entry_A.get("rating") != sA.get("rating")
            or entry_A.get("resolved") != sA.get("resolution_date")
            or entry_A.get("raw") != sA.get("raw_return")
            or _alpha(entry_A) != sA.get("outcome")
        ):
            unmatched_vintages_count += 1
            elig_rec["ineligibility_reasons"].append(
                "Contradiction between on-disk native memory log and journal record for Arm A"
            )
            continue

        if (
            entry_B.get("rating") != sB.get("rating")
            or entry_B.get("resolved") != sB.get("resolution_date")
            or entry_B.get("raw") != sB.get("raw_return")
            or _alpha(entry_B) != sB.get("outcome")
        ):
            unmatched_vintages_count += 1
            elig_rec["ineligibility_reasons"].append(
                "Contradiction between on-disk native memory log and journal record for Arm B"
            )
            continue

        # R4: Native holding window contradiction check
        planned_h = sA.get("holding_period") or sB.get("holding_period") or 5
        planned_h_str = str(planned_h).replace(" trading days", "").replace("d", "").strip()
        h_str_A = (
            str(entry_A.get("holding") or "").replace(" trading days", "").replace("d", "").strip()
        )
        h_str_B = (
            str(entry_B.get("holding") or "").replace(" trading days", "").replace("d", "").strip()
        )

        if not entry_A.get("holding") or h_str_A != planned_h_str:
            unmatched_vintages_count += 1
            elig_rec["ineligibility_reasons"].append(
                f"Contradiction in Arm A holding period: native log '{entry_A.get('holding')}' does not match planned '{planned_h}'"
            )
            continue

        if not entry_B.get("holding") or h_str_B != planned_h_str:
            unmatched_vintages_count += 1
            elig_rec["ineligibility_reasons"].append(
                f"Contradiction in Arm B holding period: native log '{entry_B.get('holding')}' does not match planned '{planned_h}'"
            )
            continue

        if h_str_A != h_str_B:
            unmatched_vintages_count += 1
            elig_rec["ineligibility_reasons"].append(
                f"Contradiction between arm holding periods in native log: '{entry_A.get('holding')}' vs '{entry_B.get('holding')}'"
            )
            continue

        # 4. Verify resolution dates match and are present
        resA = sA.get("resolution_date")
        resB = sB.get("resolution_date")
        if not resA or not resB or resA != resB:
            unmatched_vintages_count += 1
            elig_rec["ineligibility_reasons"].append(
                f"Mismatched resolution dates: {resA} vs {resB}"
            )
            continue

        # R1: Timing & Provenance eligibility verification via authentic immutable archive validation
        prov_ok, timing_ok, archive_reasons, is_v2 = _validate_archive_evidence(
            output_root,
            ticker_val,
            date_val,
            output_root / "cells" / f"{k}_A",
            output_root / "cells" / f"{k}_B",
            dA,
            dB,
            sA,
            sB,
        )
        elig_rec["diagnostic_provenance_verified"] = prov_ok
        elig_rec["diagnostic_timing_eligible"] = timing_ok
        elig_rec["provenance_verified"] = False
        elig_rec["timing_eligible"] = False
        elig_rec["is_v2"] = is_v2
        if is_v2:
            elig_rec["archive_version"] = "t13-archive/v2"
        if archive_reasons:
            elig_rec["ineligibility_reasons"].extend(archive_reasons)

        # 5. Verify holding period matches
        holdA = sA.get("holding_period")
        holdB = sB.get("holding_period")
        if (
            holdA is None
            or holdB is None
            or holdA != holdB
            or str(holdA).rstrip("d") != planned_h_str
        ):
            unmatched_vintages_count += 1
            elig_rec["ineligibility_reasons"].append(
                f"Mismatched planned holding period: {holdA} vs {holdB}"
            )
            continue

        actA = sA.get("holding_period_actual")
        actB = sB.get("holding_period_actual")
        if (actA is None) != (actB is None):
            unmatched_vintages_count += 1
            elig_rec["ineligibility_reasons"].append(
                f"Omitted actual holding period in one arm: {actA} vs {actB}"
            )
            continue

        if actA is not None:
            if actA != actB or str(actA).rstrip("d") != planned_h_str:
                unmatched_vintages_count += 1
                elig_rec["ineligibility_reasons"].append(
                    f"Mismatched actual holding period: {actA} vs {actB}"
                )
                continue

            if str(actA).rstrip("d") != h_str_A or str(actB).rstrip("d") != h_str_B:
                unmatched_vintages_count += 1
                elig_rec["ineligibility_reasons"].append(
                    "Actual holding period in journal does not match native memory log holding tag"
                )
                continue

        # 6. Verify non-news tool input/output hashes match (R2, R3)
        hA = dA.get("non_news_inputs_hash")
        hB = dB.get("non_news_inputs_hash")
        if not hA or not hB:
            elig_rec["non_news_inputs_equal"] = False
            elig_rec["ineligibility_reasons"].append(
                "Non-news tool observations missing or unrecorded in one or both arms"
            )
        elif hA != hB:
            elig_rec["non_news_inputs_equal"] = False
            elig_rec["ineligibility_reasons"].append(
                "Non-news tool inputs and consumed outputs mismatch between arms"
            )
        else:
            elig_rec["non_news_inputs_equal"] = True

        # 7. Verify benchmark identity matches
        benchA = sA.get("benchmark_name") or sA.get("benchmark")
        benchB = sB.get("benchmark_name") or sB.get("benchmark")
        if benchA != benchB:
            unmatched_vintages_count += 1
            elig_rec["ineligibility_reasons"].append(
                f"Mismatched benchmark identity: {benchA} vs {benchB}"
            )
            continue

        # 8. Exact Raw Return Float Equality (R3, R4)
        raw_float_A = sA.get("raw_return_float")
        raw_float_B = sB.get("raw_return_float")
        if raw_float_A is None or raw_float_B is None:
            unmatched_vintages_count += 1
            elig_rec["price_vintage_equal"] = False
            elig_rec["ineligibility_reasons"].append(
                "Exact raw return float evidence missing (cannot establish price vintage equality)"
            )
            continue
        if abs(raw_float_A - raw_float_B) > 1e-6:
            unmatched_vintages_count += 1
            elig_rec["price_vintage_equal"] = False
            elig_rec["ineligibility_reasons"].append(
                f"Different raw return floats: {raw_float_A} vs {raw_float_B}"
            )
            continue
        elig_rec["price_vintage_equal"] = True

        # 9. Exact Benchmark Return Float Equality (R3, R4)
        bench_float_A = sA.get("benchmark_return_float")
        bench_float_B = sB.get("benchmark_return_float")
        if bench_float_A is None or bench_float_B is None:
            unmatched_vintages_count += 1
            elig_rec["benchmark_vintage_equal"] = False
            elig_rec["ineligibility_reasons"].append(
                "Exact benchmark return float evidence missing (cannot establish benchmark vintage equality)"
            )
            continue
        if abs(bench_float_A - bench_float_B) > 1e-6:
            unmatched_vintages_count += 1
            elig_rec["benchmark_vintage_equal"] = False
            elig_rec["ineligibility_reasons"].append(
                f"Different benchmark return floats: {bench_float_A} vs {bench_float_B}"
            )
            continue
        elig_rec["benchmark_vintage_equal"] = True

        # R4: Strict native evidence gate
        if not native_evidence_available:
            elig_rec["ineligibility_reasons"].append(
                "Native backtest summary unavailable or native logs uncompiled"
            )

        is_pair_strictly_eligible = bool(
            elig_rec["decision_verified"]
            and elig_rec["native_logs_present"]
            and elig_rec["diagnostic_timing_eligible"]
            and elig_rec["diagnostic_provenance_verified"]
            and elig_rec["non_news_inputs_equal"]
            and elig_rec["price_vintage_equal"]
            and elig_rec["benchmark_vintage_equal"]
            and native_evidence_available
        )
        elig_rec["diagnostic_eligible"] = is_pair_strictly_eligible
        elig_rec["eligible"] = False
        if not is_pair_strictly_eligible:
            unmatched_vintages_count += 1
            continue

        eligible_matched_keys.append(k)

    # Paired metrics on eligible matched keys
    # R7: Exclude REVIEW from alphas and directional hits; Hold has no directional hit
    scored_keys: list[str] = []
    unscored_count = 0

    hits_A = 0
    directional_count_A = 0
    hits_B = 0
    directional_count_B = 0

    for k in eligible_matched_keys:
        rA = settled_A[k].get("rating")
        rB = settled_B[k].get("rating")
        aA = settled_A[k]["outcome"]
        aB = settled_B[k]["outcome"]

        # If either rating is REVIEW, the pair is unscored
        if rA == RATING_REVIEW or rB == RATING_REVIEW:
            unscored_count += 1
            continue

        scored_keys.append(k)

        dirA = _DIRECTION.get(rA, 0)
        dirB = _DIRECTION.get(rB, 0)

        if dirA != 0:
            directional_count_A += 1
            if (aA * dirA) > 0:
                hits_A += 1

        if dirB != 0:
            directional_count_B += 1
            if (aB * dirB) > 0:
                hits_B += 1

    alphas_A = [settled_A[k]["outcome"] for k in scored_keys]
    alphas_B = [settled_B[k]["outcome"] for k in scored_keys]

    mean_alpha_A = round(sum(alphas_A) / len(alphas_A), 3) if alphas_A else None
    mean_alpha_B = round(sum(alphas_B) / len(alphas_B), 3) if alphas_B else None
    alpha_diff = (
        round(mean_alpha_B - mean_alpha_A, 3)
        if (mean_alpha_A is not None and mean_alpha_B is not None)
        else None
    )

    hit_rate_A = round(hits_A / directional_count_A, 3) if directional_count_A > 0 else None
    hit_rate_B = round(hits_B / directional_count_B, 3) if directional_count_B > 0 else None
    summary: dict[str, Any] = {
        "manifest_hash": journal.get("manifest_hash"),
        "experiment_id": journal.get("experiment_id"),
        "planned_decisions_count": planned_decisions_count,
        "planned_cells_count": planned_cells_count,
        "planned_pairs_count": planned_pairs_count,
        "attempted_decisions_count": attempted_decisions_count,
        "unattempted_decisions_count": unattempted_decisions_count,
        "completed_decisions_A": len(completed_decisions_A),
        "completed_decisions_B": len(completed_decisions_B),
        "failed_decisions_count": len(failed_decisions),
        "matched_decision_pairs": len(matched_decision_keys),
        "planned_settlements_count": planned_cells_count,
        "attempted_settlements_count": attempted_settlements_count,
        "unattempted_settlements_count": unattempted_settlements_count,
        "failed_settlements_count": len(failed_settlements),
        "pending_window_settlements_count": len(pending_window_settlements),
        "orphaned_settlements_count": orphaned_settlements_count,
        "unmatched_vintages_count": unmatched_vintages_count,
        "settled_pairs_count": 0,
        "scored_pairs_count": 0,
        "unscored_pairs_count": unscored_count,
        "ineligible_pairs_count": planned_pairs_count,
        "strict_empirical_eligible": False,
        "availability_status": "availability_unknown",
        "provenance_verified": False,
        "timing_eligible": False,
        "non_news_inputs_equal": (
            all(pair_eligibility[k]["non_news_inputs_equal"] for k in eligible_matched_keys)
            if eligible_matched_keys
            else False
        ),
        "eligibility_fields": [
            "provenance_verified",
            "timing_eligible",
            "non_news_inputs_equal",
            "price_vintage_equal",
            "benchmark_vintage_equal",
        ],
        "pair_eligibility": pair_eligibility,
        "mean_alpha_A": None,
        "mean_alpha_B": None,
        "alpha_difference": None,
        "directional_hits_A": 0,
        "directional_count_A": 0,
        "directional_hit_rate_A": None,
        "directional_hits_B": 0,
        "directional_count_B": 0,
        "directional_hit_rate_B": None,
        "diagnostic_summary": {
            "diagnostic_scored_pairs_count": len(scored_keys),
            "diagnostic_settled_pairs_count": len(eligible_matched_keys),
            "diagnostic_mean_alpha_A": mean_alpha_A,
            "diagnostic_mean_alpha_B": mean_alpha_B,
            "diagnostic_alpha_difference": alpha_diff,
            "diagnostic_directional_hits_A": hits_A,
            "diagnostic_directional_count_A": directional_count_A,
            "diagnostic_directional_hit_rate_A": hit_rate_A,
            "diagnostic_directional_hits_B": hits_B,
            "diagnostic_directional_count_B": directional_count_B,
            "diagnostic_directional_hit_rate_B": hit_rate_B,
            "diagnostic_status": "NON_EMPIRICAL_DIAGNOSTIC_ONLY",
        },
        "native_summary_A": {
            "resolved": native_summary_A.resolved,
            "pending": native_summary_A.pending,
            "unscored": native_summary_A.unscored,
            "holding": native_summary_A.holding,
        }
        if native_summary_A
        else None,
        "native_summary_B": {
            "resolved": native_summary_B.resolved,
            "pending": native_summary_B.pending,
            "unscored": native_summary_B.unscored,
            "holding": native_summary_B.holding,
        }
        if native_summary_B
        else None,
        "limitations": [
            "Yahoo feed baseline is exploratory publication filtering without contemporaneous receipt storage.",
            "Small sample size cannot establish statistical alpha or causal decision improvement.",
            "Offline execution and synthetic tests do not represent live market outcomes.",
            "Model evaluations do not include slippage, transaction costs, or liquidity constraints.",
            "Unmatched resolution dates, price vintages, and orphaned settlements are strictly excluded from paired alpha.",
        ],
    }

    # Atomically save summary.json
    summary_path = output_root / "summary.json"
    temp_path = output_root / f"summary.json.tmp.{uuid.uuid4().hex}"
    serialized = json.dumps(summary, indent=2, sort_keys=True)
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(serialized)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, summary_path)

    return summary


# ---------------------------------------------------------------------------
# CLI Entrypoint (R8)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for T13 native A/B runner."""
    parser = argparse.ArgumentParser(description="T13 Genuine Native A/B Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate
    validate_p = subparsers.add_parser(
        "validate", help="Validate experiment manifest without side effects"
    )
    validate_p.add_argument("--manifest", required=True, help="Path to manifest JSON file")

    # run
    run_p = subparsers.add_parser("run", help="Run paired decision generation")
    run_p.add_argument("--manifest", required=True, help="Path to manifest JSON file")
    run_p.add_argument("--output", required=True, help="Output directory path")

    # settle
    settle_p = subparsers.add_parser(
        "settle", help="Settle matured decisions and record reflections"
    )
    settle_p.add_argument("--manifest", required=True, help="Path to manifest JSON file")
    settle_p.add_argument("--output", required=True, help="Output directory path")

    # summarize
    sum_p = subparsers.add_parser("summarize", help="Summarize settled experiment metrics")
    sum_p.add_argument("--output", required=True, help="Output directory path")

    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            manifest_path = Path(args.manifest)
            if not manifest_path.exists():
                print(f"Error: Manifest file not found: {manifest_path}", file=sys.stderr)
                return 1
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            validated = validate_manifest(payload)
            print(
                json.dumps(
                    {"status": "valid", "manifest_hash": compute_manifest_hash(validated)}, indent=2
                )
            )
            return 0

        elif args.command == "run":
            manifest_path = Path(args.manifest)
            if not manifest_path.exists():
                print(f"Error: Manifest file not found: {manifest_path}", file=sys.stderr)
                return 1
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            journal = run_experiment(payload, args.output)

            # Truthful CLI status and exit code (R8)
            planned = journal.get("planned_cells_count", 0)
            decisions = journal.get("decision_attempts", {})
            completed = sum(1 for d in decisions.values() if d.get("phase") == "completed")
            failed = sum(1 for d in decisions.values() if d.get("phase") == "failed")

            if completed == planned and planned > 0:
                print(
                    json.dumps(
                        {
                            "status": "run_completed",
                            "manifest_hash": journal["manifest_hash"],
                            "completed_cells": completed,
                            "planned_cells": planned,
                        },
                        indent=2,
                    )
                )
                return 0
            else:
                status_label = "run_failed" if failed > 0 else "run_incomplete"
                print(
                    json.dumps(
                        {
                            "status": status_label,
                            "manifest_hash": journal["manifest_hash"],
                            "completed_cells": completed,
                            "failed_cells": failed,
                            "planned_cells": planned,
                        },
                        indent=2,
                    )
                )
                return 1

        elif args.command == "settle":
            manifest_path = Path(args.manifest)
            if not manifest_path.exists():
                print(f"Error: Manifest file not found: {manifest_path}", file=sys.stderr)
                return 1
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            journal = settle_experiment(payload, args.output)

            # Truthful CLI status and exit code for settle (R8)
            planned_cells = journal.get("planned_cells_count") or (
                len(payload.get("pairs", [])) * 2
            )
            decisions = journal.get("decision_attempts", {})
            completed_decisions = sum(
                1 for d in decisions.values() if d.get("phase") == "completed"
            )
            failed_decisions = sum(1 for d in decisions.values() if d.get("phase") == "failed")
            settlements = journal.get("settlement_attempts", {})
            completed_settlements = sum(
                1 for s in settlements.values() if s.get("phase") == "completed"
            )
            pending_window = sum(
                1 for s in settlements.values() if s.get("phase") == "pending_window"
            )
            failed_settlements = sum(1 for s in settlements.values() if s.get("phase") == "failed")

            is_wholly_successful = (
                planned_cells > 0
                and completed_settlements == planned_cells
                and completed_decisions == planned_cells
                and failed_decisions == 0
                and failed_settlements == 0
                and pending_window == 0
            )

            if is_wholly_successful:
                print(
                    json.dumps(
                        {
                            "status": "settle_completed",
                            "manifest_hash": journal["manifest_hash"],
                            "settled_cells": completed_settlements,
                            "completed_decisions": completed_decisions,
                            "planned_cells": planned_cells,
                        },
                        indent=2,
                    )
                )
                return 0
            else:
                if pending_window > 0:
                    status_label = "settle_pending_window"
                elif failed_decisions > 0 or failed_settlements > 0:
                    status_label = "settle_failed"
                else:
                    status_label = "settle_incomplete"

                print(
                    json.dumps(
                        {
                            "status": status_label,
                            "manifest_hash": journal["manifest_hash"],
                            "settled_cells": completed_settlements,
                            "pending_window_cells": pending_window,
                            "failed_settlements": failed_settlements,
                            "failed_decisions": failed_decisions,
                            "completed_decisions": completed_decisions,
                            "planned_cells": planned_cells,
                        },
                        indent=2,
                    )
                )
                return 1

        elif args.command == "summarize":
            summary = summarize_experiment(args.output)
            print(json.dumps(summary, indent=2))
            return 0

    except (
        OwnershipError,
        JournalPersistenceError,
        JournalVerificationError,
        ManifestMismatchError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error executing %s: %s", args.command, exc)
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
