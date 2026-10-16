# T13 Genuine Native A/B Experiment Runner

The T13 genuine native runner (`experiments/t13_native_ab.py`) provides an authentic, paired A/B evaluation harness comparing default news data (Arm A: `yfinance`) with News Intelligence (Arm B: `news_intelligence`) under identical assets, evaluation dates, model controls, prompts, and schedules.

## 1. Architectural Principles & Reuse

- **Native API Reuse**: Directly invokes `TradingAgentsGraph`, `graph.propagate(...)`, `graph.save_reports(...)`, `graph.settle_pending(...)`, `TradingMemoryLog`, and native `backtest.summarize(...)`.
- **Zero Architectural Drift**: Does not implement a secondary backtester, mock cost estimator, custom portfolio simulator, or alternative database layer.
- **Sequential Cell Isolation**: Each cell `(ticker, date, arm)` executes with isolated working directories (`results/`, `data/`, `cache/`), preventing cross-contamination of native logs, checkpoints, reflections, or learned lessons.
- **Alternating Arm Order**: Alternates execution order across pairs (Pair 1: A then B; Pair 2: B then A) to prevent positional ordering bias.

## 2. Invariant Controls & Exact Native Keys (R1, R3)

The experiment is governed by an immutable JSON manifest validated by `validate_manifest()`:
- **Canonical Dates & Tickers**: Strict `YYYY-MM-DD` round-trip civil dates (rejecting non-canonical unpadded forms like `2026-08-3`), canonical ticker symbols validated via `safe_ticker_component()`, and explicit asset types (`stock` or `crypto`). Tickers containing path traversal characters (`/`, `\`, `..`, leading dot, pure dots) are rejected before directory creation. Future dates and duplicate pairs are rejected.
- **Exact Native Keys**:
  - `quick_think_llm`, `deep_think_llm` (model selection)
  - `tool_vendors`:
    - **Arm A**: `{"get_news": "yfinance", "get_global_news": "yfinance"}`
    - **Arm B**: `{"get_news": "news_intelligence", "get_global_news": "news_intelligence"}`
  - `data_cache_dir` (isolated per cell)
  - `llm_max_retries`
  - `max_recur_limit`
  - `output_language`
  - `google_thinking_level`
  - `benchmark_ticker` and `holding_period_days`
- **Native Portfolio Context (R2)**: Optional portfolio inputs are validated as `tradingagents.portfolio.PortfolioContext`. The runner passes the model instance with `.render()` and `.fingerprint()` methods to `propagate()`, and unpacks the returned `(final_state, signal)` tuple for reporting.
- **Secret & URL Rejection**: Arbitrary endpoint URLs and secrets (`api_key`, `password`, `secret`, `credentials`, `auth_token`) are rejected from the persisted manifest.

## 3. Concurrency Control, Process Contention & Crash Matrix (R5)

- **Owner Lock**: `acquire_experiment_owner()` holds an exclusive OS file lock on `.experiment-owner` beside the canonical output root via `dataflows.files.locked(..., wait=False)` for the entire mutating command.
- **Path Canonicalization**: Resolves output paths with `Path.resolve()` to eliminate path-alias races across processes.
- **Zero Lock Rotation**: The `.experiment-owner` lock base is permanent and never deleted, rotated, or replaced with the journal.
- **Subprocess Contention & Boundary Rejection**: Concurrent child processes competing for the same experiment output directory are synchronized before the synthetic graph boundary. Exactly one process acquires the lock and reaches execution; the second process fails closed with `OwnershipError` (exit code 1) before reading state, modifying files, or constructing graphs.
- **Abrupt Process Termination & Lock Recovery**: If an owner process holding the lock is terminated abruptly (SIGKILL / `proc.kill()`) after durable `started` phase has been persisted to `journal.json`:
  1. The operating system immediately releases the file lock descriptor.
  2. Subsequent runs cleanly reacquire the lock without requiring manual cleanup or stale lock file deletion.
  3. The interrupted `started` cell without on-disk reports is never replayed to paid model execution.
- **Settlement Crash & Reflection Reconciliation (Round 6 R3, Round 7 R3)**: If a crash occurs after native reflection generates on-disk memory log outcomes (`trading_memory.log`) but before `completed` phase is persisted to `journal.json`:
  1. Atomic journal persistence fails closed with `JournalPersistenceError`.
  2. On restart, `settle_experiment` inspects the on-disk memory log, detects the already settled outcome, and reconciles the cell to `completed` with zero duplicate reflection or model calls.
  3. Reconciled native outcomes on disk restore usage **only** if an authentic matching `settlement_audit_{settle_id}.json` exists with strict 5-field identity validation: non-missing `operation: "settlement"`, exact `settlement_id`, exact `cell_key`, exact `decision_attempt_id`, and exact `manifest_hash`. Settlement recovery strictly forbids borrowing or falling back to decision audit traces (`audit_trace.json`). If any field is missing or mismatched, the outcome is reconciled with `has_unknown_usage = True`, `model_calls = 0`, `model_usage = {}`, and `audit_trace_path = None`.
- **Incomplete Settlement Crash Boundaries & Usage Tracking (Round 5 R3, Round 6 R3, Round 7 R3)**:
  1. In-flight settlement records initialize with `phase: "started"` and `has_unknown_usage = True`.
  2. If execution halts or crashes during reflection or before disk outcome persistence, restart enforces zero duplicate model calls and preserves `has_unknown_usage = True`.
  3. When an immature price window is encountered without model calls (`pending_window`), it is verified with `model_calls = 0` and `has_unknown_usage = False`.
  4. Attempt audit traces (`settlement_audit_{settle_id}.json`) explicitly capture `has_unknown_usage`, `operation: "settlement"`, `cell_key`, `decision_attempt_id`, and `manifest_hash` alongside token and tool traces.
  5. The incomplete-outcome recovery branch applies the identical 5-field audit identity check; invalid or mismatched audits never attach unverified usage or audit trace provenance.

## 5. Settlement Memory Invariants, Sweep Halting & Recheck History (R2, R5, R6)

- **Single-Decision Log Invariant (R2)**: Before invoking native settlement, the runner verifies that `trading_memory.log` exists, is non-empty, and contains exactly 1 entry matching the cell's ticker and trade date. Extra or mismatched unjournaled entries fail closed.
- **Reconciliation without Provider Construction (R2)**: Cells whose native log is already settled on disk are reconciled directly without initializing provider credentials or constructing graphs.
- **Whole-Command Quota & Provider Halting (R2, R5)**: The runner inspects all entries in `Settlement.failed` as well as caught exceptions. Any rate limit, quota exhaustion (HTTP 429 / `ResourceExhausted`), or authentication failure halts the entire command loop across all pairs and arms.
- **Immutable Recheck History (R6)**: Immature price window rechecks (`pending_window`) retain full historical records in `journal["settlement_history"]` and `settle_rec["history"]`. Prior attempt UUIDs, timestamps, token usage, and errors are preserved across multiple rechecks, with attempt-specific audit traces (`settlement_audit_{settle_id}.json`).

## 6. Audit Traces, Modern Usage Metadata & Execution Clocks (R2, R3, R7)

- **Chat Model Accounting & Trace Recording (R2)**: `AuditCallbackHandler.on_chat_model_start` increments `model_calls` and `pending_calls`, captures sanitized input prompt traces, and deduplicates using `_active_run_ids` so constructor and invoke callbacks never double-count requests.
- **Parallel Tool Call Mapping by Run ID (R3)**: `_pending_tool_calls_by_run_id` dictionary keyed by callback `run_id` associates tool starts with their corresponding completions and errors, preventing interleaved parallel tool executions (e.g. stock data vs news intelligence) from misattributing outputs or corrupting non-news hashes.
- **Authentic Token Usage Extraction**: `AuditCallbackHandler` inspects modern LangChain `ChatGeneration.message.usage_metadata` (`input_tokens`, `output_tokens`, `total_tokens`) as well as `response.llm_output`.
- **Unknown Usage on Interruption/Error (R7)**: If a model call encounters an error, loses its response, or lacks usage metadata, `has_unknown_usage` is marked `True`. Partial token totals are retained without claiming complete zero usage.
- **Execution Wall Clocks vs Native Entry Evidence (Round 6 R2, Round 7 R2, Round 8 R2, Round 9 R2)**:
  - Timing eligibility is evaluated against auditable entry/availability evidence tied to the actual valid native entry bar.
  - Guessed `20:00:00Z` fallback is completely eliminated: missing entry evidence fails closed (`timing_eligible: False`).
  - Native `get_closes` captures entry timestamps only after applying native `settlement._by_day(res)` to drop invalid non-positive closes, and rejects calendar daily-bar midnight labels (`00:00:00`) as availability clocks.
  - **Comprehensive Entry Evidence Validation (Round 8 R2, Round 9 R2)**: `_validate_entry_timestamp_candidate()` strictly parses candidate ISO strings with explicit timezones. Corrupt or naive strings fail closed immediately (never silently dropped to fall back to valid aliases).
  - **Market-Exchange Timezone Daily-Bar Elimination (Round 9 R2)**: Normalizes candidate timestamps using `_get_market_tz(ticker)` to detect and reject daily-bar midnight labels in the asset's market exchange timezone (`market_dt.hour == 0 and minute == 0 and second == 0`) under any equivalent offset representation (e.g., Tokyo `00:00:00+09:00` and `15:00:00Z`; US equity `00:00:00-04:00` and `04:00:00Z`). UTC midnight labels (`00:00:00Z`) are also rejected.
  - **Offset Invariance & Availability Preservation (Round 9 R2)**: Verification proves that equivalent instant representations (e.g., `20:00:00Z` vs `00:00:00+04:00`) yield identical eligibility decisions. Legitimate availability instants across all hours and offsets (such as `04:00:00Z` for crypto or `05:00:00Z` for US equities in EDT) are preserved and pass timing gates.
  - **Cross-Candidate & Cross-Phase Consistency (Round 8 R2, Round 9 R2)**: Enforces strict timestamp equality among decision candidate aliases (`entry_time`, `entry_close`, `first_close`, `entry_deadline`), settlement entry aliases, and between decision vs settlement phases.
  - Conflicting A/B entry evidence fails closed; neither arm overrides the other.
  - In prospective execution mode (where decision execution date matches the trade date), normal in-run callback execution events between `started_at` and `completed_at` (< entry cutoff) are recognized as valid in-run activity.
  - Lookahead guards strictly reject pairs as ineligible (`timing_eligible: False`) if decisions started, decisions completed, or tool audit events occurred after the native entry cutoff (`completed_at > dt_cutoff` or `trace_timestamp > dt_cutoff`).
  - Retrospective execution (executed on a later date) strictly requires verified point-in-time contemporaneous input clock archives where publication, receipt, and enrichment clocks precede the entry cutoff.
- **Non-News Tool Input Hashing (R3)**: All tool calls to market data, fundamentals, and technical analysis are hashed to verify identical non-treatment inputs between Arm A and Arm B.
- **Secret Scrubbing**: Secrets, tokens, and credentials are automatically scrubbed from prompts, tool inputs, and outputs before persisting audit traces.

## 7. Honest Paired Metrics, Exact Float Equality & Provenance Gating (R1, R3, R4, R8, R9)

- **Native Summary Compilation (R4, R7)**: Compiles per-arm native logs into `arm_A_memory.md` and `arm_B_memory.md`, then evaluates them via `tradingagents.backtest.summarize()`. If either native summary is unavailable or uncompiled, paired scoring is blocked.
- **Strict Matched Pairing & Vintage Equality (R4)**: A pair is eligible for comparative alpha only if:
  1. Both Arm A and Arm B have matching completed decisions and matching completed settlements with linked attempt IDs.
  2. Resolution dates match (`resolution_date_A == resolution_date_B`).
  3. Planned and actual holding periods match without contradiction.
  4. Benchmark identities match (`benchmark_A == benchmark_B`).
  5. Exact raw return floats match (`raw_return_float_A == raw_return_float_B` within 1e-6), rejecting differing price vintages formatted to the same rounded string.
  6. Exact benchmark return floats match (`benchmark_return_float_A == benchmark_return_float_B` within 1e-6), rejecting differing index price bars.
  7. Non-news tool input hashes match between arms (`non_news_inputs_equal`).
  8. Native memory logs exist on disk with matching resolved entries.
- **Holding Window Invariants (R4)**: Compares native memory log holding tags (`entry_A['holding']`, `entry_B['holding']`) against planned holding periods and each other, as well as journal actual/planned holding periods. Contradictions (e.g., `2d` vs `5d`) invalidate pairing while preserving the native diagnostic summary.
- **Authentic Both-Arm Archive & Timing Validation (Round 5 R1, Round 6 R1, Round 7 R1, Round 8 R1, Round 9 R1)**: `_validate_archive_evidence`, `_verify_archive_for_arm`, and `_verify_timing_clock_file` enforce substantive evidence verification independently for BOTH arms:
  - **Fail-Closed Candidate Source Path Isolation**: Arm B candidate paths strictly isolate to Arm B sources (`cell_B_dir / data / provenance_archive.json`, `api_snapshots.json`, `cell_B_dir / data / news_intelligence_archive.json`, `output_root / news_intelligence_archive`, `output_root / provenance_archive.json`), strictly removing `output_root / baseline_archive`.
  - **Arm A (Yahoo Baseline)**: Specifically verified against Yahoo baseline evidence (e.g. `baseline_archive`). Records must declare `arm: "A"` and vendor `yfinance` or `yahoo`.
  - **Arm B (Repo 1 News Intelligence)**: Specifically verified against Repo 1 News Intelligence evidence (e.g. `provenance_archive.json`). Records must declare `arm: "B"`, vendor `news_intelligence` or `repo1`, and **mandatory** receipt timestamp (`received_at` / `receivedAt`).
  - **Strict Native Formatter Derivation & No Arbitrary Record Fields (Round 9 R1)**: Eliminates arbitrary record fields (`response`, `output`, `content`, `summary`, `text`, `title`) as candidate digests. Candidate payloads are derived exclusively from authentic native client formatters (`format_article` for Arm B, `_format_articles` for Arm A) and whole-structure serializations. Moving late responses into individual record fields fails closed.
  - **Tool Request Parameter Binding & Mandatory Run ID Linking (Round 8 R1, Round 9 R1)**: Tool start traces in `audit_trace.json` are bound to completions. When `run_id` is present on the completion, exactly one matching `tool_start` trace with that `run_id` is required (never falling back to sequence). Tool input parameters are strictly validated against target ticker (`normalize_symbol(req_sym) == normalize_symbol(ticker)`), request date window (`req_date <= date_val`), and entry cutoff (`req_cutoff <= dt_cutoff`). Absent, empty, or unparseable tool start inputs fail closed.
  - **List Archive Cell Isolation (Round 9 R1)**: In multi-item list archives, declared hashes and records are filtered strictly by `ticker`, `arm`, and `date_val` to prevent hash leakage across multi-ticker or multi-arm experiments.
  - **Recomputed Payload Digest & Hash Parity (Round 7 R1, Round 9 R1)**: The validator recomputes the SHA-256 digest of retained archive payloads directly from native formatters or canonical structures. Declared hashes, recomputed digests, and observed news tool outputs in `audit_trace.json` must strictly agree. Tampering with retained content while keeping declared hashes fails closed.
  - **Mandatory Identity and Clock Binding**: Archive records must explicitly provide `attempt_id` (matching `decision_attempt_id`), `query` (matching `ticker`), and `cutoff` (`<= entry_cutoff`).
  - **Mandatory Audit Trace Presence**: `audit_trace.json` must be present on disk in each cell's results directory and contain non-empty observed news tool executions.
  - **Canonical Clock Conflict Rejection (Round 10 R1)**: `_resolve_and_validate_clock` resolves and normalizes canonical clock representations (`publishedAt`, `receivedAt`, `enrichedAt`), rejects conflicting snake-case aliases (`published_at`, `received_at`, `enriched_at`), and verifies `<=` entry cutoff. Conflicting earlier wrapper aliases cannot override late canonical fields.
  - **Native Point-in-Time Sanitization & Enrichment Withholding (Round 10 R1)**: Applies existing native `filter_and_sanitize_event(r_norm, cutoff_dt=dt_cutoff)` at the verified entry cutoff before deriving candidate model input payloads. Missing, null, or after-cutoff enrichment availability (`enrichedAt`) withholds derived intelligence facts (`sentiment`, `summary`, `clusterId`, `symbols -> []`). If an observed tool response contains unverified derived facts, candidate digests differ and the pair fails closed.
  - **Vendor Routing Identity vs Article Publisher Disentanglement (Round 10 R2)**: Disentangles treatment/routing vendor validation (`archive.vendor`, `record.vendor`: `news_intelligence`/`repo1` for Arm B, `yfinance`/`yahoo` for Arm A) from canonical article publisher origins (`event.source`, e.g. `REVIEWER_ONLY publisher`, `Reuters`, `Bloomberg`). Canonical article publishers are preserved intact through native validation and formatting without vendor rewrites.
  - **Explicit Canonical Null Enrichment Preservation & Mandatory Fail-Closed (Round 11 R1)**: `_resolve_and_validate_clock` strictly preserves explicit canonical null enrichment timestamps (`enrichedAt: null`). Record-level explicit nulls are never backfilled from top-level wrapper payloads or fallback aliases. Null vs non-null alias conflicts reject immediately. For mandatory clocks, an explicit null fails closed. For optional clocks (`enrichedAt`), an explicit null is preserved and triggers native point-in-time sanitization (withholding derived intelligence facts: sentiment, summary, clusterId, symbols). Wrapper payload keys are copied via `_copy_top_level_wrapper_keys` to prevent alias contamination or manufactured conflicts.
  - **Native Global Market News Candidate Derivation & Multi-Tool Runs (Round 11 R2)**:
    - Extended provenance validation to support authentic native `get_global_news` tools for Arm A (`get_global_news_yfinance`) and Arm B (`NewsIntelligenceClient.get_global_news`).
    - Audit trace inspection detects observed global news tool completions and bound tool request parameters (`as_of_date`, `look_back_days`, `limit`).
    - Derives authentic `## Global Market News, from {start_date} to {as_of_date}:\n\n...` candidate digests matching native vendor outputs.
    - Multi-tool runs containing both asset-specific news (`get_news`) and global macro news (`get_global_news`) cleanly separate asset vs macro candidate sets, applying date-window filtering and authentic native formatters to verify each tool execution independently against archive evidence.
  - **Optional Null Parameter Resolution in Global News (Round 12 R1)**:
    - Resolves absent and explicit null parameters (`look_back_days=None`, `limit=None`) through the exact effective configuration defaults (`global_news_lookback_days: 7`, `global_news_article_limit: 10`) defined by `get_config()`, mirroring native producer semantics (`NewsIntelligenceClient.get_global_news` and `get_global_news_yfinance`).
    - Preserves explicitly supplied numeric values (including `0`), avoiding truthiness fallbacks that could silently alter valid integer zero.
    - Accurately validates tool request parameters before candidate reconstruction, rejecting non-integer inputs or negative values with explicit descriptive errors instead of treating reconstruction exceptions as archive tampering.
    - Produces byte-identical candidate payloads across omitted options, `(None, None)`, `(None, 10)`, `(7, None)`, and `(7, 10)` parameter variants, ensuring legitimate native tool invocations qualify reliably under strict provenance and hash verification.
  - **Bi-Temporal Clocks**: Publication, receipt, and enrichment clocks must be timezone-aware and `<= entry_cutoff` (`published_at <= dt_cutoff`, `received_at <= dt_cutoff`, `enriched_at <= dt_cutoff`).
  - **Nested Event Bundle Support**: Archive lists containing per-ticker bundles unpack nested events while inheriting parent ticker, date, arm, and source metadata.
- **Provenance & Timing Eligibility Fields (R1, R3)**: `summary.json` includes explicit audit fields:
  - `"provenance_verified"`: Boolean indicating verified immutable baseline/API response archives and auditable contemporaneous source evidence for both arms.
  - `"timing_eligible"`: Boolean indicating verified contemporaneous decision clocks (`started_at <= trade_date`) or explicit point-in-time input clock archives.
  - `"non_news_inputs_equal"`: Boolean indicating identical non-treatment tool identities, input arguments, and consumed responses between arms.
  - `"pair_eligibility"`: Per-pair breakdown of eligibility criteria and ineligibility reasons.
  - `"eligibility_fields"`: List of active eligibility verification gates.
  - `"ineligible_pairs_count"`: Number of planned pairs excluded from paired comparative alpha.
- **Rating Semantics**: `REVIEW` ratings are excluded from alpha averages and directional hit rates, recorded under `unscored_pairs_count`. `Hold` ratings have zero directional claim (`direction = 0`) and do not produce false directional hits.
- **Truthful Status & Denominator in Settle CLI (R8)**:
  - Derives command status from the immutable planned grid denominator (`planned_cells_count = len(pairs) * 2`).
  - `main(["settle", ...])` returns exit code 0 only when all planned cells are completed and settled with zero failures or pending windows.
  - Returns exit code 1 with descriptive status (`settle_failed`, `settle_pending_window`, `settle_incomplete`) on partial or immature grids, preserving the full planned denominator.

## 8. Testing Safeguards & Network Refusal (R6)

- **Socket-Level Network Refusal**: `tests/test_t13_native_ab.py` installs an autouse fixture monkeypatching `socket.socket.connect` and `requests.sessions.Session.request` to raise `NetworkAccessAttemptedError(BaseException)`.
- **Immunity to Broad Exception Handlers**: Inheriting from `BaseException` ensures unauthorized external network requests fail the test immediately without being suppressed by application try/except blocks.
- **Double Retention Across Restarts**: Synthetic graph, provider, and reflection doubles are retained across process restart scenarios, guaranteeing zero live provider calls or credential escapes.

## 9. CLI Command Reference

```bash
# 1. Validate manifest without graph construction or network calls:
uv run python -m experiments.t13_native_ab validate --manifest <path/to/manifest.json>

# 2. Execute paired decision generation under exclusive lock:
uv run python -m experiments.t13_native_ab run --manifest <path/to/manifest.json> --output <path/to/output_dir>

# 3. Settle matured holding windows and record reflections:
uv run python -m experiments.t13_native_ab settle --manifest <path/to/manifest.json> --output <path/to/output_dir>

# 4. Summarize matched pairs and performance metrics (offline/no network):
uv run python -m experiments.t13_native_ab summarize --output <path/to/output_dir>
```
