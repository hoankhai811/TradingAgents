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

## 10. Prospective Evidence Capture Protocol (Revision 6.0.0)

The prospective observation and diagnostic capture system (`ProspectiveCaptureContext`, `_verify_v2_archive`, `validate_evidence_archive`) provides verifiable, tamper-evident recording of live API interactions during decision generation without retrospective backfilling or external truth claims.

### Two-Tier Storage Layout
1. **Discrete Per-Call Raw Snapshots**:
   - Location: `cells/{cell_key}/data/responses/{seq:03d}_{tool}_{call_id}.json`
   - Content: Exact, unmodified vendor API response payload (raw JSON list/dict as returned across the transport boundary).
   - Atomic Persistence: Writes to temporary file, flushes file buffer, calls `os.fsync(fileno)`, renames via `os.replace`, and verifies the written snapshot digest matches the in-memory payload SHA-256 hash before completing the call.
2. **Cumulative Archive Envelopes (`t13-archive/v2`)**:
   - Location: `cells/{cell_key}/data/baseline_archive.json` (Arm A) and `provenance_archive.json` (Arm B).
   - Schema:
     - `version`: Strictly `"t13-archive/v2"`
     - `attempt_id`: Monotonically bound to the decision attempt UUID
     - `ticker`: Target asset symbol
     - `arm`: `"A"` or `"B"`
     - `date`: Execution civil trade date
     - `vendor`: `"yfinance"` (Arm A) or `"news_intelligence"` (Arm B)
     - `created_at`: UTC ISO timestamp
     - `calls`: Chronological array of discrete tool execution records, capturing:
       - `call_seq`: Monotonic integer sequence (1, 2, ...)
       - `call_id`: Unique UUID4 string
       - `tool`: Invoked tool name (`get_news` or `get_global_news`)
       - `run_id`: LangChain callback run identifier
       - `query`: Request query target
       - `received_at`: Local machine receipt timestamp in UTC ISO format
       - `params`: Exact dictionary of bound request arguments
       - `snapshot_path`: Relative path to discrete snapshot file
       - `raw_payload_hash`: SHA-256 digest of raw payload
       - `records`: List of normalized articles (Arm A) or sanitized events (Arm B)
       - `output_hash`: SHA-256 digest of the producer formatted text output
     - `articles` / `events`: Cumulative collection of normalized/sanitized records.

### Concurrency & Process Safety
- **In-Process Mutex**: An internal `threading.RLock()` in `ProspectiveCaptureContext` guards sequence allocation, raw snapshot write, envelope update, fsync, and read-back verification, ensuring thread-safe execution under parallel LangChain agents and tool calls.
- **Process Owner Lock**: Mutating commands (`run`, `settle`) continue to acquire the exclusive `.experiment-owner` file lock beside the output directory, preventing concurrent processes from interleaving operations on the same cell directory.

### Fail-Closed Storage Failures
- Any snapshot or envelope write, flush, fsync, or read-back verification failure immediately raises `EvidenceStorageError`.
- `run_experiment` catches `EvidenceStorageError`, marks the cell `phase: "failed"` with `status: "FAILED_INCOMPLETE"`, updates `journal.json` with durable sync, and aborts the cell without automated paid replay.

### Unbroken 5-Step Derivation Chain (`_verify_v2_archive`)
Archive envelopes declaring `version: "t13-archive/v2"` are validated against observed tool outputs in `audit_trace.json` through a strict 5-step derivation chain:
1. **Bounded Snapshot Read & SHA-256 Verification**: Reads each `snapshot_path` from disk, computes its SHA-256 digest, and enforces byte-level equality with `raw_payload_hash`.
2. **Native Normalization & Point-in-Time Sanitization**:
   - **Arm A**: Applies native `yahoo_news._extract_article_data`, filters within window via `date_window.in_window`, and truncates to configured limits.
   - **Arm B**: Validates canonical events via `validate_canonical_event` and applies point-in-time filtering via `ni_client.filter_and_sanitize_event` at the verified entry cutoff.
3. **Arm-Specific Field-by-Field Record Equality**:
   - **Arm A**: Compares `title`, `summary`, `publisher`, `link`, and `pub_date` between stored records and derived snapshot records without fabricating artificial fields.
   - **Arm B**: Compares canonical fields `id`, `publishedAt`, `receivedAt`, `enrichedAt`, and `symbols`.
4. **Complete Native Producer Output Reconstruction**:
   - Reconstructs exact producer strings including headers (`## {ticker} News, from {start} to {end}:`), window formatting, limit truncation, empty window sentinels (`No news found...`), and coverage gap warnings (`<Yahoo Finance news unavailable...>`).
5. **Exact Hash and Text Comparison**:
   - Enforces exact match between candidate output hash and observed `output_hash` in `audit_trace.json`.
   - Binds tool completions to corresponding tool starts using `run_id`.
   - Fails closed immediately on any discrepancy.

### Version Fail-Closed Dispatch Policy
Envelopes declaring `version: "t13-archive/v2"` are routed exclusively to `_verify_v2_archive`. If any step fails, verification returns `(False, reason)` immediately. The validator strictly forbids silent fallback to legacy candidate matching or heuristic searches.

### Summary Contract & Diagnostic Accounting
In accordance with the unresolved first-close availability status of historical and prospective feeds:
- `strict_empirical_eligible` is strictly `False`.
- `availability_status` is strictly `"availability_unknown"`.
- `scored_pairs_count` is strictly `0`.
- `settled_pairs_count` is strictly `0`.
- `provenance_verified` is strictly `False`.
- `timing_eligible` is strictly `False`.
- Simulated alpha metrics are strictly isolated within the `diagnostic_summary` object with `diagnostic_status = "NON_EMPIRICAL_DIAGNOSTIC_ONLY"`, ensuring zero false empirical claims in public summaries.

### Round 2 Hardened Verification & Concurrency Protocol (Findings R1 - R7)
1. **Unconditional Diagnostic Public Contract (R1)**:
   - The public summary fields (`scored_pairs_count = 0`, `settled_pairs_count = 0`, `strict_empirical_eligible = False`, `availability_status = "availability_unknown"`, `provenance_verified = False`, `timing_eligible = False`, `ineligible_pairs_count = planned_pairs_count`, `mean_alpha_* = None`, `directional_hits_* = 0`, `directional_hit_rate_* = None`) are unconditionally enforced across Version 2, legacy, unknown version, absent archive, malformed archive, and recovered states.
   - Per-pair public flags (`provenance_verified`, `timing_eligible`, `eligible`) remain strictly `False`, while internal verification metrics are cleanly routed to `diagnostic_summary` and diagnostic flags (`diagnostic_provenance_verified`, `diagnostic_timing_eligible`, `diagnostic_eligible`).
2. **Context-Wide Capture Hooks with Thread-Local Isolation (R2)**:
   - Capture hooks on Yahoo and News Intelligence are installed once per `ProspectiveCaptureContext.__enter__` across all worker threads.
   - Pending calls and collected responses use thread-local state (`threading.local()`), preventing cross-contamination or article bleeding across parallel tool invocations.
   - `__exit__` cleanly restores original methods across all error and exit paths.
3. **1-to-1 Cardinality & Run ID Bijection (R3)**:
   - Enforces exact cardinality: the number of archive calls must equal the number of observed news tool completions in `audit_trace.json` (`len(calls) == len(news_ends)`).
   - Binds each call to a tool execution by exact `run_id` without positional fallback. Duplicate `run_id` bindings, unarchived tool ends, or missing tool starts fail closed immediately.
   - Validates that requested tickers, query parameters, declared output hashes, and derived candidate hashes match observed trace events.
4. **Immediate Post-Return Receipt Clock & Entry Cutoff Enforcement (R4)**:
   - `received_at` is timestamped immediately after the vendor HTTP request returns.
   - If wall clock advances past the entry cutoff during transport, `received_at > dt_cutoff` triggers immediate verification failure.
   - Events with publication, receipt, or enrichment timestamps after the cutoff fail point-in-time sanitization.
5. **Producer Parameter Retention & Yahoo Search Sentinel (R5)**:
   - Retains original producer arguments (`producer_params`) alongside transport HTTP parameters, guaranteeing that non-default query windows round-trip cleanly without default fallback.
   - Reconstructs authentic native sentinels when Yahoo feeds are empty and search fallback is activated (`feed_source == "search"`).
6. **Snapshot Path Containment & Resource Bounds (R6)**:
   - Snapshot candidate search enforces strict containment within the cell directory (`snap_path.is_relative_to(cell_dir.resolve())`).
   - Rejects absolute paths, path traversal (`..`), symlinks, and non-regular files.
   - Enforces collection and snapshot limits: `MAX_SNAPSHOT_BYTES = 10 * 1024 * 1024`, `MAX_ARCHIVE_CALLS = 100`, `MAX_CALL_RECORDS = 500`.
7. **Full Envelope Read-Back Equality Verification (R7)**:
   - After atomic replace, the written archive envelope is fully parsed and checked for deep object equality (`read_obj == envelope`).
   - Any corruption or silent mutation in nested fields raises `EvidenceStorageError` and halts the cell.

### Round 3 Strict Request Binding & Defensive Bounds Protocol (Findings HIGH1, MED2, MED3, MED4)
1. **Full Effective Request Argument Binding & Unique Tool Starts (HIGH1)**:
   - Enforces exact 1-to-1 bijection for both starts and ends: `len(calls) == len(news_starts)` and `len(calls) == len(news_ends)`.
   - Rejects duplicate `tool_start` traces with the same `run_id`.
   - Requires `tool_start` trace input to be valid JSON dictionary; rejects null, empty, malformed, or non-dictionary types (e.g. lists/scalars).
   - Binds and validates all effective request parameters (`ticker`, `start_date`, `end_date`, `as_of_date`, `look_back_days`, `limit`) between observed tool start inputs and captured call/producer parameters, normalizing legitimate defaults.
2. **Explicit Zero Limit Preservation (MED2)**:
   - Resolves effective limits across Arm A and Arm B using explicit `is not None` checks instead of truthy `or` chaining (`val is not None`).
   - Distinguishes explicit `limit=0` from omitted/None limits, reproducing exact native sentinels (Yahoo unavailable sentinel, News Intelligence header-only) without fallback substitution.
3. **Fail-Closed Shape Validation for Nested Archive Fields (MED3)**:
   - Validates nested shapes before dereferencing: `prod_p` dict, `params` dict, `records` list of dicts with scalar timestamps within cutoff.
   - Validates snapshot payloads: Arm A must be a list of dicts, Arm B must be a dict with `events` list of dicts.
   - Fails closed locally returning `(False, reason)` without raising uncaught exceptions, preventing invalid envelopes (`records: [null]`, scalar records, malformed parameters) from crashing `summarize_experiment`.
4. **Pre-Allocation Envelope Read Bounds (MED4)**:
   - Defines and documents `MAX_ENVELOPE_BYTES = 4 * 1024 * 1024` (4 MiB) for outer archive envelopes.
   - Implements `_read_bounded_json_envelope(fpath: Path) -> tuple[Any | None, str | None]` using bounded reads before `json.loads` or memory allocation.
   - Applied at all archive input boundaries (`_verify_archive_for_arm`, `_verify_timing_clock_file`, and version scanning in `_evaluate_arm_evidence`), preventing memory exhaustion from oversized or padded envelopes while keeping the 10 MiB snapshot bound separate.

### Round 4 Native Callback Extraction, Effective Binding & Bounded Read Protocol (Findings HIGH1, HIGH2, MED3)
1. **LangGraph ToolNode Native Message Extraction & Injected State Binding (HIGH1)**:
   - ToolNode passes `ToolMessage` instances into `AuditCallbackHandler.on_tool_end`. The handler extracts `output.content` when available rather than stringifying the container `ToolMessage` (which contains object metadata, tool call IDs, and names), guaranteeing that the tool output string and its SHA-256 hash match the authentic consumed tool return.
   - Parses both strict JSON strings and Python mapping string literals (`ast.literal_eval`) emitted by LangGraph tool execution, rejecting scalars, lists, empty strings, and malformed structures without permissive regex fallback.
   - Binds model-omitted tickers from trusted cell state (`company_of_interest`) because native LangGraph tools inject the symbol via `InjectedState("company_of_interest")`, while verifying that if a model explicitly supplies a ticker, it matches the cell ticker.
   - Evaluates requested date windows through native `as_of_window` and `as_of` clamping before comparing against producer parameters, ensuring that forward dates clamped to the execution `trade_date` are correctly verified rather than falsely rejected.
2. **Mandatory Model Arguments & Cross-Transport Metadata Consistency (HIGH2)**:
   - Enforces visible mandatory arguments: `start_date` and `end_date` are strictly required for `get_news`; `curr_date` (or `as_of_date`/`date`/`asOf`) is strictly required for `get_global_news`. Omission triggers immediate fail-closed verification failure.
   - Enforces cross-transport metadata consistency across all parameters: observed model arguments, producer parameters, and transport parameters (`symbol`, `ticker`, `asOf`, `from`, `to`, `start_date`, `end_date`, `limit`) must agree. Any contradiction (such as transport `symbol` mutating the requested asset or transport `to` extending the date range) triggers verification failure.
3. **Guaranteed Bounded Allocation via MAX + 1 Stream Reading (MED3)**:
   - Defines `MAX_ENVELOPE_BYTES = 4 * 1024 * 1024` (4 MiB) and `MAX_SNAPSHOT_BYTES = 10 * 1024 * 1024` (10 MiB).
   - Replaces naive `read_bytes()` after `stat()` with `_read_bounded_bytes(fpath, max_bytes) -> tuple[bytes | None, str | None]`, which reads at most `max_bytes + 1` bytes via stream `f.read(max_bytes + 1)` and checks `len(data) <= max_bytes` before decoding or parsing.
   - Binds memory allocation even in the presence of race growth or file replacement occurring after the initial `stat()` check, applied across outer JSON envelopes, raw snapshot validation, and read-back equality checks.

### Round 5 Transport Clock Exact UTC Instant Verification Protocol (Finding 1)
1. **Full Expected Transport Instant Derivation via Native `to_iso_utc` Contract**:
   - Derives full expected transport instants using the native News Intelligence `to_iso_utc` contract:
     - Start-of-day for `from` (`YYYY-MM-DDT00:00:00Z`).
     - End-of-day for `to` and `asOf` (`YYYY-MM-DDT23:59:59.999999Z` inclusive microsecond end).
2. **Exact Instant Comparison & Same-Date Mutation Rejection**:
   - Eliminates the previous permissive same-calendar-date fallback (`str(value)[:10] != eff_date`).
   - Parses transport timestamps via `_parse_utc_iso` into canonical `datetime.astimezone(UTC)` objects and compares complete UTC instants against expected values.
   - Fails closed immediately if any transport timestamp is missing, malformed (e.g. `2026-08-03-not-a-timestamp`), timezone-unaware, or mutated to a different time on the same date (e.g. `23:59:59.999999Z` changed to `10:00:00Z`, `00:00:00Z` changed to `23:00:00Z`).
   - Supports legitimate UTC-equivalent timestamp representations (e.g., `+00:00` offset notation matching `Z`) while preserving civil-date comparisons for Arm A (Yahoo) and producer date arguments.

### Round 6 Mandatory Arm B Transport Clocks & Deletion Rejection Protocol (Finding 1)
1. **Unconditional Requirement for Native Transport Clocks**:
   - For all Arm B news calls (both asset-level `get_news` and macro `get_global_news`), all three native transport clock keys (`asOf`, `from`, `to`) are strictly mandatory in `params`.
   - Before instant parsing or comparison, `_verify_v2_archive` unconditionally checks for the presence and non-nullity of each key. If any of `asOf`, `from`, or `to` is omitted, deleted, or `None`, verification fails closed immediately with:
     `"Archive v2 verification failure: transport missing required parameter '<field>'"`
   - Eliminates the bypass vulnerability where omitting or deleting a transport clock parameter from `params` evaded instant comparison entirely.
