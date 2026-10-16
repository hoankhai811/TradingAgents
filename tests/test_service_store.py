"""SQLite transitions guard durable, bounded paid work."""

from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

REQUEST = {
    "ticker": "NVDA",
    "tradeDate": "2026-01-05",
    "assetType": "stock",
    "selectedAnalysts": ["market"],
    "researchDepth": "quick",
}


def test_create_claim_finish_survive_reopen(tmp_path):
    from tradingagents.service.store import RunStore

    path = tmp_path / "runs.sqlite3"
    store = RunStore(path)
    run = store.create("analysis", REQUEST, 2)
    UUID(run["runId"])
    claimed = store.claim_next()
    assert claimed["id"] == run["runId"] and claimed["claim_token"]
    assert store.finish(
        claimed["id"], claimed["claim_token"], "completed", {"decision": "ok"}, None
    )
    reopened = RunStore(path)
    assert reopened.get(run["runId"])["decision"] == "ok"
    assert reopened.get(run["runId"])["status"] == "completed"


def test_restart_marks_running_failed_but_keeps_unstarted_pending(tmp_path):
    from tradingagents.service.store import RunStore

    store = RunStore(tmp_path / "runs.db")
    first = store.create("analysis", REQUEST, 2)
    second = store.create("analysis", REQUEST, 2)
    claim = store.claim_next()
    store.recover_interrupted()
    assert store.get(first["runId"])["status"] == "failed"
    assert store.get(first["runId"])["error"] == "RUN_INTERRUPTED"
    assert store.get(second["runId"])["status"] == "pending"
    assert not store.finish(claim["id"], claim["claim_token"], "completed", {}, None)
    assert store.claim_next()["id"] == second["runId"]


def test_concurrent_acceptance_never_exceeds_bound(tmp_path):
    from tradingagents.service.store import QueueFull, RunStore

    store = RunStore(tmp_path / "runs.db")

    def accept(_):
        try:
            return store.create("analysis", REQUEST, 3)
        except QueueFull:
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        accepted = list(pool.map(accept, range(20)))
    assert len([job for job in accepted if job]) == 3


def test_ids_are_generated_and_untrusted_ids_never_select_paths(tmp_path):
    from tradingagents.service.store import RunStore

    store = RunStore(tmp_path / "runs.db")
    run = store.create("analysis", REQUEST | {"runId": "../../escape"}, 2)
    UUID(run["runId"])
    assert store.get("../../escape") is None
    assert not (tmp_path / "escape").exists()


def test_progress_and_terminal_race_keep_verified_counts(tmp_path):
    from tradingagents.service.store import RunStore

    store = RunStore(tmp_path / "runs.db")
    req = {
        "tickers": ["NVDA"],
        "startDate": "2026-01-05",
        "endDate": "2026-01-12",
        "everyNDays": 7,
        "assetType": "stock",
        "selectedAnalysts": ["news"],
    }
    run = store.create("backtest", req, 2)
    claim = store.claim_next()
    assert store.update_progress(claim["id"], claim["claim_token"], 1)
    progress = RunStore(tmp_path / "runs.db").get(run["backtestId"])
    assert (progress["cellsAttempted"], progress["cellsRun"]) == (1, 0)
    assert store.terminate(claim["id"], claim["claim_token"], "RUN_TIMED_OUT")
    assert not store.update_progress(claim["id"], claim["claim_token"], 2)
    assert not store.finish(claim["id"], claim["claim_token"], "completed", {"cellsRun": 2}, None)
    assert store.get(claim["id"])["error"] == "RUN_TIMED_OUT"
