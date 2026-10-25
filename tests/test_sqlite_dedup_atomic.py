"""SQLiteSessionStore.record_processed_message() is atomic (F-c, stage 1).

The duplicate check and the write are a single UPSERT, so concurrent callers (worker
threads, or several store instances on one file) can never both accept the same id.
Before, the SELECT ran in autocommit mode before the write: two threads accepted the
same wamid in 9-20% of rounds.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from chatflow_agent.core.memory import SQLiteSessionStore

ROUNDS = 150


@pytest.fixture(params=["file", "memory"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    s = SQLiteSessionStore(tmp_path / "dedup.db" if request.param == "file" else ":memory:")
    yield s
    s.close()


def _age(db: Path, message_id: str, hours: float) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE chatflow_processed_messages SET processed_at = datetime('now', ?) WHERE message_id = ?",
        (f"-{hours} hours", message_id),
    )
    conn.commit()
    conn.close()


def _ids(db: Path) -> set[str]:
    conn = sqlite3.connect(db)
    rows = {r[0] for r in conn.execute("SELECT message_id FROM chatflow_processed_messages")}
    conn.close()
    return rows


def test_new_then_duplicate(store: Any) -> None:
    assert store.record_processed_message("wamid.1") is True
    assert store.record_processed_message("wamid.1") is False
    assert store.is_message_processed("wamid.1")
    assert store.record_processed_message("wamid.2") is True


def test_expired_record_is_accepted_again_and_others_are_purged(tmp_path: Path) -> None:
    db = tmp_path / "dedup.db"
    store = SQLiteSessionStore(db, dedup_ttl_hours=24)
    assert store.record_processed_message("old") is True
    assert store.record_processed_message("other-old") is True
    _age(db, "old", 25)
    _age(db, "other-old", 25)
    assert not store.is_message_processed("old")
    assert store.record_processed_message("old") is True  # renewed
    assert store.is_message_processed("old")
    assert _ids(db) == {"old"}  # the other expired record was purged


def test_record_within_ttl_is_still_a_duplicate(tmp_path: Path) -> None:
    db = tmp_path / "dedup.db"
    store = SQLiteSessionStore(db, dedup_ttl_hours=24)
    assert store.record_processed_message("recent") is True
    _age(db, "recent", 23)
    assert store.record_processed_message("recent") is False


def test_duplicate_does_not_purge(tmp_path: Path) -> None:
    """As before: only a newly recorded id triggers the purge of expired records."""
    db = tmp_path / "dedup.db"
    store = SQLiteSessionStore(db, dedup_ttl_hours=24)
    store.record_processed_message("kept")
    store.record_processed_message("expired")
    _age(db, "expired", 30)
    assert store.record_processed_message("kept") is False
    assert _ids(db) == {"kept", "expired"}


def _record_after_barrier(
    record: Any,
    message_id: str,
    barrier: threading.Barrier,
    lock: threading.Lock,
    results: list[bool],
    errors: list[str],
) -> None:
    barrier.wait()
    try:
        ok = record(message_id)
    except Exception as err:  # noqa: BLE001
        with lock:
            errors.append(repr(err))
        return
    with lock:
        results.append(ok)


def _race(callers: list[Any], rounds: int) -> tuple[int, list[str]]:
    """Each round, every caller records the same new id at the same instant."""
    wrong = 0
    errors: list[str] = []
    lock = threading.Lock()
    for i in range(rounds):
        barrier = threading.Barrier(len(callers))
        results: list[bool] = []
        threads = [
            threading.Thread(
                target=_record_after_barrier,
                args=(c, f"wamid.{i}", barrier, lock, results, errors),
            )
            for c in callers
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if results.count(True) != 1:
            wrong += 1
    return wrong, errors


def test_concurrent_threads_accept_an_id_exactly_once(store: Any) -> None:
    wrong, errors = _race([store.record_processed_message] * 2, ROUNDS)
    assert errors == []
    assert wrong == 0


def test_four_threads_accept_an_id_exactly_once(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "dedup.db")
    wrong, errors = _race([store.record_processed_message] * 4, ROUNDS // 3)
    assert errors == []
    assert wrong == 0


def test_two_store_instances_on_one_file_accept_an_id_exactly_once(tmp_path: Path) -> None:
    a = SQLiteSessionStore(tmp_path / "dedup.db")
    b = SQLiteSessionStore(tmp_path / "dedup.db")
    wrong, errors = _race([a.record_processed_message, b.record_processed_message], ROUNDS)
    assert errors == []
    assert wrong == 0
