"""Thread-safety tests for the in-memory SessionStore deduplication registry."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Any

import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.memory import SessionStore


class _TrackingLock:
    """Lock test double that records ownership by the current thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self.acquisitions = 0

    def __enter__(self) -> None:
        self._lock.acquire()
        self._local.held = True
        self.acquisitions += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._local.held = False
        self._lock.release()

    def assert_held(self) -> None:
        assert getattr(self._local, "held", False), "dedup state accessed without its lock"


class _GuardedDict(dict[str, float]):
    """Dictionary that rejects every relevant operation outside the dedup lock."""

    def __init__(self, lock: _TrackingLock, values: dict[str, float] | None = None) -> None:
        super().__init__(values or {})
        self._lock = lock

    def get(self, key: str, default: float | None = None) -> float | None:
        self._lock.assert_held()
        return super().get(key, default)

    def __getitem__(self, key: str) -> float:
        self._lock.assert_held()
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: float) -> None:
        self._lock.assert_held()
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._lock.assert_held()
        super().__delitem__(key)

    def __iter__(self) -> Iterator[str]:
        self._lock.assert_held()
        return super().__iter__()

    def __len__(self) -> int:
        self._lock.assert_held()
        return super().__len__()

    def items(self) -> Any:
        self._lock.assert_held()
        return super().items()


class _GuardedPurgeStore(SessionStore):
    """SessionStore that also verifies _last_purge access is locked."""

    _purge_lock_probe: _TrackingLock | None = None

    def __getattribute__(self, name: str) -> Any:
        if name == "_last_purge":
            probe = object.__getattribute__(self, "_purge_lock_probe")
            if probe is not None:
                probe.assert_held()
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_last_purge":
            probe = getattr(self, "_purge_lock_probe", None)
            if probe is not None:
                probe.assert_held()
        super().__setattr__(name, value)


def _instrument(store: _GuardedPurgeStore, values: dict[str, float]) -> _TrackingLock:
    lock = _TrackingLock()
    store._dedup_lock = lock  # type: ignore[assignment]
    store._processed_ids = _GuardedDict(lock, values)
    store._purge_lock_probe = lock
    return lock


def test_all_dedup_state_accesses_share_one_lock() -> None:
    now = time.monotonic()
    store = _GuardedPurgeStore(dedup_ttl_hours=1)
    lock = _instrument(store, {"fresh": now, "stale": now - 7200})

    assert store.is_message_processed("missing") is False
    assert store.is_message_processed("fresh") is True
    assert store.is_message_processed("stale") is False
    assert store.record_processed_message("fresh") is False
    assert store.record_processed_message("new") is True
    assert lock.acquisitions == 5


def test_purge_and_capacity_enforcement_run_under_the_dedup_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10_000.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    store = _GuardedPurgeStore(dedup_ttl_hours=1)
    lock = _instrument(store, {"fresh": now, "stale": now - 7200})
    object.__setattr__(store, "_last_purge", now - store._DEDUP_PURGE_INTERVAL)

    assert store.record_processed_message("after-purge") is True
    assert "stale" not in store._processed_ids

    store._DEDUP_MAX_ENTRIES = 2
    store._processed_ids = _GuardedDict(lock, {"oldest": now - 2, "newer": now - 1})
    object.__setattr__(store, "_last_purge", now)

    assert store.record_processed_message("at-capacity") is True
    with lock:
        assert "oldest" not in store._processed_ids
        assert set(store._processed_ids) == {"newer", "at-capacity"}


def _record_after_barrier(store: SessionStore, barrier: threading.Barrier) -> bool:
    barrier.wait(timeout=5)
    return store.record_processed_message("same-message")


def test_concurrent_threads_accept_the_same_id_exactly_once() -> None:
    workers = 16
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _ in range(100):
                store = SessionStore()
                barrier = threading.Barrier(workers)
                futures = [
                    executor.submit(_record_after_barrier, store, barrier) for _ in range(workers)
                ]
                results = [future.result(timeout=5) for future in futures]
                assert results.count(True) == 1
                assert results.count(False) == workers - 1
    finally:
        sys.setswitchinterval(previous_interval)


@pytest.mark.asyncio
async def test_dedup_lock_is_independent_from_the_async_session_lock() -> None:
    store = SessionStore()
    session = store.get_or_create("session", Agent(name="Agent", instructions="test"))

    async with session.lock:
        assert await asyncio.wait_for(
            asyncio.to_thread(store.record_processed_message, "message"), timeout=1
        )
