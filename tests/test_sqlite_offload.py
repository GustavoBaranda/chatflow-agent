"""Private offload of blocking SQLite calls: _offloads_blocking_io / _run_blocking (F-c).

Deterministic: blocking calls wait on threading.Event objects set by the test, and
"has started" is observed through events, not sleeps.
"""

import asyncio
import contextvars
import gc
import logging
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.memory import (
    _BLOCKING_IO_WORKERS,
    SessionStore,
    SQLiteSessionStore,
    _offloads_blocking_io,
    _run_blocking,
)

SAFETY = 10.0
MAIN = threading.get_ident()


class _SubclassStore(SQLiteSessionStore):
    pass


class _CustomStore:
    """Stands in for a third-party store: not a SQLiteSessionStore."""


@pytest.fixture
def file_store(tmp_path: Path) -> Any:
    store = SQLiteSessionStore(tmp_path / "offload.db")
    yield store
    store.close()


async def bounded(aw: Any) -> Any:
    return await asyncio.wait_for(aw, SAFETY)


async def wait_event(ev: threading.Event) -> None:
    """Wait for a threading.Event without blocking the loop."""
    deadline = time.monotonic() + SAFETY
    while not ev.is_set():
        assert time.monotonic() < deadline, "event never set"
        await asyncio.sleep(0.005)


class Blocker:
    """A blocking call that records that it started and waits until released."""

    def __init__(self, result: Any = "done", error: BaseException | None = None) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.result = result
        self.error = error
        self.thread_name = ""

    def __call__(self) -> Any:
        self.thread_name = threading.current_thread().name
        self.started.set()
        assert self.release.wait(SAFETY), "never released"
        self.finished.set()
        if self.error is not None:
            raise self.error
        return self.result


# ---------------------------------------------------------------- which stores


def test_only_a_file_backed_sqlite_store_offloads(tmp_path: Path) -> None:
    assert _offloads_blocking_io(SQLiteSessionStore(tmp_path / "a.db"))
    assert not _offloads_blocking_io(SQLiteSessionStore(":memory:"))
    assert not _offloads_blocking_io(_SubclassStore(tmp_path / "b.db"))
    assert not _offloads_blocking_io(SessionStore())
    assert not _offloads_blocking_io(_CustomStore())


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["memory_sqlite", "subclass", "session_store", "custom"])
async def test_other_stores_run_on_the_event_loop_thread(kind: str, tmp_path: Path) -> None:
    store: Any = {
        "memory_sqlite": lambda: SQLiteSessionStore(":memory:"),
        "subclass": lambda: _SubclassStore(tmp_path / "sub.db"),
        "session_store": SessionStore,
        "custom": _CustomStore,
    }[kind]()
    assert await _run_blocking(store, threading.get_ident) == MAIN
    if isinstance(store, SQLiteSessionStore):
        assert store._executor is None  # never created


@pytest.mark.asyncio
async def test_file_store_runs_calls_in_its_own_lazy_pool(file_store: Any) -> None:
    assert file_store._executor is None
    name = await _run_blocking(file_store, lambda: threading.current_thread().name)
    assert name.startswith("chatflow-sqlite")
    assert file_store._executor is not None
    assert file_store._executor._max_workers == _BLOCKING_IO_WORKERS == 4


@pytest.mark.asyncio
async def test_result_exception_and_context_are_carried(file_store: Any) -> None:
    var: contextvars.ContextVar[str] = contextvars.ContextVar("fc_test", default="unset")
    var.set("from-caller")
    assert await _run_blocking(file_store, var.get) == "from-caller"
    assert await _run_blocking(file_store, lambda: 42) == 42

    def boom() -> None:
        raise LookupError("in the worker")

    with pytest.raises(LookupError, match="in the worker"):
        await _run_blocking(file_store, boom)


@pytest.mark.asyncio
async def test_real_store_calls_work_offloaded(file_store: Any) -> None:
    ctx = await _run_blocking(file_store, lambda: file_store.get_or_create("s", Agent(name="A", instructions="x")))
    assert ctx.session_id == "s"
    assert await _run_blocking(file_store, lambda: file_store.record_processed_message("w")) is True
    assert await _run_blocking(file_store, lambda: file_store.record_processed_message("w")) is False


# ---------------------------------------------------------------- cancellation


@pytest.mark.asyncio
async def test_cancel_while_running_waits_for_the_call_then_cancels(file_store: Any) -> None:
    call = Blocker()
    task = asyncio.ensure_future(_run_blocking(file_store, call))
    await wait_event(call.started)
    task.cancel()
    for _ in range(20):
        await asyncio.sleep(0)
    assert not task.done(), "must not return while the call is still running"
    call.release.set()
    with pytest.raises(asyncio.CancelledError):
        await bounded(task)
    assert call.finished.is_set()


@pytest.mark.asyncio
async def test_repeated_cancellations_still_wait_and_cancel_once(file_store: Any) -> None:
    call = Blocker()
    task = asyncio.ensure_future(_run_blocking(file_store, call))
    await wait_event(call.started)
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
    call.release.set()
    with pytest.raises(asyncio.CancelledError):
        await bounded(task)
    assert call.finished.is_set()


@pytest.mark.asyncio
async def test_cancel_before_a_queued_call_starts_withdraws_it(file_store: Any) -> None:
    """Condition 5: a queued call that is cancelled never runs (no late write)."""
    busy = [Blocker() for _ in range(_BLOCKING_IO_WORKERS)]
    running = [asyncio.ensure_future(_run_blocking(file_store, b)) for b in busy]
    for b in busy:
        await wait_event(b.started)
    queued = Blocker()
    task = asyncio.ensure_future(_run_blocking(file_store, queued))
    await asyncio.sleep(0.05)
    assert not queued.started.is_set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bounded(task)  # returns at once: nothing to wait for
    for b in busy:
        b.release.set()
    await bounded(asyncio.gather(*running))
    await _run_blocking(file_store, lambda: None)  # the pool has drained its queue
    assert not queued.started.is_set(), "a withdrawn call must never run"


@pytest.mark.asyncio
async def test_must_run_call_is_not_withdrawn_when_cancelled_while_queued(file_store: Any) -> None:
    busy = [Blocker() for _ in range(_BLOCKING_IO_WORKERS)]
    running = [asyncio.ensure_future(_run_blocking(file_store, b)) for b in busy]
    for b in busy:
        await wait_event(b.started)
    rollback = Blocker()
    rollback.release.set()
    task = asyncio.ensure_future(_run_blocking(file_store, rollback, must_run=True))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done(), "a must_run call is waited for, not withdrawn"
    for b in busy:
        b.release.set()
    with pytest.raises(asyncio.CancelledError):
        await bounded(task)
    assert rollback.finished.is_set()
    await bounded(asyncio.gather(*running))


@pytest.mark.asyncio
async def test_error_of_a_call_finished_after_cancellation_is_observed(
    file_store: Any, caplog: pytest.LogCaptureFixture
) -> None:
    call = Blocker(error=RuntimeError("late failure"))
    task = asyncio.ensure_future(_run_blocking(file_store, call))
    await wait_event(call.started)
    task.cancel()
    await asyncio.sleep(0.01)
    call.release.set()
    with pytest.raises(asyncio.CancelledError):
        await bounded(task)
    with caplog.at_level(logging.ERROR, logger="asyncio"):
        del task
        gc.collect()
        await asyncio.sleep(0)
    assert "never retrieved" not in caplog.text


# ---------------------------------------------------------------- close() and isolation


@pytest.mark.asyncio
async def test_close_does_not_wait_for_running_calls_and_later_calls_run_inline(tmp_path: Path) -> None:
    """Condition 4: close() with a call in flight and new calls arriving afterwards."""
    store = SQLiteSessionStore(tmp_path / "close.db")
    call = Blocker(result="finished after close")
    task = asyncio.ensure_future(_run_blocking(store, call))
    await wait_event(call.started)
    t0 = time.perf_counter()
    store.close()
    assert time.perf_counter() - t0 < 0.5, "close() must not block on running calls"
    assert store._executor is None
    assert await _run_blocking(store, threading.get_ident) == MAIN  # inline after close
    with pytest.raises(RuntimeError, match="closed"):
        await _run_blocking(store, lambda: store.record_processed_message("x"))
    call.release.set()
    assert await bounded(task) == "finished after close"


@pytest.mark.asyncio
async def test_submission_racing_with_close_falls_back_inline(tmp_path: Path, monkeypatch: Any) -> None:
    store = SQLiteSessionStore(tmp_path / "race.db")
    executor = store._blocking_executor()
    assert executor is not None
    executor.shutdown(wait=True)  # as if close() ran between lookup and submit
    monkeypatch.setattr(store, "_blocking_executor", lambda: executor)
    assert await _run_blocking(store, threading.get_ident) == MAIN


@pytest.mark.asyncio
async def test_stuck_store_calls_do_not_starve_the_default_executor(file_store: Any) -> None:
    """Report 38 test 17: more stuck calls than the store's workers; to_thread work
    (sync tools) still runs at once."""
    stuck = [Blocker() for _ in range(_BLOCKING_IO_WORKERS * 2)]
    tasks = [asyncio.ensure_future(_run_blocking(file_store, b)) for b in stuck]
    for b in stuck[:_BLOCKING_IO_WORKERS]:
        await wait_event(b.started)
    t0 = time.perf_counter()
    await bounded(asyncio.to_thread(lambda: None))
    assert time.perf_counter() - t0 < 1.0
    for b in stuck:
        b.release.set()
    assert await bounded(asyncio.gather(*tasks)) == ["done"] * len(stuck)


@pytest.mark.asyncio
async def test_failed_must_run_call_is_raised_instead_of_the_cancellation(file_store: Any) -> None:
    """Condition 3: a rollback that fails after a cancellation is never reported as done."""
    call = Blocker(error=RuntimeError("rollback failed"))
    task = asyncio.ensure_future(_run_blocking(file_store, call, must_run=True))
    await wait_event(call.started)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    call.release.set()
    with pytest.raises(RuntimeError, match="rollback failed") as raised:
        await bounded(task)
    assert isinstance(raised.value.__cause__, asyncio.CancelledError)
