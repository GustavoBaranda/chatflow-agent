#!/usr/bin/env python3
"""Reproducible offline benchmark for SQLiteSessionStore and Runner turns (#174).

Measures SQLite connection counts, write-transaction counts, median/p95/mean
latency, and throughput for individual store operations and complete offline
Runner turns, comparing the pre-#174 baseline against the optimized Phase 1
implementation within the same process.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from chatflow_agent import Agent, Runner, SessionStore
from chatflow_agent.core._sync_work import _track_sync_work, _wait_for_sync_work
from chatflow_agent.core.engines.base import BaseEngine, EngineTurnResult
from chatflow_agent.core.memory import (
    SessionContext,
    SQLiteSessionContext,
    SQLiteSessionStore,
    _offloads_blocking_io,
    _run_blocking,
    dumps_json,
)
from chatflow_agent.core.runner import (
    _hold_session_lock,
    _refuse_nested_turn,
    _TurnTimeout,
    _within,
)
from chatflow_agent.types import AgentResponse, Message, Role


class EchoEngine(BaseEngine):
    """Zero-latency offline engine for isolating storage and orchestration cost."""

    async def generate_turn_async(
        self,
        agent: Agent,
        history: list[Message],
    ) -> EngineTurnResult:
        return EngineTurnResult(text="ack")


DUMMY_AGENT = Agent(name="assistant", instructions="Echo", engine=EchoEngine())


def summarize_ms(samples_sec: list[float]) -> dict[str, float]:
    ms = [s * 1000.0 for s in samples_sec]
    ms_sorted = sorted(ms)
    n = len(ms_sorted)
    p95_idx = min(n - 1, max(0, int(round(0.95 * (n - 1)))))
    total_sec = sum(samples_sec)
    return {
        "n": float(n),
        "median_ms": statistics.median(ms_sorted),
        "p95_ms": ms_sorted[p95_idx],
        "mean_ms": statistics.mean(ms_sorted),
        "ops_per_sec": (n / total_sec) if total_sec > 0 else 0.0,
    }


def _legacy_get_or_create(
    store: SQLiteSessionStore,
    session_id: str,
    default_agent: Agent,
    metadata: dict[str, Any] | None = None,
) -> SessionContext:
    """Pre-#174 get_or_create implementation (unconditional INSERT ON CONFLICT DO NOTHING)."""
    store._ensure_open()
    with store._connection() as conn:
        meta_json = dumps_json(metadata or {}, what="Session metadata")
        conn.execute(
            """
            INSERT INTO chatflow_sessions (session_id, active_agent_name, metadata_json)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO NOTHING
            """,
            (session_id, default_agent.name, meta_json),
        )
        row = conn.execute(
            "SELECT active_agent_name, metadata_json FROM chatflow_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        agent_name = row["active_agent_name"]
        meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        active_agent = default_agent
        if store._agent_resolver:
            resolved = store._agent_resolver(agent_name)
            if resolved is not None:
                active_agent = resolved
        msg_rows = conn.execute(
            "SELECT message_json FROM chatflow_messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        history = [Message.model_validate_json(r["message_json"]) for r in msg_rows]
        return SQLiteSessionContext(
            session_id=session_id,
            initial_agent=active_agent,
            db_path=store._conn_path,
            max_turns=store.default_max_turns,
            metadata=meta,
            preloaded_history=history,
            store=store if store._in_memory else None,
            owner=store,
            default_agent=default_agent,
        )


def _legacy_get(store: SQLiteSessionStore, session_id: str) -> SessionContext | None:
    """Pre-#174 two-connection get() implementation."""
    store._ensure_open()
    with store._connection() as conn:
        row = conn.execute(
            "SELECT active_agent_name FROM chatflow_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        agent_name = row["active_agent_name"]

    active_agent = None
    if store._agent_resolver:
        active_agent = store._agent_resolver(agent_name)
    if active_agent is None:
        active_agent = Agent(name=agent_name, instructions="", model="gemini-2.5-flash")
    return _legacy_get_or_create(store, session_id, default_agent=active_agent)


class _LegacyRunner(Runner):
    """Runner executing the exact pre-#174 run_async() path (get_or_create + refresh)."""

    async def run_async(
        self,
        session_id: str,
        user_message: str,
        metadata: dict[str, Any] | None = None,
        *,
        turn_timeout: float | None = None,
    ) -> AgentResponse:
        _refuse_nested_turn(self.session_store, session_id)
        store = self.session_store
        session: SessionContext | None = None
        if _offloads_blocking_io(store):
            lock = store._session_lock(session_id)  # type: ignore[attr-defined]
        else:
            session = _legacy_get_or_create(
                store,  # type: ignore[arg-type]
                session_id=session_id,
                default_agent=self.starting_agent,
                metadata=metadata,
            )
            lock = session.lock

        async with _hold_session_lock(store, session_id, lock, wait_deadline=None):
            await _wait_for_sync_work(store, session_id)
            if session is None:
                session = await _run_blocking(
                    store,
                    functools.partial(
                        _legacy_get_or_create,
                        store,  # type: ignore[arg-type]
                        session_id=session_id,
                        default_agent=self.starting_agent,
                        metadata=metadata,
                    ),
                )
            refresh = getattr(session, "refresh", None)
            if callable(refresh):
                await _run_blocking(store, refresh)
            with _track_sync_work(store, session_id):
                return await _within(
                    turn_timeout, self._execute_turn(session, user_message, metadata), _TurnTimeout
                )


def measure_connections(fn: Any) -> tuple[int, int]:
    """Return (total_connections, write_connections) opened during fn()."""
    real_connect = sqlite3.connect
    traces: list[list[str]] = []

    def wrapped(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)
        stmts: list[str] = []
        traces.append(stmts)
        conn.set_trace_callback(lambda sql: stmts.append(sql.strip().upper()))
        return conn

    with patch("chatflow_agent.core.memory.sqlite3.connect", side_effect=wrapped):
        fn()
    write_conns = sum(
        1
        for stmts in traces
        if any(s.startswith(("INSERT", "UPDATE", "DELETE")) for s in stmts)
    )
    return len(traces), write_conns


async def measure_connections_async(coro_fn: Any) -> tuple[int, int]:
    """Return (total_connections, write_connections) opened during await coro_fn()."""
    real_connect = sqlite3.connect
    traces: list[list[str]] = []

    def wrapped(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)
        stmts: list[str] = []
        traces.append(stmts)
        conn.set_trace_callback(lambda sql: stmts.append(sql.strip().upper()))
        return conn

    with patch("chatflow_agent.core.memory.sqlite3.connect", side_effect=wrapped):
        await coro_fn()
    write_conns = sum(
        1
        for stmts in traces
        if any(s.startswith(("INSERT", "UPDATE", "DELETE")) for s in stmts)
    )
    return len(traces), write_conns


async def run_suite(iterations: int) -> None:
    print(f"Platform: {platform.platform()}", flush=True)
    print(f"Python: {sys.version.split()[0]} | SQLite: {sqlite3.sqlite_version}", flush=True)
    print(f"Iterations per workload: {iterations}\n", flush=True)

    with tempfile.TemporaryDirectory(prefix="chatflow_bench_174_") as tmpdir:
        db_ops = Path(tmpdir) / "ops.db"
        with SQLiteSessionStore(db_ops) as store:
            ctx = store.get_or_create("s1", default_agent=DUMMY_AGENT, metadata={"k": "v"})
            for i in range(10):
                ctx.add_message(Message(role=Role.USER, content=f"msg {i}"))

            # Connection counts: Before vs After
            b_get_c, b_get_w = measure_connections(lambda: _legacy_get(store, "s1"))
            a_get_c, a_get_w = measure_connections(lambda: store.get("s1"))

            b_goc_c, b_goc_w = measure_connections(
                lambda: _legacy_get_or_create(store, "s1", DUMMY_AGENT)
            )
            a_goc_c, a_goc_w = measure_connections(
                lambda: store.get_or_create("s1", DUMMY_AGENT)
            )

            # Interleaved latency measurements for store.get and store.get_or_create
            samples_get_before: list[float] = []
            samples_get_after: list[float] = []
            samples_goc_before: list[float] = []
            samples_goc_after: list[float] = []

            for _ in range(iterations):
                t0 = time.perf_counter()
                _legacy_get(store, "s1")
                samples_get_before.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                store.get("s1")
                samples_get_after.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                _legacy_get_or_create(store, "s1", DUMMY_AGENT)
                samples_goc_before.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                store.get_or_create("s1", DUMMY_AGENT)
                samples_goc_after.append(time.perf_counter() - t0)

        # Turn benchmarks (Before vs After in interleaved pairs on identical DBs)
        db_before = Path(tmpdir) / "turns_before.db"
        db_after = Path(tmpdir) / "turns_after.db"
        with SQLiteSessionStore(db_before) as s_before, SQLiteSessionStore(db_after) as s_after:
            r_before = _LegacyRunner(DUMMY_AGENT, session_store=s_before, engine=EchoEngine())
            r_after = Runner(DUMMY_AGENT, session_store=s_after, engine=EchoEngine())

            await r_before.run_async("s1", "warmup")
            await r_after.run_async("s1", "warmup")

            b_turn_nm_c, b_turn_nm_w = await measure_connections_async(
                lambda: r_before.run_async("s1", "probe")
            )
            a_turn_nm_c, a_turn_nm_w = await measure_connections_async(
                lambda: r_after.run_async("s1", "probe")
            )

            b_turn_m_c, b_turn_m_w = await measure_connections_async(
                lambda: r_before.run_async("s1", "probe", metadata={"channel": "tg"})
            )
            a_turn_m_c, a_turn_m_w = await measure_connections_async(
                lambda: r_after.run_async("s1", "probe", metadata={"channel": "tg"})
            )

            samples_nm_before: list[float] = []
            samples_nm_after: list[float] = []
            samples_m_before: list[float] = []
            samples_m_after: list[float] = []

            for i in range(iterations):
                t0 = time.perf_counter()
                await r_before.run_async("s1", f"m-{i}")
                samples_nm_before.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                await r_after.run_async("s1", f"m-{i}")
                samples_nm_after.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                await r_before.run_async(
                    "s1", f"mm-{i}", metadata={"channel": "tg", "seq": i}
                )
                samples_m_before.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                await r_after.run_async(
                    "s1", f"mm-{i}", metadata={"channel": "tg", "seq": i}
                )
                samples_m_after.append(time.perf_counter() - t0)

        # Reference baselines: SessionStore and SQLiteSessionStore(":memory:")
        mem_dict_store = SessionStore()
        r_dict = Runner(DUMMY_AGENT, session_store=mem_dict_store, engine=EchoEngine())
        await r_dict.run_async("s1", "warmup")
        samples_dict_turn: list[float] = []
        for i in range(iterations):
            t0 = time.perf_counter()
            await r_dict.run_async("s1", f"m-{i}", metadata={"channel": "tg", "seq": i})
            samples_dict_turn.append(time.perf_counter() - t0)

        with SQLiteSessionStore(":memory:") as mem_sql_store:
            r_memsql = Runner(DUMMY_AGENT, session_store=mem_sql_store, engine=EchoEngine())
            await r_memsql.run_async("s1", "warmup")
            samples_memsql_turn: list[float] = []
            for i in range(iterations):
                t0 = time.perf_counter()
                await r_memsql.run_async("s1", f"m-{i}", metadata={"channel": "tg", "seq": i})
                samples_memsql_turn.append(time.perf_counter() - t0)

        rows = [
            (
                "store.get('existing', 10 msgs) [Before]",
                b_get_c,
                b_get_w,
                summarize_ms(samples_get_before),
            ),
            (
                "store.get('existing', 10 msgs) [After]",
                a_get_c,
                a_get_w,
                summarize_ms(samples_get_after),
            ),
            (
                "store.get_or_create('existing') [Before]",
                b_goc_c,
                b_goc_w,
                summarize_ms(samples_goc_before),
            ),
            (
                "store.get_or_create('existing') [After]",
                a_goc_c,
                a_goc_w,
                summarize_ms(samples_goc_after),
            ),
            (
                "Runner.run_async (no metadata) [Before]",
                b_turn_nm_c,
                b_turn_nm_w,
                summarize_ms(samples_nm_before),
            ),
            (
                "Runner.run_async (no metadata) [After]",
                a_turn_nm_c,
                a_turn_nm_w,
                summarize_ms(samples_nm_after),
            ),
            (
                "Runner.run_async (with metadata) [Before]",
                b_turn_m_c,
                b_turn_m_w,
                summarize_ms(samples_m_before),
            ),
            (
                "Runner.run_async (with metadata) [After]",
                a_turn_m_c,
                a_turn_m_w,
                summarize_ms(samples_m_after),
            ),
            (
                "Runner.run_async (SQLite :memory: ref)",
                1,
                1,
                summarize_ms(samples_memsql_turn),
            ),
            (
                "Runner.run_async (SessionStore RAM ref)",
                0,
                0,
                summarize_ms(samples_dict_turn),
            ),
        ]

        print(
            "| Workload | Conns | Write Conns | Median ms | p95 ms | Mean ms | Ops/sec |",
            flush=True,
        )
        print("|---|---:|---:|---:|---:|---:|---:|", flush=True)
        for label, conns, wconns, st in rows:
            print(
                f"| {label} | {conns} | {wconns} | {st['median_ms']:.2f} | "
                f"{st['p95_ms']:.2f} | {st['mean_ms']:.2f} | {st['ops_per_sec']:.1f} |",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=40, help="Iterations per workload")
    args = parser.parse_args()
    # Keep temporary files inside a writable directory when TMPDIR is not set
    if "TMPDIR" not in os.environ and "TEMP" not in os.environ:
        os.environ["TMPDIR"] = str(Path.cwd())
    asyncio.run(run_suite(args.iterations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
