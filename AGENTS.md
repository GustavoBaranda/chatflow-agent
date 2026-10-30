# AGENTS.md

This file applies to the entire repository. It is the working contract for AI coding
agents contributing to `chatflow-agent`. Follow the user's task first, then this file,
and use the implementation, tests, and project configuration as the source of truth when
documentation is stale or ambiguous.

## 1. Project overview

`chatflow-agent` is a lightweight, typed, async-first Python framework for conversational
agents. It provides multi-agent handoffs, tool schema generation and wrapped tool execution,
provider-independent LLM engines, in-memory and SQLite session stores, and CLI, Telegram, and
WhatsApp channels.

Keep the project small and modular. Preserve clear boundaries between orchestration,
providers, tools, persistence, and channels. Do not add speculative architecture, planned
backends, or integrations that are not present in the repository.

Supported Python versions are 3.10 through 3.13. CI runs on Ubuntu and Windows for every
supported version. The package is typed (`py.typed`), Ruff targets Python 3.10, and Mypy is
configured in strict mode for the `chatflow_agent` package.

## 2. Repository map

Make changes in the component that owns the behavior:

| Area | Location | Responsibility |
| --- | --- | --- |
| Public package API | `src/chatflow_agent/__init__.py`, `src/chatflow_agent/memory.py` | Top-level exports, compatibility aliases, package version |
| Agents and handoffs | `src/chatflow_agent/core/agent.py` | Agent configuration, tool registration, handoff declarations |
| Turn orchestration | `src/chatflow_agent/core/runner.py` | Session locking, engine selection, tools, handoffs, timeouts, cancellation, rollback, resource scopes |
| Synchronous tool barrier | `src/chatflow_agent/core/_sync_work.py` | Tracks sync tools that outlive their awaiting task and prevents unsafe same-session progress |
| Tools | `src/chatflow_agent/core/tools.py` | Decorator, introspection, JSON schema validation, sync/async execution |
| Engines | `src/chatflow_agent/core/engines/` | `BaseEngine`, Gemini REST, OpenAI-compatible REST, Anthropic REST, engine resolution |
| Session state | `src/chatflow_agent/core/memory.py` | `SessionContext`, `SessionStore`, SQLite contexts/stores, deduplication, transactions |
| Domain models | `src/chatflow_agent/types.py` | Messages, roles, tool calls/results, handoffs, responses |
| Exceptions | `src/chatflow_agent/exceptions.py` | Public error hierarchy |
| Channels | `src/chatflow_agent/channels/` | Base channel, CLI, Telegram, WhatsApp/FastAPI webhook |
| Tests | `tests/` | Unit, integration, concurrency, security, compatibility, and regression coverage |
| Examples | `examples/` | Executable public-API examples; do not use them as proof of behavior without checking code |
| Documentation | `README.md`, `README.es.md`, `docs/`, `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md` | English and Spanish user, architecture, migration, contribution, release, and security guidance |
| CI and release | `.github/workflows/`, `scripts/verify_release.py` | Test matrix and release consistency/ancestry checks |

There is no built-in generic `WebhookChannel` or distributed session backend. The custom
FastAPI webhook is an example application; do not document examples or proposed features as
shipped package components.

## 3. Architectural Boundaries

These are ownership constraints, not a duplicate design specification. See
`docs/ARCHITECTURE.md` for detailed flows, and verify its details against the implementation
and tests when they differ.

- **Runner:** Owns turn orchestration and the execution lifecycle, including handoff application,
  session coordination, timeouts, cancellation, and rollback. Do not duplicate orchestration in
  channels, engines, or tools.
- **Agent:** Declares agent configuration, instructions, available tools, and permitted handoff
  targets. Keep transport and persistence responsibilities outside `Agent`.
- **Engines:** Encapsulate provider-specific communication, request formatting, response
  normalization, and provider error handling. Preserve provider independence and keep this logic
  out of orchestration.
- **Tools:** Represent callable capabilities executed through the established schema-generation,
  execution, error-handling, and concurrency contracts. Conversational rollback cannot undo a
  tool's external side effects.
- **Session stores:** Own session state, persistence, and message deduplication. Keep storage
  independent of channels and providers, and preserve the distinct in-memory and SQLite
  guarantees documented below.
- **Channels:** Adapt external transports and messaging platforms, then dispatch through the
  attached `Runner` and session store contracts. Do not create parallel orchestration or
  persistence paths.
- **Handoffs:** Keep handoff declarations on `Agent` and their authorization and state transition
  in `Runner`. Do not implement alternative handoff mechanisms in channels or engines.

Before adding an abstraction, service, manager, or execution path, identify the component that
already owns the responsibility and whether the capability already exists. Prefer extending that
component over creating a competing implementation. This does not prohibit a legitimate
architectural change explicitly authorized by the maintainer.

## 4. Engineering principles

- Make the smallest focused change that fully solves the assigned problem. Avoid unrelated
  refactors, formatting churn, and new abstraction layers without a demonstrated need.
- Keep interfaces explicit and component ownership clear. Provider-specific payload logic
  belongs in an engine; transport-specific logic belongs in a channel; turn semantics belong
  in `Runner`; persistence belongs in session stores.
- Do not change production behavior solely to make a test pass. Reproduce the defect, fix its
  cause, and keep the regression test meaningful.
- Avoid silent fallbacks that hide invalid configuration, security failures, malformed
  provider responses, or unsupported schemas. Use the existing public exception hierarchy.
- Preserve typing. Add explicit annotations to production functions, methods, and classes.
  Do not weaken types with broad `Any`, casts, or ignores merely to silence Mypy.
- Core dependencies must remain lightweight. Add channel-specific libraries under the
  appropriate optional extra in `pyproject.toml`; do not add a dependency when the standard
  library or an existing dependency is sufficient.
- Preserve the MIT license and existing attribution. Do not copy incompatible code or assets.

## 5. Public API and compatibility

Treat the exports in `src/chatflow_agent/__init__.py`, `src/chatflow_agent/memory.py`, and
documented constructors and methods as public compatibility surfaces.

- Preserve names, parameter order, keyword behavior, defaults, return types, exceptions, and
  side effects unless a breaking change is explicitly authorized by the maintainer.
- Never remove a constructor parameter in a way that silently reinterprets an old positional
  argument as a different parameter. Add an explicit compatibility safeguard, normally a
  keyword-only boundary, and test both positional and keyword behavior.
- New optional parameters should normally be keyword-only when positional expansion could
  make future evolution unsafe.
- Keep top-level exports and `__all__` intentional. Adding an import path can create a public
  API commitment; removing or moving one can break users.
- A breaking change requires explicit authorization, focused API regression tests, migration
  guidance, and corresponding changelog/release documentation. Do not infer authorization
  from a request to investigate or prepare a proposal.
- Preserve SQLite files created by supported earlier releases. Schema or serialization changes
  require an explicit migration/backward-compatibility design and tests using real legacy data,
  including `tests/fixtures/sqlite_0_3_4/` when relevant.
- Validate examples against actual public imports and signatures. Do not rely on private
  underscore-prefixed helpers in user documentation.

## 6. Async execution and concurrency

The execution contract is tested extensively. Changes in this area require focused regression
tests, not reasoning from a single happy path.

- `Runner.run_async()` is the async entry point. `Runner.run()` owns a temporary loop and must
  continue to reject calls made inside an already running event loop.
- Turns for one session serialize on an `asyncio.Lock`; different sessions should remain able
  to progress independently. For SQLite, the shared lock guarantee is scoped to a store,
  session ID, and running event loop. Do not claim cross-process or cross-loop serialization.
- Queue waiting and active turn execution are separate budgets. Queue wait must not consume the
  turn timeout. A queue timeout must not start or write the turn.
- On an unhandled error, cancellation, or turn timeout after a checkpoint, restore history,
  active agent, and metadata before surfacing the failure. Never report a successful rollback
  when persistence restoration failed.
- Preserve `ReentrantTurnError` for same-session turns started from work descended from the
  current turn. Use `Runner.spawn_turn()` for a deferred follow-up; do not bypass the lineage
  checks or session lock.
- Synchronous tools run in threads and cannot be forcibly cancelled once started. Preserve the
  per-session sync-work barrier so timeout/cancellation cannot let later same-session work race
  with a still-running tool.
- Do not perform blocking file SQLite I/O on the event-loop thread. File-backed
  `SQLiteSessionStore` operations use the store's dedicated executor through `_run_blocking()`;
  `:memory:` stores and custom stores follow their documented inline behavior.
- Never hold a `threading.Lock` across `await`. Protect shared thread state with short critical
  sections and keep `asyncio.Lock` ownership on its event loop.
- Close resources deterministically. Use `async with runner:` when testing pooled HTTP clients
  and `with SQLiteSessionStore(...)` or `store.close()` for SQLite lifecycle coverage.

## 7. Session storage and persistence

Implemented stores are `SessionStore` and `SQLiteSessionStore`, both implementing
`BaseSessionStore`.

- `SessionStore` keeps sessions in memory. Its deduplication state is thread-safe; keep
  check/update/purge/capacity operations atomic under `_dedup_lock`. Do not infer the same
  cross-thread guarantee for the session registry. Session turns use per-session async locks.
- `SQLiteSessionStore` supports file paths and isolated shared-cache `:memory:` databases.
  File-backed connections use WAL, `synchronous=NORMAL`, and a busy timeout. In-memory
  configurations use shared cache and a busy timeout, but not WAL. Both configurations use the
  same schema, transaction boundaries, and per-session turn journaling.
- Keep session IDs isolated. Concurrent work for one session must not modify another session.
- Preserve atomic writes and rollback boundaries. Use explicit transactions where several SQL
  statements form one invariant; verify rollback paths and database integrity after failures.
- Deduplication is an atomic check-and-record operation. Preserve TTL expiration, capacity or
  purge semantics, and exactly-one acceptance under concurrency. Channels must use the attached
  store rather than maintain a second deduplication source of truth.
- For `SQLiteSessionStore` and `SQLiteSessionContext`, do not weaken `SessionDeletedError`
  behavior. A running turn must not recreate or leave rows behind after its session is deleted;
  a queued turn may subsequently start a new empty session. Do not infer this deletion guard for
  the in-memory `SessionStore`.
- `SQLiteSessionStore` is not a distributed coordination service. Multi-process and multi-node
  deployments need an application-provided external store; do not imply stronger guarantees.
- Any schema, `PRAGMA user_version`, timestamp, JSON serialization, or legacy loading change
  needs disk and `:memory:` coverage, reopen tests, and backward-compatibility tests.

## 8. Providers, tools, and channels

### Providers and tools

- Every engine implements `BaseEngine.generate_turn_async()` and returns `EngineTurnResult`.
  Keep provider request formatting, response validation, retries, and tool-call normalization in
  the relevant engine module.
- Preserve engine-resolution precedence in `src/chatflow_agent/core/engines/factory.py`. Do not
  leak one provider's environment key to another provider, a custom host, or a local endpoint.
- Do not follow credential-bearing redirects. Redact secrets and sensitive response details in
  errors and logs using the existing helpers.
- Tool schemas are generated at registration. Unrepresentable union members raise
  `ToolSchemaError`; unknown non-union annotations retain the backward-compatible `STRING`
  fallback, and unresolved annotations emit a warning and retain that fallback.
- Prior to tool execution (sync, async, and direct callable invocation), LLM-provided arguments
  are validated and coerced against parameter annotations via Pydantic (`ToolArgumentValidationError`,
  subclass of `ToolExecutionError`). Validation is enabled by default (`validate_args=True`). Numeric
  types coerce integer and float strings and reject booleans, NaN, and infinity; string types coerce
  numbers but reject booleans and collections. Parameters with leading underscores or named
  `model_config` validate without shadowing parent BaseModel attributes. Omitted parameters with
  defaults preserve native Python parameter binding and mutable default semantics. Synchronous tools
  are never dispatched to worker threads when validation fails. Built-in validation errors and
  exception tracebacks strictly redact raw inputs, untrusted parameter keys, and nested paths, with error
  messages strictly bounded (max 3 errors, max 300 characters). Validation can be disabled per tool
  with `validate_args=False` or overridden via `add_tool()`.
- A conversational rollback cannot undo a tool's external side effects. Tools that charge,
  send, mutate, or publish should use application-level idempotency keys and retry-safe design.

### Channels

- Extend `BaseChannel` and preserve `attach()`, `dispatch_async()`, and sync-dispatch semantics.
  Keep provider- or transport-specific behavior out of `Runner` and storage.
- Optional channel imports must continue to fail with an actionable `DependencyError`, not make
  the core installation require FastAPI, Uvicorn, or `python-telegram-bot`.
- WhatsApp webhooks must retain request-size checks, fail-closed HMAC-SHA256 verification by
  default, constant-time token/signature comparison, safe malformed-payload handling, atomic
  `wamid` deduplication before background dispatch, and masked phone numbers in logs.
- WhatsApp's at-most-once acknowledgment tradeoff is deliberate: recording an ID before the
  background turn can lose that turn after a process crash. Do not relabel it exactly-once or
  change the contract without explicit design approval.
- Telegram group reset authorization must remain fail closed when membership checks fail.
- Respect platform message-length rules independently. Do not assume WhatsApp and Telegram use
  identical limits or units.

## 9. Testing requirements

Install all optional and development dependencies in an isolated environment:

```bash
pip install -e ".[all,dev]"
```

Use the same primary commands as CI:

```bash
ruff check .
python -m mypy
pytest -v
```

CI runs these commands on Python 3.10, 3.11, 3.12, and 3.13 on Ubuntu and Windows. Mypy is
strict but configured for the `chatflow_agent` package; do not claim tests were type-checked by
that command.

Run the smallest relevant tests while iterating, then broaden validation in proportion to risk:

- Runner, timeouts, cancellation, rollback, and sync tools: `tests/test_runner*.py`,
  `tests/test_sqlite_session_concurrency.py`, `tests/test_sqlite_event_loop.py`,
  `tests/test_sqlite_offload.py`.
- Session stores and deduplication: `tests/test_memory*.py`, `tests/test_sqlite_*.py`.
- Tools and schemas: `tests/test_tools*.py`, `tests/test_schema_normalization.py`.
- Providers: `tests/test_gemini*.py`, `tests/test_multi_engine.py`,
  `tests/test_engine_credentials.py`, provider resilience/malformed-response tests.
- Channels: the corresponding `tests/test_whatsapp*.py`, `tests/test_telegram*.py`, or
  `tests/test_cli_channel.py` modules.
- Migration claims and legacy SQLite: `tests/test_migration_guide_claims.py` and
  `tests/test_sqlite_034_compat.py`.

Testing rules:

- Every bug fix needs a regression test that fails for the original defect. Public API changes,
  concurrency changes, persistence changes, and provider/channel changes need direct coverage.
- Prefer events, barriers, injected clocks, controlled timestamps, and deterministic fakes over
  real-time sleeps. Give boundary assertions a safe margin when a real clock is unavoidable.
- Tests must stay offline. Use `httpx.MockTransport`, fake engines, and mocked channel clients;
  never require paid credentials or live provider calls.
- Exercise cancellation and failure paths, not only successful completion. For concurrency bugs,
  prove the relevant ordering or atomicity rather than relying on stress alone.
- Do not delete, weaken, skip, or over-mock a meaningful regression merely to make CI green.
- Report exactly what ran, including failures, skips, platform, and any suite not executed.
  Never say a check passed when it was not run.

## 10. Documentation

- English is the primary user-documentation language. Keep the relevant Spanish counterpart
  semantically equivalent: `README.md` / `README.es.md`, `docs/ARCHITECTURE.md` /
  `docs/es/ARCHITECTURE.md`, and `docs/MIGRATION_0.4.md` /
  `docs/es/MIGRATION_0.4.md`. `CONTRIBUTING.md` and `CHANGELOG.md` contain both languages.
- Documentation must describe implemented behavior. Verify signatures, defaults, errors,
  timeout semantics, persistence guarantees, and examples against code and tests.
- Update migration guidance and the changelog for authorized compatibility changes. Keep release
  candidates and stable-version wording accurate to the repository state.
- Use public APIs in examples. Ensure snippets import, construct, and clean up resources correctly.
- Avoid duplicating large architecture or contribution sections; link to the canonical document.
- Do not translate code identifiers, exception names, paths, commands, or configuration keys.

## 11. Security and reliability

- Never commit, print, or include real API keys, tokens, app secrets, passwords, webhook payloads,
  or personal phone numbers. Use sanitized fixtures and mock values.
- Treat provider responses, webhook bodies, tool arguments, persisted JSON, and external errors as
  untrusted input. Validate shapes and types before indexing or executing.
- Preserve fail-closed webhook authentication and credential isolation. Disabling verification is
  for explicit local/test configuration only and must remain visibly warned.
- Keep logs useful without exposing secrets or message contents. Preserve URL/body redaction and
  phone masking.
- Treat model-supplied tool arguments as untrusted. Do not assume the provider enforced the
  advertised schema; preserve `ToolExecutionError` wrapping, and treat any new runtime coercion
  or validation behavior as a compatibility-sensitive design change.
- Review dependency changes for necessity, license, maintenance, and attack surface. Keep optional
  integrations optional.
- Report security vulnerabilities through the private process in `SECURITY.md`, never a public
  issue. Do not overstate guarantees: document known trust and deployment boundaries.

## 12. Git, pull requests, and releases

- Before editing, inspect the branch, exact `HEAD`, worktree status, existing instructions, and
  the target files. Preserve unrelated user changes, untracked files, branches, tags, and stashes.
- Use the base explicitly specified by the task. Otherwise, verify the authoritative remote base
  before new development: `upstream/main` in the documented contributor-fork workflow, or
  `origin/main` in a typical maintainer checkout. Do not update refs, switch branches, pull,
  rebase, merge, or create a worktree without task authorization, except when an isolated
  worktree is necessary to perform authorized work without disturbing the user's checkout.
- Keep the diff scoped to the task. Review both the file list and full diff before completion.
- Use the PR and issue templates. Link the issue, explain compatibility impact, and list commands
  actually executed.
- Do not commit, push, force-push, open or update issues/PRs, merge, create tags/releases, or
  publish packages without explicit authorization for that action. Preparing a change is not
  authorization to publish it.
- Do not alter or delete stashes. Never use destructive Git commands to discard user work.
- Releases are published by `.github/workflows/publish.yml` through PyPI trusted publishing.
  Release work must keep the tag, `pyproject.toml` version, `chatflow_agent.__version__`, checkout,
  and `main` ancestry consistent with `scripts/verify_release.py`.
- When a change affects public APIs, component ownership or architecture, concurrency or timeout
  guarantees, session storage, supported Python versions, CI requirements, or relevant
  engineering policies, review whether this file remains accurate and complete. Do not update
  `AGENTS.md` mechanically for every PR; change it only when its guidance would otherwise become
  inaccurate or incomplete.

## 13. Completion checklist

Before reporting a task complete:

- [ ] Confirm the final diff contains only authorized files and behavior.
- [ ] Re-check public API signatures, defaults, exceptions, and compatibility impact.
- [ ] Add or update deterministic regression tests where behavior changed.
- [ ] Run focused tests and the applicable Ruff/Mypy/full-suite checks.
- [ ] Verify async cancellation, rollback, cleanup, and session isolation when affected.
- [ ] Verify disk and `:memory:` SQLite behavior and legacy compatibility when affected.
- [ ] Update English and relevant Spanish documentation when user-facing behavior changed.
- [ ] Check examples and referenced paths against real public APIs.
- [ ] Report commands, pass/fail counts, skips, warnings, and checks not run honestly.
- [ ] Confirm no secrets, unrelated edits, remote operations, or publication actions occurred.

