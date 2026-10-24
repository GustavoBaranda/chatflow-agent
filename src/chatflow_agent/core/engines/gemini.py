"""Gemini LLM engine for chatflow-agent, using the Gemini Developer API (v1beta) over httpx."""

import asyncio
import contextvars
import dataclasses
import datetime
import email.utils
import http.cookiejar
import json
import logging
import math
import os
import random
import re
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, quote_plus, urlsplit

import httpx
from pydantic import BaseModel

from chatflow_agent.core.engines.base import BaseEngine, EngineTurnResult, system_prompt
from chatflow_agent.core.engines.openai_compatible import _body_excerpt
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role, ToolCall

logger = logging.getLogger("chatflow_agent.engine.gemini")

if TYPE_CHECKING:
    from chatflow_agent.core.agent import Agent

_API_VERSION = "v1beta"
# Module-level so tests can point the engine at a local server.
_API_BASE_URL = f"https://generativelanguage.googleapis.com/{_API_VERSION}"
_RETRIABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# The key travels in an HTTP header (or the URL), so only visible ASCII characters are valid.
_API_KEY_PATTERN = re.compile(r"[\x21-\x7e]+")
# A model ID such as "gemini-2.5-flash"; "models/" or "tunedModels/" may precede it.
_MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _new_attempt_client(timeout: float) -> httpx.AsyncClient:
    """Create the client used by a single HTTP attempt (replaced in tests)."""
    return httpx.AsyncClient(timeout=timeout)


def _new_pooled_client() -> httpx.AsyncClient:
    """Create a Runner's pooled client (replaced in tests). Timeouts are set per request.

    The client is shared by engines with different API keys, so it keeps no cookies: one
    set on a response (e.g. by a gateway) must not be sent with another engine's request.
    """
    no_cookies = http.cookiejar.CookieJar(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
    # Every Gemini request on the loop shares this client, so httpx's default cap of 100
    # connections would make request 101+ queue for a connection within its own timeout.
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=20)
    return httpx.AsyncClient(cookies=no_cookies, limits=limits)


async def _complete_despite_cancellation(cleanup: Coroutine[Any, Any, None]) -> None:
    """Run ``cleanup`` to completion even if the caller is cancelled, once or repeatedly.

    The cleanup runs in its own task, awaited through ``asyncio.shield`` until it is done; a
    cancellation received meanwhile is re-raised afterwards. Otherwise a second cancellation
    arriving during ``AsyncClient.aclose()`` would abandon the remaining open connections.
    """
    task = asyncio.ensure_future(cleanup)
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as err:
            cancellation = err
        except BaseException:
            if cancellation is None:
                raise
            # A cancellation received earlier takes precedence over the cleanup's own error
            # (shield has already marked that error as retrieved).
    if cancellation is not None:
        raise cancellation
    task.result()


class _ClientPool:
    """The client a Runner owns for one event loop while ``async with runner:`` is open.

    It is lent to every GeminiEngine HTTP attempt the Runner makes on that loop, so their
    connections are reused. Closing stops lending at once, waits for the attempts that
    already use the client, then closes it. Only the loop that created the pool may use it.
    """

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._client: httpx.AsyncClient | None = None
        self._in_use = 0
        self._closing = False
        self._idle = asyncio.Event()
        self._idle.set()

    def acquire(self) -> httpx.AsyncClient | None:
        """Lend the client, creating it on first use; None once closing has started."""
        if self._closing:
            return None
        if self._client is None:
            self._client = _new_pooled_client()
        self._in_use += 1
        self._idle.clear()
        return self._client

    def release(self) -> None:
        """Return the client lent by ``acquire``."""
        self._in_use -= 1
        if self._in_use == 0:
            self._idle.set()

    def stop_lending(self) -> None:
        """Make later ``acquire`` calls return None. Synchronous, so it precedes any await."""
        self._closing = True

    async def aclose(self) -> None:
        """Stop lending, wait for attempts using the client, then close it.

        The client is closed even if this wait is cancelled; attempts still using it then
        fail with an httpx error instead of keeping a connection open. Closing itself runs
        to completion through repeated cancellation, which is re-raised afterwards.
        """
        self.stop_lending()
        try:
            await self._idle.wait()
        finally:
            if self._client is not None:
                await _complete_despite_cancellation(self._client.aclose())


# The pool lent by the Runner for the engine call that is running. Set around each call in
# the turn's own task; copies of it seen by other loops (e.g. a thread running asyncio.run)
# are ignored by the loop check in GeminiEngine._post.
_lent_pool: contextvars.ContextVar[_ClientPool | None] = contextvars.ContextVar(
    "chatflow_gemini_client_pool", default=None
)


class _AttemptTrace:
    """Records httpcore's ``trace`` events for one HTTP attempt.

    They show whether an attempt that ran out of time was still opening a new connection
    (TCP connect or TLS handshake) and had not started sending the request. Transports that
    emit no events, such as ``httpx.MockTransport``, give no such evidence.
    """

    def __init__(self) -> None:
        self.connecting = False
        self.sending = False

    async def __call__(self, event: str, info: dict[str, Any]) -> None:
        if event.startswith("connection."):
            self.connecting = True
        elif event.endswith(".send_request_headers.started"):
            self.sending = True

    @property
    def request_unsent(self) -> bool:
        return self.connecting and not self.sending


def _client_creation_failed(err: Exception) -> ProviderError:
    """The error for an httpx client that could not be created from the environment.

    Only the exception type is given: the original message may contain a proxy URL with
    credentials or a certificate path.
    """
    return ProviderError(
        f"Gemini API HTTP client could not be created ({type(err).__name__}). Check the proxy "
        "and TLS environment variables (HTTPS_PROXY, HTTP_PROXY, ALL_PROXY, SSL_CERT_FILE, "
        "SSL_CERT_DIR)."
    )


def _describe(err: BaseException) -> str:
    """Error text for messages; some exceptions (e.g. timeouts) have an empty str()."""
    return str(err) or type(err).__name__


def _clean_key(value: str | None) -> str | None:
    """Strip surrounding whitespace, such as the newline a secret file or CRLF .env leaves."""
    return (value or "").strip() or None


def _model_path(model: str) -> str:
    """Resource path of a model name, resolved as google-genai did, after validating it."""
    if model.startswith("tunedModels/"):
        collection, model_id = "tunedModels", model[len("tunedModels/"):]
    else:
        collection, model_id = "models", model.removeprefix("models/")
    if ".." in model_id or not _MODEL_ID_PATTERN.fullmatch(model_id):
        raise ProviderError(
            f"Invalid Gemini model name {model!r}: expected an ID such as 'gemini-2.5-flash', "
            "optionally prefixed with 'models/' or 'tunedModels/'."
        )
    return f"{collection}/{model_id}"


def _api_base_url() -> str:
    """Versioned API root. GOOGLE_GEMINI_BASE_URL replaces the host root, as in google-genai."""
    override = os.environ.get("GOOGLE_GEMINI_BASE_URL", "").strip()
    if not override:
        return _API_BASE_URL
    try:
        parts = urlsplit(override)
        valid = (
            parts.scheme in ("http", "https")
            and bool(parts.netloc)
            and not parts.query
            and not parts.fragment
            and all(c.isprintable() and not c.isspace() for c in override)
        )
    except ValueError:
        valid = False
    if not valid:
        # The value is not echoed: it may embed credentials.
        raise ProviderError("GOOGLE_GEMINI_BASE_URL is not a valid http(s) base URL.")
    return f"{override.rstrip('/')}/{_API_VERSION}"


def _retry_after_seconds(value: str) -> float:
    """Seconds requested by a Retry-After header: a number, or an HTTP-date (a past date is 0).
    Digits too many for a float (RFC 9110 delay-seconds has no limit) give inf; an unusable
    value gives NaN."""
    value = value.strip()
    if re.fullmatch(r"[0-9]+", value):
        return float(value)
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        # Other numbers, such as 2.5, are accepted too, but not inf, nan or 1e999.
        return seconds if math.isfinite(seconds) else math.nan
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return max(0.0, (when - datetime.datetime.now(datetime.timezone.utc)).total_seconds())


def _json_default(value: Any) -> Any:
    """Encode what json cannot: structurally where possible, as google-genai did, else as text."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    return str(value)


class GeminiEngine(BaseEngine):
    """Coordinates API requests with Google Gemini through the generateContent REST endpoint.

    Only the Gemini Developer API (API key) is supported; Vertex AI is not.

    The API key is sent in the ``x-goog-api-key`` header. Pass ``api_key_in_query=True`` to
    send it as the ``?key=`` query parameter instead; the key is then part of the request URL,
    which httpx logs at INFO level, so the header is preferred.

    An ``http_client`` passed in belongs to the caller: it is used for every request and never
    closed. Otherwise, inside ``async with runner:`` the Runner lends its pooled client to the
    engine's requests, and outside it each HTTP attempt uses its own client, closed when the
    attempt ends.

    Like google-genai, the ``GOOGLE_GEMINI_BASE_URL`` environment variable replaces the API
    host root (e.g. for a proxy); ``/v1beta`` is appended to it.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        http_client: httpx.AsyncClient | None = None,
        *,
        api_key_in_query: bool = False,
    ) -> None:
        # google-genai also read GOOGLE_API_KEY when GEMINI_API_KEY was unset.
        self.api_key = (
            _clean_key(api_key)
            or _clean_key(os.environ.get("GEMINI_API_KEY"))
            or _clean_key(os.environ.get("GOOGLE_API_KEY"))
        )
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.api_key_in_query = api_key_in_query
        self._external_client = http_client

    def _format_history_to_contents(self, history: list[Message]) -> list[dict[str, Any]]:
        """Convert chatflow Message objects into Gemini ``contents`` entries."""
        contents: list[dict[str, Any]] = []

        for msg in history:
            parts: list[dict[str, Any]] = []

            # 1. User text
            if msg.role == Role.USER and msg.content:
                parts.append({"text": msg.content})
                contents.append({"role": "user", "parts": parts})

            # 2. Model text or tool calls
            elif msg.role == Role.MODEL:
                if msg.content:
                    parts.append({"text": msg.content})
                if msg.tool_calls:
                    for call in msg.tool_calls:
                        function_call = {"name": call.name, "args": call.args}
                        parts.append({"functionCall": _with_provider_id(function_call, call.id, call.name)})
                if parts:
                    contents.append({"role": "model", "parts": parts})

            # 3. Tool results returned back to model
            elif msg.role == Role.TOOL and msg.tool_results:
                for res in msg.tool_results:
                    payload = res.content if isinstance(res.content, dict) else {"output": res.content}
                    function_response = {"name": res.name, "response": payload}
                    parts.append(
                        {"functionResponse": _with_provider_id(function_response, res.tool_call_id, res.name)}
                    )
                if parts:
                    contents.append({"role": "user", "parts": parts})

        return contents

    def _build_request_body(self, agent: "Agent", history: list[Message]) -> dict[str, Any]:
        """Build the generateContent request body for the active agent."""
        body: dict[str, Any] = {"contents": self._format_history_to_contents(history)}

        instruction = system_prompt(agent.get_instructions(), history)
        if instruction is not None:
            body["systemInstruction"] = {"parts": [{"text": instruction}]}

        all_tools = agent.get_all_tools()
        if all_tools:
            body["tools"] = [
                {"functionDeclarations": [t.to_gemini_declaration() for t in all_tools]}
            ]

        return body

    def _redact(self, text: str) -> str:
        """Remove the API key from text that ends up in errors or logs.

        Errors may quote the key as is, escaped (e.g. a bytes repr of a header value) or
        percent-encoded (a URL), so every one of those forms is replaced.
        """
        key = self.api_key
        if not key:
            return text
        forms = {
            key,
            key.encode("unicode_escape").decode("ascii"),
            ascii(key)[1:-1],
            quote(key, safe=""),
            quote_plus(key, safe=""),
        }
        for form in sorted(forms, key=len, reverse=True):
            text = text.replace(form, "***")
        return text

    def _retry_delay(self, retries: int, retry_after: str | None = None) -> float:
        """Exponential backoff with jitter; a valid Retry-After (seconds or an HTTP-date)
        replaces the base. The delay is capped at max_delay either way, however long the server
        asks to wait."""
        base: float = self.base_delay * (2 ** retries)
        if retry_after:
            parsed = _retry_after_seconds(retry_after)
            if parsed >= 0:  # NaN (unusable) is not
                # Capped before the jitter, which would otherwise turn inf into nan.
                base = min(parsed, self.max_delay)
        return min(self.max_delay, base + random.uniform(0.0, 0.25 * base))

    async def _post(
        self,
        url: str,
        content: bytes,
        headers: dict[str, str],
        params: dict[str, str] | None,
        trace: _AttemptTrace,
    ) -> httpx.Response:
        """Send one HTTP attempt with the injected client, the Runner's lent pool, or a client
        of its own that is closed however the attempt ends.

        Redirects are never followed, even by an injected client configured to: httpx keeps
        x-goog-api-key on a cross-origin redirect.
        """
        extensions = {"trace": trace}
        if self._external_client is not None:
            return await self._external_client.post(
                url,
                content=content,
                headers=headers,
                params=params,
                follow_redirects=False,
                extensions=extensions,
            )
        # A client that cannot be created raises after the except block, so the original error
        # (which may quote a proxy URL or a certificate path) is not kept as __context__ either.
        failure: ProviderError | None = None
        pool = _lent_pool.get()
        if pool is not None and pool.loop is asyncio.get_running_loop():
            try:
                pooled = pool.acquire()
            except Exception as err:
                failure = _client_creation_failed(err)
            if failure is not None:
                raise failure from None
            if pooled is not None:
                try:
                    return await pooled.post(
                        url,
                        content=content,
                        headers=headers,
                        params=params,
                        timeout=self.timeout_seconds,
                        follow_redirects=False,
                        extensions=extensions,
                    )
                finally:
                    pool.release()
        try:
            client = _new_attempt_client(self.timeout_seconds)
        except Exception as err:
            failure = _client_creation_failed(err)
        if failure is not None:
            raise failure from None
        async with client:
            return await client.post(
                url,
                content=content,
                headers=headers,
                params=params,
                follow_redirects=False,
                extensions=extensions,
            )

    async def generate_turn_async(
        self,
        agent: "Agent",
        history: list[Message],
    ) -> EngineTurnResult:
        """Call Gemini API asynchronously with active agent's tools and system instructions."""
        api_key = self.api_key
        if not api_key:
            raise ProviderError(
                "Gemini API client is not configured. Set GEMINI_API_KEY environment variable "
                "or pass api_key to Runner."
            )
        if not _API_KEY_PATTERN.fullmatch(api_key):
            # Checked before any request: httpx would quote the invalid value in its error.
            raise ProviderError(
                "Gemini API key is invalid: it contains spaces, control characters or "
                "non-ASCII characters. Check GEMINI_API_KEY or the api_key passed to Runner."
            )

        url = f"{_api_base_url()}/{_model_path(agent.model)}:generateContent"
        body = self._build_request_body(agent, history)
        try:
            # Tool results that json cannot encode (pydantic models, dataclasses, datetimes,
            # ...) are converted by _json_default instead of failing the turn.
            content = json.dumps(body, default=_json_default).encode("utf-8")
        except Exception as err:
            # Only the exception type is given: the message may quote tool results.
            raise ProviderError(
                f"Gemini API request body could not be encoded as JSON ({type(err).__name__}): "
                "a tool result or tool call argument contains a circular reference or a "
                "dict key that is not a string, number, boolean or None."
            ) from None
        headers = {"Content-Type": "application/json"}
        params: dict[str, str] | None = None
        if self.api_key_in_query:
            params = {"key": api_key}
        else:
            headers["x-goog-api-key"] = api_key

        retries = 0
        while True:
            attempt = _AttemptTrace()
            # The original exception, the httpcore exception chained to it and its request may
            # all hold the key. The error is therefore raised after the except block, where it
            # gets no __context__, and "from None", so it has no __cause__ either.
            failure: ProviderError | None = None
            try:
                # timeout_seconds bounds the whole attempt, as with the previous SDK client.
                resp = await asyncio.wait_for(
                    self._post(url, content, headers, params, attempt),
                    timeout=self.timeout_seconds,
                )
            except (httpx.ConnectError, httpx.TimeoutException, asyncio.TimeoutError) as err:
                # Only connection failures are retried: the request never reached Gemini. Other
                # timeouts are not, since it may have; that includes an attempt that ran out of
                # time, unless it was still opening a connection.
                connection_failed = isinstance(err, (httpx.ConnectError, httpx.ConnectTimeout)) or (
                    isinstance(err, asyncio.TimeoutError) and attempt.request_unsent
                )
                reason = self._redact(_describe(err))
                if connection_failed and retries < self.max_retries:
                    delay = self._retry_delay(retries)
                    logger.warning(
                        f"Gemini API transient error ({reason}). Retrying in {delay:.2f}s (attempt {retries + 1}/{self.max_retries})..."
                    )
                    await asyncio.sleep(delay)
                    retries += 1
                    continue
                failure = ProviderError(f"Gemini API request failed ({type(err).__name__}): {reason}")
            except httpx.RequestError as err:
                failure = ProviderError(
                    f"Gemini API connection error ({type(err).__name__}): {self._redact(_describe(err))}"
                )
            except httpx.InvalidURL as err:
                failure = ProviderError(
                    f"Gemini API request URL is invalid: {self._redact(_describe(err))}"
                )
            except RuntimeError as err:
                # httpx raises RuntimeError for a closed client or one bound to a closed loop.
                if self._external_client is None:
                    raise
                failure = ProviderError(
                    "Gemini API request failed: the provided http_client is closed or bound to "
                    f"another event loop ({self._redact(_describe(err))})"
                )
            if failure is not None:
                raise failure from None

            if resp.status_code in _RETRIABLE_STATUS_CODES and retries < self.max_retries:
                delay = self._retry_delay(retries, resp.headers.get("Retry-After"))
                logger.warning(
                    f"Gemini API returned HTTP {resp.status_code}. Retrying in {delay:.2f}s (attempt {retries + 1}/{self.max_retries})..."
                )
                await asyncio.sleep(delay)
                retries += 1
                continue
            if not resp.is_success:
                # Includes 1xx/3xx: a redirect (e.g. from a proxy) is not a Gemini response.
                # Redact before truncating, so a key cut at the limit cannot leak as a fragment.
                raise ProviderError(
                    f"Gemini API request failed ({resp.status_code}): "
                    f"{_body_excerpt(self._redact(resp.text))}"
                )
            break

        try:
            data = resp.json()
        except ValueError as err:
            raise ProviderError("Gemini API returned a response body that is not valid JSON.") from err
        if not isinstance(data, dict):
            raise ProviderError("Gemini API returned a malformed response: expected a JSON object.")

        return self._parse_response(data)

    def _parse_response(self, data: dict[str, Any]) -> EngineTurnResult:
        """Extract text and tool calls from the first candidate of a generateContent response."""
        feedback = data.get("promptFeedback")
        if feedback is not None and not isinstance(feedback, dict):
            raise _malformed("'promptFeedback' is not an object")
        if feedback and feedback.get("blockReason"):
            raise ProviderError(f"Gemini blocked the prompt (blockReason: {feedback['blockReason']}).")

        extracted_text: str | None = None
        extracted_calls: list[ToolCall] = []

        candidates = _optional_list(data, "candidates")
        if candidates:
            candidate = candidates[0]
            if not isinstance(candidate, dict):
                raise _malformed("a candidate is not an object")
            content = candidate.get("content")
            if content is not None and not isinstance(content, dict):
                raise _malformed("'content' is not an object")
            parts = _optional_list(content or {}, "parts")
            text_chunks: list[str] = []
            for idx, part in enumerate(parts):
                if not isinstance(part, dict):
                    raise _malformed("a part is not an object")
                text = part.get("text")
                if text:
                    if not isinstance(text, str):
                        raise _malformed("'text' is not a string")
                    text_chunks.append(text)
                function_call = part.get("functionCall")
                if function_call:
                    extracted_calls.append(_parse_function_call(function_call, idx))
            if text_chunks:
                extracted_text = "".join(text_chunks)

        return EngineTurnResult(
            text=extracted_text,
            tool_calls=extracted_calls,
            raw_response=data,
        )


def _malformed(detail: str) -> ProviderError:
    return ProviderError(f"Gemini API returned a malformed response: {detail}.")


def _optional_list(obj: dict[str, Any], key: str) -> list[Any]:
    """Return ``obj[key]`` as a list; a missing or null value is an empty list."""
    value = obj.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise _malformed(f"'{key}' is not a list")
    return value


def _parse_function_call(function_call: Any, part_idx: int) -> ToolCall:
    """Convert a ``functionCall`` part into a ToolCall. Gemini's ``id`` is kept; without one,
    the ID is synthetic, in the engine's historical format."""
    if not isinstance(function_call, dict):
        raise _malformed("'functionCall' is not an object")
    name = function_call.get("name")
    if name is not None and not isinstance(name, str):
        raise _malformed("'functionCall.name' is not a string")
    args = function_call.get("args")
    if args is not None and not isinstance(args, dict):
        raise _malformed("'functionCall.args' is not an object")
    call_id = function_call.get("id")
    if call_id is not None and not isinstance(call_id, str):
        raise _malformed("'functionCall.id' is not a string")
    return ToolCall(
        id=call_id or f"call_{name}_{part_idx}",
        name=name or "",
        args=dict(args or {}),
    )


def _with_provider_id(part: dict[str, Any], call_id: str, name: str) -> dict[str, Any]:
    """Add ``id`` to a functionCall/functionResponse part, unless the ID is a synthetic one."""
    if not re.fullmatch(rf"call_{re.escape(name)}_\d+", call_id):
        part["id"] = call_id
    return part
