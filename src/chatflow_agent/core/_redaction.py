"""Internal helpers for redacting credentials from error messages, response bodies, and logs."""

import json
import re
from collections.abc import Iterable
from urllib.parse import parse_qsl, quote, quote_plus, unquote, urlsplit

from chatflow_agent.exceptions import ProviderError

# Visible ASCII only: headers must not contain whitespace, control characters, or non-ASCII.
_HEADER_SECRET_PATTERN = re.compile(r"[\x21-\x7e]+")

# Matches scheme://userinfo@host in free-form error text (e.g. httpx/proxy error strings).
_URL_USERINFO_PATTERN = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^\s/@]+@)")

# Splits a percent-encoded string into literal segments and %XX escape tokens.
_PERCENT_ESCAPE_SPLIT = re.compile(r"(%[0-9A-Fa-f]{2})")

# Query parameter names that commonly carry credentials or session secrets.
_SENSITIVE_QUERY_KEY_PATTERN = re.compile(
    r"(?:key|token|secret|auth|pass|pwd|cred|sig|signature|jwt|bearer|session|access|private)",
    re.IGNORECASE,
)

# Common non-secret ISO date / API-version values (e.g. Azure OpenAI ?api-version=2024-02-01-preview).
# Strictly bounded to known release tags so arbitrary secret suffixes (e.g. 2024-01-01-SECRET...) never match.
_ISO_DATE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:-(?:preview|alpha|beta|ga|stable|rc\d{0,2}))?$",
    re.IGNORECASE,
)

# Well-known credential prefixes that should be redacted even under generic query param names.
_SECRET_VALUE_PREFIXES = ("sk-", "pk-", "rk-", "AIza", "EAAG", "ghp_", "gho_", "ghu_", "eyJ")


def _is_sensitive_query_param(key: str, value: str) -> bool:
    """Return True if a URL query parameter should have its value redacted globally."""
    if not value:
        return False
    if _SENSITIVE_QUERY_KEY_PATTERN.search(key):
        # Protect short tokens under sensitive names; 1-2 char values are still redacted as k=v.
        return len(value) >= 3
    if _ISO_DATE_PATTERN.fullmatch(value):
        return False
    if value.startswith(_SECRET_VALUE_PREFIXES):
        return True
    if len(value) >= 16:
        return True
    return False


def _compile_percent_pattern(encoded: str) -> re.Pattern[str]:
    """Match percent-encoded hex escapes case-insensitively while keeping literal chars exact."""
    parts = _PERCENT_ESCAPE_SPLIT.split(encoded)
    pattern = "".join(
        f"(?i:{re.escape(part)})" if len(part) == 3 and part.startswith("%") else re.escape(part)
        for part in parts
    )
    return re.compile(pattern)


def describe_error(err: BaseException) -> str:
    """Return non-empty error text; some exceptions (e.g. timeouts) have an empty str()."""
    return str(err) or type(err).__name__


def extract_url_secrets(url: str | None) -> list[str]:
    """Extract embedded userinfo credentials and sensitive query values from a URL for redaction."""
    if not url:
        return []
    try:
        parsed = urlsplit(url)
    except ValueError:
        return []

    secrets: list[str] = []
    if parsed.password:
        secrets.append(parsed.password)
        decoded_pw = unquote(parsed.password)
        if decoded_pw and decoded_pw != parsed.password:
            secrets.append(decoded_pw)
        if parsed.username:
            secrets.append(f"{parsed.username}:{parsed.password}")
    elif parsed.username:
        # Tokens are commonly passed as username-only userinfo (e.g. https://TOKEN@host).
        secrets.append(parsed.username)
        decoded_user = unquote(parsed.username)
        if decoded_user and decoded_user != parsed.username:
            secrets.append(decoded_user)
    if parsed.query:
        for key, value in parse_qsl(parsed.query, keep_blank_values=False):
            if _is_sensitive_query_param(key, value):
                secrets.append(value)
        for pair in parsed.query.split("&"):
            if "=" in pair:
                raw_key, raw_value = pair.split("=", 1)
                if not raw_value:
                    continue
                decoded_key = unquote(raw_key)
                decoded_value = unquote(raw_value)
                if _SENSITIVE_QUERY_KEY_PATTERN.search(decoded_key):
                    # Always redact the k=v pair (covers even 1-2 char tokens in URLs).
                    secrets.append(pair)
                    if len(decoded_value) >= 3:
                        secrets.append(raw_value)
                        if decoded_value != raw_value:
                            secrets.append(decoded_value)
                elif _is_sensitive_query_param(decoded_key, decoded_value):
                    secrets.append(raw_value)
                    if decoded_value != raw_value:
                        secrets.append(decoded_value)
    return secrets


def redact_secrets(text: str, secrets: Iterable[str | None]) -> str:
    """Replace all plain, escaped, JSON-encoded, and percent-encoded forms of secrets with ``***``.

    Also scrubs any ``scheme://userinfo@`` credentials embedded in URLs inside ``text``.
    """
    redacted = _URL_USERINFO_PATTERN.sub(r"\1***@", text)

    forms: set[str] = set()
    percent_encoded: set[str] = set()
    for secret in secrets:
        if not secret:
            continue
        forms.add(secret)
        forms.add(secret.encode("unicode_escape").decode("ascii"))
        forms.add(ascii(secret)[1:-1])
        json_ascii = json.dumps(secret)[1:-1]
        json_unicode = json.dumps(secret, ensure_ascii=False)[1:-1]
        for json_form in (json_ascii, json_unicode):
            forms.add(json_form)
            if "/" in json_form:
                forms.add(json_form.replace("/", r"\/"))
                forms.add(json_form.replace("/", r"\u002f"))
                forms.add(json_form.replace("/", r"\u002F"))
        for encoded in (
            quote(secret, safe=""),
            quote_plus(secret, safe=""),
            quote(secret),
            quote_plus(secret, safe="/"),
        ):
            forms.add(encoded)
            if "%" in encoded:
                percent_encoded.add(encoded)

    for form in sorted(forms, key=len, reverse=True):
        if form:
            redacted = redacted.replace(form, "***")
    for encoded in sorted(percent_encoded, key=len, reverse=True):
        redacted = _compile_percent_pattern(encoded).sub("***", redacted)
    return redacted


def body_excerpt(text: str | None, limit: int = 1000) -> str:
    """Return a bounded excerpt of an error response body."""
    body = text or ""
    return body if len(body) <= limit else body[:limit] + "… (truncated)"


def sanitize_error_body(
    text: str | None,
    secrets: Iterable[str | None],
    *,
    limit: int = 1000,
) -> str:
    """Redact credentials BEFORE truncating so a secret at the boundary cannot leak a fragment."""
    return body_excerpt(redact_secrets(text or "", secrets), limit=limit)


def is_valid_header_secret(secret: str) -> bool:
    """Return True if ``secret`` contains only visible ASCII characters safe for HTTP headers."""
    return bool(_HEADER_SECRET_PATTERN.fullmatch(secret))


def validate_header_secret(secret: str, *, label: str) -> None:
    """Raise ``ProviderError`` if a non-empty HTTP header credential contains unsafe characters.

    Checked before calling httpx so neither httpx nor ``UnicodeEncodeError.__repr__`` can echo
    the raw secret.
    """
    if secret and not is_valid_header_secret(secret):
        raise ProviderError(
            f"{label} is invalid: it contains spaces, control characters or non-ASCII characters."
        )
