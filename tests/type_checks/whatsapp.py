"""Static type checks for ``WhatsAppChannel.llm_timeout_seconds`` (#200).

Checked in CI with ``python -m mypy --strict tests/type_checks``; pytest does
not collect this module. ``llm_timeout_seconds`` must accept ``float | None``
and keep the attribute typed as ``float | None``. If it regresses to ``float``
or ``Any``, the ``type: ignore[...]`` checks below fail under ``--strict``.
"""

from chatflow_agent import WhatsAppChannel

ch_none = WhatsAppChannel(
    verify_token="tok",
    verify_signature=False,
    llm_timeout_seconds=None,
)
ch_float = WhatsAppChannel(
    verify_token="tok",
    verify_signature=False,
    llm_timeout_seconds=30.0,
)
ch_int = WhatsAppChannel(
    verify_token="tok",
    verify_signature=False,
    llm_timeout_seconds=30,
)
ch_default = WhatsAppChannel(
    verify_token="tok",
    verify_signature=False,
)

timeout_from_none: float | None = ch_none.llm_timeout_seconds
timeout_from_float: float | None = ch_float.llm_timeout_seconds
timeout_from_int: float | None = ch_int.llm_timeout_seconds
timeout_from_default: float | None = ch_default.llm_timeout_seconds

# Negative check before reassignment so mypy does not narrow the attribute to float.
_narrow_float: float = ch_default.llm_timeout_seconds  # type: ignore[assignment]
_bad_type = WhatsAppChannel(
    verify_token="tok",
    verify_signature=False,
    llm_timeout_seconds="120",  # type: ignore[arg-type]
)

# Reassigning both None and float must type-check.
ch_default.llm_timeout_seconds = None
ch_default.llm_timeout_seconds = 60.0
