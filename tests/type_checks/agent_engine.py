"""Static type checks for ``Agent.engine`` (#220).

Checked in CI with ``python -m mypy --strict tests/type_checks``; pytest does
not collect this module. ``engine`` must accept ``BaseEngine | None`` and keep
the attribute typed as ``BaseEngine | None``. If either regresses to ``Any``,
the ``type: ignore[...]`` comments below become unused and ``--strict`` fails.
"""

from chatflow_agent import Agent, BaseEngine, GeminiEngine

with_engine = Agent(name="Bot", instructions="x", engine=GeminiEngine())
with_none = Agent(name="Bot", instructions="x", engine=None)
with_default = Agent(name="Bot", instructions="x")

engine_attr: BaseEngine | None = with_engine.engine

invalid_engine = Agent(name="Bot", instructions="x", engine="invalid")  # type: ignore[arg-type]
if with_engine.engine is not None:
    invalid_attr: str = with_engine.engine  # type: ignore[assignment]
