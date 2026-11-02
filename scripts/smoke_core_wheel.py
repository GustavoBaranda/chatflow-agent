#!/usr/bin/env python3
"""Smoke-test a core-only chatflow-agent install (no optional extras).

Used by CI after building and installing the wheel without extras, and runnable
locally the same way. Exit 0 only when:

- ``import chatflow_agent`` works
- ``CLIChannel`` constructs
- ``TelegramChannel`` / ``WhatsAppChannel`` raise ``DependencyError``
"""

from __future__ import annotations

import sys


def main() -> int:
    import chatflow_agent
    from chatflow_agent import CLIChannel, TelegramChannel, WhatsAppChannel
    from chatflow_agent.exceptions import DependencyError

    if not hasattr(chatflow_agent, "Agent"):
        print("chatflow_agent imported but Agent is missing", file=sys.stderr)
        return 1

    CLIChannel()

    try:
        TelegramChannel(token="smoke-token")
    except DependencyError as err:
        if err.feature != "TelegramChannel":
            print(
                f"unexpected Telegram DependencyError feature: {err.feature!r}",
                file=sys.stderr,
            )
            return 1
    else:
        print(
            "TelegramChannel did not raise DependencyError without extras",
            file=sys.stderr,
        )
        return 1

    try:
        # Pass verify_signature=False so __init__ reaches the optional-dep probe
        # before the app_secret fail-closed ValueError.
        WhatsAppChannel(verify_token="smoke-token", verify_signature=False)
    except DependencyError as err:
        if err.feature != "WhatsAppChannel":
            print(
                f"unexpected WhatsApp DependencyError feature: {err.feature!r}",
                file=sys.stderr,
            )
            return 1
    else:
        print(
            "WhatsAppChannel did not raise DependencyError without extras",
            file=sys.stderr,
        )
        return 1

    print("core-only wheel smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
