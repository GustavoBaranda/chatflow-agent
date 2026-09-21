"""Production Telegram Bot with multi-agent handoffs and SQLite persistence.

Features:
- Continuous session memory across bot restarts via SQLiteSessionStore.
- Preserves active specialist agent across restarts (user remains with Billing or Support).
- Standard python-telegram-bot integration with zero emoji clutter.
"""

import os
import sys

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from chatflow_agent import (
    Agent,
    Runner,
    SQLiteSessionStore,
    TelegramChannel,
)

# 1. Specialist: Technical Support
support_agent = Agent(
    name="Technical Support",
    instructions=(
        "You assist users with technical incidents, API connectivity, and status checks. "
        "Be analytical and brief."
    ),
)

@support_agent.tool
def get_service_status(service_name: str) -> dict:
    """Check service uptime and operational status."""
    return {"service": service_name, "status": "Operational", "latency_ms": 18}


# 2. Specialist: Billing & Subscriptions
billing_agent = Agent(
    name="Billing Specialist",
    instructions=(
        "You handle subscription inquiries, plan changes, and payment queries."
    ),
)

@billing_agent.tool
def get_account_plan(account_id: str) -> dict:
    """Retrieve subscription tier and renewal date."""
    return {
        "account_id": account_id,
        "plan": "Enterprise Scale",
        "renewal_date": "2026-12-31",
        "active_seats": 25,
    }


# 3. Frontline Concierge
triage_agent = Agent(
    name="Telegram Concierge",
    instructions=(
        "You are the frontline assistant. Greet users and identify their requirements. "
        "Route technical questions to Technical Support. "
        "Route subscription and payment queries to Billing Specialist."
    ),
    handoffs=[support_agent, billing_agent],
)

# 4. Persistent Runner
session_store = SQLiteSessionStore(
    db_path="telegram_sessions.db",
    default_max_turns=25,
)

runner = Runner(
    starting_agent=triage_agent,
    session_store=session_store,
)

# 5. Telegram Channel
telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
channel = TelegramChannel(
    token=telegram_token,
    start_message="Welcome. How can we assist you today?",
)
channel.attach(runner)

if __name__ == "__main__":
    print("Starting Telegram bot polling with SQLite persistence...")
    print("Database: telegram_sessions.db")
    channel.run()
