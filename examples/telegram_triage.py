"""Telegram Bot multi-agent triage example for chatflow-agent."""

import os
import sys

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from chatflow_agent import Agent, Runner, TelegramChannel

# 1. Specialist Agent: Billing
billing_agent = Agent(
    name="Billing Agent",
    instructions="You handle payments, invoices, refunds, and subscription renewals.",
)

@billing_agent.tool
def get_invoice(invoice_id: str) -> dict:
    """Retrieve billing status for an invoice ID."""
    return {"invoice_id": invoice_id, "status": "PAID", "amount_usd": 120.00}


# 2. Specialist Agent: Tech Support
tech_agent = Agent(
    name="Technical Support",
    instructions="You assist users with technical outages, credentials, and connectivity.",
)

@tech_agent.tool
def test_connection(host: str) -> dict:
    """Check ping latency and connectivity to a remote host."""
    return {"host": host, "ping_ms": 24, "status": "REACHABLE"}


# 3. Frontline Triage Agent
triage_agent = Agent(
    name="Telegram Concierge",
    instructions=(
        "You are the frontline assistant on Telegram. Greet users warmly. "
        "Hand off billing questions to Billing Agent, and technical questions to Technical Support."
    ),
    handoffs=[billing_agent, tech_agent],
)

# 4. Initialize Runner and Telegram Channel
runner = Runner(starting_agent=triage_agent)

telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
channel = TelegramChannel(
    token=telegram_token,
    start_message="Welcome to ChatFlow Telegram Bot. How can we help you today?",
)
channel.attach(runner)

if __name__ == "__main__":
    print("Starting Telegram bot polling... Press Ctrl+C to stop.")
    channel.run()
