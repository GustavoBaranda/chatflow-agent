"""Interactive CLI demo for multi-agent Swarm with autonomous handoffs."""

import os
import sys

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from chatflow_agent import Agent, CLIChannel, Runner

# 1. Specialist Agent: Billing
billing_agent = Agent(
    name="Billing Agent",
    instructions="You handle invoices, subscriptions, and payments. Always be professional and concise.",
)

@billing_agent.tool
def get_invoice_status(invoice_id: str) -> dict:
    """Retrieve billing status for an invoice ID."""
    return {"invoice_id": invoice_id, "status": "PAID", "amount_usd": 150.00}


# 2. Specialist Agent: Tech Support
tech_agent = Agent(
    name="Technical Support",
    instructions="You diagnose server infrastructure incidents and service status.",
)

@tech_agent.tool
def check_server_uptime(service_name: str) -> dict:
    """Check the uptime and health of an infrastructure service."""
    return {"service": service_name, "status": "ONLINE", "uptime_percentage": 99.98}


# 3. Root Frontline Triage Agent
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are the frontline assistant. Understand user intent and hand off "
        "to Billing Agent for payment/invoice questions, or Technical Support for infrastructure issues."
    ),
    handoffs=[billing_agent, tech_agent],
)

# 4. Initialize Runner and CLI Channel
runner = Runner(starting_agent=triage_agent)
cli = CLIChannel()
cli.attach(runner)

if __name__ == "__main__":
    # To run with live Gemini API, ensure GEMINI_API_KEY environment variable is set.
    cli.run()
