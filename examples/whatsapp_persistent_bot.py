"""Production WhatsApp Bot with Meta Cloud API and SQLite persistence.

Features:
- Omnichannel SQLite persistence across server restarts.
- Multi-agent handoffs between Frontline Concierge, Order Tracker, and Billing.
- Concurrency locks per phone number and anti-500 webhook shield.
"""

import os
import sys

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from chatflow_agent import (
    Agent,
    Runner,
    SQLiteSessionStore,
    WhatsAppChannel,
)

# 1. Specialist Agent: Order Tracking
order_agent = Agent(
    name="Order Specialist",
    instructions=(
        "You assist customers with order status, tracking numbers, and delivery ETAs. "
        "Keep responses direct, professional, and concise."
    ),
)

@order_agent.tool
def get_order_tracking(order_id: str) -> dict:
    """Look up delivery status and estimated arrival time for an order."""
    return {
        "order_id": order_id,
        "status": "In Transit",
        "carrier": "Express Freight",
        "eta": "Next business day by 4:00 PM",
    }


# 2. Specialist Agent: Billing and Invoices
billing_agent = Agent(
    name="Billing Specialist",
    instructions=(
        "You assist customers with payment confirmation, invoices, and billing issues. "
        "Be courteous and precise."
    ),
)

@billing_agent.tool
def lookup_invoice(invoice_id: str) -> dict:
    """Retrieve billing status and balance for an invoice."""
    return {
        "invoice_id": invoice_id,
        "status": "Settled",
        "amount_usd": 85.50,
        "receipt_url": f"https://billing.example.com/receipts/{invoice_id}.pdf",
    }


# 3. Frontline Concierge Agent
concierge_agent = Agent(
    name="WhatsApp Concierge",
    instructions=(
        "You are the frontline assistant on WhatsApp. Greet users warmly and identify their need. "
        "Hand off delivery and shipping inquiries to Order Specialist. "
        "Hand off invoices, receipts, and payment queries to Billing Specialist."
    ),
    handoffs=[order_agent, billing_agent],
)

# 4. Initialize persistent SQLite session store and Runner
# Note: SQLite with WAL mode is designed for single-process deployments (1 Uvicorn worker).
# All conversation history and active agent states persist across restarts.
session_store = SQLiteSessionStore(
    db_path="whatsapp_sessions.db",
    default_max_turns=20,
)

runner = Runner(
    starting_agent=concierge_agent,
    session_store=session_store,
)

# 5. WhatsApp Channel configuration
whatsapp = WhatsAppChannel(
    verify_token=os.environ.get("WHATSAPP_VERIFY_TOKEN", "secure_verification_token"),
    access_token=os.environ.get("WHATSAPP_ACCESS_TOKEN"),
    app_secret=os.environ.get("WHATSAPP_APP_SECRET"),  # Required in production
    phone_number_id=os.environ.get("WHATSAPP_PHONE_ID"),
)
whatsapp.attach(runner)

if __name__ == "__main__":
    print("Starting WhatsApp webhook service on port 8000...")
    print("Database: whatsapp_sessions.db (SQLite)")
    print("Webhook endpoint: POST /webhook")
    print("Verification endpoint: GET /webhook")
    whatsapp.run(host="0.0.0.0", port=8000)
