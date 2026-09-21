"""WhatsApp Webhook Service example for chatflow-agent using Meta Cloud API."""

import os
import sys

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from chatflow_agent import Agent, Runner, WhatsAppChannel

# 1. Specialist Agent: Orders and Deliveries
order_agent = Agent(
    name="Order Tracking Agent",
    instructions="You provide updates on order dispatch, tracking numbers, and delivery ETAs.",
)

@order_agent.tool
def track_order(order_id: str) -> dict:
    """Retrieve tracking status and ETA for an order ID."""
    return {
        "order_id": order_id,
        "status": "In Transit",
        "carrier": "Express Logistics",
        "eta": "Today by 6:00 PM",
    }


# 2. Specialist Agent: Human Escalation / General Support
support_agent = Agent(
    name="Customer Support",
    instructions="You assist with returns, store hours, and warranty questions.",
)

@support_agent.tool
def get_store_info() -> dict:
    """Return physical store address and opening hours."""
    return {"hours": "Mon-Fri 9AM - 8PM", "address": "Av. Central 1234"}


# 3. Frontline Triage Agent
triage_agent = Agent(
    name="WhatsApp Frontline",
    instructions=(
        "You are the 24/7 WhatsApp concierge. Greet warmly. Route order inquiries to "
        "Order Tracking Agent, or store/general questions to Customer Support."
    ),
    handoffs=[order_agent, support_agent],
)

# 4. Attach runner to WhatsApp Channel
runner = Runner(starting_agent=triage_agent)

# Configuration from environment variables
whatsapp = WhatsAppChannel(
    verify_token=os.environ.get("WHATSAPP_VERIFY_TOKEN", "my_secure_webhook_token"),
    access_token=os.environ.get("WHATSAPP_ACCESS_TOKEN"),
    app_secret=os.environ.get("WHATSAPP_APP_SECRET"),  # Required in production
    phone_number_id=os.environ.get("WHATSAPP_PHONE_ID"),
)
whatsapp.attach(runner)

if __name__ == "__main__":
    print("Starting WhatsApp webhook listener on http://localhost:8000...")
    print("Meta Webhook Callback URL: http://<your-public-domain>/webhook")
    whatsapp.run(host="0.0.0.0", port=8000)
