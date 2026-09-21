"""Production REST Webhook and API Service for web and mobile frontends.

Provides a clean JSON API endpoint compatible with any frontend (React, Next.js, Vue, mobile apps).
Uses native SQLiteSessionStore for cross-request session persistence and autonomous multi-agent handoffs.
"""

import os
import sys

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from chatflow_agent import (
    Agent,
    Runner,
    SQLiteSessionStore,
)

# 1. Specialist Agents
support_agent = Agent(
    name="Support Specialist",
    instructions="You resolve technical issues, troubleshooting, and bug inquiries.",
)

sales_agent = Agent(
    name="Sales Specialist",
    instructions="You provide enterprise pricing, feature walkthroughs, and custom quotes.",
)

@sales_agent.tool
def get_tier_pricing(seats: int) -> dict:
    """Calculate pricing tier according to required team seats."""
    tier = "Enterprise" if seats >= 50 else "Professional"
    price_per_seat = 20 if seats >= 50 else 25
    return {"tier": tier, "price_per_seat": price_per_seat, "monthly_total": seats * price_per_seat}


# 2. Frontline Receptionist
triage_agent = Agent(
    name="Frontline API Receptionist",
    instructions=(
        "You are the main digital receptionist. Understand incoming user requests. "
        "Hand off to Support Specialist for technical issues, or Sales Specialist for quotes and pricing."
    ),
    handoffs=[support_agent, sales_agent],
)

# 3. Persistent Runner with SQLite
session_store = SQLiteSessionStore(
    db_path="api_sessions.db",
    default_max_turns=30,
)

runner = Runner(
    starting_agent=triage_agent,
    session_store=session_store,
)


# 4. HTTP Route Handlers
async def chat_endpoint(request: Request) -> JSONResponse:
    """Handle incoming message turn."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    session_id = data.get("session_id")
    message = data.get("message")
    metadata = data.get("metadata", {})

    if not session_id or not message:
        return JSONResponse(
            {"error": "Fields 'session_id' and 'message' are required"},
            status_code=422,
        )

    response = await runner.run_async(
        session_id=str(session_id),
        user_message=str(message),
        metadata=metadata,
    )

    return JSONResponse({
        "session_id": session_id,
        "content": response.content,
        "active_agent": response.active_agent_name,
        "tool_calls": [tc.name for tc in response.tool_calls],
        "handoff": response.handoff.target_agent if response.handoff else None,
    })


async def get_session_history(request: Request) -> JSONResponse:
    """Retrieve persisted conversation history and active agent."""
    session_id = request.path_params["session_id"]
    session = runner.get_session(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    return JSONResponse({
        "session_id": session.session_id,
        "active_agent": session.active_agent.name,
        "messages_count": len(session.history),
        "history": [
            {"role": msg.role.value, "content": msg.content}
            for msg in session.history
        ],
        "metadata": session.metadata,
    })


async def health_check(request: Request) -> JSONResponse:
    """Liveness probe."""
    return JSONResponse({"status": "healthy", "service": "chatflow-agent-api"})


# 5. Starlette Application Assembly
routes = [
    Route("/api/chat", endpoint=chat_endpoint, methods=["POST"]),
    Route("/api/sessions/{session_id}", endpoint=get_session_history, methods=["GET"]),
    Route("/health", endpoint=health_check, methods=["GET"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(routes=routes, middleware=middleware)

if __name__ == "__main__":
    print("Starting Webhook API Service on http://0.0.0.0:8000...")
    print("POST /api/chat - Execute agent turn")
    print("GET  /api/sessions/{session_id} - Inspect persisted session")
    print("GET  /health - Liveness check")
    uvicorn.run(app, host="0.0.0.0", port=8000)
