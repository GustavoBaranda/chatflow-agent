"""Example: Heterogeneous Multi-Provider Swarm with chatflow-agent.

Wires agents powered by different AI providers (Google Gemini, local Gemma via
Ollama, xAI Grok, and Anthropic Claude) into one handoff graph, then runs a
short demo conversation. The frontline Receptionist (Gemini) greets the user
and may hand the turn to a specialist, so a single conversation can cross
providers. Requires GEMINI_API_KEY (plus XAI_API_KEY / ANTHROPIC_API_KEY and a
running Ollama only if you edit the message to reach those specialists).
"""

import asyncio
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from chatflow_agent import Agent, Runner

# 1. Local Gemma Agent (Runs 100% locally via Ollama / vLLM without external API costs)
local_gemma = Agent(
    name="LocalProcessor",
    provider="ollama",          # Connects to http://localhost:11434/v1
    model="gemma2:9b",          # Or "gemma2:2b", "llama3.2", "mistral", "qwen2.5"
    instructions="You are a local data processor running on offline Google Gemma.",
)

@local_gemma.tool
def calculate_local_metrics(data_points: list[int]) -> dict:
    """Computes basic offline statistics."""
    total = sum(data_points)
    avg = total / len(data_points) if data_points else 0
    return {"total": total, "average": avg, "points": len(data_points)}

# 2. xAI Grok Agent (Specialist in technical research and real-time reasoning)
grok_analyst = Agent(
    name="GrokAnalyst",
    provider="grok",            # Connects to https://api.x.ai/v1 (uses XAI_API_KEY)
    model="grok-2-latest",
    instructions="You are a deep-dive research specialist powered by xAI Grok.",
)

# 3. Anthropic Claude Agent (Specialist in compliance and auditing)
claude_auditor = Agent(
    name="ClaudeAuditor",
    provider="anthropic",       # Connects to Anthropic Messages API (uses ANTHROPIC_API_KEY)
    model="claude-3-5-sonnet-latest",
    instructions="You are a strict compliance and security auditor powered by Claude.",
)

# 4. Frontline Receptionist powered by Google Gemini
triage_agent = Agent(
    name="Receptionist",
    provider="gemini",          # Default Google Gemini (uses GEMINI_API_KEY)
    model="gemini-2.5-flash",   # Ultra fast and economical
    instructions="You are the main customer concierge. Greet the user and route them to specialists.",
    handoffs=[local_gemma, grok_analyst, claude_auditor],
)

async def main() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set. The Receptionist runs on Google Gemini,")
        print("so this demo needs it before it can run a conversation.")
        print("Set it and retry, or point the Receptionist at a local provider.")
        return
    runner = Runner(starting_agent=triage_agent)
    direct_handoffs = [a.name for a in runner.starting_agent.handoffs]
    print(f"Starting Agent: {runner.starting_agent.name} | Direct Handoffs: {direct_handoffs}")
    print("Running a short demo conversation through the multi-provider swarm...\n")
    response = await runner.run_async(
        session_id="swarm-demo",
        user_message="Hello! Please have the right specialist give me a quick summary of what this swarm can do.",
    )
    print(f"Replied by: {response.active_agent_name}")
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
