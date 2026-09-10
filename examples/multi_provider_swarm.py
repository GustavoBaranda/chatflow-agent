"""Example: Heterogeneous Multi-Provider Swarm with chatflow-agent.

Demonstrates how agents powered by different AI providers (Google Gemini,
Google Gemma local via Ollama, xAI Grok, OpenAI GPT-4o, and Anthropic Claude)
can collaborate and hand off seamlessly to one another in the same conversation.
"""

import asyncio
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
    model="grok-2",
    instructions="You are a deep-dive research specialist powered by xAI Grok.",
)

# 3. Anthropic Claude Agent (Specialist in compliance and auditing)
claude_auditor = Agent(
    name="ClaudeAuditor",
    provider="anthropic",       # Connects to Anthropic Messages API (uses ANTHROPIC_API_KEY)
    model="claude-3-5-sonnet",
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
    runner = Runner(starting_agent=triage_agent)
    print("Multi-Provider Swarm Initialized!")
    print(f"Known Agents: {list(runner._agents_registry.keys())}")
    print("Ready to route between Gemini, Gemma (Ollama), Grok, and Claude.")

if __name__ == "__main__":
    asyncio.run(main())
