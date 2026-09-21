"""Local Offline Multi-Agent CLI powered by Google Gemma 2 via Ollama.

Features:
- Runs 100% offline with zero cloud API keys, zero external network calls, and zero costs.
- Native SQLiteSessionStore ensures terminal sessions persist across script invocations.
- Autonomous handoffs between local specialist agents.

Prerequisites:
- Install Ollama (https://ollama.ai)
- Pull Gemma 2: `ollama run gemma2:2b` or `ollama run gemma2:9b`
"""

import os
import sys

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from chatflow_agent import (
    Agent,
    CLIChannel,
    Runner,
    SQLiteSessionStore,
)

# 1. Specialist: Local Code Reviewer
code_specialist = Agent(
    name="Code Reviewer",
    provider="ollama",
    model="gemma2:2b",  # or "gemma2:9b"
    instructions=(
        "You are an offline code auditor running on local Google Gemma. "
        "Review code snippets for performance, safety, and readability. "
        "Keep recommendations clean and succinct."
    ),
)

@code_specialist.tool
def analyze_complexity(lines_count: int, cyclomatic_complexity: int) -> dict:
    """Analyze code maintainability metrics."""
    risk = "Low" if cyclomatic_complexity <= 5 else "Medium" if cyclomatic_complexity <= 10 else "High"
    return {
        "lines": lines_count,
        "complexity": cyclomatic_complexity,
        "risk_level": risk,
        "refactor_recommended": risk == "High",
    }


# 2. Specialist: Local System Diagnostics
sys_specialist = Agent(
    name="System Diagnostics",
    provider="ollama",
    model="gemma2:2b",
    instructions=(
        "You inspect local machine state and operational statistics. "
        "Provide factual, technical diagnostic reports."
    ),
)

@sys_specialist.tool
def get_system_snapshot() -> dict:
    """Inspect local execution environment summary."""
    import platform
    return {
        "os": platform.system(),
        "release": platform.release(),
        "python_version": platform.python_version(),
        "mode": "Offline Local Gemma",
    }


# 3. Frontline Coordinator
coordinator = Agent(
    name="Local Coordinator",
    provider="ollama",
    model="gemma2:2b",
    instructions=(
        "You are the primary offline terminal assistant powered by Google Gemma. "
        "Greet the user. If they ask about code inspection or architecture, hand off to Code Reviewer. "
        "If they ask about environment, host platform, or system stats, hand off to System Diagnostics."
    ),
    handoffs=[code_specialist, sys_specialist],
)

# 4. Persistent Runner
session_store = SQLiteSessionStore(
    db_path="gemma_sessions.db",
    default_max_turns=30,
)

runner = Runner(
    starting_agent=coordinator,
    session_store=session_store,
)

# 5. CLI Channel
cli = CLIChannel()
cli.attach(runner)

if __name__ == "__main__":
    print("=" * 60)
    print("Local Google Gemma 2 Offline Multi-Agent CLI")
    print("Provider: Ollama (http://localhost:11434/v1)")
    print("Storage: gemma_sessions.db (SQLite)")
    print("Type your message, or 'exit' / 'quit' to terminate.")
    print("=" * 60)
    cli.run()
