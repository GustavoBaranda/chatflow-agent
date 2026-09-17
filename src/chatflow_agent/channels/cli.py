"""Interactive Command-Line Interface (CLI) channel for chatflow-agent."""

import sys
from typing import Any

from chatflow_agent.channels.base import BaseChannel
from chatflow_agent.types import AgentResponse


class CLIChannel(BaseChannel):
    """Terminal-based interactive channel for real-time multi-agent testing and debugging."""

    def __init__(
        self,
        session_id: str = "cli_user",
        prompt_prefix: str = "\nYou: ",
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.prompt_prefix = prompt_prefix

    def send_message(self, text: str) -> AgentResponse:
        """Convenience method to dispatch a single message synchronously."""
        return self.dispatch(self.session_id, text)

    def run(self, **kwargs: Any) -> None:
        """Start an interactive terminal conversation session."""
        runner = self._ensure_runner_attached()

        print("=" * 65)
        print(f"ChatFlow CLI Session Started [Starting: {runner.starting_agent.name}]")
        print("Type 'exit', 'quit' or 'salir' to end the session.")
        print("=" * 65)

        while True:
            try:
                user_input = input(self.prompt_prefix).strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nSession ended by user.")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "salir"):
                print("\nGoodbye!")
                break

            # Dispatch to runner
            try:
                response = self.dispatch(self.session_id, user_input)
                # Show handoff indicator if a transfer occurred
                if response.handoff:
                    print(f"[Transferred to {response.handoff.target_agent_name}]")
                    if response.handoff.reason:
                        print(f"   Reason: {response.handoff.reason}")

                # Print final text response
                print(f"\n[{response.active_agent_name}]: {response.content}")

            except Exception as err:
                print(f"\nError during processing: {err}", file=sys.stderr)
