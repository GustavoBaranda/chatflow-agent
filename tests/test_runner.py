"""Tests for chatflow-agent Runner loop, tool calling, and multi-agent handoffs."""


import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.engines import BaseEngine
from chatflow_agent.core.memory import SessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.types import Handoff, Message, Role, ToolCall


class MockEngine(GeminiEngine):
    """Mock engine that simulates Gemini turn evaluations deterministically."""

    def __init__(self, turns_sequence: list[EngineTurnResult]) -> None:
        super().__init__()
        self.turns = list(turns_sequence)
        self.call_history: list[dict] = []

    async def generate_turn_async(
        self,
        agent: Agent,
        history: list[Message],
    ) -> EngineTurnResult:
        self.call_history.append({"agent": agent.name, "history_len": len(history)})
        if not self.turns:
            return EngineTurnResult(text="No more turns queued.")
        return self.turns.pop(0)


@pytest.mark.asyncio
async def test_runner_direct_text_response() -> None:
    agent = Agent(name="Greeter", instructions="Greet the user.")
    mock_engine = MockEngine([EngineTurnResult(text="Hello! How can I help you today?")])

    runner = Runner(starting_agent=agent, engine=mock_engine)
    response = await runner.run_async(session_id="user_1", user_message="Hi")

    assert response.content == "Hello! How can I help you today?"
    assert response.active_agent_name == "Greeter"
    assert response.handoff is None
    assert len(response.tool_calls_executed) == 0

    # Verify session history has 2 messages: user and model
    session = runner.get_session("user_1")
    assert session is not None
    assert len(session.history) == 2
    assert session.history[0].role == Role.USER
    assert session.history[1].role == Role.MODEL


@pytest.mark.asyncio
async def test_runner_with_tool_execution() -> None:
    agent = Agent(name="Calculator", instructions="Compute math.")

    @agent.tool
    def multiply(a: int, b: int) -> int:
        return a * b

    # Turn 1: Model requests tool call
    # Turn 2: Model receives tool result and produces final answer
    mock_engine = MockEngine([
        EngineTurnResult(
            tool_calls=[ToolCall(id="call_1", name="multiply", args={"a": 6, "b": 7})]
        ),
        EngineTurnResult(text="6 multiplied by 7 is 42."),
    ])

    runner = Runner(starting_agent=agent, engine=mock_engine)
    response = await runner.run_async(session_id="user_2", user_message="What is 6 * 7?")

    assert response.content == "6 multiplied by 7 is 42."
    assert "multiply" in response.tool_calls_executed
    assert response.active_agent_name == "Calculator"


@pytest.mark.asyncio
async def test_runner_multiagent_handoff_flow() -> None:
    # 1. Define specialist agent
    billing_agent = Agent(
        name="Billing Agent",
        instructions="Handle invoices and refunds.",
    )

    @billing_agent.tool
    def get_invoice(invoice_id: str) -> dict:
        return {"id": invoice_id, "amount": 100}

    # 2. Define triage agent with handoff
    triage_agent = Agent(
        name="Triage Agent",
        instructions="Route users to correct agent.",
        handoffs=[billing_agent],
    )

    # Turn 1: Triage decides to hand off to Billing Agent
    # Turn 2: Billing agent receives handoff result and calls get_invoice
    # Turn 3: Billing agent provides final answer
    mock_engine = MockEngine([
        EngineTurnResult(
            tool_calls=[
                ToolCall(
                    id="call_h1",
                    name="transfer_to_billing_agent",
                    args={"reason": "Customer invoice query"},
                )
            ]
        ),
        EngineTurnResult(
            tool_calls=[
                ToolCall(id="call_inv1", name="get_invoice", args={"invoice_id": "INV-10"})
            ]
        ),
        EngineTurnResult(text="Here is your invoice INV-10 for $100."),
    ])

    runner = Runner(starting_agent=triage_agent, engine=mock_engine)
    response = await runner.run_async(
        session_id="user_3", user_message="I need help with invoice INV-10"
    )

    # Assert active agent switched to Billing Agent!
    assert response.active_agent_name == "Billing Agent"
    assert response.content == "Here is your invoice INV-10 for $100."
    assert response.handoff is not None
    assert response.handoff.target_agent_name == "Billing Agent"
    assert response.handoff.reason == "Customer invoice query"

    # Assert session retains Billing Agent for future turns
    session = runner.get_session("user_3")
    assert session is not None
    assert session.active_agent.name == "Billing Agent"


@pytest.mark.asyncio
async def test_runner_handoff_context_updates_are_available_to_recipient_tools() -> None:
    billing_agent = Agent(name="Billing Agent", instructions="Handle invoices and refunds.")
    triage_agent = Agent(
        name="Triage Agent",
        instructions="Route users to the correct specialist.",
        handoffs=[billing_agent],
    )

    @triage_agent.tool
    def transfer_with_context() -> Handoff:
        return Handoff(
            target_agent_name="Billing Agent",
            context_updates={"customer_tier": "VIP", "invoice_id": "INV-101"},
        )

    session_store = SessionStore()
    session = session_store.get_or_create(
        session_id="user_context_handoff",
        default_agent=triage_agent,
        metadata={"customer_id": "C-1", "customer_tier": "standard"},
    )

    @billing_agent.tool
    def get_customer_context() -> dict[str, str]:
        return {
            "customer_id": session.metadata["customer_id"],
            "customer_tier": session.metadata["customer_tier"],
            "invoice_id": session.metadata["invoice_id"],
        }

    mock_engine = MockEngine([
        EngineTurnResult(
            tool_calls=[ToolCall(id="call_handoff", name="transfer_with_context", args={})]
        ),
        EngineTurnResult(
            tool_calls=[ToolCall(id="call_context", name="get_customer_context", args={})]
        ),
        EngineTurnResult(text="Billing context received."),
    ])
    runner = Runner(starting_agent=triage_agent, session_store=session_store, engine=mock_engine)

    response = await runner.run_async(
        session_id="user_context_handoff", user_message="Please review invoice INV-101."
    )

    assert response.active_agent_name == "Billing Agent"
    assert session.metadata == {
        "customer_id": "C-1",
        "customer_tier": "VIP",
        "invoice_id": "INV-101",
    }


@pytest.mark.asyncio
async def test_runner_does_not_apply_context_updates_for_rejected_handoff() -> None:
    billing_agent = Agent(name="Billing Agent", instructions="Handle invoices and refunds.")
    triage_agent = Agent(
        name="Triage Agent",
        instructions="Route users to the correct specialist.",
        handoffs=[billing_agent],
    )

    @triage_agent.tool
    def transfer_with_context() -> Handoff:
        return Handoff(
            target_agent_name="Billing Agent",
            context_updates={"customer_tier": "VIP"},
        )

    session_store = SessionStore()
    session = session_store.get_or_create(
        session_id="user_rejected_handoff",
        default_agent=triage_agent,
        metadata={"customer_tier": "standard"},
    )
    mock_engine = MockEngine([
        EngineTurnResult(
            tool_calls=[ToolCall(id="call_handoff", name="transfer_with_context", args={})]
        ),
        EngineTurnResult(text="The transfer limit was reached."),
    ])
    runner = Runner(
        starting_agent=triage_agent,
        session_store=session_store,
        engine=mock_engine,
        max_handoffs_per_turn=0,
    )

    response = await runner.run_async(
        session_id="user_rejected_handoff", user_message="Please escalate this invoice."
    )

    assert response.active_agent_name == "Triage Agent"
    assert response.handoff is None
    assert session.active_agent.name == "Triage Agent"
    assert session.metadata == {"customer_tier": "standard"}


@pytest.mark.asyncio
@pytest.mark.parametrize("max_handoffs", [0, 1])
@pytest.mark.parametrize("max_tool_iterations", [1, 2])
async def test_runner_handoff_limit_preserves_tool_result_parity(
    max_handoffs: int, max_tool_iterations: int
) -> None:
    billing_agent = Agent(name="Billing", instructions="Handle invoices.")
    triage_agent = Agent(
        name="Triage", instructions="Route requests.", handoffs=[billing_agent]
    )
    executed: list[str] = []

    @triage_agent.tool
    def record_action(value: str) -> str:
        executed.append(value)
        return value

    @triage_agent.tool
    def transfer_with_context(value: str) -> Handoff:
        executed.append(value)
        return Handoff(target_agent_name="Billing", context_updates={"transfer": value})

    calls = [ToolCall(id="before", name="record_action", args={"value": "before"})]
    if max_handoffs:
        calls.append(
            ToolCall(id="accepted", name="transfer_with_context", args={"value": "accepted"})
        )
    calls.extend([
        ToolCall(id="rejected", name="transfer_with_context", args={"value": "rejected"}),
        ToolCall(id="skipped_action", name="record_action", args={"value": "after"}),
        ToolCall(id="skipped_handoff", name="transfer_with_context", args={"value": "after"}),
    ])

    class ValidatingEngine(BaseEngine):
        def __init__(self) -> None:
            self.calls = 0

        async def generate_turn_async(
            self, agent: Agent, history: list[Message]
        ) -> EngineTurnResult:
            self.calls += 1
            # A provider must see a result for every call before the next generation.
            if history[-1].role == Role.TOOL:
                assert history[-1].tool_results is not None
                assert [result.tool_call_id for result in history[-1].tool_results] == [
                    call.id for call in calls
                ]
                return EngineTurnResult(text="The transfer limit was reached.")
            return EngineTurnResult(tool_calls=calls)

    engine = ValidatingEngine()
    runner = Runner(
        starting_agent=triage_agent,
        engine=engine,
        max_handoffs_per_turn=max_handoffs,
        max_tool_iterations=max_tool_iterations,
    )

    response = await runner.run_async(session_id="limit_parity", user_message="Help with billing.")

    session = runner.get_session("limit_parity")
    assert session is not None
    results = session.history[2].tool_results
    assert results is not None
    assert [(result.tool_call_id, result.name) for result in results] == [
        (call.id, call.name) for call in calls
    ]
    assert results[0].content == "before"
    assert [result.is_error for result in results] == [False] * (1 + max_handoffs) + [True] * 3
    assert "Maximum agent transfers exceeded" in results[-3].content
    assert all("skipped" in result.content.lower() for result in results[-2:])
    assert executed == ["before"] + (["accepted"] if max_handoffs else []) + ["rejected"]
    assert response.tool_calls_executed == [call.name for call in calls[:-2]]
    assert response.active_agent_name == ("Billing" if max_handoffs else "Triage")
    assert session.metadata == ({"transfer": "accepted"} if max_handoffs else {})
    if max_handoffs:
        assert response.handoff is not None
        assert response.handoff.context_updates == {"transfer": "accepted"}
    else:
        assert response.handoff is None
    if max_tool_iterations == 1:
        assert response.content.startswith("Execution iteration limit reached")
    else:
        assert response.content == "The transfer limit was reached."
    assert engine.calls == max_tool_iterations


def test_runner_sync_wrapper() -> None:
    agent = Agent(name="Echo", instructions="Echo input.")
    mock_engine = MockEngine([EngineTurnResult(text="Echo reply.")])

    runner = Runner(starting_agent=agent, engine=mock_engine)
    response = runner.run(session_id="user_4", user_message="Testing sync")

    assert response.content == "Echo reply."
    assert response.active_agent_name == "Echo"


def test_runner_run_raises_inside_running_loop() -> None:
    """Runner.run() must not be called from a running event loop."""
    import asyncio
    import warnings

    agent = Agent(name="Echo", instructions="Echo input.")
    mock_engine = MockEngine([EngineTurnResult(text="should not reach")])
    runner = Runner(starting_agent=agent, engine=mock_engine)

    async def call_sync_from_async() -> None:
        with pytest.raises(RuntimeError, match="cannot be called from a running event loop"):
            runner.run(session_id="user_async", user_message="hi")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(call_sync_from_async())
        unawaited = [
            w
            for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "never awaited" in str(w.message)
        ]
        assert unawaited == [], f"Unexpected unawaited coroutine warnings: {unawaited}"


@pytest.mark.asyncio
async def test_runner_resolves_handoff_added_after_initialization() -> None:
    """Peers registered via add_handoff after Runner creation must resolve successfully."""
    from chatflow_agent.exceptions import AgentHandoffError

    triage = Agent(name="Triage", instructions="Frontline triage")
    billing = Agent(name="Billing", instructions="Billing specialist")
    nested = Agent(name="Nested Billing", instructions="Nested specialist")
    billing.add_handoff(nested)

    mock_engine = MockEngine(
        [
            EngineTurnResult(
                tool_calls=[
                    ToolCall(
                        id="call_h1",
                        name="transfer_to_billing",
                        args={"reason": "Invoice question"},
                    )
                ]
            ),
            EngineTurnResult(text="Billing can help with that invoice."),
        ]
    )

    runner = Runner(starting_agent=triage, engine=mock_engine)
    assert "Billing" not in runner._agents_registry

    triage.add_handoff(billing)

    response = await runner.run_async(
        session_id="user_late_handoff",
        user_message="I need help with my bill",
    )

    assert response.active_agent_name == "Billing"
    assert response.content == "Billing can help with that invoice."
    assert response.handoff is not None
    assert response.handoff.target_agent_name == "Billing"
    assert "Billing" in runner._agents_registry
    assert "Nested Billing" in runner._agents_registry
    # Guard against regressions that drop the fail-closed path.
    with pytest.raises(AgentHandoffError):
        runner._resolve_handoff_target(triage, "Ghost Agent")


@pytest.mark.asyncio
async def test_runner_rejects_unauthorized_handoff_not_in_current_handoffs() -> None:
    """Arbitrary Handoff targets outside current_agent.handoffs must fail closed."""
    from chatflow_agent.exceptions import AgentHandoffError

    triage = Agent(name="Triage", instructions="Frontline triage")
    billing = Agent(name="Billing", instructions="Billing specialist")

    @triage.tool
    def sneak_to_billing() -> Handoff:
        return Handoff(target_agent_name="Billing", reason="Unauthorized path")

    mock_engine = MockEngine(
        [
            EngineTurnResult(
                tool_calls=[ToolCall(id="call_bad", name="sneak_to_billing", args={})]
            ),
        ]
    )

    runner = Runner(starting_agent=triage, engine=mock_engine)
    # Billing exists but was never authorized on triage.handoffs.
    runner.register_agent(billing)
    # Remove billing from the registry to force the late-bind authorization check.
    runner._agents_registry.pop("Billing", None)

    with pytest.raises(AgentHandoffError, match="Billing"):
        await runner.run_async(
            session_id="user_unauthorized_handoff",
            user_message="Bypass triage routing",
        )

    assert "Billing" not in runner._agents_registry


@pytest.mark.asyncio
async def test_runner_concurrent_late_handoff_registration() -> None:
    """Concurrent sessions must safely register the same late-bound peer."""
    import asyncio

    triage = Agent(name="Triage", instructions="Frontline triage")
    billing = Agent(name="Billing", instructions="Billing specialist")

    class LateHandoffEngine(BaseEngine):
        async def generate_turn_async(
            self,
            agent: Agent,
            history: list[Message],
        ) -> EngineTurnResult:
            if agent.name == "Triage":
                return EngineTurnResult(
                    tool_calls=[
                        ToolCall(
                            id="call_h",
                            name="transfer_to_billing",
                            args={"reason": "parallel"},
                        )
                    ]
                )
            return EngineTurnResult(text=f"Handled by {agent.name}.")

    runner = Runner(starting_agent=triage, engine=LateHandoffEngine())
    triage.add_handoff(billing)

    results = await asyncio.gather(
        *[
            runner.run_async(session_id=f"concurrent-{i}", user_message="bill help")
            for i in range(8)
        ]
    )

    assert all(r.active_agent_name == "Billing" for r in results)
    assert all(r.handoff is not None for r in results)
    assert runner._agents_registry["Billing"] is billing


@pytest.mark.asyncio
async def test_runner_rejects_cross_branch_handoff_to_registered_agent() -> None:
    """A registered agent is not a valid target unless current_agent declares it.

    Triage -> Sales, Support; Support -> AdminAgent. Sales has no handoff to
    AdminAgent, so a Sales tool returning Handoff("AdminAgent") must fail even
    though AdminAgent is already present in the Runner registry.
    """
    from chatflow_agent.exceptions import AgentHandoffError

    triage = Agent(name="Triage", instructions="Frontline triage")
    sales = Agent(name="Sales", instructions="Sales specialist")
    support = Agent(name="Support", instructions="Support specialist")
    admin = Agent(name="AdminAgent", instructions="Privileged admin")
    support.add_handoff(admin)
    triage.add_handoff(sales)
    triage.add_handoff(support)

    @sales.tool
    def escalate_to_admin() -> Handoff:
        return Handoff(target_agent_name="AdminAgent", reason="Cross-branch jump")

    mock_engine = MockEngine(
        [
            EngineTurnResult(
                tool_calls=[
                    ToolCall(id="call_s", name="transfer_to_sales", args={"reason": "buy"})
                ]
            ),
            EngineTurnResult(
                tool_calls=[ToolCall(id="call_x", name="escalate_to_admin", args={})]
            ),
        ]
    )

    runner = Runner(starting_agent=triage, engine=mock_engine)
    assert "AdminAgent" in runner._agents_registry

    with pytest.raises(AgentHandoffError, match="AdminAgent"):
        await runner.run_async(
            session_id="user_cross_branch",
            user_message="Give me admin access",
        )

    # Direct resolution: Sales is rejected, Support (which declares it) is allowed.
    with pytest.raises(AgentHandoffError):
        runner._resolve_handoff_target(sales, "AdminAgent")
    assert runner._resolve_handoff_target(support, "AdminAgent") is admin
