"""Tests for the example webhook API service."""

import json
from pathlib import Path

import pytest
from examples import webhook_api_service
from starlette.requests import Request

from chatflow_agent import SessionStore
from chatflow_agent.types import AgentResponse, Handoff, Message, Role


def make_request(payload: dict[str, object]) -> Request:
    body = json.dumps(payload).encode()

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": [],
        },
        receive,
    )


async def test_chat_endpoint_returns_executed_tool_names(monkeypatch) -> None:
    async def fake_run_async(**kwargs) -> AgentResponse:
        return AgentResponse(
            content="Here is the result.",
            active_agent_name="Frontline API Receptionist",
            tool_calls_executed=["get_tier_pricing"],
        )

    monkeypatch.setattr(webhook_api_service.runner, "run_async", fake_run_async)

    response = await webhook_api_service.chat_endpoint(
        make_request({"session_id": "test-tools", "message": "How much does it cost?"})
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {
        "session_id": "test-tools",
        "content": "Here is the result.",
        "active_agent": "Frontline API Receptionist",
        "tool_calls": ["get_tier_pricing"],
        "handoff": None,
    }


async def test_chat_endpoint_returns_handoff_target_name(monkeypatch) -> None:
    async def fake_run_async(**kwargs) -> AgentResponse:
        return AgentResponse(
            content="I am connecting you with support.",
            active_agent_name="Support Specialist",
            handoff=Handoff(target_agent_name="Support Specialist"),
        )

    monkeypatch.setattr(webhook_api_service.runner, "run_async", fake_run_async)

    response = await webhook_api_service.chat_endpoint(
        make_request({"session_id": "test-handoff", "message": "I need technical help."})
    )

    assert response.status_code == 200
    assert json.loads(response.body)["handoff"] == "Support Specialist"


def make_raw_request(
    *,
    method: str = "POST",
    path: str = "/api/chat",
    body: bytes = b"",
    path_params: dict[str, str] | None = None,
) -> Request:
    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [],
            "path_params": path_params or {},
        },
        receive,
    )


async def test_chat_endpoint_rejects_invalid_json_body() -> None:
    response = await webhook_api_service.chat_endpoint(make_raw_request(body=b"{not valid json"))

    assert response.status_code == 400
    assert json.loads(response.body) == {"error": "Invalid JSON body"}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": "Hello"},
        {"session_id": "test-missing-message"},
        {"session_id": "", "message": "Hello"},
        {"session_id": "test-empty-message", "message": ""},
    ],
)
async def test_chat_endpoint_requires_session_id_and_message(
    payload: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    async def fake_run_async(**kwargs: object) -> AgentResponse:
        calls.append(kwargs)
        return AgentResponse(content="unexpected", active_agent_name="Unexpected")

    monkeypatch.setattr(webhook_api_service.runner, "run_async", fake_run_async)

    response = await webhook_api_service.chat_endpoint(make_request(payload))

    assert response.status_code == 422
    assert json.loads(response.body) == {"error": "Fields 'session_id' and 'message' are required"}
    assert calls == []


async def test_get_session_history_returns_404_for_unknown_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(webhook_api_service.runner, "session_store", SessionStore())

    response = await webhook_api_service.get_session_history(
        make_raw_request(
            method="GET",
            path="/api/sessions/missing-session",
            path_params={"session_id": "missing-session"},
        )
    )

    assert response.status_code == 404
    assert json.loads(response.body) == {"error": "Session not found"}


async def test_get_session_history_returns_session_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore()
    session = store.get_or_create("history-session", webhook_api_service.triage_agent)
    session.add_message(Message(role=Role.USER, content="I need a quote."))
    session.add_message(Message(role=Role.MODEL, content="Happy to help with pricing."))
    session.update_metadata({"channel": "web"})
    monkeypatch.setattr(webhook_api_service.runner, "session_store", store)

    response = await webhook_api_service.get_session_history(
        make_raw_request(
            method="GET",
            path="/api/sessions/history-session",
            path_params={"session_id": "history-session"},
        )
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {
        "session_id": "history-session",
        "active_agent": "Frontline API Receptionist",
        "messages_count": 2,
        "history": [
            {"role": "user", "content": "I need a quote."},
            {"role": "model", "content": "Happy to help with pricing."},
        ],
        "metadata": {"channel": "web"},
    }


async def test_health_check_returns_healthy_status() -> None:
    response = await webhook_api_service.health_check(
        make_raw_request(method="GET", path="/health")
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {"status": "healthy", "service": "chatflow-agent-api"}


def test_example_database_is_kept_out_of_the_repository_root() -> None:
    db_path = Path(webhook_api_service.session_store.db_path).resolve()
    assert db_path.parent != Path.cwd().resolve()
    assert not Path("api_sessions.db").exists()
