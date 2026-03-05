from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tinyharness.gateway_debug import ProxyResponse, create_app
from tinyharness.harbor_agents import encode_proxy_token


@pytest.mark.parametrize("debug_enabled", [True, False])
def test_gateway_debug_requests_and_replay_require_debug_mode(debug_enabled: bool) -> None:
    app = create_app(
        api_key="secret-token",
        litellm_forwarder=lambda *args, **kwargs: None,
        llama_forwarder=lambda *args, **kwargs: None,
        debug_enabled=debug_enabled,
    )
    client = TestClient(app)

    response = client.get(
        "/debug/requests",
        params={"trial_name": "trial-1"},
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == (200 if debug_enabled else 404)


def test_gateway_debug_merges_inbound_effective_and_response_by_request_id() -> None:
    litellm_calls: list[tuple[str, dict[str, str], dict[str, object]]] = []
    llama_calls: list[tuple[str, dict[str, str], dict[str, object]]] = []

    async def litellm_forwarder(*, method: str, path: str, headers: dict[str, str], body: bytes) -> ProxyResponse:
        litellm_calls.append((path, headers, json.loads(body.decode("utf-8"))))
        return ProxyResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps({"id": "msg-1", "stop_reason": "end_turn", "usage": {"input_tokens": 10}}).encode("utf-8"),
        )

    async def llama_forwarder(*, method: str, path: str, headers: dict[str, str], body: bytes) -> ProxyResponse:
        llama_calls.append((path, headers, json.loads(body.decode("utf-8"))))
        return ProxyResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps(
                {
                    "id": "chatcmpl-1",
                    "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 2},
                }
            ).encode("utf-8"),
        )

    app = create_app(
        api_key="secret-token",
        litellm_forwarder=litellm_forwarder,
        llama_forwarder=llama_forwarder,
        debug_enabled=True,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={
            "Authorization": "Bearer secret-token",
            "x-tinyharness-job-name": "smoke-v0",
            "x-tinyharness-trial-name": "cancel-async-tasks__trial",
        },
        json={"model": "qwen", "messages": [{"role": "user", "content": "fix it"}]},
    )
    assert response.status_code == 200

    forwarded_headers = litellm_calls[0][1]
    request_id = forwarded_headers["x-tinyharness-request-id"]
    assert forwarded_headers["x-tinyharness-trial-name"] == "cancel-async-tasks__trial"
    assert forwarded_headers["x-tinyharness-job-name"] == "smoke-v0"

    openai_response = client.post(
        "/openai-proxy/v1/chat/completions",
        headers={
            "Authorization": "Bearer secret-token",
            "x-tinyharness-request-id": request_id,
            "x-tinyharness-trial-name": "cancel-async-tasks__trial",
            "x-tinyharness-job-name": "smoke-v0",
        },
        json={"messages": [{"role": "user", "content": "fix it"}]},
    )
    assert openai_response.status_code == 200
    assert llama_calls[0][0] == "v1/chat/completions"

    bundle = client.get(
        "/debug/requests",
        params={"trial_name": "cancel-async-tasks__trial"},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert bundle.status_code == 200
    payload = bundle.json()
    assert payload["trial_name"] == "cancel-async-tasks__trial"
    assert payload["requests"][0]["request_id"] == request_id
    assert payload["requests"][0]["job_name"] == "smoke-v0"
    assert payload["requests"][0]["inbound_request"]["sha256"]
    assert payload["requests"][0]["effective_request"]["sha256"]
    assert payload["requests"][0]["response_summary"]["status_code"] == 200

    replay = client.post(
        "/debug/replay",
        headers={"Authorization": "Bearer secret-token"},
        json={"request_id": request_id, "count": 3},
    )
    assert replay.status_code == 200
    replay_payload = replay.json()
    assert replay_payload["request_id"] == request_id
    assert len(replay_payload["responses"]) == 3
    assert len({item["response_sha256"] for item in replay_payload["responses"]}) == 1


def test_gateway_debug_extracts_trial_context_from_encoded_api_key() -> None:
    litellm_calls: list[tuple[str, dict[str, str], dict[str, object]]] = []
    llama_calls: list[tuple[str, dict[str, str], dict[str, object]]] = []

    async def litellm_forwarder(*, method: str, path: str, headers: dict[str, str], body: bytes) -> ProxyResponse:
        litellm_calls.append((path, headers, json.loads(body.decode("utf-8"))))
        return ProxyResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps({"id": "msg-1", "stop_reason": "end_turn"}).encode("utf-8"),
        )

    async def llama_forwarder(*, method: str, path: str, headers: dict[str, str], body: bytes) -> ProxyResponse:
        llama_calls.append((path, headers, json.loads(body.decode("utf-8"))))
        return ProxyResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps({"choices": [{"finish_reason": "stop"}]}).encode("utf-8"),
        )

    app = create_app(
        api_key="secret-token",
        litellm_forwarder=litellm_forwarder,
        llama_forwarder=llama_forwarder,
        debug_enabled=True,
    )
    client = TestClient(app)

    encoded_token = encode_proxy_token(
        "secret-token",
        job_name="smoke-v0",
        trial_name="cancel-async-tasks__trial",
        correlation_id="cid-123",
    )
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": encoded_token},
        json={"model": "qwen", "messages": [{"role": "user", "content": "fix it"}]},
    )

    assert response.status_code == 200
    forwarded_headers = litellm_calls[0][1]
    assert forwarded_headers["x-api-key"] == "secret-token"
    assert forwarded_headers["x-tinyharness-job-name"] == "smoke-v0"
    assert forwarded_headers["x-tinyharness-trial-name"] == "cancel-async-tasks__trial"
    assert forwarded_headers["x-tinyharness-correlation-id"] == "cid-123"

    openai_response = client.post(
        "/openai-proxy/v1/chat/completions",
        headers={"x-api-key": "secret-token"},
        json={"messages": [{"role": "user", "content": "fix it"}]},
    )
    assert openai_response.status_code == 200
    assert llama_calls[0][0] == "v1/chat/completions"

    bundle = client.get(
        "/debug/requests",
        params={"trial_name": "cancel-async-tasks__trial"},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert bundle.status_code == 200
    record = bundle.json()["requests"][0]
    assert record["job_name"] == "smoke-v0"
    assert record["trial_name"] == "cancel-async-tasks__trial"
    assert record["correlation_id"] == "cid-123"
    assert record["effective_request"]["path"] == "v1/chat/completions"

    all_requests = client.get(
        "/debug/requests",
        headers={"Authorization": "Bearer secret-token"},
    )
    assert all_requests.status_code == 200
    assert len(all_requests.json()["requests"]) == 1


def test_gateway_debug_replay_reinjects_api_key_for_effective_request() -> None:
    async def litellm_forwarder(*, method: str, path: str, headers: dict[str, str], body: bytes) -> ProxyResponse:
        return ProxyResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps({"id": "msg-1", "stop_reason": "end_turn"}).encode("utf-8"),
        )

    async def llama_forwarder(*, method: str, path: str, headers: dict[str, str], body: bytes) -> ProxyResponse:
        api_key = headers.get("x-api-key")
        authorization = headers.get("authorization")
        if api_key != "secret-token" and authorization != "Bearer secret-token":
            return ProxyResponse(
                status_code=401,
                headers={"content-type": "application/json"},
                body=json.dumps(
                    {
                        "error": {
                            "message": "Invalid API Key",
                            "type": "authentication_error",
                            "code": 401,
                        }
                    }
                ).encode("utf-8"),
            )
        return ProxyResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps(
                {
                    "id": "chatcmpl-1",
                    "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                }
            ).encode("utf-8"),
        )

    app = create_app(
        api_key="secret-token",
        litellm_forwarder=litellm_forwarder,
        llama_forwarder=llama_forwarder,
        debug_enabled=True,
    )
    client = TestClient(app)

    inbound_response = client.post(
        "/v1/messages",
        headers={
            "Authorization": "Bearer secret-token",
            "x-tinyharness-job-name": "smoke-v0",
            "x-tinyharness-trial-name": "cancel-async-tasks__trial",
        },
        json={"model": "qwen", "messages": [{"role": "user", "content": "fix it"}]},
    )
    assert inbound_response.status_code == 200

    openai_response = client.post(
        "/openai-proxy/v1/chat/completions",
        headers={
            "Authorization": "Bearer secret-token",
            "x-tinyharness-job-name": "smoke-v0",
            "x-tinyharness-trial-name": "cancel-async-tasks__trial",
        },
        json={"messages": [{"role": "user", "content": "fix it"}]},
    )
    assert openai_response.status_code == 200

    replay = client.post(
        "/debug/replay",
        headers={"Authorization": "Bearer secret-token"},
        json={"trial_name": "cancel-async-tasks__trial", "count": 2},
    )
    assert replay.status_code == 200
    responses = replay.json()["responses"]
    assert len(responses) == 2
    assert {item["status_code"] for item in responses} == {200}
