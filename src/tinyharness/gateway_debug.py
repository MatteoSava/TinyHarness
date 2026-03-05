from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse


@dataclass(frozen=True)
class ProxyResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes


Forwarder = Callable[..., Awaitable[ProxyResponse]]
_PROXY_TOKEN_MARKER = "::tinyharness::"


def _epoch_ms() -> int:
    return int(time.time() * 1000)


def _normalize_headers(headers: dict[str, str]) -> dict[str, str]:
    ignored = {"authorization", "content-length", "host", "connection"}
    return {key.lower(): value for key, value in headers.items() if key.lower() not in ignored}


def _decode_payload(body: bytes, content_type: str | None) -> Any:
    if not body:
        return None
    text = body.decode("utf-8", errors="replace")
    if content_type and "json" in content_type.lower():
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _sha256_payload(payload: Any) -> str:
    if isinstance(payload, (dict, list)):
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    elif payload is None:
        encoded = b""
    else:
        encoded = str(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _response_summary(*, status_code: int, headers: dict[str, str], body: bytes) -> dict[str, Any]:
    payload = _decode_payload(body, headers.get("content-type"))
    summary: dict[str, Any] = {
        "status_code": status_code,
        "content_type": headers.get("content-type", ""),
        "response_sha256": _sha256_payload(payload),
    }
    if isinstance(payload, dict):
        summary["payload"] = payload
        usage = payload.get("usage")
        if usage is not None:
            summary["usage"] = usage
        stop_reason = payload.get("stop_reason")
        if stop_reason is not None:
            summary["stop_reason"] = stop_reason
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict) and first_choice.get("finish_reason") is not None:
                summary["finish_reason"] = first_choice["finish_reason"]
    else:
        summary["payload"] = payload
    return summary


def _path_with_query(path: str, query: str) -> str:
    return f"{path}?{query}" if query else path


def _filtered_response_headers(headers: dict[str, str]) -> dict[str, str]:
    ignored = {"content-length", "transfer-encoding", "connection"}
    return {key: value for key, value in headers.items() if key.lower() not in ignored}


class GatewayDebugStore:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._request_order: list[str] = []

    def _ensure_record(
        self,
        request_id: str,
        *,
        job_name: str | None,
        trial_name: str | None,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        record = self._records.setdefault(
            request_id,
            {
                "request_id": request_id,
                "job_name": job_name,
                "trial_name": trial_name,
                "correlation_id": correlation_id,
                "created_at_epoch_ms": _epoch_ms(),
                "inbound_request": None,
                "effective_request": None,
                "response_summary": None,
            },
        )
        if request_id not in self._request_order:
            self._request_order.append(request_id)
        if job_name and not record.get("job_name"):
            record["job_name"] = job_name
        if trial_name and not record.get("trial_name"):
            record["trial_name"] = trial_name
        if correlation_id and not record.get("correlation_id"):
            record["correlation_id"] = correlation_id
        return record

    def record_inbound(
        self,
        *,
        request_id: str,
        job_name: str | None,
        trial_name: str | None,
        correlation_id: str | None,
        method: str,
        path: str,
        headers: dict[str, str],
        payload: Any,
    ) -> None:
        record = self._ensure_record(
            request_id,
            job_name=job_name,
            trial_name=trial_name,
            correlation_id=correlation_id,
        )
        record["inbound_request"] = {
            "method": method,
            "path": path,
            "headers": _normalize_headers(headers),
            "payload": payload,
            "sha256": _sha256_payload(payload),
        }

    def record_effective(
        self,
        *,
        request_id: str,
        job_name: str | None,
        trial_name: str | None,
        correlation_id: str | None,
        method: str,
        path: str,
        headers: dict[str, str],
        payload: Any,
    ) -> None:
        record = self._ensure_record(
            request_id,
            job_name=job_name,
            trial_name=trial_name,
            correlation_id=correlation_id,
        )
        record["effective_request"] = {
            "method": method,
            "path": path,
            "headers": _normalize_headers(headers),
            "payload": payload,
            "sha256": _sha256_payload(payload),
        }

    def record_response(self, *, request_id: str, status_code: int, headers: dict[str, str], body: bytes) -> None:
        record = self._ensure_record(request_id, job_name=None, trial_name=None)
        record["response_summary"] = _response_summary(status_code=status_code, headers=headers, body=body)

    def get_trial_requests(self, trial_name: str) -> list[dict[str, Any]]:
        records = [record for record in self._records.values() if record.get("trial_name") == trial_name]
        return sorted(records, key=lambda item: int(item.get("created_at_epoch_ms", 0)))

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        return self._records.get(request_id)

    def latest_unpaired_request_id(self) -> str | None:
        for request_id in reversed(self._request_order):
            record = self._records.get(request_id)
            if record is None:
                continue
            if record.get("inbound_request") is not None and record.get("effective_request") is None:
                return request_id
        return None

    def all_requests(self) -> list[dict[str, Any]]:
        return [self._records[request_id] for request_id in self._request_order if request_id in self._records]


def _build_http_forwarder(base_url: str) -> Forwarder:
    async def _forward(*, method: str, path: str, headers: dict[str, str], body: bytes) -> ProxyResponse:
        timeout = httpx.Timeout(60.0, read=60.0 * 20.0)
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
            response = await client.request(method, path, headers=headers, content=body)
            return ProxyResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                body=response.content,
            )

    return _forward


def _authorized(request: Request, api_key: str) -> bool:
    auth_header = request.headers.get("authorization", "")
    if auth_header == f"Bearer {api_key}":
        return True
    if request.headers.get("x-api-key") == api_key:
        return True
    return False


def _extract_api_key(headers: dict[str, str]) -> str | None:
    auth_header = headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.removeprefix("Bearer ")
    return headers.get("x-api-key")


def _parse_proxy_token(token: str | None, *, api_key: str) -> tuple[str | None, str | None, str | None, str | None]:
    if not token:
        return None, None, None, None
    if token == api_key:
        return api_key, None, None, None
    if not token.startswith(f"{api_key}{_PROXY_TOKEN_MARKER}"):
        return None, None, None, None
    query = token.split(_PROXY_TOKEN_MARKER, 1)[1]
    parsed = parse_qs(query, keep_blank_values=False)
    return (
        api_key,
        parsed.get("job", [None])[0],
        parsed.get("trial", [None])[0],
        parsed.get("cid", [None])[0],
    )


def create_app(
    *,
    api_key: str,
    litellm_forwarder: Forwarder | None = None,
    llama_forwarder: Forwarder | None = None,
    debug_enabled: bool = True,
    litellm_base_url: str | None = None,
    llama_base_url: str | None = None,
) -> FastAPI:
    if litellm_forwarder is None:
        if litellm_base_url is None:
            raise ValueError("litellm_base_url is required when litellm_forwarder is not provided")
        litellm_forwarder = _build_http_forwarder(litellm_base_url)
    if llama_forwarder is None:
        if llama_base_url is None:
            raise ValueError("llama_base_url is required when llama_forwarder is not provided")
        llama_forwarder = _build_http_forwarder(llama_base_url)

    app = FastAPI()
    store = GatewayDebugStore()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/debug/requests")
    async def debug_requests(request: Request, trial_name: str | None = None) -> JSONResponse:
        if not debug_enabled:
            raise HTTPException(status_code=404, detail="Gateway debug mode is disabled.")
        if not _authorized(request, api_key):
            raise HTTPException(status_code=401, detail="Unauthorized.")
        if trial_name is None:
            return JSONResponse({"trial_name": None, "requests": store.all_requests()})
        return JSONResponse({"trial_name": trial_name, "requests": store.get_trial_requests(trial_name)})

    @app.post("/debug/replay")
    async def debug_replay(request: Request) -> JSONResponse:
        if not debug_enabled:
            raise HTTPException(status_code=404, detail="Gateway debug mode is disabled.")
        if not _authorized(request, api_key):
            raise HTTPException(status_code=401, detail="Unauthorized.")
        payload = await request.json()
        headers = dict(request.headers)
        resolved_api_key = _extract_api_key(headers) or api_key
        request_id = payload.get("request_id")
        trial_name = payload.get("trial_name")
        count = int(payload.get("count", 3))
        if request_id is None:
            if not isinstance(trial_name, str):
                raise HTTPException(status_code=400, detail="Provide request_id or trial_name.")
            requests_for_trial = store.get_trial_requests(trial_name)
            if not requests_for_trial:
                raise HTTPException(status_code=404, detail="No requests found for trial.")
            request_id = requests_for_trial[-1]["request_id"]
        record = store.get_request(str(request_id))
        if record is None or not isinstance(record.get("effective_request"), dict):
            raise HTTPException(status_code=404, detail="No effective request found.")

        effective_request = record["effective_request"]
        responses: list[dict[str, Any]] = []
        for _ in range(max(1, count)):
            headers = dict(effective_request["headers"])
            if "authorization" not in headers and "x-api-key" not in headers:
                headers["authorization"] = f"Bearer {resolved_api_key}"
                headers["x-api-key"] = resolved_api_key
            response = await llama_forwarder(
                method=str(effective_request["method"]),
                path=str(effective_request["path"]),
                headers=headers,
                body=json.dumps(effective_request["payload"], ensure_ascii=False).encode("utf-8"),
            )
            responses.append(_response_summary(status_code=response.status_code, headers=response.headers, body=response.body))
        return JSONResponse(
            {
                "request_id": request_id,
                "trial_name": record.get("trial_name"),
                "responses": responses,
            }
        )

    async def _proxy_to_litellm(
        request: Request,
        *,
        upstream_path: str,
        job_name: str | None,
        trial_name: str | None,
    ) -> Response:
        body = await request.body()
        inbound_headers = dict(request.headers)
        resolved_api_key, token_job_name, token_trial_name, correlation_id = _parse_proxy_token(
            _extract_api_key(inbound_headers),
            api_key=api_key,
        )
        job_name = job_name or token_job_name
        trial_name = trial_name or token_trial_name
        request_id = request.headers.get("x-tinyharness-request-id") or str(uuid.uuid4())
        headers = dict(inbound_headers)
        headers["x-tinyharness-request-id"] = request_id
        if job_name is not None:
            headers["x-tinyharness-job-name"] = job_name
        if trial_name is not None:
            headers["x-tinyharness-trial-name"] = trial_name
        if correlation_id is not None:
            headers["x-tinyharness-correlation-id"] = correlation_id
        if resolved_api_key is not None:
            headers["x-api-key"] = resolved_api_key
            if "authorization" in headers:
                headers["authorization"] = f"Bearer {resolved_api_key}"
        parsed_payload = _decode_payload(body, request.headers.get("content-type"))
        if debug_enabled:
            store.record_inbound(
                request_id=request_id,
                job_name=job_name,
                trial_name=trial_name,
                correlation_id=correlation_id,
                method=request.method,
                path=upstream_path,
                headers=headers,
                payload=parsed_payload,
            )
        response = await litellm_forwarder(
            method=request.method,
            path=_path_with_query(upstream_path, request.url.query),
            headers=headers,
            body=body,
        )
        if debug_enabled:
            store.record_response(
                request_id=request_id,
                status_code=response.status_code,
                headers=response.headers,
                body=response.body,
            )
        return Response(
            content=response.body,
            status_code=response.status_code,
            headers=_filtered_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )

    @app.api_route("/openai-proxy/{upstream_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy_openai(request: Request, upstream_path: str) -> Response:
        body = await request.body()
        headers = dict(request.headers)
        request_id = headers.get("x-tinyharness-request-id") or store.latest_unpaired_request_id() or str(uuid.uuid4())
        job_name = headers.get("x-tinyharness-job-name")
        trial_name = headers.get("x-tinyharness-trial-name")
        correlation_id = headers.get("x-tinyharness-correlation-id")
        parsed_payload = _decode_payload(body, request.headers.get("content-type"))
        if debug_enabled:
            store.record_effective(
                request_id=request_id,
                job_name=job_name,
                trial_name=trial_name,
                correlation_id=correlation_id,
                method=request.method,
                path=upstream_path,
                headers=headers,
                payload=parsed_payload,
            )
        response = await llama_forwarder(
            method=request.method,
            path=_path_with_query(upstream_path, request.url.query),
            headers=headers,
            body=body,
        )
        return Response(
            content=response.body,
            status_code=response.status_code,
            headers=_filtered_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )

    @app.api_route("/{upstream_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy_passthrough(request: Request, upstream_path: str) -> Response:
        return await _proxy_to_litellm(
            request,
            upstream_path=upstream_path,
            job_name=None,
            trial_name=None,
        )

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the TinyHarness gateway debug proxy.")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--litellm-port", type=int, required=True)
    parser.add_argument("--llama-port", type=int, required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--debug-enabled", type=int, default=1)
    args = parser.parse_args(argv)

    app = create_app(
        api_key=args.api_key,
        litellm_base_url=f"http://127.0.0.1:{args.litellm_port}",
        llama_base_url=f"http://127.0.0.1:{args.llama_port}",
        debug_enabled=bool(args.debug_enabled),
    )
    uvicorn.run(app, host="0.0.0.0", port=args.listen_port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
