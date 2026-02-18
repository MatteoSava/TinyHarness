from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)
from mlflow.entities.trace_location import MlflowExperimentLocation

CLAUDE_CODE_PRESET = {"type": "preset", "preset": "claude_code"}


def build_sdk_options(
    *,
    cwd: str,
    max_turns: int | None,
    max_thinking_tokens: int | None,
    model: str | None,
) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        cwd=cwd,
        model=model,
        permission_mode="bypassPermissions",
        setting_sources=[],
        system_prompt=CLAUDE_CODE_PRESET,
        tools=CLAUDE_CODE_PRESET,
        max_turns=max_turns,
        max_thinking_tokens=max_thinking_tokens,
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in vars(value).items()}
    return value


def _message_to_dict(message: Any) -> dict[str, Any]:
    payload = _jsonable(message)
    payload["message_type"] = message.__class__.__name__
    return payload


def _assistant_text(messages: Iterable[Any]) -> str:
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                chunks.append(block.text.rstrip())
    return "\n\n".join(chunks)


def _usage_tokens(result: ResultMessage | None) -> tuple[int | None, int | None, int | None]:
    if result is None or not result.usage:
        return None, None, None

    usage = result.usage
    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")
    cache = usage.get("cache_read_input_tokens")

    return (
        int(prompt) if prompt is not None else None,
        int(completion) if completion is not None else None,
        int(cache) if cache is not None else None,
    )


def _now_epoch_ms() -> int:
    return int(time.time() * 1000)


def _iso_from_epoch_ms(value: int | None) -> str | None:
    if value is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(value / 1000.0)) + f".{value % 1000:03d}Z"


def _extract_tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block in payload.get("content", []):
        if not isinstance(block, dict):
            continue
        tool_id = block.get("id")
        tool_name = block.get("name")
        if not isinstance(tool_id, str) or not isinstance(tool_name, str):
            continue
        calls.append(
            {
                "tool_use_id": tool_id,
                "tool_name": tool_name,
                "input": block.get("input"),
            }
        )
    return calls


def _has_assistant_text(message: Any) -> bool:
    if not isinstance(message, AssistantMessage):
        return False
    return any(isinstance(block, TextBlock) and block.text.strip() for block in message.content)


def _tool_result_info(payload: dict[str, Any]) -> tuple[str | None, int]:
    tool_use_id = payload.get("tool_use_id")
    if not isinstance(tool_use_id, str):
        tool_use_id = payload.get("parent_tool_use_id")
    result_payload = _tool_result_payload(payload)
    if result_payload is None:
        result_payload = payload.get("content")
        if isinstance(result_payload, list):
            for block in result_payload:
                if not isinstance(block, dict):
                    continue
                block_tool_use_id = block.get("tool_use_id")
                if isinstance(block_tool_use_id, str):
                    tool_use_id = block_tool_use_id
                    break
    encoded = json.dumps(result_payload, ensure_ascii=False).encode("utf-8")
    return (tool_use_id if isinstance(tool_use_id, str) else None, len(encoded))


def _tool_result_payload(payload: dict[str, Any]) -> Any:
    result_payload = payload.get("tool_use_result")
    if result_payload is None:
        result_payload = payload.get("content")
        if isinstance(result_payload, list):
            for block in result_payload:
                if not isinstance(block, dict):
                    continue
                if "content" in block:
                    return block["content"]
    return result_payload


def _is_tool_result_message(payload: dict[str, Any]) -> bool:
    if payload.get("tool_use_result") is not None:
        return True
    if isinstance(payload.get("tool_use_id"), str) or isinstance(payload.get("parent_tool_use_id"), str):
        return True
    content = payload.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and isinstance(block.get("tool_use_id"), str) for block in content)


def _task_identity(logs_dir: Path) -> tuple[str, str]:
    trial_name = logs_dir.parent.name
    if "__" in trial_name:
        task_name = trial_name.split("__", 1)[0]
    else:
        task_name = trial_name
    return task_name, trial_name


@dataclass
class _SpanHandle:
    manager: Any
    span: Any
    name: str
    started_ms: int

    @classmethod
    def start(cls, name: str, span_type: str, *, started_ms: int, attributes: dict[str, Any] | None = None) -> "_SpanHandle":
        manager = mlflow.start_span(name=name, span_type=span_type, attributes=attributes or {})
        span = manager.__enter__()
        return cls(manager=manager, span=span, name=name, started_ms=started_ms)

    def close(self, *, ended_ms: int, attributes: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        attributes = dict(attributes or {})
        attributes.setdefault("start_epoch_ms", self.started_ms)
        attributes.setdefault("end_epoch_ms", ended_ms)
        attributes.setdefault("duration_ms", max(0, ended_ms - self.started_ms))
        self.span.set_attributes(attributes)
        if error is not None:
            self.span.record_exception(error)
            self.span.set_status("ERROR")
            self.manager.__exit__(type(error), error, error.__traceback__)
            return
        self.manager.__exit__(None, None, None)


@dataclass
class _TurnState:
    index: int
    started_ms: int
    pending_tool_results: int = 0
    tool_calls: int = 0
    shell_commands: int = 0
    span: _SpanHandle | None = None


@dataclass
class _StreamState:
    request_started_ms: int
    trace_path: Path
    first_event_ms: int | None = None
    first_text_ms: int | None = None
    response_completed_ms: int | None = None
    event_index: int = 0
    active_turn: _TurnState | None = None
    next_turn_index: int = 1
    turn_latencies_ms: list[int] = field(default_factory=list)
    turns: list[dict[str, int]] = field(default_factory=list)
    tool_call_count: int = 0
    shell_command_count: int = 0
    tool_output_bytes: int = 0
    active_tool_spans: dict[str, _SpanHandle] = field(default_factory=dict)

    def _write_event(
        self,
        *,
        payload: dict[str, Any],
        received_ms: int,
        turn_index: int | None,
        tool_name: str | None = None,
        tool_use_id: str | None = None,
        is_tool_call: bool = False,
        is_tool_result: bool = False,
    ) -> None:
        event = {
            "event_index": self.event_index,
            "received_at_iso": _iso_from_epoch_ms(received_ms),
            "received_at_epoch_ms": received_ms,
            "elapsed_ms_from_request_start": received_ms - self.request_started_ms,
            "message_type": payload.get("message_type"),
            "sdk_subtype": payload.get("subtype"),
            "turn_index": turn_index,
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "is_tool_call": is_tool_call,
            "is_tool_result": is_tool_result,
            "payload": payload,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.event_index += 1

    def start_turn(self, received_ms: int) -> _TurnState:
        turn = _TurnState(
            index=self.next_turn_index,
            started_ms=received_ms,
            span=_SpanHandle.start(
                name=f"turn.{self.next_turn_index}",
                span_type="CHAIN",
                started_ms=received_ms,
                attributes={"turn_index": self.next_turn_index},
            ),
        )
        self.next_turn_index += 1
        self.active_turn = turn
        return turn

    def finish_turn(self, received_ms: int) -> None:
        if self.active_turn is None:
            return
        latency_ms = max(0, received_ms - self.active_turn.started_ms)
        self.turn_latencies_ms.append(latency_ms)
        self.turns.append(
            {
                "turn_index": self.active_turn.index,
                "turn_start_epoch_ms": self.active_turn.started_ms,
                "turn_end_epoch_ms": received_ms,
                "turn_latency_ms": latency_ms,
                "tool_calls_in_turn": self.active_turn.tool_calls,
                "shell_commands_in_turn": self.active_turn.shell_commands,
            }
        )
        if self.active_turn.span is not None:
            self.active_turn.span.close(
                ended_ms=received_ms,
                attributes={
                    "turn_index": self.active_turn.index,
                    "tool_calls": self.active_turn.tool_calls,
                    "shell_commands": self.active_turn.shell_commands,
                },
            )
        self.active_turn = None


async def _run_sdk(
    *,
    instruction: str,
    cwd: str,
    logs_dir: Path,
    max_turns: int | None,
    max_thinking_tokens: int | None,
    model: str | None,
) -> int:
    logs_dir.mkdir(parents=True, exist_ok=True)
    trace_path = logs_dir / "sdk-trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    options = build_sdk_options(
        cwd=cwd,
        max_turns=max_turns,
        max_thinking_tokens=max_thinking_tokens,
        model=model,
    )

    raw_messages: list[Any] = []
    result_message: ResultMessage | None = None
    request_started_ms = _now_epoch_ms()
    stream_state = _StreamState(request_started_ms=request_started_ms, trace_path=trace_path)
    task_name, trial_name = _task_identity(logs_dir)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    tracking_enabled = bool(tracking_uri)
    root_span: _SpanHandle | None = None
    active_run = None
    mlflow_run_id: str | None = None
    trace_id: str | None = None

    if tracking_enabled:
        experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID") or None
        parent_run_id = os.environ.get("TINYHARNESS_PARENT_RUN_ID") or None
        mlflow.set_tracking_uri(tracking_uri)
        if experiment_id is not None:
            mlflow.tracing.set_destination(MlflowExperimentLocation(experiment_id), context_local=True)
        active_run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=trial_name,
            nested=parent_run_id is not None,
            parent_run_id=parent_run_id,
            tags={
                "task_name": task_name,
                "trial_name": trial_name,
                "job_name": os.environ.get("TINYHARNESS_JOB_NAME", ""),
                "runner": "sdk_runner",
            },
        )
        mlflow_run_id = active_run.info.run_id
        mlflow.log_params(
            {
                "task_name": task_name,
                "trial_name": trial_name,
                "model_name": model or "",
                "max_turns": max_turns or 0,
                "max_thinking_tokens": max_thinking_tokens or 0,
            }
        )
        root_span = _SpanHandle.start(
            name=f"task.{task_name}",
            span_type="AGENT",
            started_ms=request_started_ms,
            attributes={
                "task_name": task_name,
                "trial_name": trial_name,
                "mlflow_run_id": mlflow_run_id or "",
                "request_start_epoch_ms": request_started_ms,
            },
        )
        root_span.span.set_inputs(
            {
                "instruction": instruction,
            }
        )
        trace_id = str(root_span.span.trace_id)
        mlflow.set_tag("trace_id", trace_id)
        mlflow.set_trace_tag(trace_id, "mlflow.run_id", mlflow_run_id or "")
        mlflow.set_trace_tag(trace_id, "task_name", task_name)
        mlflow.set_trace_tag(trace_id, "trial_name", trial_name)
        mlflow.set_trace_tag(trace_id, "job_name", os.environ.get("TINYHARNESS_JOB_NAME", ""))

    client = ClaudeSDKClient(options=options)
    await client.connect()
    error: Exception | None = None
    assistant_text = ""
    try:
        await client.query(instruction)
        with (logs_dir / "sdk-messages.jsonl").open("w", encoding="utf-8") as raw_handle:
            async for message in client.receive_response():
                raw_messages.append(message)
                payload = _message_to_dict(message)
                raw_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                raw_handle.flush()

                received_ms = _now_epoch_ms()
                if stream_state.first_event_ms is None:
                    stream_state.first_event_ms = received_ms
                if stream_state.first_text_ms is None and _has_assistant_text(message):
                    stream_state.first_text_ms = received_ms

                message_type = payload.get("message_type")
                tool_calls = _extract_tool_calls(payload)

                if message_type == "AssistantMessage":
                    if stream_state.active_turn is None:
                        stream_state.start_turn(received_ms)
                    turn_index = stream_state.active_turn.index if stream_state.active_turn else None
                    if tool_calls:
                        for call in tool_calls:
                            stream_state.tool_call_count += 1
                            if stream_state.active_turn is not None:
                                stream_state.active_turn.tool_calls += 1
                                stream_state.active_turn.pending_tool_results += 1
                            if call["tool_name"] == "Bash":
                                stream_state.shell_command_count += 1
                                if stream_state.active_turn is not None:
                                    stream_state.active_turn.shell_commands += 1
                            stream_state._write_event(
                                payload=payload,
                                received_ms=received_ms,
                                turn_index=turn_index,
                                tool_name=call["tool_name"],
                                tool_use_id=call["tool_use_id"],
                                is_tool_call=True,
                            )
                            if root_span is not None and stream_state.active_turn is not None:
                                tool_span = _SpanHandle.start(
                                    name=f"tool.{call['tool_name']}",
                                    span_type="TOOL",
                                    started_ms=received_ms,
                                    attributes={
                                        "turn_index": stream_state.active_turn.index,
                                        "tool_name": call["tool_name"],
                                        "tool_use_id": call["tool_use_id"],
                                        "is_shell_command": int(call["tool_name"] == "Bash"),
                                    },
                                )
                                tool_span.span.set_inputs(call["input"])
                                stream_state.active_tool_spans[call["tool_use_id"]] = tool_span
                    else:
                        stream_state._write_event(
                            payload=payload,
                            received_ms=received_ms,
                            turn_index=turn_index,
                        )
                elif message_type == "UserMessage" and _is_tool_result_message(payload):
                    tool_use_id, output_bytes = _tool_result_info(payload)
                    stream_state.tool_output_bytes += output_bytes
                    turn_index = stream_state.active_turn.index if stream_state.active_turn else None
                    tool_name = None
                    if tool_use_id and tool_use_id in stream_state.active_tool_spans:
                        tool_span = stream_state.active_tool_spans.pop(tool_use_id)
                        tool_name = tool_span.name.removeprefix("tool.")
                        tool_span.span.set_outputs(_tool_result_payload(payload))
                        tool_span.close(
                            ended_ms=received_ms,
                            attributes={
                                "turn_index": turn_index,
                                "tool_output_bytes": output_bytes,
                            },
                        )
                    stream_state._write_event(
                        payload=payload,
                        received_ms=received_ms,
                        turn_index=turn_index,
                        tool_name=tool_name,
                        tool_use_id=tool_use_id,
                        is_tool_result=True,
                    )
                    if stream_state.active_turn is not None:
                        if stream_state.active_turn.pending_tool_results > 0:
                            stream_state.active_turn.pending_tool_results -= 1
                        if stream_state.active_turn.pending_tool_results <= 0:
                            stream_state.finish_turn(received_ms)
                else:
                    turn_index = stream_state.active_turn.index if stream_state.active_turn else None
                    stream_state._write_event(
                        payload=payload,
                        received_ms=received_ms,
                        turn_index=turn_index,
                    )

                if isinstance(message, ResultMessage):
                    result_message = message
                    stream_state.response_completed_ms = received_ms
                    stream_state.finish_turn(received_ms)
    except Exception as exc:
        error = exc
        raise
    finally:
        await client.disconnect()
        if stream_state.response_completed_ms is None:
            stream_state.response_completed_ms = _now_epoch_ms()
        for tool_span in list(stream_state.active_tool_spans.values()):
            tool_span.close(
                ended_ms=stream_state.response_completed_ms,
                attributes={"tool_output_bytes": 0},
                error=error,
            )
        stream_state.active_tool_spans.clear()
        stream_state.finish_turn(stream_state.response_completed_ms)
        if root_span is not None:
            assistant_text = _assistant_text(raw_messages)
            root_span.span.set_outputs({"assistant_text": assistant_text})
            root_span.close(
                ended_ms=stream_state.response_completed_ms,
                attributes={
                    "task_name": task_name,
                    "trial_name": trial_name,
                    "tool_call_count": stream_state.tool_call_count,
                    "shell_command_count": stream_state.shell_command_count,
                    "turn_count": len(stream_state.turn_latencies_ms),
                    "tool_output_bytes": stream_state.tool_output_bytes,
                },
                error=error,
            )
        if active_run is not None:
            mlflow.end_run()

    if assistant_text:
        print(assistant_text)

    prompt_tokens, completion_tokens, cache_tokens = _usage_tokens(result_message)
    average_turn_latency_ms = (
        sum(stream_state.turn_latencies_ms) / len(stream_state.turn_latencies_ms)
        if stream_state.turn_latencies_ms
        else None
    )
    max_turn_latency_ms = max(stream_state.turn_latencies_ms) if stream_state.turn_latencies_ms else None
    tool_output_tokens_estimate = math.ceil(stream_state.tool_output_bytes / 4) if stream_state.tool_output_bytes else 0

    telemetry = {
        "request_started_at_iso": _iso_from_epoch_ms(request_started_ms),
        "request_started_at_epoch_ms": request_started_ms,
        "first_event_at_iso": _iso_from_epoch_ms(stream_state.first_event_ms),
        "first_event_at_epoch_ms": stream_state.first_event_ms,
        "first_event_latency_ms": (
            stream_state.first_event_ms - request_started_ms if stream_state.first_event_ms is not None else None
        ),
        "first_text_at_iso": _iso_from_epoch_ms(stream_state.first_text_ms),
        "first_text_at_epoch_ms": stream_state.first_text_ms,
        "first_text_latency_ms": (
            stream_state.first_text_ms - request_started_ms if stream_state.first_text_ms is not None else None
        ),
        "response_completed_at_iso": _iso_from_epoch_ms(stream_state.response_completed_ms),
        "response_completed_at_epoch_ms": stream_state.response_completed_ms,
        "response_complete_latency_ms": (
            stream_state.response_completed_ms - request_started_ms
            if stream_state.response_completed_ms is not None
            else None
        ),
        "turn_count": len(stream_state.turn_latencies_ms),
        "turns": stream_state.turns,
        "per_turn_latencies_ms": stream_state.turn_latencies_ms,
        "average_turn_latency_ms": average_turn_latency_ms,
        "max_turn_latency_ms": max_turn_latency_ms,
        "tool_call_count": stream_state.tool_call_count,
        "shell_command_count": stream_state.shell_command_count,
        "tool_output_bytes": stream_state.tool_output_bytes,
        "tool_output_tokens_estimate": tool_output_tokens_estimate,
    }

    metadata = {
        "duration_ms": result_message.duration_ms if result_message else None,
        "duration_api_ms": result_message.duration_api_ms if result_message else None,
        "num_turns": result_message.num_turns if result_message else None,
        "stop_reason": result_message.stop_reason if result_message else None,
        "session_id": result_message.session_id if result_message else None,
        "result": result_message.result if result_message else None,
        "is_error": result_message.is_error if result_message else True,
        "gateway_base_url": os.environ.get("ANTHROPIC_BASE_URL"),
        "model": model,
        "instruction": instruction,
        "mlflow_run_id": mlflow_run_id,
        "trace_id": trace_id,
        "telemetry": telemetry,
    }
    summary = {
        "n_input_tokens": prompt_tokens,
        "n_cache_tokens": cache_tokens,
        "n_output_tokens": completion_tokens,
        "cost_usd": result_message.total_cost_usd if result_message else None,
        "is_error": result_message.is_error if result_message else True,
        "metadata": metadata,
    }

    (logs_dir / "assistant.txt").write_text(assistant_text, encoding="utf-8")
    (logs_dir / "telemetry.json").write_text(
        json.dumps(telemetry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (logs_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if tracking_enabled and mlflow_run_id is not None:
        with mlflow.start_run(run_id=mlflow_run_id):
            metrics = {
                "turn_count": len(stream_state.turn_latencies_ms),
                "tool_call_count": stream_state.tool_call_count,
                "shell_command_count": stream_state.shell_command_count,
                "tool_output_bytes": stream_state.tool_output_bytes,
                "tool_output_tokens_estimate": tool_output_tokens_estimate,
                "prompt_tokens": prompt_tokens or 0,
                "cache_tokens": cache_tokens or 0,
                "output_tokens": completion_tokens or 0,
            }
            if telemetry["first_event_latency_ms"] is not None:
                metrics["first_event_latency_ms"] = telemetry["first_event_latency_ms"]
            if telemetry["first_text_latency_ms"] is not None:
                metrics["first_text_latency_ms"] = telemetry["first_text_latency_ms"]
            if telemetry["response_complete_latency_ms"] is not None:
                metrics["response_complete_latency_ms"] = telemetry["response_complete_latency_ms"]
            if average_turn_latency_ms is not None:
                metrics["average_turn_latency_ms"] = average_turn_latency_ms
            if max_turn_latency_ms is not None:
                metrics["max_turn_latency_ms"] = max_turn_latency_ms
            if result_message and result_message.duration_ms and completion_tokens:
                metrics["tokens_per_second"] = completion_tokens / (result_message.duration_ms / 1000.0)
            mlflow.log_metrics(metrics)

    if result_message and result_message.is_error:
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Claude Agent SDK against one instruction.")
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-thinking-tokens", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("instruction")
    args = parser.parse_args(argv)

    exit_code = asyncio.run(
        _run_sdk(
            instruction=args.instruction,
            cwd=args.cwd,
            logs_dir=Path(args.logs_dir),
            max_turns=args.max_turns,
            max_thinking_tokens=args.max_thinking_tokens,
            model=args.model,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
