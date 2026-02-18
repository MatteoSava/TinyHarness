from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolResultBlock, ToolUseBlock, UserMessage

from tinyharness import sdk_runner


class _FakeClaudeSDKClient:
    messages = []

    def __init__(self, options) -> None:
        self.options = options
        self.connected = False
        self.queried_instruction: str | None = None

    async def connect(self) -> None:
        self.connected = True

    async def query(self, instruction: str) -> None:
        self.queried_instruction = instruction

    async def receive_response(self):
        for message in list(self.messages):
            yield message

    async def disconnect(self) -> None:
        self.connected = False


class _FakeSpan:
    def __init__(self, name: str, span_type: str, attributes: dict[str, object], trace_id: str) -> None:
        self.name = name
        self.span_type = span_type
        self.attributes = dict(attributes)
        self.trace_id = trace_id
        self.status: str | None = None
        self.recorded_exception: Exception | None = None
        self.inputs: object | None = None
        self.outputs: object | None = None

    def set_attributes(self, attributes: dict[str, object]) -> None:
        self.attributes.update(attributes)

    def set_inputs(self, inputs: object) -> None:
        self.inputs = inputs

    def set_outputs(self, outputs: object) -> None:
        self.outputs = outputs

    def record_exception(self, exc: Exception) -> None:
        self.recorded_exception = exc

    def set_status(self, status: str) -> None:
        self.status = status


class _FakeSpanManager:
    def __init__(self, collector: dict[str, object], name: str, span_type: str, attributes: dict[str, object]) -> None:
        self.collector = collector
        self.span = _FakeSpan(name, span_type, attributes, trace_id="trace-1")

    def __enter__(self) -> _FakeSpan:
        self.collector["started_spans"].append(self.span)
        return self.span

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.collector["ended_spans"].append(self.span)
        return False


class _FakeActiveRun:
    def __init__(self, run_id: str, collector: dict[str, object]) -> None:
        self.info = SimpleNamespace(run_id=run_id)
        self.collector = collector

    def __enter__(self) -> "_FakeActiveRun":
        self.collector["entered_runs"].append(self.info.run_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.collector["exited_runs"].append(self.info.run_id)
        return False


def test_sdk_runner_logs_live_mlflow_metrics_and_trace_artifacts(monkeypatch, tmp_path: Path) -> None:
    collector: dict[str, object] = {
        "start_run_calls": [],
        "entered_runs": [],
        "exited_runs": [],
        "ended_runs": 0,
        "params": [],
        "metrics": [],
        "trace_tags": [],
        "run_tags": [],
        "tracking_uris": [],
        "trace_destinations": [],
        "started_spans": [],
        "ended_spans": [],
    }

    def fake_start_run(**kwargs):
        collector["start_run_calls"].append(kwargs)
        if kwargs.get("run_id"):
            return _FakeActiveRun(kwargs["run_id"], collector)
        return _FakeActiveRun("child-run-1", collector)

    def fake_end_run() -> None:
        collector["ended_runs"] += 1

    def fake_start_span(name: str, span_type: str, attributes: dict[str, object] | None = None, **_kwargs):
        return _FakeSpanManager(collector, name, span_type, attributes or {})

    def fake_set_tracking_uri(uri: str) -> None:
        collector["tracking_uris"].append(uri)

    def fake_log_params(params: dict[str, object]) -> None:
        collector["params"].append(dict(params))

    def fake_log_metrics(metrics: dict[str, object]) -> None:
        collector["metrics"].append(dict(metrics))

    def fake_set_tag(key: str, value: str) -> None:
        collector["run_tags"].append((key, value))

    def fake_set_trace_tag(trace_id: str, key: str, value: str) -> None:
        collector["trace_tags"].append((trace_id, key, value))

    def fake_set_destination(destination, *, context_local: bool = False) -> None:
        collector["trace_destinations"].append((destination.experiment_id, context_local))

    times = iter([1_000, 1_100, 1_300, 1_600, 2_000])
    monkeypatch.setattr(sdk_runner, "_now_epoch_ms", lambda: next(times))
    monkeypatch.setattr(sdk_runner, "ClaudeSDKClient", _FakeClaudeSDKClient)
    monkeypatch.setattr(sdk_runner.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(sdk_runner.mlflow, "end_run", fake_end_run)
    monkeypatch.setattr(sdk_runner.mlflow, "start_span", fake_start_span)
    monkeypatch.setattr(sdk_runner.mlflow, "set_tracking_uri", fake_set_tracking_uri)
    monkeypatch.setattr(sdk_runner.mlflow, "log_params", fake_log_params)
    monkeypatch.setattr(sdk_runner.mlflow, "log_metrics", fake_log_metrics)
    monkeypatch.setattr(sdk_runner.mlflow, "set_tag", fake_set_tag)
    monkeypatch.setattr(sdk_runner.mlflow, "set_trace_tag", fake_set_trace_tag)
    monkeypatch.setattr(sdk_runner.mlflow.tracing, "set_destination", fake_set_destination)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "exp-123")
    monkeypatch.setenv("TINYHARNESS_PARENT_RUN_ID", "parent-run-1")
    monkeypatch.setenv("TINYHARNESS_JOB_NAME", "smoke-v0-20260312-120000")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")

    _FakeClaudeSDKClient.messages = [
        AssistantMessage(
            content=[
                TextBlock(text="Inspecting the workspace."),
                ToolUseBlock(id="tool-1", name="Bash", input={"command": "ls -la"}),
            ],
            model="qwen",
        ),
        UserMessage(
            content=[ToolResultBlock(tool_use_id="tool-1", content="file.txt")],
            parent_tool_use_id="tool-1",
            tool_use_result={"stdout": "file.txt"},
        ),
        AssistantMessage(content=[TextBlock(text="Done.")], model="qwen"),
        ResultMessage(
            subtype="success",
            duration_ms=1200,
            duration_api_ms=1100,
            is_error=False,
            num_turns=2,
            session_id="session-1",
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 4, "cache_read_input_tokens": 2},
            result="ok",
        ),
    ]

    logs_dir = tmp_path / "cancel-async-tasks__trial" / "agent"
    exit_code = asyncio.run(
        sdk_runner._run_sdk(
            instruction="fix the task",
            cwd="/app",
            logs_dir=logs_dir,
            max_turns=8,
            max_thinking_tokens=512,
            model="qwen3.5-35b-a3b-ud-iq3_s",
        )
    )

    assert exit_code == 0
    assert collector["tracking_uris"] == ["https://mlflow.example"]
    assert collector["trace_destinations"] == [("exp-123", True)]
    assert collector["ended_runs"] == 1

    create_run_call = collector["start_run_calls"][0]
    assert create_run_call["experiment_id"] == "exp-123"
    assert create_run_call["parent_run_id"] == "parent-run-1"
    assert create_run_call["nested"] is True

    trace_tags = set(collector["trace_tags"])
    assert ("trace-1", "mlflow.run_id", "child-run-1") in trace_tags
    assert ("trace-1", "task_name", "cancel-async-tasks") in trace_tags
    root_span = collector["started_spans"][0]
    assert root_span.inputs == {"instruction": "fix the task"}
    assert root_span.outputs == {"assistant_text": "Inspecting the workspace.\n\nDone."}
    tool_span = next(span for span in collector["started_spans"] if span.name == "tool.Bash")
    assert tool_span.inputs == {"command": "ls -la"}
    assert tool_span.outputs == {"stdout": "file.txt"}

    summary = json.loads((logs_dir / "summary.json").read_text(encoding="utf-8"))
    telemetry = summary["metadata"]["telemetry"]
    assert summary["metadata"]["mlflow_run_id"] == "child-run-1"
    assert summary["metadata"]["trace_id"] == "trace-1"
    assert telemetry["turn_count"] == 2
    assert telemetry["tool_call_count"] == 1
    assert telemetry["shell_command_count"] == 1
    assert telemetry["first_event_latency_ms"] == 100
    assert telemetry["response_complete_latency_ms"] == 1000
    assert telemetry["tool_output_bytes"] > 0
    assert telemetry["turns"][0]["tool_calls_in_turn"] == 1
    assert telemetry["turns"][0]["shell_commands_in_turn"] == 1

    trace_lines = [
        json.loads(line)
        for line in (logs_dir / "sdk-trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tool_result_events = [event for event in trace_lines if event["is_tool_result"]]
    assert len(tool_result_events) == 1
    assert tool_result_events[0]["tool_use_id"] == "tool-1"

    telemetry_artifact = json.loads((logs_dir / "telemetry.json").read_text(encoding="utf-8"))
    assert telemetry_artifact["tool_call_count"] == 1

    metric_keys = {key for payload in collector["metrics"] for key in payload}
    assert "tool_output_bytes" in metric_keys
    assert "average_turn_latency_ms" in metric_keys
