from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.server import get_app_client

from tinyharness import __version__
from tinyharness.config import (
    AppConfig,
    BenchmarkMode,
    ConfigError,
    TrackingConfig,
    ensure_state_dirs,
    resolve_mlflow_password,
    resolve_tracking_uri,
)
from tinyharness.results import JobSummary, TrialSummary


DEFAULT_MLFLOW_ADMIN_PASSWORD = "password1234"  # bootstrap default, replaced on first deploy


@dataclass(frozen=True)
class ParentRunInfo:
    run_id: str
    experiment_id: str
    tracking_uri: str


@dataclass(frozen=True)
class ChildRunInfo:
    run_id: str
    trace_id: str | None


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    value = (result.stdout or "").strip()
    return value or "dev"


def _package_version(name: str, default: str = "unknown") -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return default


def _terminal_bench_commit(config: AppConfig) -> str:
    return os.environ.get("TINYHARNESS_TERMINAL_BENCH_COMMIT", config.benchmark.dataset)


@contextlib.contextmanager
def _tracking_credentials(config: TrackingConfig) -> Iterator[None]:
    previous_user = os.environ.get("MLFLOW_TRACKING_USERNAME")
    previous_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    os.environ["MLFLOW_TRACKING_USERNAME"] = config.admin_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = resolve_mlflow_password(config)
    try:
        yield
    finally:
        if previous_user is None:
            os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
        else:
            os.environ["MLFLOW_TRACKING_USERNAME"] = previous_user
        if previous_password is None:
            os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
        else:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = previous_password


@contextlib.contextmanager
def _specific_tracking_credentials(username: str, password: str) -> Iterator[None]:
    previous_user = os.environ.get("MLFLOW_TRACKING_USERNAME")
    previous_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    try:
        yield
    finally:
        if previous_user is None:
            os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
        else:
            os.environ["MLFLOW_TRACKING_USERNAME"] = previous_user
        if previous_password is None:
            os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
        else:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = previous_password


def _tracking_uri(config: AppConfig) -> str:
    ensure_state_dirs(config)
    uri = resolve_tracking_uri(config.tracking)
    mlflow.set_tracking_uri(uri)
    return uri


def _is_remote_tracking_uri(tracking_uri: str) -> bool:
    return tracking_uri.startswith("http://") or tracking_uri.startswith("https://")


def _ensure_experiment(config: AppConfig) -> tuple[str, str]:
    uri = _tracking_uri(config)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(config.tracking.experiment_name)
    if experiment is not None:
        return experiment.experiment_id, uri

    if uri.startswith("http://") or uri.startswith("https://"):
        experiment_id = client.create_experiment(name=config.tracking.experiment_name)
    else:
        experiment_id = client.create_experiment(
            name=config.tracking.experiment_name,
            artifact_location=config.tracking.artifact_root.resolve().as_uri(),
        )
    return experiment_id, uri


def _log_common_tags(config: AppConfig, *, live_tracing_enabled: bool) -> None:
    mlflow.set_tags(
        {
            "benchmark_suite": config.benchmark.dataset,
            "task_set": config.benchmark.task_set_name,
            "frontend": "claude_code_preset",
            "backend_model": config.model.model_alias,
            "gpu": config.model.gpu,
            "runner": config.benchmark.runner,
            "run_mode": config.benchmark.mode.value,
            "gateway_debug_enabled": str(bool(config.model.gateway_debug)).lower(),
            "live_tracing_enabled": str(bool(live_tracing_enabled)).lower(),
        }
    )


def _task_names_param(tasks: tuple[str, ...] | None) -> str:
    if not tasks:
        return ""
    return ",".join(tasks)


def _task_selection_param(config: AppConfig) -> str:
    if config.benchmark.tasks:
        return "explicit"
    return "all"


def _benchmark_metric_name(name: str) -> str:
    return f"benchmark.{name}"


def _trial_metric_name(name: str) -> str:
    return f"trial.{name}"


def _harbor_config_sha256(harbor_config_path: Path) -> str:
    payload = json.loads(harbor_config_path.read_text(encoding="utf-8"))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _live_tracing_enabled(config: AppConfig) -> bool:
    return config.benchmark.mode == BenchmarkMode.DEBUG


def _log_trial_artifacts_enabled(config: AppConfig) -> bool:
    return config.benchmark.mode == BenchmarkMode.DEBUG


def create_parent_run(
    *,
    config: AppConfig,
    job_name: str,
    server_config_path: Path,
    harbor_config_path: Path,
) -> ParentRunInfo:
    with _tracking_credentials(config.tracking):
        experiment_id, tracking_uri = _ensure_experiment(config)
        live_tracing_enabled = _live_tracing_enabled(config) and _is_remote_tracking_uri(tracking_uri)
        with mlflow.start_run(experiment_id=experiment_id, run_name=job_name) as run:
            _log_common_tags(config, live_tracing_enabled=live_tracing_enabled)
            mlflow.set_tag("run_kind", "benchmark")
            mlflow.log_params(
                {
                    "run_mode": config.benchmark.mode.value,
                    "model_repo": config.model.hf_repo_id,
                    "model_file": config.model.hf_filename,
                    "model_alias": config.model.model_alias,
                    "context_window": config.model.context_window,
                    "temperature": config.model.temperature,
                    "top_p": config.model.top_p,
                    "top_k": config.model.top_k,
                    "seed": config.model.seed,
                    "cache_prompt": config.model.cache_prompt,
                    "parallel_requests": config.model.parallel_requests,
                    "gateway_max_containers": config.model.max_containers,
                    "gateway_scaledown_window_sec": config.model.scaledown_window_sec,
                    "gateway_debug_enabled": int(bool(config.model.gateway_debug)),
                    "live_tracing_enabled": int(live_tracing_enabled),
                    "mlflow_server_max_containers": config.tracking.server_max_containers,
                    "mlflow_server_scaledown_window_sec": config.tracking.server_scaledown_window_sec,
                    "task_selection": _task_selection_param(config),
                    "task_names": _task_names_param(config.benchmark.tasks),
                    "configured_n_tasks": config.benchmark.n_tasks or 0,
                    "benchmark_dataset": config.benchmark.dataset,
                    "benchmark_runner": config.benchmark.runner,
                    "agent_import_path": config.agent.import_path,
                    "max_turns": config.agent.max_turns,
                    "max_thinking_tokens": config.agent.max_thinking_tokens,
                    "tinyharness_version": _git_commit() or __version__,
                    "claude_agent_sdk_version": _package_version("claude-agent-sdk"),
                    "harbor_version": _package_version("harbor"),
                    "terminal_bench_commit": _terminal_bench_commit(config),
                    "mlflow_tracking_uri": tracking_uri,
                }
            )
            mlflow.log_artifact(server_config_path.as_posix(), artifact_path="run")
            if harbor_config_path.exists():
                mlflow.log_artifact(harbor_config_path.as_posix(), artifact_path="run")
            return ParentRunInfo(
                run_id=run.info.run_id,
                experiment_id=experiment_id,
                tracking_uri=tracking_uri,
            )


def _trial_results_row(trial: TrialSummary, child_run: ChildRunInfo) -> dict[str, object | None]:
    return {
        "task_name": trial.task_name,
        "trial_name": trial.trial_name,
        "mlflow_run_id": child_run.run_id,
        "trace_id": child_run.trace_id,
        "model_name": trial.model_name,
        "passed": trial.passed,
        "solved": trial.solved,
        "pass_at_1": trial.pass_at_1,
        "timeout": trial.timeout,
        "failure_type": trial.failure_type,
        "run_mode": trial.run_mode,
        "score": trial.score,
        "wall_clock_seconds": trial.wall_clock_seconds,
        "duration_ms": trial.duration_ms,
        "duration_api_ms": trial.duration_api_ms,
        "prompt_tokens": trial.prompt_tokens,
        "cache_tokens": trial.cache_tokens,
        "output_tokens": trial.output_tokens,
        "tokens_per_second": trial.tokens_per_second,
        "turn_count": trial.turn_count,
        "average_turn_latency_ms": trial.average_turn_latency_ms,
        "max_turn_latency_ms": trial.max_turn_latency_ms,
        "tool_call_count": trial.tool_call_count,
        "shell_command_count": trial.shell_command_count,
        "tool_output_bytes": trial.tool_output_bytes,
        "tool_output_tokens_estimate": trial.tool_output_tokens_estimate,
        "first_event_latency_ms": trial.first_event_latency_ms,
        "first_text_latency_ms": trial.first_text_latency_ms,
        "response_complete_latency_ms": trial.response_complete_latency_ms,
    }


def _write_trial_results_table(summary: JobSummary, child_runs: dict[str, ChildRunInfo]) -> Path:
    output_path = summary.job_dir / "trial-results.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for trial in summary.trials:
            child_run = child_runs[trial.trial_name]
            handle.write(json.dumps(_trial_results_row(trial, child_run), ensure_ascii=False) + "\n")
    return output_path


def tracking_environment(config: AppConfig, parent_run: ParentRunInfo, *, job_name: str) -> dict[str, str]:
    if not _live_tracing_enabled(config):
        return {}
    if not _is_remote_tracking_uri(parent_run.tracking_uri):
        return {}
    return {
        "MLFLOW_TRACKING_URI": parent_run.tracking_uri,
        "MLFLOW_TRACKING_USERNAME": config.tracking.admin_username,
        "MLFLOW_TRACKING_PASSWORD": resolve_mlflow_password(config.tracking),
        "MLFLOW_EXPERIMENT_ID": parent_run.experiment_id,
        "TINYHARNESS_PARENT_RUN_ID": parent_run.run_id,
        "TINYHARNESS_JOB_NAME": job_name,
    }


def _load_sdk_trace_events(trial: TrialSummary) -> list[dict[str, object]]:
    trace_path = trial.trial_dir / "agent" / "sdk-trace.jsonl"
    if not trace_path.exists():
        return []
    events: list[dict[str, object]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _tool_result_payload(payload: dict[str, object]) -> object | None:
    result_payload = payload.get("tool_use_result")
    if result_payload is not None:
        return result_payload
    content = payload.get("content")
    if not isinstance(content, list):
        return content
    for block in content:
        if isinstance(block, dict) and "content" in block:
            return block["content"]
    return content


def _tool_result_use_id(event: dict[str, object]) -> str | None:
    tool_use_id = event.get("tool_use_id")
    if isinstance(tool_use_id, str):
        return tool_use_id
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    parent_tool_use_id = payload.get("parent_tool_use_id")
    if isinstance(parent_tool_use_id, str):
        return parent_tool_use_id
    content = payload.get("content")
    if not isinstance(content, list):
        return None
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("tool_use_id"), str):
            return block["tool_use_id"]
    return None


def _tool_call_input(payload: dict[str, object], tool_use_id: str) -> object | None:
    content = payload.get("content")
    if not isinstance(content, list):
        return None
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("id") == tool_use_id:
            return block.get("input")
    return None


def _trial_instruction(trial: TrialSummary, metadata: dict[str, object]) -> str | None:
    instruction = metadata.get("instruction")
    if isinstance(instruction, str) and instruction:
        return instruction

    command_path = trial.trial_dir / "agent" / "command-0" / "command.txt"
    if not command_path.exists():
        return None

    command = command_path.read_text(encoding="utf-8")
    marker = " 2>&1 | stdbuf -oL tee "
    if marker not in command:
        return None
    prefix = command.split(marker, 1)[0].rstrip()
    python_command = prefix.rsplit(";", 1)[-1].strip()
    if not python_command:
        return None
    parts = shlex.split(python_command)
    return parts[-1] if parts else None


def _posthoc_trace(
    *,
    experiment_id: str,
    run_id: str,
    trial: TrialSummary,
) -> str | None:
    metadata = trial.raw.agent_result.metadata if trial.raw.agent_result and trial.raw.agent_result.metadata else {}
    telemetry = metadata.get("telemetry") if isinstance(metadata, dict) else {}
    if not isinstance(telemetry, dict):
        telemetry = {}
    instruction = _trial_instruction(trial, metadata if isinstance(metadata, dict) else {})
    assistant_text_path = trial.trial_dir / "agent" / "assistant.txt"
    assistant_text = assistant_text_path.read_text(encoding="utf-8") if assistant_text_path.exists() else ""

    started_ms = telemetry.get("request_started_at_epoch_ms")
    completed_ms = telemetry.get("response_completed_at_epoch_ms")
    turns = telemetry.get("turns", [])
    if started_ms is None or completed_ms is None:
        return None

    root_span = mlflow.start_span_no_context(
        name=f"task.{trial.task_name}",
        span_type="AGENT",
        experiment_id=experiment_id,
        start_time_ns=int(started_ms) * 1_000_000,
        attributes={
            "task_name": trial.task_name,
            "trial_name": trial.trial_name,
            "mlflow_run_id": run_id,
        },
    )
    if instruction is not None:
        root_span.set_inputs({"instruction": instruction})
    if assistant_text:
        root_span.set_outputs({"assistant_text": assistant_text})
    trace_id = str(root_span.trace_id)
    mlflow.set_trace_tag(trace_id, "mlflow.run_id", run_id)
    mlflow.set_trace_tag(trace_id, "task_name", trial.task_name)
    mlflow.set_trace_tag(trace_id, "trial_name", trial.trial_name)

    for turn in turns if isinstance(turns, list) else []:
        if not isinstance(turn, dict):
            continue
        start_ms = turn.get("turn_start_epoch_ms")
        end_ms = turn.get("turn_end_epoch_ms")
        if start_ms is None or end_ms is None:
            continue
        turn_span = mlflow.start_span_no_context(
            name=f"turn.{turn.get('turn_index', 'unknown')}",
            span_type="CHAIN",
            parent_span=root_span,
            start_time_ns=int(start_ms) * 1_000_000,
            attributes={
                "turn_index": turn.get("turn_index"),
                "tool_calls": turn.get("tool_calls_in_turn", 0),
                "shell_commands": turn.get("shell_commands_in_turn", 0),
            },
        )
        turn_span.end(
            attributes={
                "duration_ms": max(0, int(end_ms) - int(start_ms)),
            },
            end_time_ns=int(end_ms) * 1_000_000,
        )

    active_tool_spans: dict[str, object] = {}
    for event in _load_sdk_trace_events(trial):
        received_ms = event.get("received_at_epoch_ms")
        tool_use_id = event.get("tool_use_id")
        tool_name = event.get("tool_name") or "tool"
        if not isinstance(received_ms, int):
            continue
        if event.get("is_tool_call") is True:
            if not isinstance(tool_use_id, str):
                continue
            tool_span = mlflow.start_span_no_context(
                name=f"tool.{tool_name}",
                span_type="TOOL",
                parent_span=root_span,
                start_time_ns=received_ms * 1_000_000,
                attributes={
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "turn_index": event.get("turn_index"),
                    "is_shell_command": int(tool_name == "Bash"),
                },
            )
            payload = event.get("payload")
            if isinstance(payload, dict):
                tool_input = _tool_call_input(payload, tool_use_id)
                if tool_input is not None:
                    tool_span.set_inputs(tool_input)
            active_tool_spans[tool_use_id] = tool_span
        elif event.get("is_tool_result") is True:
            tool_use_id = _tool_result_use_id(event)
            if not isinstance(tool_use_id, str) or tool_use_id not in active_tool_spans:
                continue
            tool_span = active_tool_spans.pop(tool_use_id)
            payload = event.get("payload")
            if isinstance(payload, dict):
                tool_output = _tool_result_payload(payload)
                if tool_output is not None:
                    tool_span.set_outputs(tool_output)
            output_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            tool_span.end(
                attributes={
                    "tool_output_bytes": output_bytes,
                    "turn_index": event.get("turn_index"),
                },
                end_time_ns=received_ms * 1_000_000,
            )

    for tool_span in active_tool_spans.values():
        tool_span.end(end_time_ns=int(completed_ms) * 1_000_000)

    root_span.end(
        attributes={
            "tool_call_count": trial.tool_call_count,
            "shell_command_count": trial.shell_command_count,
            "turn_count": trial.turn_count,
            "tool_output_bytes": trial.tool_output_bytes,
        },
        end_time_ns=int(completed_ms) * 1_000_000,
    )
    return trace_id


def _is_missing_run_error(exc: MlflowException) -> bool:
    error_code = getattr(exc, "error_code", None)
    if error_code == "RESOURCE_DOES_NOT_EXIST":
        return True
    return "run with id=" in str(exc).lower() and "not found" in str(exc).lower()


def _find_existing_child_run(parent_run: ParentRunInfo, trial: TrialSummary):
    client = MlflowClient(tracking_uri=parent_run.tracking_uri)
    runs = client.search_runs(
        [parent_run.experiment_id],
        filter_string=(
            f"tags.mlflow.parentRunId = '{parent_run.run_id}' "
            f"AND tags.trial_name = '{trial.trial_name}'"
        ),
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def _existing_trace_id(*, experiment_id: str, run_id: str) -> str | None:
    traces = mlflow.search_traces(
        run_id=run_id,
        locations=[experiment_id],
        return_type="list",
        max_results=1,
    )
    if not traces:
        return None
    return traces[0].info.trace_id


def _ensure_trace(*, experiment_id: str, run_id: str, trial: TrialSummary) -> str | None:
    trace_id = _existing_trace_id(experiment_id=experiment_id, run_id=run_id)
    if trace_id is not None:
        return trace_id
    return _posthoc_trace(experiment_id=experiment_id, run_id=run_id, trial=trial)


def _create_child_run(config: AppConfig, parent_run: ParentRunInfo, trial: TrialSummary) -> ChildRunInfo:
    existing_run = _find_existing_child_run(parent_run, trial)
    if existing_run is not None:
        with mlflow.start_run(run_id=existing_run.info.run_id):
            _log_trial_metrics(trial)
            trace_id = (
                _ensure_trace(
                    experiment_id=parent_run.experiment_id,
                    run_id=existing_run.info.run_id,
                    trial=trial,
                )
                if _live_tracing_enabled(config)
                else None
            )
            if trace_id is not None:
                mlflow.set_tag("trace_id", trace_id)
            if _log_trial_artifacts_enabled(config):
                mlflow.log_artifacts(trial.trial_dir.as_posix(), artifact_path="trial")
        return ChildRunInfo(run_id=existing_run.info.run_id, trace_id=trace_id)

    with mlflow.start_run(
        experiment_id=parent_run.experiment_id,
        run_name=trial.trial_name,
        nested=True,
        parent_run_id=parent_run.run_id,
    ) as child_run:
        _log_trial_metrics(trial)
        trace_id = (
            _ensure_trace(
                experiment_id=parent_run.experiment_id,
                run_id=child_run.info.run_id,
                trial=trial,
            )
            if _live_tracing_enabled(config)
            else None
        )
        if trace_id is not None:
            mlflow.set_tag("trace_id", trace_id)
        if _log_trial_artifacts_enabled(config):
            mlflow.log_artifacts(trial.trial_dir.as_posix(), artifact_path="trial")
        return ChildRunInfo(run_id=child_run.info.run_id, trace_id=trace_id)


def _log_trial_run(config: AppConfig, parent_run: ParentRunInfo, trial: TrialSummary) -> ChildRunInfo:
    if trial.mlflow_run_id is not None:
        try:
            with mlflow.start_run(run_id=trial.mlflow_run_id):
                _log_trial_metrics(trial)
                trace_id = (
                    _ensure_trace(
                        experiment_id=parent_run.experiment_id,
                        run_id=trial.mlflow_run_id,
                        trial=trial,
                    )
                    if _live_tracing_enabled(config)
                    else None
                )
                if trace_id is not None:
                    mlflow.set_tag("trace_id", trace_id)
                if _log_trial_artifacts_enabled(config):
                    mlflow.log_artifacts(trial.trial_dir.as_posix(), artifact_path="trial")
            return ChildRunInfo(run_id=trial.mlflow_run_id, trace_id=trace_id)
        except MlflowException as exc:
            if not _is_missing_run_error(exc):
                raise

    return _create_child_run(config, parent_run, trial)


def _log_parent_metrics(summary: JobSummary) -> None:
    n_trials = len(summary.trials)
    if n_trials == 0:
        return
    timeout_count = sum(trial.timeout for trial in summary.trials)
    mlflow.log_metric(_benchmark_metric_name("solved_pct"), sum(trial.solved for trial in summary.trials) / n_trials)
    mlflow.log_metric(_benchmark_metric_name("pass_at_1"), sum(trial.pass_at_1 for trial in summary.trials) / n_trials)
    mlflow.log_metric(_benchmark_metric_name("timeout_pct"), timeout_count / n_trials)
    mlflow.log_metric(_benchmark_metric_name("mean_score"), summary.mean_score)
    mlflow.log_metric(_benchmark_metric_name("n_trials"), n_trials)
    mlflow.log_metric(_benchmark_metric_name("n_errors"), summary.job_result.stats.n_errors)
    mlflow.log_metric(
        _benchmark_metric_name("avg_wall_clock_seconds"),
        sum((trial.wall_clock_seconds or 0.0) for trial in summary.trials) / n_trials,
    )
    mlflow.log_metric(
        _benchmark_metric_name("avg_turns_per_task"),
        sum(trial.turn_count for trial in summary.trials) / n_trials,
    )
    mlflow.log_metric(
        _benchmark_metric_name("avg_tool_calls_per_task"),
        sum(trial.tool_call_count for trial in summary.trials) / n_trials,
    )
    mlflow.log_metric(
        _benchmark_metric_name("avg_shell_commands_per_task"),
        sum(trial.shell_command_count for trial in summary.trials) / n_trials,
    )
    mlflow.log_metric(
        _benchmark_metric_name("avg_tool_output_bytes_per_task"),
        sum(trial.tool_output_bytes for trial in summary.trials) / n_trials,
    )
    mlflow.log_metric(
        _benchmark_metric_name("avg_tool_output_tokens_estimate_per_task"),
        sum(trial.tool_output_tokens_estimate for trial in summary.trials) / n_trials,
    )
    mlflow.log_metric(
        _benchmark_metric_name("avg_first_event_latency_ms"),
        sum((trial.first_event_latency_ms or 0) for trial in summary.trials) / n_trials,
    )
    mlflow.log_metric(
        _benchmark_metric_name("avg_first_text_latency_ms"),
        sum((trial.first_text_latency_ms or 0) for trial in summary.trials) / n_trials,
    )
    mlflow.log_metric(
        _benchmark_metric_name("avg_response_complete_latency_ms"),
        sum((trial.response_complete_latency_ms or 0) for trial in summary.trials) / n_trials,
    )
    failure_counts: dict[str, int] = {}
    for trial in summary.trials:
        failure_counts[trial.failure_type] = failure_counts.get(trial.failure_type, 0) + 1
    for failure_type, count in sorted(failure_counts.items()):
        mlflow.log_metric(_benchmark_metric_name(f"failure_count.{failure_type}"), count)


def _log_trial_metrics(trial: TrialSummary) -> None:
    mlflow.set_tags(
        {
            "run_kind": "task_trial",
            "trial_name": trial.trial_name,
            "task_name": trial.task_name,
            "failure_type": trial.failure_type,
            "run_mode": trial.run_mode,
            "gateway_debug_enabled": str(bool(trial.gateway_debug_enabled)).lower(),
            "live_tracing_enabled": str(bool(trial.live_tracing_enabled)).lower(),
        }
    )
    mlflow.log_metric(_trial_metric_name("solved"), trial.solved)
    mlflow.log_metric(_trial_metric_name("pass_at_1"), trial.pass_at_1)
    mlflow.log_metric(_trial_metric_name("timeout"), trial.timeout)
    mlflow.log_metric(_trial_metric_name("score"), trial.score)
    if trial.wall_clock_seconds is not None:
        mlflow.log_metric(_trial_metric_name("wall_clock_seconds"), trial.wall_clock_seconds)
    mlflow.log_metric(_trial_metric_name("turn_count"), trial.turn_count)
    mlflow.log_metric(_trial_metric_name("tool_call_count"), trial.tool_call_count)
    mlflow.log_metric(_trial_metric_name("shell_command_count"), trial.shell_command_count)
    mlflow.log_metric(_trial_metric_name("tool_output_bytes"), trial.tool_output_bytes)
    mlflow.log_metric(_trial_metric_name("tool_output_tokens_estimate"), trial.tool_output_tokens_estimate)
    mlflow.log_metric(_trial_metric_name("prompt_tokens"), trial.prompt_tokens)
    mlflow.log_metric(_trial_metric_name("cache_tokens"), trial.cache_tokens)
    mlflow.log_metric(_trial_metric_name("output_tokens"), trial.output_tokens)
    if trial.average_turn_latency_ms is not None:
        mlflow.log_metric(_trial_metric_name("average_turn_latency_ms"), trial.average_turn_latency_ms)
    if trial.max_turn_latency_ms is not None:
        mlflow.log_metric(_trial_metric_name("max_turn_latency_ms"), trial.max_turn_latency_ms)
    if trial.first_event_latency_ms is not None:
        mlflow.log_metric(_trial_metric_name("first_event_latency_ms"), trial.first_event_latency_ms)
    if trial.first_text_latency_ms is not None:
        mlflow.log_metric(_trial_metric_name("first_text_latency_ms"), trial.first_text_latency_ms)
    if trial.response_complete_latency_ms is not None:
        mlflow.log_metric(_trial_metric_name("response_complete_latency_ms"), trial.response_complete_latency_ms)
    if trial.tokens_per_second is not None:
        mlflow.log_metric(_trial_metric_name("tokens_per_second"), trial.tokens_per_second)


def finalize_benchmark_run(
    *,
    config: AppConfig,
    parent_run: ParentRunInfo,
    summary: JobSummary,
    server_config_path: Path,
    server_state_path: Path | None = None,
) -> None:
    with _tracking_credentials(config.tracking):
        child_runs: dict[str, ChildRunInfo] = {}
        for trial in summary.trials:
            child_runs[trial.trial_name] = _log_trial_run(config, parent_run, trial)

        with mlflow.start_run(run_id=parent_run.run_id):
            _log_common_tags(
                config,
                live_tracing_enabled=_live_tracing_enabled(config) and _is_remote_tracking_uri(parent_run.tracking_uri),
            )
            mlflow.set_tag("run_kind", "benchmark")
            harbor_config_path = summary.job_dir / "harbor-job-config.json"
            if harbor_config_path.exists():
                mlflow.log_param("harbor_config_sha256", _harbor_config_sha256(harbor_config_path))
            _log_parent_metrics(summary)
            summary_path = summary.job_dir / "summary.md"
            if summary_path.exists():
                mlflow.log_artifact(summary_path.as_posix(), artifact_path="run")
            trial_results_path = _write_trial_results_table(summary, child_runs)
            mlflow.log_artifact(trial_results_path.as_posix(), artifact_path="run")
            for extra in ("harbor.stdout.txt", "harbor.stderr.txt", "job.log", "result.json", "mlflow.json", "harbor-job-config.json"):
                artifact_path = summary.job_dir / extra
                if artifact_path.exists():
                    mlflow.log_artifact(artifact_path.as_posix(), artifact_path="run")
            mlflow.log_artifact(server_config_path.as_posix(), artifact_path="run")
            if server_state_path is not None and server_state_path.exists():
                mlflow.log_artifact(server_state_path.as_posix(), artifact_path="run")


def wait_for_server_ready(tracking_uri: str, timeout_sec: float = 180.0) -> None:
    deadline = time.time() + timeout_sec
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(tracking_uri, timeout=5) as response:
                if response.status < 500:
                    return
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                return
            last_error = exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2.0)
    raise RuntimeError(f"MLflow server at {tracking_uri} did not become ready: {last_error}")


def bootstrap_basic_auth(config: TrackingConfig, tracking_uri: str) -> str:
    target_password = resolve_mlflow_password(config)

    with _specific_tracking_credentials(config.admin_username, target_password):
        client = get_app_client("basic-auth", tracking_uri=tracking_uri)
        try:
            client.get_user(config.admin_username)
            return "configured"
        except Exception:  # noqa: BLE001
            pass

    with _specific_tracking_credentials(config.admin_username, DEFAULT_MLFLOW_ADMIN_PASSWORD):
        client = get_app_client("basic-auth", tracking_uri=tracking_uri)
        try:
            client.get_user(config.admin_username)
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(
                "Unable to authenticate to MLflow using either the configured admin password or the default bootstrap password."
            ) from exc
        client.update_user_password(config.admin_username, target_password)
        return "rotated"
