from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import mlflow
import pytest
from mlflow import MlflowClient

from harbor.models.agent.context import AgentContext
from harbor.models.job.result import JobResult
from harbor.models.task.id import LocalTaskId
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ModelInfo, TimingInfo, TrialResult
from harbor.models.verifier.result import VerifierResult

from tinyharness.config import AppConfig, ConfigError
from tinyharness.mlflow_tracking import (
    bootstrap_basic_auth,
    create_parent_run,
    finalize_benchmark_run,
    tracking_environment,
)
from tinyharness.results import load_job_summary, write_json, write_markdown_summary


def _write_sample_run(tmp_path: Path, child_run_ids: dict[str, str] | None = None) -> Path:
    run_dir = tmp_path / "artifacts" / "runs" / "smoke-v0-20260312-120000"
    run_dir.mkdir(parents=True)
    started_at = datetime.now(UTC)
    finished_at = started_at + timedelta(seconds=4)

    for task_name in ("cancel-async-tasks", "filter-js-from-html", "sqlite-db-truncate"):
        trial_dir = run_dir / f"{task_name}-trial"
        trial_dir.mkdir()
        agent_dir = trial_dir / "agent"
        agent_dir.mkdir()
        trial = TrialResult(
            task_name=task_name,
            trial_name=f"{task_name}-trial",
            trial_uri=trial_dir.as_posix(),
            task_id=LocalTaskId(path=Path(task_name)),
            source="terminal-bench",
            task_checksum="checksum",
            config=TrialConfig(task=TaskConfig(path=Path(task_name))),
            agent_info=AgentInfo(
                name="qwen-claude-sdk",
                version="0.1.48",
                model_info=ModelInfo(name="qwen3.5-35b-a3b-ud-iq3_s", provider="qwen"),
            ),
            agent_result=AgentContext(
                n_input_tokens=100,
                n_cache_tokens=10,
                n_output_tokens=50,
                metadata={
                    "instruction": "fix the task",
                    "duration_ms": 2000,
                    "duration_api_ms": 1500,
                    "telemetry": {
                        "request_started_at_epoch_ms": 1_000,
                        "response_completed_at_epoch_ms": 1_900,
                        "turn_count": 2,
                        "turns": [
                            {
                                "turn_index": 1,
                                "turn_start_epoch_ms": 1_100,
                                "turn_end_epoch_ms": 1_400,
                                "tool_calls_in_turn": 1,
                                "shell_commands_in_turn": 1,
                            },
                            {
                                "turn_index": 2,
                                "turn_start_epoch_ms": 1_500,
                                "turn_end_epoch_ms": 1_900,
                                "tool_calls_in_turn": 0,
                                "shell_commands_in_turn": 0,
                            },
                        ],
                        "per_turn_latencies_ms": [300, 400],
                        "average_turn_latency_ms": 350,
                        "max_turn_latency_ms": 400,
                        "tool_call_count": 1,
                        "shell_command_count": 1,
                        "tool_output_bytes": 32,
                        "tool_output_tokens_estimate": 8,
                        "first_event_latency_ms": 120,
                        "first_text_latency_ms": 150,
                        "response_complete_latency_ms": 900,
                    },
                    **(
                        {
                            "mlflow_run_id": child_run_ids[task_name],
                            "trace_id": f"trace-{task_name}",
                        }
                        if child_run_ids is not None
                        else {}
                    ),
                },
            ),
            verifier_result=VerifierResult(rewards={"reward": 1}),
            started_at=started_at,
            finished_at=finished_at,
            agent_execution=TimingInfo(started_at=started_at, finished_at=finished_at),
        )
        (trial_dir / "result.json").write_text(trial.model_dump_json(indent=2), encoding="utf-8")
        (trial_dir / "stdout.txt").write_text("ok", encoding="utf-8")
        (agent_dir / "sdk-trace.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "received_at_epoch_ms": 1100,
                            "turn_index": 1,
                            "tool_name": "Bash",
                            "tool_use_id": "tool-1",
                            "is_tool_call": True,
                            "is_tool_result": False,
                            "payload": {
                                "content": [
                                    {
                                        "id": "tool-1",
                                        "name": "Bash",
                                        "input": {"command": "ls -la"},
                                    }
                                ]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "received_at_epoch_ms": 1400,
                            "turn_index": 1,
                            "tool_name": "Bash",
                            "tool_use_id": None,
                            "is_tool_call": False,
                            "is_tool_result": True,
                            "payload": {
                                "content": [
                                    {
                                        "tool_use_id": "tool-1",
                                        "content": "ok",
                                        "is_error": False,
                                    }
                                ],
                                "tool_use_result": {"stdout": "ok"},
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (agent_dir / "assistant.txt").write_text("Done.", encoding="utf-8")

    job = JobResult(
        id=uuid4(),
        started_at=started_at,
        finished_at=finished_at,
        n_total_trials=3,
        stats={"n_trials": 3, "n_errors": 0, "evals": {}},
    )
    (run_dir / "result.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text("# Summary\n", encoding="utf-8")
    return run_dir


def test_mlflow_logs_parent_and_updates_live_child_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TINYHARNESS_MLFLOW_ADMIN_PASSWORD", "secret-password")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    db_path = tmp_path / "state" / "mlflow" / "mlflow.db"
    artifact_root = tmp_path / "artifacts" / "mlflow"
    config = AppConfig.from_env(
        {
            "TINYHARNESS_JOBS_DIR": (tmp_path / "artifacts" / "runs").as_posix(),
            "TINYHARNESS_MLFLOW_DB_PATH": db_path.as_posix(),
            "TINYHARNESS_MLFLOW_ARTIFACT_ROOT": artifact_root.as_posix(),
        }
    )

    server_config_path = tmp_path / "server-config.json"
    harbor_config_path = tmp_path / "harbor-job-config.json"
    write_json(server_config_path, {"gateway_url": "https://gateway.example"})
    write_json(harbor_config_path, {"job_name": "smoke"})

    parent_run = create_parent_run(
        config=config,
        job_name="smoke-v0-20260312-120000",
        server_config_path=server_config_path,
        harbor_config_path=harbor_config_path,
    )

    mlflow.set_tracking_uri(parent_run.tracking_uri)
    child_run_ids: dict[str, str] = {}
    for task_name in ("cancel-async-tasks", "filter-js-from-html", "sqlite-db-truncate"):
        with mlflow.start_run(
            experiment_id=parent_run.experiment_id,
            run_name=task_name,
            parent_run_id=parent_run.run_id,
        ) as child_run:
            child_run_ids[task_name] = child_run.info.run_id

    run_dir = _write_sample_run(tmp_path, child_run_ids)
    summary = load_job_summary(run_dir)
    write_markdown_summary(summary)

    finalize_benchmark_run(
        config=config,
        parent_run=parent_run,
        summary=summary,
        server_config_path=server_config_path,
    )

    client = MlflowClient(tracking_uri=parent_run.tracking_uri)
    experiment = client.get_experiment_by_name(config.tracking.experiment_name)
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    run_ids = {run.info.run_id for run in runs}
    assert parent_run.run_id in run_ids
    assert len(runs) == 4

    parent = client.get_run(parent_run.run_id)
    assert parent.data.metrics["solved_pct"] == 1.0
    assert parent.data.metrics["avg_tool_calls_per_task"] == 1.0

    child = client.get_run(child_run_ids["cancel-async-tasks"])
    assert child.data.metrics["solved"] == 1.0
    assert child.data.metrics["tool_call_count"] == 1.0
    assert child.data.tags["failure_type"] == "passed"


def test_tracking_environment_skips_local_sqlite_for_remote_sandboxes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TINYHARNESS_MLFLOW_ADMIN_PASSWORD", "secret-password")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    config = AppConfig.from_env(
        {
            "TINYHARNESS_MLFLOW_DB_PATH": (tmp_path / "state" / "mlflow.db").as_posix(),
        }
    )
    server_config_path = tmp_path / "server-config.json"
    harbor_config_path = tmp_path / "harbor-job-config.json"
    write_json(server_config_path, {"gateway_url": "https://gateway.example"})
    write_json(harbor_config_path, {"job_name": "smoke"})
    parent_run = create_parent_run(
        config=config,
        job_name="smoke-v0-20260312-120000",
        server_config_path=server_config_path,
        harbor_config_path=harbor_config_path,
    )

    env = tracking_environment(config, parent_run, job_name="smoke-v0-20260312-120000")

    assert env == {}


def test_mlflow_creates_child_runs_post_hoc_when_live_run_ids_are_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TINYHARNESS_MLFLOW_ADMIN_PASSWORD", "secret-password")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    db_path = tmp_path / "state" / "mlflow" / "mlflow.db"
    artifact_root = tmp_path / "artifacts" / "mlflow"
    config = AppConfig.from_env(
        {
            "TINYHARNESS_JOBS_DIR": (tmp_path / "artifacts" / "runs").as_posix(),
            "TINYHARNESS_MLFLOW_DB_PATH": db_path.as_posix(),
            "TINYHARNESS_MLFLOW_ARTIFACT_ROOT": artifact_root.as_posix(),
        }
    )

    server_config_path = tmp_path / "server-config.json"
    harbor_config_path = tmp_path / "harbor-job-config.json"
    write_json(server_config_path, {"gateway_url": "https://gateway.example"})
    write_json(harbor_config_path, {"job_name": "smoke"})

    parent_run = create_parent_run(
        config=config,
        job_name="smoke-v0-20260312-120000",
        server_config_path=server_config_path,
        harbor_config_path=harbor_config_path,
    )

    run_dir = _write_sample_run(tmp_path, child_run_ids=None)
    summary = load_job_summary(run_dir)
    write_markdown_summary(summary)

    finalize_benchmark_run(
        config=config,
        parent_run=parent_run,
        summary=summary,
        server_config_path=server_config_path,
    )

    client = MlflowClient(tracking_uri=parent_run.tracking_uri)
    experiment = client.get_experiment_by_name(config.tracking.experiment_name)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])

    assert len(runs) == 4
    child_runs = [run for run in runs if run.info.run_id != parent_run.run_id]
    assert len(child_runs) == 3
    assert all(run.data.tags["failure_type"] == "passed" for run in child_runs)
    traces = mlflow.search_traces(
        locations=[experiment.experiment_id],
        max_results=10,
        return_type="list",
    )
    assert len(traces) == 3
    assert all(trace.info.request_preview == '{"instruction": "fix the task"}' for trace in traces)
    assert all(trace.info.response_preview == '{"assistant_text": "Done."}' for trace in traces)
    bash_spans = [
        span
        for trace in traces
        for span in trace.data.spans
        if span.name == "tool.Bash"
    ]
    assert len(bash_spans) == 3
    assert all(span.inputs == {"command": "ls -la"} for span in bash_spans)
    assert all(span.outputs == {"stdout": "ok"} for span in bash_spans)


def test_mlflow_falls_back_when_trial_run_ids_are_unknown_locally(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TINYHARNESS_MLFLOW_ADMIN_PASSWORD", "secret-password")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    db_path = tmp_path / "state" / "mlflow" / "mlflow.db"
    artifact_root = tmp_path / "artifacts" / "mlflow"
    config = AppConfig.from_env(
        {
            "TINYHARNESS_JOBS_DIR": (tmp_path / "artifacts" / "runs").as_posix(),
            "TINYHARNESS_MLFLOW_DB_PATH": db_path.as_posix(),
            "TINYHARNESS_MLFLOW_ARTIFACT_ROOT": artifact_root.as_posix(),
        }
    )

    server_config_path = tmp_path / "server-config.json"
    harbor_config_path = tmp_path / "harbor-job-config.json"
    write_json(server_config_path, {"gateway_url": "https://gateway.example"})
    write_json(harbor_config_path, {"job_name": "smoke"})

    parent_run = create_parent_run(
        config=config,
        job_name="smoke-v0-20260312-120000",
        server_config_path=server_config_path,
        harbor_config_path=harbor_config_path,
    )

    bogus_child_run_ids = {
        "cancel-async-tasks": "bogus-cancel-run-id",
        "filter-js-from-html": "bogus-filter-run-id",
        "sqlite-db-truncate": "bogus-sqlite-run-id",
    }
    run_dir = _write_sample_run(tmp_path, child_run_ids=bogus_child_run_ids)
    summary = load_job_summary(run_dir)
    write_markdown_summary(summary)

    finalize_benchmark_run(
        config=config,
        parent_run=parent_run,
        summary=summary,
        server_config_path=server_config_path,
    )

    client = MlflowClient(tracking_uri=parent_run.tracking_uri)
    experiment = client.get_experiment_by_name(config.tracking.experiment_name)
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 4
    child_runs = [run for run in runs if run.info.run_id != parent_run.run_id]
    assert len(child_runs) == 3
    assert all(run.data.tags["failure_type"] == "passed" for run in child_runs)

    traces = mlflow.search_traces(
        locations=[experiment.experiment_id],
        max_results=10,
        return_type="list",
    )
    assert len(traces) == 3
    assert all(trace.info.request_preview == '{"instruction": "fix the task"}' for trace in traces)
    assert all(trace.info.response_preview == '{"assistant_text": "Done."}' for trace in traces)
    bash_spans = [
        span
        for trace in traces
        for span in trace.data.spans
        if span.name == "tool.Bash"
    ]
    assert len(bash_spans) == 3
    assert all(span.inputs == {"command": "ls -la"} for span in bash_spans)
    assert all(span.outputs == {"stdout": "ok"} for span in bash_spans)


def test_mlflow_finalize_is_idempotent_for_local_posthoc_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TINYHARNESS_MLFLOW_ADMIN_PASSWORD", "secret-password")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    db_path = tmp_path / "state" / "mlflow" / "mlflow.db"
    artifact_root = tmp_path / "artifacts" / "mlflow"
    config = AppConfig.from_env(
        {
            "TINYHARNESS_JOBS_DIR": (tmp_path / "artifacts" / "runs").as_posix(),
            "TINYHARNESS_MLFLOW_DB_PATH": db_path.as_posix(),
            "TINYHARNESS_MLFLOW_ARTIFACT_ROOT": artifact_root.as_posix(),
        }
    )

    server_config_path = tmp_path / "server-config.json"
    harbor_config_path = tmp_path / "harbor-job-config.json"
    write_json(server_config_path, {"gateway_url": "https://gateway.example"})
    write_json(harbor_config_path, {"job_name": "smoke"})

    parent_run = create_parent_run(
        config=config,
        job_name="smoke-v0-20260312-120000",
        server_config_path=server_config_path,
        harbor_config_path=harbor_config_path,
    )

    run_dir = _write_sample_run(tmp_path, child_run_ids=None)
    summary = load_job_summary(run_dir)
    write_markdown_summary(summary)

    finalize_benchmark_run(
        config=config,
        parent_run=parent_run,
        summary=summary,
        server_config_path=server_config_path,
    )
    finalize_benchmark_run(
        config=config,
        parent_run=parent_run,
        summary=summary,
        server_config_path=server_config_path,
    )

    client = MlflowClient(tracking_uri=parent_run.tracking_uri)
    experiment = client.get_experiment_by_name(config.tracking.experiment_name)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 4
    traces = mlflow.search_traces(
        locations=[experiment.experiment_id],
        max_results=10,
        return_type="list",
    )
    assert len(traces) == 3


class _FakeAuthClient:
    def __init__(self, behavior: str, calls: list[tuple[str, str]]) -> None:
        self.behavior = behavior
        self.calls = calls

    def get_user(self, username: str):
        self.calls.append(("get_user", username))
        if self.behavior in {"configured", "default"}:
            return {"username": username}
        raise RuntimeError("unauthorized")

    def update_user_password(self, username: str, new_password: str) -> None:
        self.calls.append(("update_user_password", f"{username}:{new_password}"))


def test_bootstrap_basic_auth_returns_configured_when_target_password_already_works(monkeypatch) -> None:
    monkeypatch.setenv("TINYHARNESS_MLFLOW_ADMIN_PASSWORD", "secret-password")
    config = AppConfig.from_env({}).tracking
    calls: list[tuple[str, str]] = []

    def fake_get_app_client(_app_name: str, *, tracking_uri: str):
        assert tracking_uri == "https://mlflow.example"
        return _FakeAuthClient("configured", calls)

    monkeypatch.setattr("tinyharness.mlflow_tracking.get_app_client", fake_get_app_client)

    status = bootstrap_basic_auth(config, "https://mlflow.example")

    assert status == "configured"
    assert calls == [("get_user", "admin")]


def test_bootstrap_basic_auth_rotates_default_password(monkeypatch) -> None:
    monkeypatch.setenv("TINYHARNESS_MLFLOW_ADMIN_PASSWORD", "secret-password")
    config = AppConfig.from_env({}).tracking
    calls: list[tuple[str, str]] = []
    attempts = iter(["configured-fails", "default"])

    def fake_get_app_client(_app_name: str, *, tracking_uri: str):
        assert tracking_uri == "https://mlflow.example"
        behavior = next(attempts)
        if behavior == "configured-fails":
            return _FakeAuthClient("fail", calls)
        return _FakeAuthClient("default", calls)

    monkeypatch.setattr("tinyharness.mlflow_tracking.get_app_client", fake_get_app_client)

    status = bootstrap_basic_auth(config, "https://mlflow.example")

    assert status == "rotated"
    assert ("update_user_password", "admin:secret-password") in calls


def test_bootstrap_basic_auth_raises_when_neither_password_works(monkeypatch) -> None:
    monkeypatch.setenv("TINYHARNESS_MLFLOW_ADMIN_PASSWORD", "secret-password")
    config = AppConfig.from_env({}).tracking

    def fake_get_app_client(_app_name: str, *, tracking_uri: str):
        assert tracking_uri == "https://mlflow.example"
        return _FakeAuthClient("fail", [])

    monkeypatch.setattr("tinyharness.mlflow_tracking.get_app_client", fake_get_app_client)

    with pytest.raises(ConfigError):
        bootstrap_basic_auth(config, "https://mlflow.example")
