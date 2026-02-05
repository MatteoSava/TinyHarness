from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import mlflow
from mlflow import MlflowClient

from harbor.models.agent.context import AgentContext
from harbor.models.job.result import JobResult
from harbor.models.task.id import LocalTaskId
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ModelInfo, TimingInfo, TrialResult
from harbor.models.verifier.result import VerifierResult

from tinyharness.config import AppConfig
from tinyharness.mlflow_tracking import log_benchmark_run
from tinyharness.results import load_job_summary, write_json, write_markdown_summary


def _write_sample_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "artifacts" / "runs" / "smoke-v0-20260312-120000"
    run_dir.mkdir(parents=True)
    started_at = datetime.now(UTC)
    finished_at = started_at + timedelta(seconds=4)

    for task_name in ("cancel-async-tasks", "filter-js-from-html", "sqlite-db-truncate"):
        trial_dir = run_dir / f"{task_name}-trial"
        trial_dir.mkdir()
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
                metadata={"duration_ms": 2000, "duration_api_ms": 1500},
            ),
            verifier_result=VerifierResult(rewards={"reward": 1}),
            started_at=started_at,
            finished_at=finished_at,
            agent_execution=TimingInfo(started_at=started_at, finished_at=finished_at),
        )
        (trial_dir / "result.json").write_text(trial.model_dump_json(indent=2), encoding="utf-8")
        (trial_dir / "stdout.txt").write_text("ok", encoding="utf-8")

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


def test_mlflow_logs_parent_and_nested_trials(tmp_path: Path) -> None:
    run_dir = _write_sample_run(tmp_path)
    summary = load_job_summary(run_dir)
    write_markdown_summary(summary)

    db_path = tmp_path / "state" / "mlflow" / "mlflow.db"
    artifact_root = tmp_path / "artifacts" / "mlflow"
    config = AppConfig.from_env(
        {
            "TINYHARNESS_JOBS_DIR": (tmp_path / "artifacts" / "runs").as_posix(),
            "TINYHARNESS_MLFLOW_DB_PATH": db_path.as_posix(),
            "TINYHARNESS_MLFLOW_ARTIFACT_ROOT": artifact_root.as_posix(),
        }
    )
    server_config_path = run_dir / "server-config.json"
    write_json(server_config_path, {"gateway_url": "https://gateway.example"})

    parent_run_id = log_benchmark_run(
        config=config,
        summary=summary,
        server_config_path=server_config_path,
    )

    mlflow.set_tracking_uri(config.tracking.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.tracking.experiment_name)
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    run_ids = {run.info.run_id for run in runs}
    assert parent_run_id in run_ids
    assert len(runs) == 4

