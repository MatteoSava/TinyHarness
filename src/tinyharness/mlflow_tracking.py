from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow import MlflowClient

from tinyharness.config import AppConfig, ensure_state_dirs
from tinyharness.constants import DEFAULT_GPU_TYPE
from tinyharness.results import JobSummary, TrialSummary


def _ensure_experiment(config: AppConfig) -> str:
    ensure_state_dirs(config)
    mlflow.set_tracking_uri(config.tracking.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.tracking.experiment_name)
    if experiment is not None:
        return experiment.experiment_id

    return client.create_experiment(
        name=config.tracking.experiment_name,
        artifact_location=config.tracking.artifact_root.resolve().as_uri(),
    )


def _log_common_tags(config: AppConfig) -> None:
    mlflow.set_tags(
        {
            "benchmark_suite": config.benchmark.dataset,
            "task_set": config.benchmark.task_set_name,
            "frontend": "claude_code_preset",
            "backend_model": config.model.model_alias,
            "gpu": config.model.gpu or DEFAULT_GPU_TYPE,
            "runner": config.benchmark.runner,
        }
    )


def _log_parent_metrics(summary: JobSummary) -> None:
    mlflow.log_metric("pass_rate", summary.pass_rate)
    mlflow.log_metric("mean_score", summary.mean_score)
    mlflow.log_metric("n_trials", len(summary.trials))
    mlflow.log_metric("n_errors", summary.job_result.stats.n_errors)


def _log_trial_metrics(trial: TrialSummary) -> None:
    mlflow.log_metric("passed", trial.passed)
    mlflow.log_metric("score", trial.score)
    mlflow.log_metric("prompt_tokens", trial.prompt_tokens)
    mlflow.log_metric("cache_tokens", trial.cache_tokens)
    mlflow.log_metric("output_tokens", trial.output_tokens)
    if trial.duration_seconds is not None:
        mlflow.log_metric("duration_seconds", trial.duration_seconds)
    if trial.duration_ms is not None:
        mlflow.log_metric("request_latency_ms", trial.duration_ms)
    if trial.duration_api_ms is not None:
        mlflow.log_metric("request_api_latency_ms", trial.duration_api_ms)
    if trial.tokens_per_second is not None:
        mlflow.log_metric("tokens_per_second", trial.tokens_per_second)


def log_benchmark_run(
    *,
    config: AppConfig,
    summary: JobSummary,
    server_config_path: Path,
    server_state_path: Path | None = None,
) -> str:
    experiment_id = _ensure_experiment(config)

    with mlflow.start_run(experiment_id=experiment_id, run_name=summary.run_id) as parent_run:
        _log_common_tags(config)
        mlflow.log_params(
            {
                "model_repo": config.model.hf_repo_id,
                "model_file": config.model.hf_filename,
                "model_alias": config.model.model_alias,
                "context_window": config.model.context_window,
                "parallel_requests": config.model.parallel_requests,
                "task_names": ",".join(config.benchmark.tasks),
                "agent_import_path": config.agent.import_path,
                "max_turns": config.agent.max_turns,
                "max_thinking_tokens": config.agent.max_thinking_tokens,
            }
        )
        _log_parent_metrics(summary)
        mlflow.log_artifact(server_config_path.as_posix(), artifact_path="run")
        summary_path = summary.job_dir / "summary.md"
        if summary_path.exists():
            mlflow.log_artifact(summary_path.as_posix(), artifact_path="run")
        for extra in ("harbor.stdout.txt", "harbor.stderr.txt", "job.log", "result.json"):
            artifact_path = summary.job_dir / extra
            if artifact_path.exists():
                mlflow.log_artifact(artifact_path.as_posix(), artifact_path="run")
        if server_state_path is not None and server_state_path.exists():
            mlflow.log_artifact(server_state_path.as_posix(), artifact_path="run")

        for trial in summary.trials:
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=trial.task_name,
                nested=True,
            ):
                mlflow.set_tags(
                    {
                        "trial_name": trial.trial_name,
                        "task_name": trial.task_name,
                    }
                )
                mlflow.log_params(
                    {
                        "task_name": trial.task_name,
                        "model_name": trial.model_name or config.model.model_alias,
                    }
                )
                _log_trial_metrics(trial)
                mlflow.log_artifacts(trial.trial_dir.as_posix(), artifact_path="trial")

        return parent_run.info.run_id

