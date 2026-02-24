from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig, RegistryDatasetConfig, RemoteRegistryInfo
from harbor.models.trial.config import AgentConfig as HarborAgentConfig
from harbor.models.trial.config import EnvironmentConfig as HarborEnvironmentConfig

from tinyharness.config import AppConfig, ensure_state_dirs, resolve_gateway_base_url, resolve_proxy_token
from tinyharness.constants import MODAL_STATE_PATH
from tinyharness.mlflow_tracking import create_parent_run, finalize_benchmark_run, tracking_environment
from tinyharness.results import JobSummary, load_job_summary, write_json, write_markdown_summary

_UNSET = object()


def build_job_name(task_set_name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{task_set_name}-{timestamp}"


def _resolve_task_names(tasks: tuple[str, ...] | None) -> list[str] | None:
    if not tasks:
        return None
    return list(tasks)


def build_harbor_job_config(
    config: AppConfig,
    *,
    base_url: str,
    proxy_token: str,
    job_name: str,
    tracking_env: dict[str, str] | None = None,
) -> JobConfig:
    dataset_name, version = config.benchmark.dataset.split("@", 1)
    agent_env = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": proxy_token,
        "ANTHROPIC_MODEL": config.model.model_alias,
        "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
    }
    if tracking_env:
        agent_env.update(tracking_env)
    return JobConfig(
        job_name=job_name,
        jobs_dir=config.benchmark.jobs_dir,
        agents=[
            HarborAgentConfig(
                import_path=config.agent.import_path,
                model_name=config.model.model_alias,
                kwargs={
                    "max_turns": config.agent.max_turns,
                    "max_thinking_tokens": config.agent.max_thinking_tokens,
                    "workspace_cwd": config.agent.workspace_cwd,
                },
                env=agent_env,
            )
        ],
        environment=HarborEnvironmentConfig(
            type=EnvironmentType.MODAL,
            kwargs={
                "sandbox_timeout_secs": config.benchmark.sandbox_timeout_secs,
                "sandbox_idle_timeout_secs": config.benchmark.sandbox_idle_timeout_secs,
            },
        ),
        datasets=[
            RegistryDatasetConfig(
                registry=RemoteRegistryInfo(),
                name=dataset_name,
                version=version,
                task_names=_resolve_task_names(config.benchmark.tasks),
                n_tasks=config.benchmark.n_tasks,
            )
        ],
    )


def _write_harbor_subprocess_logs(run_dir: Path, result: subprocess.CompletedProcess[str]) -> None:
    (run_dir / "harbor.stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (run_dir / "harbor.stderr.txt").write_text(result.stderr or "", encoding="utf-8")


def _run_harbor(config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "harbor", "run", "--config", config_path.as_posix()],
        text=True,
        capture_output=True,
        check=False,
    )


def run_benchmark(
    config: AppConfig,
    *,
    task_set_name: str | None = None,
    tasks: tuple[str, ...] | None | object = _UNSET,
    n_tasks: int | None = None,
) -> JobSummary:
    if task_set_name is not None or tasks is not _UNSET or n_tasks is not None:
        config = replace(
            config,
            benchmark=replace(
                config.benchmark,
                task_set_name=task_set_name or config.benchmark.task_set_name,
                tasks=config.benchmark.tasks if tasks is _UNSET else tasks,
                n_tasks=n_tasks if n_tasks is not None else config.benchmark.n_tasks,
            ),
        )

    ensure_state_dirs(config)
    base_url = resolve_gateway_base_url()
    proxy_token = resolve_proxy_token(config.agent)
    job_name = build_job_name(config.benchmark.task_set_name)
    run_dir = config.benchmark.jobs_dir / job_name
    run_dir.mkdir(parents=True, exist_ok=True)

    server_config_path = run_dir / "server-config.json"
    write_json(
        server_config_path,
        {
            "modal_app_name": config.model.modal_app_name,
            "modal_function_name": config.model.modal_function_name,
            "gateway_url": base_url,
            "model_repo": config.model.hf_repo_id,
            "model_file": config.model.hf_filename,
            "model_alias": config.model.model_alias,
            "gpu": config.model.gpu,
            "gateway_max_containers": config.model.max_containers,
            "gateway_scaledown_window_sec": config.model.scaledown_window_sec,
            "parallel_requests": config.model.parallel_requests,
        },
    )

    provisional_job_config = build_harbor_job_config(
        config,
        base_url=base_url,
        proxy_token=proxy_token,
        job_name=job_name,
    )
    harbor_config_path = run_dir / "harbor-job-config.json"
    harbor_config_path.write_text(provisional_job_config.model_dump_json(indent=2), encoding="utf-8")
    parent_run = create_parent_run(
        config=config,
        job_name=job_name,
        server_config_path=server_config_path,
        harbor_config_path=harbor_config_path,
    )
    final_job_config = build_harbor_job_config(
        config,
        base_url=base_url,
        proxy_token=proxy_token,
        job_name=job_name,
        tracking_env=tracking_environment(config, parent_run, job_name=job_name),
    )
    harbor_config_path.write_text(final_job_config.model_dump_json(indent=2), encoding="utf-8")
    result = _run_harbor(harbor_config_path)
    _write_harbor_subprocess_logs(run_dir, result)
    if result.returncode != 0:
        raise RuntimeError(
            f"Harbor run failed with exit code {result.returncode}. See {run_dir / 'harbor.stderr.txt'}"
        )

    summary = load_job_summary(run_dir)
    write_markdown_summary(summary)
    finalize_benchmark_run(
        config=config,
        parent_run=parent_run,
        summary=summary,
        server_config_path=server_config_path,
        server_state_path=MODAL_STATE_PATH if MODAL_STATE_PATH.exists() else None,
    )
    write_json(run_dir / "mlflow.json", {"run_id": parent_run.run_id, "tracking_uri": parent_run.tracking_uri})
    return summary


def run_smoke_benchmark(config: AppConfig) -> JobSummary:
    return run_benchmark(config)
