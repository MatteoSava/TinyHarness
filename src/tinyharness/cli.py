from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import webbrowser
from dataclasses import replace
from pathlib import Path

from tinyharness.benchmark import run_benchmark, run_smoke_benchmark
from tinyharness.config import (
    AppConfig,
    BenchmarkMode,
    ConfigError,
    ensure_state_dirs,
    local_tracking_uri,
    resolve_remote_tracking_uri,
)
from tinyharness.constants import MLFLOW_MODAL_STATE_PATH, MODAL_STATE_PATH, PROJECT_ROOT
from tinyharness.dspy_prompt import build_agent_prompt_config
from tinyharness.env import load_dotenv
from tinyharness.gepa_prompt_compiler import DEFAULT_GEPA_PROMPT_DIR, compile_prompt
from tinyharness.mlflow_server import resolve_web_url as resolve_mlflow_web_url
from tinyharness.mlflow_tracking import bootstrap_basic_auth, wait_for_server_ready
from tinyharness.modal_server import resolve_web_url
from tinyharness.results import load_job_summary


def _load_config() -> AppConfig:
    load_dotenv(PROJECT_ROOT / ".env")
    config = AppConfig.from_env()
    ensure_state_dirs(config)
    return config


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_interactive_command(args: list[str]) -> int:
    result = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=False,
        check=False,
    )
    return result.returncode


def _benchmark_mode_choices() -> tuple[str, ...]:
    return tuple(mode.value for mode in BenchmarkMode)


def _resolve_mode_arg(value: str | None, *, default: BenchmarkMode) -> BenchmarkMode:
    if value is None:
        return default
    return BenchmarkMode(value)


def serve_qwen(config: AppConfig, *, dev: bool) -> int:
    required = [config.agent.proxy_token_env]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    command = ["uv", "run", "modal", "serve" if dev else "deploy", "-m", "tinyharness.modal_server"]
    if not dev:
        command.extend(["--name", config.model.modal_app_name])

    result = _run_command(command)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode

    web_url = resolve_web_url(config.model)
    MODAL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODAL_STATE_PATH.write_text(
        json.dumps(
            {
                "app_name": config.model.modal_app_name,
                "function_name": config.model.modal_function_name,
                "web_url": web_url,
                "dev_mode": dev,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(web_url)
    return 0


def serve_mlflow(config: AppConfig) -> int:
    required = [
        "TINYHARNESS_MLFLOW_BACKEND_STORE_URI",
        config.tracking.admin_password_env,
        config.tracking.flask_secret_key_env,
    ]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    command = [
        "uv",
        "run",
        "modal",
        "deploy",
        "-m",
        "tinyharness.mlflow_server",
        "--name",
        config.tracking.modal_app_name,
    ]
    result = _run_command(command)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode

    web_url = resolve_mlflow_web_url(config.tracking)
    wait_for_server_ready(web_url)
    bootstrap_status = bootstrap_basic_auth(config.tracking, web_url)
    MLFLOW_MODAL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MLFLOW_MODAL_STATE_PATH.write_text(
        json.dumps(
            {
                "app_name": config.tracking.modal_app_name,
                "function_name": config.tracking.modal_function_name,
                "web_url": web_url,
                "admin_username": config.tracking.admin_username,
                "bootstrap_status": bootstrap_status,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(web_url)
    return 0


def fetch_results(run_id: str, config: AppConfig) -> int:
    run_dir = config.benchmark.jobs_dir / run_id
    summary = load_job_summary(run_dir)
    summary_path = run_dir / "summary.md"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
    else:
        from tinyharness.results import build_markdown_summary

        print(build_markdown_summary(summary))
    return 0


def mlflow_ui(config: AppConfig, *, remote: bool) -> int:
    if remote:
        tracking_uri = resolve_remote_tracking_uri()
        print(tracking_uri)
        webbrowser.open(tracking_uri)
        return 0

    tracking_uri = local_tracking_uri(config.tracking)
    print(f"http://127.0.0.1:{config.tracking.port}")
    return _run_interactive_command(
        [
            "uv",
            "run",
            "mlflow",
            "ui",
            "--backend-store-uri",
            tracking_uri,
            "--host",
            "127.0.0.1",
            "--port",
            str(config.tracking.port),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TinyHarness operator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve-qwen", help="Deploy or hot-serve the Modal Qwen gateway")
    serve_parser.add_argument("--dev", action="store_true", help="Use `modal serve` instead of `modal deploy`.")

    subparsers.add_parser("serve-mlflow", help="Deploy the remote MLflow tracking server on Modal.")
    benchmark_parser = subparsers.add_parser("run-benchmark", help="Run a configurable Terminal-Bench slice.")
    benchmark_parser.add_argument("--task-set", help="Name used for the benchmark run and MLflow tags.")
    benchmark_parser.add_argument(
        "--tasks",
        help="Comma-separated task names. Omit to use the configured task set or pair with --n-tasks to slice the dataset.",
    )
    benchmark_parser.add_argument("--n-tasks", type=int, help="Run only the first N tasks after dataset filtering.")
    benchmark_parser.add_argument(
        "--mode",
        choices=_benchmark_mode_choices(),
        help="Benchmark runtime mode: `lean` skips heavy tracing/artifacts, `debug` keeps full observability.",
    )
    smoke_parser = subparsers.add_parser("run-smoke", help="Run the fixed Terminal-Bench smoke subset through Harbor.")
    smoke_parser.add_argument(
        "--mode",
        choices=_benchmark_mode_choices(),
        help="Smoke runtime mode. Defaults to `debug`.",
    )

    fetch_parser = subparsers.add_parser("fetch-results", help="Print the markdown summary for one benchmark run.")
    fetch_parser.add_argument("run_id")

    prompt_parser = subparsers.add_parser("agent-prompt", help="Print the DSPy/GEPA agent prompt config.")
    prompt_parser.add_argument("instruction")

    compile_parser = subparsers.add_parser("compile-gepa-prompt", help="Compile the agent prompt with DSPy GEPA.")
    compile_parser.add_argument("--max-metric-calls", type=int, default=8)
    compile_parser.add_argument("--max-tokens", type=int, default=1536)
    compile_parser.add_argument("--output-dir", type=Path, default=DEFAULT_GEPA_PROMPT_DIR)

    mlflow_parser = subparsers.add_parser("mlflow-ui", help="Launch the local MLflow UI or open the remote one.")
    mlflow_parser.add_argument("--remote", action="store_true", help="Open the deployed remote MLflow UI.")

    return parser


def _parse_tasks_arg(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or None


def _default_task_set_name(config: AppConfig, *, tasks: tuple[str, ...] | None, n_tasks: int | None) -> str:
    if tasks is not None:
        return "custom-v0"
    if n_tasks is not None:
        return f"tb{n_tasks}-v0"
    return config.benchmark.task_set_name


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _load_config()

    if args.command == "serve-qwen":
        return serve_qwen(config, dev=args.dev)
    if args.command == "serve-mlflow":
        return serve_mlflow(config)
    if args.command == "run-benchmark":
        tasks = _parse_tasks_arg(args.tasks)
        mode = _resolve_mode_arg(args.mode, default=BenchmarkMode.LEAN)
        summary = run_benchmark(
            replace(config, benchmark=replace(config.benchmark, mode=mode)),
            task_set_name=args.task_set or _default_task_set_name(config, tasks=tasks, n_tasks=args.n_tasks),
            tasks=tasks,
            n_tasks=args.n_tasks,
        )
        print(summary.job_dir.as_posix())
        return 0
    if args.command == "run-smoke":
        mode = _resolve_mode_arg(args.mode, default=BenchmarkMode.DEBUG)
        summary = run_smoke_benchmark(replace(config, benchmark=replace(config.benchmark, mode=mode)))
        print(summary.job_dir.as_posix())
        return 0
    if args.command == "fetch-results":
        return fetch_results(args.run_id, config)
    if args.command == "agent-prompt":
        prompt_config = build_agent_prompt_config(args.instruction)
        print(
            json.dumps(
                {
                    "source": prompt_config.source,
                    "tools": list(prompt_config.tools),
                    "system_prompt": prompt_config.system_prompt,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "compile-gepa-prompt":
        compile_prompt(max_metric_calls=args.max_metric_calls, output_dir=args.output_dir, max_tokens=args.max_tokens)
        return 0
    if args.command == "mlflow-ui":
        return mlflow_ui(config, remote=args.remote)

    parser.error(f"Unknown command {args.command}")
    return 2
