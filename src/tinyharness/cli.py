from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tinyharness.config import AppConfig, ConfigError, ensure_state_dirs
from tinyharness.constants import MODAL_STATE_PATH, PROJECT_ROOT
from tinyharness.env import load_dotenv
from tinyharness.modal_server import resolve_web_url
from tinyharness.benchmark import run_smoke_benchmark
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


def mlflow_ui(config: AppConfig) -> int:
    command = [
        "uv",
        "run",
        "mlflow",
        "ui",
        "--backend-store-uri",
        config.tracking.tracking_uri,
        "--host",
        "127.0.0.1",
        "--port",
        str(config.tracking.port),
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TinyHarness operator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve-qwen", help="Deploy or hot-serve the Modal Qwen gateway")
    serve_parser.add_argument("--dev", action="store_true", help="Use `modal serve` instead of `modal deploy`.")

    subparsers.add_parser("run-smoke", help="Run the fixed Terminal-Bench smoke subset through Harbor.")

    fetch_parser = subparsers.add_parser("fetch-results", help="Print the markdown summary for one benchmark run.")
    fetch_parser.add_argument("run_id")

    subparsers.add_parser("mlflow-ui", help="Start a local MLflow UI against the project tracking store.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _load_config()

    if args.command == "serve-qwen":
        return serve_qwen(config, dev=args.dev)
    if args.command == "run-smoke":
        summary = run_smoke_benchmark(config)
        print(summary.job_dir.as_posix())
        return 0
    if args.command == "fetch-results":
        return fetch_results(args.run_id, config)
    if args.command == "mlflow-ui":
        return mlflow_ui(config)

    parser.error(f"Unknown command {args.command}")
    return 2
