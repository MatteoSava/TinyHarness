from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from tinyharness.cli import main, mlflow_ui
from tinyharness.config import AppConfig


def test_mlflow_ui_runs_local_server_by_default(monkeypatch, tmp_path: Path) -> None:
    config = AppConfig.from_env(
        {
            "TINYHARNESS_MLFLOW_DB_PATH": (tmp_path / "state" / "mlflow.db").as_posix(),
            "TINYHARNESS_MLFLOW_PORT": "5055",
        }
    )
    called: dict[str, object] = {}

    def fake_run(args, cwd, text, capture_output, check):
        called["args"] = args
        called["cwd"] = cwd
        called["text"] = text
        called["capture_output"] = capture_output
        called["check"] = check
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr("tinyharness.cli.subprocess.run", fake_run)

    exit_code = mlflow_ui(config, remote=False)

    assert exit_code == 0
    assert called["args"] == [
        "uv",
        "run",
        "mlflow",
        "ui",
        "--backend-store-uri",
        f"sqlite:///{config.tracking.backend_store_path.resolve()}",
        "--host",
        "127.0.0.1",
        "--port",
        "5055",
    ]
    assert called["capture_output"] is False


def test_mlflow_ui_opens_remote_url_when_requested(monkeypatch) -> None:
    config = AppConfig.from_env({})
    opened: list[str] = []

    monkeypatch.setattr("tinyharness.cli.resolve_remote_tracking_uri", lambda: "https://mlflow.example")
    monkeypatch.setattr("tinyharness.cli.webbrowser.open", lambda url: opened.append(url))

    exit_code = mlflow_ui(config, remote=True)

    assert exit_code == 0
    assert opened == ["https://mlflow.example"]


def test_main_dispatches_run_benchmark_with_n_tasks(monkeypatch, capsys) -> None:
    config = AppConfig.from_env({})
    called: dict[str, object] = {}

    def fake_load_config() -> AppConfig:
        return config

    def fake_run_benchmark(config_arg, *, task_set_name, tasks, n_tasks):
        called["config"] = config_arg
        called["task_set_name"] = task_set_name
        called["tasks"] = tasks
        called["n_tasks"] = n_tasks
        return SimpleNamespace(job_dir=Path("/tmp/tb10-v0-20260312-120000"))

    monkeypatch.setattr("tinyharness.cli._load_config", fake_load_config)
    monkeypatch.setattr("tinyharness.cli.run_benchmark", fake_run_benchmark)

    exit_code = main(["run-benchmark", "--task-set", "tb10-v0", "--n-tasks", "10"])

    assert exit_code == 0
    assert called == {
        "config": config,
        "task_set_name": "tb10-v0",
        "tasks": None,
        "n_tasks": 10,
    }
    assert "/tmp/tb10-v0-20260312-120000" in capsys.readouterr().out
