from __future__ import annotations

import subprocess
from pathlib import Path

from tinyharness.cli import mlflow_ui
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
