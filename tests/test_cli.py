from __future__ import annotations

import subprocess
import json
from pathlib import Path
from types import SimpleNamespace

from tinyharness.cli import main, mlflow_ui
from tinyharness.config import AppConfig, BenchmarkMode


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
    assert called["config"].benchmark.mode == BenchmarkMode.LEAN
    assert called["task_set_name"] == "tb10-v0"
    assert called["tasks"] is None
    assert called["n_tasks"] == 10
    assert "/tmp/tb10-v0-20260312-120000" in capsys.readouterr().out


def test_main_defaults_run_benchmark_to_lean_mode(monkeypatch, capsys) -> None:
    config = AppConfig.from_env({})
    called: dict[str, object] = {}

    monkeypatch.setattr("tinyharness.cli._load_config", lambda: config)

    def fake_run_benchmark(config_arg, *, task_set_name, tasks, n_tasks):
        called["mode"] = config_arg.benchmark.mode
        return SimpleNamespace(job_dir=Path("/tmp/tb10-v1-20260313-120000"))

    monkeypatch.setattr("tinyharness.cli.run_benchmark", fake_run_benchmark)

    exit_code = main(["run-benchmark", "--n-tasks", "10"])

    assert exit_code == 0
    assert called["mode"] == BenchmarkMode.LEAN
    assert "/tmp/tb10-v1-20260313-120000" in capsys.readouterr().out


def test_main_defaults_run_smoke_to_debug_mode(monkeypatch, capsys) -> None:
    config = AppConfig.from_env({"TINYHARNESS_BENCHMARK_MODE": "lean"})
    called: dict[str, object] = {}

    monkeypatch.setattr("tinyharness.cli._load_config", lambda: config)

    def fake_run_smoke(config_arg):
        called["mode"] = config_arg.benchmark.mode
        return SimpleNamespace(job_dir=Path("/tmp/smoke-v0-20260313-120000"))

    monkeypatch.setattr("tinyharness.cli.run_smoke_benchmark", fake_run_smoke)

    exit_code = main(["run-smoke"])

    assert exit_code == 0
    assert called["mode"] == BenchmarkMode.DEBUG
    assert "/tmp/smoke-v0-20260313-120000" in capsys.readouterr().out


def test_main_allows_explicit_benchmark_mode_override(monkeypatch, capsys) -> None:
    config = AppConfig.from_env({})
    called: dict[str, object] = {}

    monkeypatch.setattr("tinyharness.cli._load_config", lambda: config)

    def fake_run_benchmark(config_arg, *, task_set_name, tasks, n_tasks):
        called["mode"] = config_arg.benchmark.mode
        return SimpleNamespace(job_dir=Path("/tmp/custom-v0-20260313-120000"))

    monkeypatch.setattr("tinyharness.cli.run_benchmark", fake_run_benchmark)

    exit_code = main(["run-benchmark", "--tasks", "cancel-async-tasks", "--mode", "debug"])

    assert exit_code == 0
    assert called["mode"] == BenchmarkMode.DEBUG
    assert "/tmp/custom-v0-20260313-120000" in capsys.readouterr().out


def test_main_prints_agent_prompt_config(monkeypatch, capsys) -> None:
    monkeypatch.setattr("tinyharness.cli._load_config", lambda: AppConfig.from_env({}))

    exit_code = main(["agent-prompt", "fix one benchmark task"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "dspy-gepa-seed"
    assert payload["tools"] == ["Bash", "Read", "Edit", "Write", "Grep", "Glob", "LS"]
    assert "fix one benchmark task" in payload["system_prompt"]


def test_main_dispatches_compile_gepa_prompt(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("tinyharness.cli._load_config", lambda: AppConfig.from_env({}))
    called: dict[str, object] = {}

    def fake_compile_prompt(*, max_metric_calls: int, output_dir: Path, max_tokens: int) -> Path:
        called["max_metric_calls"] = max_metric_calls
        called["output_dir"] = output_dir
        called["max_tokens"] = max_tokens
        return output_dir / "compiled-agent-prompt.txt"

    monkeypatch.setattr("tinyharness.cli.compile_prompt", fake_compile_prompt)

    exit_code = main(
        [
            "compile-gepa-prompt",
            "--max-metric-calls",
            "4",
            "--max-tokens",
            "512",
            "--output-dir",
            tmp_path.as_posix(),
        ]
    )

    assert exit_code == 0
    assert called == {
        "max_metric_calls": 4,
        "output_dir": tmp_path,
        "max_tokens": 512,
    }
