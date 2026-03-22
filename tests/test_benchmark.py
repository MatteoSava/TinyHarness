from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import tinyharness.benchmark as benchmark_module
from tinyharness.benchmark import build_harbor_job_config
from tinyharness.config import AppConfig, BenchmarkMode


def test_build_harbor_job_config_uses_modal_and_custom_agent() -> None:
    config = AppConfig.from_env({})

    job_config = build_harbor_job_config(
        config,
        base_url="https://gateway.example",
        proxy_token="secret-token",
        job_name="smoke-v0-20260312-120000",
        tracking_env={
            "MLFLOW_TRACKING_URI": "https://mlflow.example",
            "MLFLOW_TRACKING_USERNAME": "admin",
            "MLFLOW_TRACKING_PASSWORD": "secret-password",
            "MLFLOW_EXPERIMENT_ID": "123",
            "TINYHARNESS_PARENT_RUN_ID": "parent-run",
            "TINYHARNESS_JOB_NAME": "smoke-v0-20260312-120000",
        },
    )

    assert job_config.environment.type.value == "modal"
    assert job_config.agents[0].import_path == "tinyharness.harbor_agents:QwenClaudeSDKAgent"
    assert job_config.agents[0].env["ANTHROPIC_BASE_URL"] == "https://gateway.example"
    assert job_config.agents[0].env["TINYHARNESS_RUN_MODE"] == "debug"
    assert job_config.agents[0].env["MLFLOW_TRACKING_URI"] == "https://mlflow.example"
    assert job_config.agents[0].env["TINYHARNESS_PARENT_RUN_ID"] == "parent-run"
    assert job_config.agents[0].kwargs["benchmark_mode"] == "debug"
    assert job_config.datasets[0].name == "terminal-bench"
    assert job_config.datasets[0].task_names == list(config.benchmark.tasks)


def test_build_harbor_job_config_records_compiled_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "compiled-agent-prompt.txt"
    prompt_path.write_text("Compiled prompt from GEPA.", encoding="utf-8")
    monkeypatch.setenv("TINYHARNESS_DSPY_COMPILED_PROMPT_PATH", prompt_path.as_posix())

    job_config = build_harbor_job_config(
        AppConfig.from_env({}),
        base_url="https://gateway.example",
        proxy_token="secret-token",
        job_name="gepa-v0-20260312-120000",
    )

    agent_env = job_config.agents[0].env
    assert agent_env["TINYHARNESS_AGENT_PROMPT_MODE"] == "dspy-gepa"
    assert agent_env["TINYHARNESS_DSPY_COMPILED_PROMPT_PATH"] == prompt_path.as_posix()
    assert agent_env["TINYHARNESS_DSPY_COMPILED_PROMPT"] == "Compiled prompt from GEPA."
    assert agent_env["TINYHARNESS_DSPY_COMPILED_PROMPT_SHA256"] == hashlib.sha256(
        b"Compiled prompt from GEPA."
    ).hexdigest()


def test_build_harbor_job_config_supports_first_n_tasks_without_explicit_task_names() -> None:
    config = AppConfig.from_env({})
    config = replace(
        config,
        benchmark=replace(
            config.benchmark,
            task_set_name="tb10-v0",
            tasks=None,
            n_tasks=10,
        ),
    )

    job_config = build_harbor_job_config(
        config,
        base_url="https://gateway.example",
        proxy_token="secret-token",
        job_name="tb10-v0-20260312-120000",
    )

    assert job_config.datasets[0].task_names is None
    assert job_config.datasets[0].n_tasks == 10


def test_run_harbor_uses_tinyharness_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(list(args[0]))
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(benchmark_module.subprocess, "run", _fake_run)

    result = benchmark_module._run_harbor(Path("/tmp/job-config.json"))

    assert result.returncode == 0
    assert calls == [
        [
            "uv",
            "run",
            "python",
            "-m",
            "tinyharness.harbor_runner",
            "--config",
            "/tmp/job-config.json",
        ]
    ]


def test_fetch_trial_gateway_debug_artifacts_writes_requests_and_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    class _FakeClient:
        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def get(self, url: str, *, headers: dict[str, str], params: dict[str, str]) -> _FakeResponse:
            assert url == "https://gateway.example/debug/requests"
            assert headers["Authorization"] == "Bearer secret-token"
            assert params["trial_name"] == "cancel-async-tasks__trial"
            return _FakeResponse(
                {
                    "trial_name": "cancel-async-tasks__trial",
                    "requests": [
                        {
                            "request_id": "req-1",
                            "inbound_request": {"sha256": "inbound-sha"},
                            "effective_request": {"sha256": "effective-sha"},
                            "response_summary": {"status_code": 200},
                        }
                    ],
                }
            )

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> _FakeResponse:
            assert url == "https://gateway.example/debug/replay"
            assert headers["Authorization"] == "Bearer secret-token"
            assert json == {"trial_name": "cancel-async-tasks__trial", "count": 3}
            return _FakeResponse(
                {
                    "trial_name": "cancel-async-tasks__trial",
                    "responses": [{"response_sha256": "resp-sha"}] * 3,
                }
            )

    monkeypatch.setattr(benchmark_module.httpx, "Client", lambda timeout, follow_redirects: _FakeClient())

    trial_dir = tmp_path / "cancel-async-tasks__trial"
    trial_dir.mkdir()

    benchmark_module._fetch_trial_gateway_debug_artifacts(
        base_url="https://gateway.example",
        proxy_token="secret-token",
        trial_name="cancel-async-tasks__trial",
        trial_dir=trial_dir,
        replay_count=3,
    )

    requests_path = trial_dir / "gateway" / "requests.jsonl"
    replay_path = trial_dir / "gateway" / "replay.json"
    assert requests_path.exists()
    assert replay_path.exists()
    request_rows = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert request_rows[0]["request_id"] == "req-1"
    assert json.loads(replay_path.read_text(encoding="utf-8"))["responses"][0]["response_sha256"] == "resp-sha"


def test_fetch_trial_gateway_debug_artifacts_defaults_to_single_replay_with_longer_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    class _FakeClient:
        def __init__(self, *, timeout: object, follow_redirects: bool) -> None:
            captured["timeout"] = timeout
            captured["follow_redirects"] = follow_redirects

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def get(self, url: str, *, headers: dict[str, str], params: dict[str, str]) -> _FakeResponse:
            return _FakeResponse({"trial_name": params["trial_name"], "requests": []})

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> _FakeResponse:
            captured["replay_payload"] = json
            return _FakeResponse({"trial_name": json["trial_name"], "responses": []})

    monkeypatch.setattr(
        benchmark_module.httpx,
        "Client",
        lambda timeout, follow_redirects: _FakeClient(timeout=timeout, follow_redirects=follow_redirects),
    )

    trial_dir = tmp_path / "cancel-async-tasks__trial"
    trial_dir.mkdir()

    benchmark_module._fetch_trial_gateway_debug_artifacts(
        base_url="https://gateway.example",
        proxy_token="secret-token",
        trial_name="cancel-async-tasks__trial",
        trial_dir=trial_dir,
    )

    timeout = captured["timeout"]
    assert isinstance(timeout, benchmark_module.httpx.Timeout)
    assert timeout.read == 300.0
    assert captured["follow_redirects"] is True
    assert captured["replay_payload"] == {"trial_name": "cancel-async-tasks__trial", "count": 1}


def test_run_benchmark_skips_gateway_fetch_in_lean_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = AppConfig.from_env(
        {
            "TINYHARNESS_JOBS_DIR": (tmp_path / "artifacts" / "runs").as_posix(),
        }
    )
    config = replace(config, benchmark=replace(config.benchmark, mode=BenchmarkMode.LEAN))
    fetched_trials: list[str] = []

    monkeypatch.setattr(benchmark_module, "resolve_gateway_base_url", lambda: "https://gateway.example")
    monkeypatch.setattr(benchmark_module, "resolve_proxy_token", lambda _agent: "secret-token")
    monkeypatch.setattr(
        benchmark_module,
        "create_parent_run",
        lambda **kwargs: SimpleNamespace(run_id="parent-run", experiment_id="exp-1", tracking_uri="sqlite:///tmp/mlflow.db"),
    )
    monkeypatch.setattr(benchmark_module, "tracking_environment", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        benchmark_module,
        "_run_harbor",
        lambda _config_path: subprocess.CompletedProcess(args=["uv"], returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_fetch_trial_gateway_debug_artifacts",
        lambda **kwargs: fetched_trials.append(kwargs["trial_name"]),
    )
    monkeypatch.setattr(benchmark_module, "write_markdown_summary", lambda summary: summary.job_dir / "summary.md")
    monkeypatch.setattr(benchmark_module, "finalize_benchmark_run", lambda **_kwargs: None)

    def fake_load_job_summary(_run_dir: Path):
        trial_dir = tmp_path / "artifacts" / "runs" / "tb10-v0-20260312-120000" / "trial-a"
        trial_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            job_dir=trial_dir.parent,
            trials=[SimpleNamespace(trial_name="trial-a", trial_dir=trial_dir)],
        )

    monkeypatch.setattr(benchmark_module, "load_job_summary", fake_load_job_summary)
    monkeypatch.setattr(benchmark_module, "build_job_name", lambda _task_set_name: "tb10-v0-20260312-120000")

    summary = benchmark_module.run_benchmark(config)

    assert summary.job_dir.name == "tb10-v0-20260312-120000"
    assert fetched_trials == []
