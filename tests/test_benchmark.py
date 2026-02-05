from __future__ import annotations

from tinyharness.benchmark import build_harbor_job_config
from tinyharness.config import AppConfig


def test_build_harbor_job_config_uses_modal_and_custom_agent() -> None:
    config = AppConfig.from_env({})

    job_config = build_harbor_job_config(
        config,
        base_url="https://gateway.example",
        proxy_token="secret-token",
        job_name="smoke-v0-20260312-120000",
    )

    assert job_config.environment.type.value == "modal"
    assert job_config.agents[0].import_path == "tinyharness.harbor_agents:QwenClaudeSDKAgent"
    assert job_config.agents[0].env["ANTHROPIC_BASE_URL"] == "https://gateway.example"
    assert job_config.datasets[0].name == "terminal-bench"
    assert job_config.datasets[0].task_names == list(config.benchmark.tasks)
