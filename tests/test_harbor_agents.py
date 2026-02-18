from __future__ import annotations

from pathlib import Path

from tinyharness.harbor_agents import QwenClaudeSDKAgent
from tinyharness.sdk_runner import CLAUDE_CODE_PRESET, build_sdk_options


def test_build_sdk_options_uses_claude_code_preset_and_no_settings() -> None:
    options = build_sdk_options(
        cwd="/app",
        max_turns=12,
        max_thinking_tokens=2048,
        model="qwen3.5-35b-a3b-ud-iq3_s",
    )

    assert options.system_prompt == CLAUDE_CODE_PRESET
    assert options.tools == CLAUDE_CODE_PRESET
    assert options.setting_sources == []
    assert options.permission_mode == "bypassPermissions"


def test_agent_command_includes_gateway_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-token")
    monkeypatch.setenv("ANTHROPIC_MODEL", "qwen3.5-35b-a3b-ud-iq3_s")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret-password")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "123")
    monkeypatch.setenv("TINYHARNESS_PARENT_RUN_ID", "parent-run")
    monkeypatch.setenv("TINYHARNESS_JOB_NAME", "smoke-v0-20260312-120000")

    agent = QwenClaudeSDKAgent(logs_dir=tmp_path, workspace_cwd="/app")
    commands = agent.create_run_agent_commands("fix the task")

    assert len(commands) == 1
    command = commands[0]
    assert command.cwd == "/app"
    assert command.env["ANTHROPIC_BASE_URL"] == "https://gateway.example"
    assert command.env["ANTHROPIC_API_KEY"] == "secret-token"
    assert command.env["MLFLOW_TRACKING_URI"] == "https://mlflow.example"
    assert command.env["TINYHARNESS_PARENT_RUN_ID"] == "parent-run"
    assert command.env["IS_SANDBOX"] == "1"
    assert "--max-turns 20" in command.command
    assert "--logs-dir /logs/agent" in command.command
    assert "sdk_runner.py" in command.command
    assert "fix the task" in command.command


def test_install_template_installs_mlflow_for_sdk_runner(tmp_path: Path) -> None:
    agent = QwenClaudeSDKAgent(logs_dir=tmp_path, workspace_cwd="/app")

    template = agent._install_agent_template_path.read_text(encoding="utf-8")

    assert "claude-agent-sdk" in template
    assert "mlflow" in template
