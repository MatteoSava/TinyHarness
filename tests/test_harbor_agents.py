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

    agent = QwenClaudeSDKAgent(logs_dir=tmp_path, workspace_cwd="/app")
    commands = agent.create_run_agent_commands("fix the task")

    assert len(commands) == 1
    command = commands[0]
    assert command.cwd == "/app"
    assert command.env["ANTHROPIC_BASE_URL"] == "https://gateway.example"
    assert command.env["ANTHROPIC_API_KEY"] == "secret-token"
    assert command.env["IS_SANDBOX"] == "1"
    assert "--max-turns 20" in command.command
    assert "--logs-dir /logs/agent" in command.command
    assert "sdk_runner.py" in command.command
    assert "fix the task" in command.command
