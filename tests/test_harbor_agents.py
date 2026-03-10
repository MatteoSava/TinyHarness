from __future__ import annotations

import uuid
from pathlib import Path

from tinyharness.harbor_agents import QwenClaudeSDKAgent, encode_proxy_token
from tinyharness.dspy_prompt import DEFAULT_AGENT_TOOLS
from tinyharness.sdk_runner import build_sdk_options


def test_build_sdk_options_uses_dspy_gepa_prompt_and_dedicated_tools() -> None:
    options = build_sdk_options(
        instruction="fix the benchmark task",
        cwd="/app",
        max_turns=12,
        max_thinking_tokens=2048,
        model="qwen3.5-35b-a3b-ud-iq3_s",
    )

    assert isinstance(options.system_prompt, str)
    assert "TinyHarness benchmark agent" in options.system_prompt
    assert "fix the benchmark task" in options.system_prompt
    assert options.tools == list(DEFAULT_AGENT_TOOLS)
    assert options.allowed_tools == list(DEFAULT_AGENT_TOOLS)
    assert options.setting_sources == []
    assert options.permission_mode == "bypassPermissions"


def test_agent_command_includes_gateway_env(monkeypatch, tmp_path: Path) -> None:
    trial_logs_dir = tmp_path / "cancel-async-tasks__trial" / "agent"
    trial_logs_dir.mkdir(parents=True)
    agent = QwenClaudeSDKAgent(
        logs_dir=trial_logs_dir,
        workspace_cwd="/app",
        benchmark_mode="debug",
        extra_env={
            "ANTHROPIC_BASE_URL": "https://gateway.example",
            "ANTHROPIC_API_KEY": "secret-token",
            "ANTHROPIC_MODEL": "qwen3.5-35b-a3b-ud-iq3_s",
            "MLFLOW_TRACKING_URI": "https://mlflow.example",
            "MLFLOW_TRACKING_USERNAME": "admin",
            "MLFLOW_TRACKING_PASSWORD": "secret-password",
            "MLFLOW_EXPERIMENT_ID": "123",
            "TINYHARNESS_PARENT_RUN_ID": "parent-run",
            "TINYHARNESS_JOB_NAME": "smoke-v0-20260312-120000",
        },
    )
    commands = agent.create_run_agent_commands("fix the task")

    assert len(commands) == 1
    command = commands[0]
    expected_correlation_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, "smoke-v0-20260312-120000:cancel-async-tasks__trial")
    )
    assert command.cwd == "/app"
    assert command.env["ANTHROPIC_BASE_URL"] == "https://gateway.example"
    assert command.env["ANTHROPIC_API_KEY"] == encode_proxy_token(
        "secret-token",
        job_name="smoke-v0-20260312-120000",
        trial_name="cancel-async-tasks__trial",
        correlation_id=expected_correlation_id,
    )
    assert command.env["MLFLOW_TRACKING_URI"] == "https://mlflow.example"
    assert command.env["TINYHARNESS_PARENT_RUN_ID"] == "parent-run"
    assert command.env["TINYHARNESS_JOB_NAME"] == "smoke-v0-20260312-120000"
    assert command.env["TINYHARNESS_TASK_NAME"] == "cancel-async-tasks"
    assert command.env["TINYHARNESS_TRIAL_NAME"] == "cancel-async-tasks__trial"
    assert command.env["TINYHARNESS_CORRELATION_ID"] == expected_correlation_id
    assert command.env["TINYHARNESS_RUN_MODE"] == "debug"
    assert command.env["TINYHARNESS_AGENT_PROMPT_MODE"] == "dspy-gepa"
    assert command.env["IS_SANDBOX"] == "1"
    assert "--max-turns 20" in command.command
    assert "--logs-dir /logs/agent" in command.command
    assert "sdk_runner.py" in command.command
    assert "fix the task" in command.command


def test_agent_command_embeds_compiled_prompt_from_host_path(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "compiled-agent-prompt.txt"
    prompt_path.write_text("Compiled GEPA prompt.", encoding="utf-8")
    trial_logs_dir = tmp_path / "cancel-async-tasks__trial" / "agent"
    trial_logs_dir.mkdir(parents=True)
    monkeypatch.setenv("TINYHARNESS_DSPY_COMPILED_PROMPT_PATH", prompt_path.as_posix())

    agent = QwenClaudeSDKAgent(logs_dir=trial_logs_dir, workspace_cwd="/app", benchmark_mode="lean")

    command = agent.create_run_agent_commands("fix the task")[0]

    assert command.env["TINYHARNESS_DSPY_COMPILED_PROMPT_PATH"] == prompt_path.as_posix()
    assert command.env["TINYHARNESS_DSPY_COMPILED_PROMPT"] == "Compiled GEPA prompt."


def test_install_template_installs_mlflow_for_sdk_runner(tmp_path: Path) -> None:
    agent = QwenClaudeSDKAgent(logs_dir=tmp_path, workspace_cwd="/app", benchmark_mode="debug")

    variables = agent._template_variables

    assert variables["dspy_version"]
    assert variables["gepa_version"]
    assert "class AgentPromptProgram" in variables["dspy_prompt_script"]
    assert 'mlflow==3.10.1' in variables["extra_python_packages"]


def test_agent_command_omits_mlflow_env_in_lean_mode(tmp_path: Path) -> None:
    trial_logs_dir = tmp_path / "cancel-async-tasks__trial" / "agent"
    trial_logs_dir.mkdir(parents=True)
    agent = QwenClaudeSDKAgent(
        logs_dir=trial_logs_dir,
        workspace_cwd="/app",
        benchmark_mode="lean",
        extra_env={
            "ANTHROPIC_BASE_URL": "https://gateway.example",
            "ANTHROPIC_API_KEY": "secret-token",
            "ANTHROPIC_MODEL": "qwen3.5-35b-a3b-ud-iq3_s",
            "MLFLOW_TRACKING_URI": "https://mlflow.example",
            "MLFLOW_TRACKING_USERNAME": "admin",
            "MLFLOW_TRACKING_PASSWORD": "secret-password",
            "MLFLOW_EXPERIMENT_ID": "123",
            "TINYHARNESS_PARENT_RUN_ID": "parent-run",
            "TINYHARNESS_JOB_NAME": "tb10-v1-20260313-120000",
        },
    )

    command = agent.create_run_agent_commands("fix the task")[0]

    assert command.env["TINYHARNESS_RUN_MODE"] == "lean"
    assert "MLFLOW_TRACKING_URI" not in command.env
    assert "TINYHARNESS_PARENT_RUN_ID" not in command.env


def test_install_template_skips_mlflow_in_lean_mode(tmp_path: Path) -> None:
    agent = QwenClaudeSDKAgent(logs_dir=tmp_path, workspace_cwd="/app", benchmark_mode="lean")

    variables = agent._template_variables

    assert variables["extra_python_packages"] == ""
