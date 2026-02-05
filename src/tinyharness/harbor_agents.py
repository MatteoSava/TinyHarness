from __future__ import annotations

import importlib.metadata
import json
import os
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths


class QwenClaudeSDKAgent(BaseInstalledAgent):
    def __init__(
        self,
        max_turns: int = 20,
        max_thinking_tokens: int = 8192,
        workspace_cwd: str = "/app",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._max_turns = max_turns
        self._max_thinking_tokens = max_thinking_tokens
        self._workspace_cwd = workspace_cwd
        self._sdk_version = importlib.metadata.version("claude-agent-sdk")

    @staticmethod
    def name() -> str:
        return "qwen-claude-sdk"

    @property
    def _install_agent_template_path(self) -> Path:
        return Path(__file__).parent / "templates" / "install-qwen-claude-sdk.sh.j2"

    @property
    def _runner_script_path(self) -> Path:
        return Path(__file__).parent / "sdk_runner.py"

    @property
    def _template_variables(self) -> dict[str, str]:
        variables = super()._template_variables
        variables.update(
            {
                "sdk_version": self._sdk_version,
                "runner_script": self._runner_script_path.read_text(encoding="utf-8"),
            }
        )
        return variables

    def version(self) -> str | None:
        return self._sdk_version

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        model_name = env_model = os.environ.get("ANTHROPIC_MODEL", self.model_name or "")
        env = {
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL", ""),
            "ANTHROPIC_MODEL": env_model,
            "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "IS_SANDBOX": "1",
            "HOME": os.environ.get("HOME", "/root"),
        }
        env = {key: value for key, value in env.items() if value}

        command = (
            'export PATH="$HOME/.local/bin:$PATH"; '
            "/installed-agent/.venv/bin/python /installed-agent/sdk_runner.py "
            f"--logs-dir {shlex.quote(EnvironmentPaths.agent_dir.as_posix())} "
            f"--cwd {shlex.quote(self._workspace_cwd)} "
            f"--max-turns {self._max_turns} "
            f"--max-thinking-tokens {self._max_thinking_tokens} "
        )
        if model_name:
            command += f"--model {shlex.quote(model_name)} "
        command += (
            f"{shlex.quote(instruction)} "
            "2>&1 | stdbuf -oL tee /logs/agent/qwen-claude-sdk.txt"
        )

        return [ExecInput(command=command, cwd=self._workspace_cwd, env=env)]

    def populate_context_post_run(self, context: AgentContext) -> None:
        summary_path = self.logs_dir / "summary.json"
        if not summary_path.exists():
            return

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        context.n_input_tokens = summary.get("n_input_tokens")
        context.n_cache_tokens = summary.get("n_cache_tokens")
        context.n_output_tokens = summary.get("n_output_tokens")
        context.cost_usd = summary.get("cost_usd")
        context.metadata = summary.get("metadata")
