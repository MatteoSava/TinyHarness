from __future__ import annotations

import importlib.metadata
import json
import os
import shlex
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths


_PROXY_TOKEN_MARKER = "::tinyharness::"


def encode_proxy_token(
    base_token: str,
    *,
    job_name: str,
    trial_name: str,
    correlation_id: str,
) -> str:
    if not base_token:
        return base_token
    metadata = urlencode(
        {
            "job": job_name,
            "trial": trial_name,
            "cid": correlation_id,
        }
    )
    return f"{base_token}{_PROXY_TOKEN_MARKER}{metadata}"


class QwenClaudeSDKAgent(BaseInstalledAgent):
    def __init__(
        self,
        max_turns: int = 20,
        max_thinking_tokens: int = 8192,
        workspace_cwd: str = "/app",
        benchmark_mode: str = "debug",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        extra_env = kwargs.pop("extra_env", None)
        self._agent_env = dict(extra_env or {})
        super().__init__(*args, extra_env=None, **kwargs)
        self._max_turns = max_turns
        self._max_thinking_tokens = max_thinking_tokens
        self._workspace_cwd = workspace_cwd
        self._benchmark_mode = benchmark_mode
        self._sdk_version = importlib.metadata.version("claude-agent-sdk")
        self._dspy_version = importlib.metadata.version("dspy")
        self._gepa_version = importlib.metadata.version("gepa")

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
                "dspy_version": self._dspy_version,
                "gepa_version": self._gepa_version,
                "runner_script": self._runner_script_path.read_text(encoding="utf-8"),
                "dspy_prompt_script": (Path(__file__).parent / "dspy_prompt.py").read_text(encoding="utf-8"),
                "extra_python_packages": (
                    ' \\\n  "mlflow==3.10.1"'
                    if self._benchmark_mode == "debug"
                    else ""
                ),
            }
        )
        return variables

    def version(self) -> str | None:
        return self._sdk_version

    def _task_identity(self) -> tuple[str, str]:
        trial_name = self.logs_dir.parent.name
        if "__" in trial_name:
            task_name = trial_name.split("__", 1)[0]
        else:
            task_name = trial_name
        return task_name, trial_name

    @staticmethod
    def _correlation_id(job_name: str, trial_name: str) -> str:
        namespace = job_name or "tinyharness"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{trial_name}"))

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        task_name, trial_name = self._task_identity()
        base_env = {**self._agent_env, **os.environ}
        job_name = base_env.get("TINYHARNESS_JOB_NAME", "")
        correlation_id = self._correlation_id(job_name, trial_name)
        base_url = base_env.get("ANTHROPIC_BASE_URL", "")
        proxy_token = encode_proxy_token(
            base_env.get("ANTHROPIC_API_KEY", ""),
            job_name=job_name,
            trial_name=trial_name,
            correlation_id=correlation_id,
        )
        model_name = env_model = base_env.get("ANTHROPIC_MODEL", self.model_name or "")
        env = {
            "ANTHROPIC_API_KEY": proxy_token,
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_MODEL": env_model,
            "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "IS_SANDBOX": "1",
            "HOME": base_env.get("HOME", os.environ.get("HOME", "/root")),
            "TINYHARNESS_JOB_NAME": job_name,
            "TINYHARNESS_TASK_NAME": task_name,
            "TINYHARNESS_TRIAL_NAME": trial_name,
            "TINYHARNESS_CORRELATION_ID": correlation_id,
            "TINYHARNESS_RUN_MODE": self._benchmark_mode,
            "TINYHARNESS_AGENT_PROMPT_MODE": base_env.get("TINYHARNESS_AGENT_PROMPT_MODE", "dspy-gepa"),
            "TINYHARNESS_DSPY_COMPILED_PROMPT_PATH": base_env.get("TINYHARNESS_DSPY_COMPILED_PROMPT_PATH", ""),
        }
        if self._benchmark_mode == "debug":
            env.update(
                {
                    "MLFLOW_TRACKING_URI": base_env.get("MLFLOW_TRACKING_URI", ""),
                    "MLFLOW_TRACKING_USERNAME": base_env.get("MLFLOW_TRACKING_USERNAME", ""),
                    "MLFLOW_TRACKING_PASSWORD": base_env.get("MLFLOW_TRACKING_PASSWORD", ""),
                    "MLFLOW_EXPERIMENT_ID": base_env.get("MLFLOW_EXPERIMENT_ID", ""),
                    "TINYHARNESS_PARENT_RUN_ID": base_env.get("TINYHARNESS_PARENT_RUN_ID", ""),
                }
            )
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
