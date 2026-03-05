from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from harbor.models.agent.context import AgentContext
from harbor.models.task.id import LocalTaskId
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, TrialResult
from harbor.models.verifier.result import VerifierResult

from tinyharness.results import _classify_failure


def _sample_trial(task_name: str = "filter-js-from-html") -> TrialResult:
    started_at = datetime.now(UTC)
    return TrialResult(
        task_name=task_name,
        trial_name=f"{task_name}-trial",
        trial_uri=f"file:///tmp/{task_name}-trial",
        task_id=LocalTaskId(path=Path(task_name)),
        source="terminal-bench",
        task_checksum="checksum",
        config=TrialConfig(task=TaskConfig(path=Path(task_name))),
        agent_info=AgentInfo(name="qwen-claude-sdk", version="0.1.48", model_info=None),
        agent_result=AgentContext(metadata={}),
        verifier_result=VerifierResult(rewards={"reward": 0}),
        started_at=started_at,
        finished_at=started_at,
    )


def test_classify_failure_ignores_modal_teardown_timeout_when_verifier_failed(
    tmp_path: Path,
) -> None:
    trial = _sample_trial()
    (tmp_path / "trial.log").write_text(
        "Modal sandbox wait timeout after 30.0s\n",
        encoding="utf-8",
    )
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    (verifier_dir / "test-stdout.txt").write_text(
        "AssertionError: sanitizer changed clean html\n",
        encoding="utf-8",
    )
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "qwen-claude-sdk.txt").write_text("", encoding="utf-8")

    assert _classify_failure(trial, tmp_path) == "verifier_assertion_failed"
