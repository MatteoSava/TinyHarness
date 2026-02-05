from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from harbor.models.job.result import JobResult
from harbor.models.trial.result import TrialResult


def _parse_timestamp(value: datetime | None) -> datetime | None:
    return value


def _duration_seconds(trial: TrialResult) -> float | None:
    start = _parse_timestamp(trial.started_at)
    finish = _parse_timestamp(trial.finished_at)
    if start is None or finish is None:
        return None
    return (finish - start).total_seconds()


def _reward_score(trial: TrialResult) -> float:
    rewards = trial.verifier_result.rewards if trial.verifier_result else None
    if not rewards:
        return 0.0
    return float(sum(float(value) for value in rewards.values()) / len(rewards))


def _trial_passed(trial: TrialResult) -> int:
    rewards = trial.verifier_result.rewards if trial.verifier_result else None
    if not rewards:
        return 0
    return int(all(float(value) > 0 for value in rewards.values()))


@dataclass(frozen=True)
class TrialSummary:
    trial_name: str
    task_name: str
    model_name: str | None
    passed: int
    score: float
    duration_seconds: float | None
    prompt_tokens: int
    cache_tokens: int
    output_tokens: int
    duration_ms: int | None
    duration_api_ms: int | None
    tokens_per_second: float | None
    trial_dir: Path
    raw: TrialResult


@dataclass(frozen=True)
class JobSummary:
    job_dir: Path
    job_result: JobResult
    trials: list[TrialSummary]

    @property
    def run_id(self) -> str:
        return self.job_dir.name

    @property
    def pass_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(trial.passed for trial in self.trials) / len(self.trials)

    @property
    def mean_score(self) -> float:
        if not self.trials:
            return 0.0
        return sum(trial.score for trial in self.trials) / len(self.trials)


def load_job_summary(job_dir: Path) -> JobSummary:
    job_result_path = job_dir / "result.json"
    if not job_result_path.exists():
        raise FileNotFoundError(f"Harbor result.json not found in {job_dir}")

    job_result = JobResult.model_validate_json(job_result_path.read_text(encoding="utf-8"))
    trials: list[TrialSummary] = []

    for child in sorted(job_dir.iterdir()):
        if not child.is_dir():
            continue

        trial_result_path = child / "result.json"
        if not trial_result_path.exists():
            continue

        trial = TrialResult.model_validate_json(trial_result_path.read_text(encoding="utf-8"))
        prompt_tokens = trial.agent_result.n_input_tokens if trial.agent_result and trial.agent_result.n_input_tokens else 0
        cache_tokens = trial.agent_result.n_cache_tokens if trial.agent_result and trial.agent_result.n_cache_tokens else 0
        output_tokens = trial.agent_result.n_output_tokens if trial.agent_result and trial.agent_result.n_output_tokens else 0
        metadata = trial.agent_result.metadata if trial.agent_result and trial.agent_result.metadata else {}
        duration_ms = metadata.get("duration_ms")
        duration_api_ms = metadata.get("duration_api_ms")
        duration_seconds = _duration_seconds(trial)
        tokens_per_second = None
        if duration_ms and output_tokens:
            tokens_per_second = output_tokens / (float(duration_ms) / 1000.0)

        trials.append(
            TrialSummary(
                trial_name=trial.trial_name,
                task_name=trial.task_name,
                model_name=trial.agent_info.model_info.name if trial.agent_info.model_info else None,
                passed=_trial_passed(trial),
                score=_reward_score(trial),
                duration_seconds=duration_seconds,
                prompt_tokens=prompt_tokens,
                cache_tokens=cache_tokens,
                output_tokens=output_tokens,
                duration_ms=int(duration_ms) if duration_ms is not None else None,
                duration_api_ms=int(duration_api_ms) if duration_api_ms is not None else None,
                tokens_per_second=tokens_per_second,
                trial_dir=child,
                raw=trial,
            )
        )

    return JobSummary(job_dir=job_dir, job_result=job_result, trials=trials)


def build_markdown_summary(summary: JobSummary) -> str:
    lines = [
        f"# Run {summary.run_id}",
        "",
        f"- Total trials: {len(summary.trials)}",
        f"- Pass rate: {summary.pass_rate:.2%}",
        f"- Mean score: {summary.mean_score:.3f}",
        f"- Harbor recorded errors: {summary.job_result.stats.n_errors}",
        "",
        "| Task | Passed | Score | Duration (s) | Prompt | Output |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for trial in summary.trials:
        duration = f"{trial.duration_seconds:.1f}" if trial.duration_seconds is not None else "-"
        lines.append(
            f"| {trial.task_name} | {trial.passed} | {trial.score:.3f} | {duration} | "
            f"{trial.prompt_tokens} | {trial.output_tokens} |"
        )

    return "\n".join(lines) + "\n"


def write_markdown_summary(summary: JobSummary) -> Path:
    output_path = summary.job_dir / "summary.md"
    output_path.write_text(build_markdown_summary(summary), encoding="utf-8")
    return output_path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

