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


def _metadata(trial: TrialResult) -> dict[str, Any]:
    if trial.agent_result and trial.agent_result.metadata:
        return dict(trial.agent_result.metadata)
    return {}


def _telemetry(trial: TrialResult) -> dict[str, Any]:
    metadata = _metadata(trial)
    value = metadata.get("telemetry")
    return value if isinstance(value, dict) else {}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _contains_timeout(text: str) -> bool:
    lowered = text.lower()
    return "timed out" in lowered or "timeout" in lowered


def _classify_failure(trial: TrialResult, trial_dir: Path) -> str:
    if _trial_passed(trial):
        return "passed"

    trial_log = _read_text(trial_dir / "trial.log")
    verifier_log = _read_text(trial_dir / "verifier" / "test-stdout.txt")
    stderr_log = _read_text(trial_dir / "agent" / "qwen-claude-sdk.txt")
    metadata = _metadata(trial)
    exception_type = (
        trial.exception_info.exception_type
        if trial.exception_info is not None
        else None
    )

    if exception_type == "EnvironmentStartTimeoutError":
        return "environment_timeout"
    if exception_type == "VerifierTimeoutError":
        return "verifier_timeout"
    if exception_type == "AgentTimeoutError":
        return "agent_timeout"
    if _contains_timeout(verifier_log):
        return "verifier_timeout"
    if _contains_timeout(stderr_log):
        return "agent_timeout"
    if "ALERT DETECTED!" in verifier_log:
        return "xss_detected"
    if "No such file or directory" in verifier_log and "/app/" in verifier_log:
        return "missing_output"
    if metadata.get("is_error") is True:
        return "agent_error"
    if trial.exception_info is not None:
        return "infra_error"
    if "AssertionError" in verifier_log:
        return "verifier_assertion_failed"
    if verifier_log.strip():
        return "verifier_failed"
    return "unknown_failure"


@dataclass(frozen=True)
class TrialSummary:
    trial_name: str
    task_name: str
    model_name: str | None
    passed: int
    solved: int
    pass_at_1: int
    timeout: int
    failure_type: str
    score: float
    duration_seconds: float | None
    wall_clock_seconds: float | None
    prompt_tokens: int
    cache_tokens: int
    output_tokens: int
    duration_ms: int | None
    duration_api_ms: int | None
    tokens_per_second: float | None
    turn_count: int
    average_turn_latency_ms: float | None
    max_turn_latency_ms: int | None
    per_turn_latencies_ms: list[int]
    tool_call_count: int
    shell_command_count: int
    tool_output_bytes: int
    tool_output_tokens_estimate: int
    first_event_latency_ms: int | None
    first_text_latency_ms: int | None
    response_complete_latency_ms: int | None
    run_mode: str
    gateway_debug_enabled: bool
    live_tracing_enabled: bool
    mlflow_run_id: str | None
    trace_id: str | None
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
        metadata = _metadata(trial)
        telemetry = _telemetry(trial)
        agent_env = trial.config.agent.env if trial.config.agent and trial.config.agent.env else {}
        run_mode = (
            metadata.get("run_mode")
            if isinstance(metadata.get("run_mode"), str)
            else agent_env.get("TINYHARNESS_RUN_MODE", "debug")
        )
        gateway_debug_enabled = metadata.get("gateway_debug_enabled")
        live_tracing_enabled = metadata.get("live_tracing_enabled")
        duration_ms = metadata.get("duration_ms")
        duration_api_ms = metadata.get("duration_api_ms")
        duration_seconds = _duration_seconds(trial)
        tokens_per_second = None
        if duration_ms and output_tokens:
            tokens_per_second = output_tokens / (float(duration_ms) / 1000.0)

        failure_type = _classify_failure(trial, child)

        trials.append(
            TrialSummary(
                trial_name=trial.trial_name,
                task_name=trial.task_name,
                model_name=trial.agent_info.model_info.name if trial.agent_info.model_info else None,
                passed=_trial_passed(trial),
                solved=_trial_passed(trial),
                pass_at_1=_trial_passed(trial),
                timeout=int(failure_type.endswith("timeout")),
                failure_type=failure_type,
                score=_reward_score(trial),
                duration_seconds=duration_seconds,
                wall_clock_seconds=duration_seconds,
                prompt_tokens=prompt_tokens,
                cache_tokens=cache_tokens,
                output_tokens=output_tokens,
                duration_ms=int(duration_ms) if duration_ms is not None else None,
                duration_api_ms=int(duration_api_ms) if duration_api_ms is not None else None,
                tokens_per_second=tokens_per_second,
                turn_count=int(telemetry.get("turn_count") or 0),
                average_turn_latency_ms=(
                    float(telemetry["average_turn_latency_ms"])
                    if telemetry.get("average_turn_latency_ms") is not None
                    else None
                ),
                max_turn_latency_ms=(
                    int(telemetry["max_turn_latency_ms"])
                    if telemetry.get("max_turn_latency_ms") is not None
                    else None
                ),
                per_turn_latencies_ms=[int(value) for value in telemetry.get("per_turn_latencies_ms", [])],
                tool_call_count=int(telemetry.get("tool_call_count") or 0),
                shell_command_count=int(telemetry.get("shell_command_count") or 0),
                tool_output_bytes=int(telemetry.get("tool_output_bytes") or 0),
                tool_output_tokens_estimate=int(telemetry.get("tool_output_tokens_estimate") or 0),
                first_event_latency_ms=(
                    int(telemetry["first_event_latency_ms"])
                    if telemetry.get("first_event_latency_ms") is not None
                    else None
                ),
                first_text_latency_ms=(
                    int(telemetry["first_text_latency_ms"])
                    if telemetry.get("first_text_latency_ms") is not None
                    else None
                ),
                response_complete_latency_ms=(
                    int(telemetry["response_complete_latency_ms"])
                    if telemetry.get("response_complete_latency_ms") is not None
                    else None
                ),
                run_mode=str(run_mode),
                gateway_debug_enabled=(
                    bool(gateway_debug_enabled)
                    if gateway_debug_enabled is not None
                    else bool(run_mode == "debug")
                ),
                live_tracing_enabled=(
                    bool(live_tracing_enabled)
                    if live_tracing_enabled is not None
                    else bool(run_mode == "debug")
                ),
                mlflow_run_id=metadata.get("mlflow_run_id") if isinstance(metadata.get("mlflow_run_id"), str) else None,
                trace_id=metadata.get("trace_id") if isinstance(metadata.get("trace_id"), str) else None,
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
        "| Task | Passed | Score | Failure | Duration (s) | Prompt | Output |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for trial in summary.trials:
        duration = f"{trial.duration_seconds:.1f}" if trial.duration_seconds is not None else "-"
        lines.append(
            f"| {trial.task_name} | {trial.passed} | {trial.score:.3f} | {trial.failure_type} | {duration} | "
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
