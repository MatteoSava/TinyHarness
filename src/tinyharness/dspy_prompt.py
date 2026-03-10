from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import dspy

DEFAULT_AGENT_TOOLS = ("Bash", "Read", "Edit", "Write", "Grep", "Glob", "LS")

DEFAULT_AGENT_CONSTRAINTS = (
    "Inspect the workspace before editing.",
    "Make the smallest code change that solves the task.",
    "Verify behavior with focused commands before declaring completion.",
    "Keep all benchmark artifacts under artifacts/ or state/.",
    "Do not call Anthropic-hosted models from the benchmark path.",
)


@dataclass(frozen=True)
class AgentPromptSpec:
    task: str
    tools: tuple[str, ...] = DEFAULT_AGENT_TOOLS
    constraints: tuple[str, ...] = DEFAULT_AGENT_CONSTRAINTS


@dataclass(frozen=True)
class AgentPromptConfig:
    system_prompt: str
    tools: tuple[str, ...]
    source: str


class AgentPromptSignature(dspy.Signature):
    """Generate a compact system prompt for a coding benchmark agent."""

    task: str = dspy.InputField(desc="Terminal-Bench task instruction the agent must solve.")
    tools: str = dspy.InputField(desc="Dedicated tool names available to the agent.")
    constraints: str = dspy.InputField(desc="Operational constraints the agent must follow.")
    system_prompt: str = dspy.OutputField(desc="Production-ready system prompt for the agent.")


class AgentPromptProgram(dspy.Module):
    def __init__(self, *, seed_prompt: str) -> None:
        super().__init__()
        signature = AgentPromptSignature.with_instructions(seed_prompt)
        self.generate = dspy.Predict(signature)

    def forward(self, *, task: str, tools: str, constraints: str) -> Any:
        return self.generate(task=task, tools=tools, constraints=constraints)


def build_seed_agent_prompt(spec: AgentPromptSpec) -> str:
    task = spec.task.strip() or "Solve the assigned Terminal-Bench task."
    tools = ", ".join(spec.tools)
    constraints = "\n".join(f"- {item}" for item in spec.constraints)
    return "\n".join(
        [
            "You are the TinyHarness benchmark agent.",
            "",
            f"Task: {task}",
            f"Dedicated tools: {tools}",
            "",
            "Use tool calls deliberately: inspect, edit, and verify through the dedicated tools only.",
            "Prefer simple, maintainable fixes over broad rewrites.",
            "",
            "Constraints:",
            constraints,
            "",
            "Return a concise final answer with the files changed and the verification run.",
        ]
    )


def build_agent_prompt_config(
    instruction: str,
    *,
    tools: Sequence[str] = DEFAULT_AGENT_TOOLS,
    compiled_prompt_path: str | Path | None = None,
) -> AgentPromptConfig:
    resolved_tools = tuple(tools)
    compiled_prompt = os.environ.get("TINYHARNESS_DSPY_COMPILED_PROMPT")
    if compiled_prompt and compiled_prompt.strip():
        return AgentPromptConfig(
            system_prompt=compiled_prompt.strip(),
            tools=resolved_tools,
            source="dspy-gepa-compiled",
        )

    prompt_path = compiled_prompt_path or os.environ.get("TINYHARNESS_DSPY_COMPILED_PROMPT_PATH")
    if prompt_path:
        path = Path(prompt_path)
        if path.exists():
            return AgentPromptConfig(
                system_prompt=path.read_text(encoding="utf-8").strip(),
                tools=resolved_tools,
                source="dspy-gepa-compiled",
            )

    spec = AgentPromptSpec(task=instruction, tools=resolved_tools)
    return AgentPromptConfig(
        system_prompt=build_seed_agent_prompt(spec),
        tools=resolved_tools,
        source="dspy-gepa-seed",
    )


def _as_lm(model: str | Any) -> Any:
    if isinstance(model, str):
        return dspy.LM(model)
    return model


def compile_gepa_agent_prompt(
    spec: AgentPromptSpec,
    *,
    trainset: Sequence[Any],
    task_lm: str | Any,
    reflection_lm: str | Any,
    metric: Callable[..., Any],
    valset: Sequence[Any] | None = None,
    auto: str = "light",
    num_threads: int = 1,
) -> str:
    if not trainset:
        raise ValueError("GEPA prompt compilation requires at least one training example.")

    dspy.configure(lm=_as_lm(task_lm))
    reflection_model = _as_lm(reflection_lm)
    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_model,
        auto=auto,
        num_threads=num_threads,
        track_stats=True,
    )
    program = AgentPromptProgram(seed_prompt=build_seed_agent_prompt(spec))
    optimized = optimizer.compile(program, trainset=list(trainset), valset=list(valset) if valset is not None else None)
    instructions = getattr(optimized.generate.signature, "instructions", "")
    if not isinstance(instructions, str) or not instructions.strip():
        raise RuntimeError("GEPA did not return optimized prompt instructions.")
    return instructions.strip()


def agent_prompt_feedback_metric(example: Any, pred: Any, trace: Any | None = None) -> dspy.Prediction:
    expected = str(getattr(example, "expected", "") or "")
    prompt = str(getattr(pred, "system_prompt", "") or "")
    missing: list[str] = []
    for term in ("tool", "verify", "edit"):
        if term not in prompt.lower():
            missing.append(term)

    score = 1.0 if expected and expected.lower() in prompt.lower() and not missing else 0.0
    if score:
        feedback = "Prompt includes the expected benchmark behavior and operational tool guidance."
    else:
        feedback = (
            "Prompt needs clearer benchmark-specific instructions. "
            f"Missing terms: {', '.join(missing) if missing else 'expected reference behavior'}."
        )
    return dspy.Prediction(score=score, feedback=feedback)
