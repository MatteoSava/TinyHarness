from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import dspy

from tinyharness.constants import PROJECT_ROOT
from tinyharness.dspy_prompt import (
    DEFAULT_AGENT_CONSTRAINTS,
    DEFAULT_AGENT_TOOLS,
    AgentPromptProgram,
    AgentPromptSpec,
    build_seed_agent_prompt,
)
from tinyharness.env import load_dotenv


DEFAULT_GEPA_PROMPT_DIR = PROJECT_ROOT / "state" / "dspy" / "gepa-agent-prompt"


def _gateway_lm(*, max_tokens: int) -> dspy.LM:
    load_dotenv(PROJECT_ROOT / ".env")
    state_path = PROJECT_ROOT / "state" / "modal" / "qwen-server.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    token = os.environ.get("TINYHARNESS_PROXY_TOKEN")
    if not token:
        raise RuntimeError("TINYHARNESS_PROXY_TOKEN is required in .env or the shell.")
    return dspy.LM(
        "openai/qwen3.5-35b-a3b-ud-iq3_s",
        api_base=state["web_url"].rstrip("/") + "/openai-proxy/v1",
        api_key=token,
        temperature=0.0,
        max_tokens=max_tokens,
        cache=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def _trainset() -> list[dspy.Example]:
    tools = ", ".join(DEFAULT_AGENT_TOOLS)
    constraints = "\n".join(f"- {item}" for item in DEFAULT_AGENT_CONSTRAINTS)
    return [
        dspy.Example(
            task="Implement async run_tasks with bounded concurrency and cancellation-safe cleanup.",
            tools=tools,
            constraints=constraints,
            expected_terms=[
                "bounded concurrency",
                "cancel queued tasks",
                "await cleanup",
                "KeyboardInterrupt",
                "verify cancellation",
            ],
        ).with_inputs("task", "tools", "constraints"),
        dspy.Example(
            task="Solve a Terminal-Bench coding task by inspecting files, editing minimally, and running verifier-focused tests.",
            tools=tools,
            constraints=constraints,
            expected_terms=["inspect first", "minimal edit", "run tests", "tool calls", "final verification"],
        ).with_inputs("task", "tools", "constraints"),
    ]


def _metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: object | None = None,
    pred_name: str | None = None,
    pred_trace: object | None = None,
) -> dspy.Prediction:
    prompt = str(getattr(pred, "system_prompt", "") or "")
    lower = prompt.lower()
    expected = list(getattr(example, "expected_terms", []) or [])
    hits = [term for term in expected if term.lower() in lower]
    common_terms = ["tool", "verify", "test", "minimal", "cancellation"]
    common_hits = [term for term in common_terms if term in lower]
    score = min(1.0, (len(hits) / max(1, len(expected))) * 0.75 + (len(common_hits) / len(common_terms)) * 0.25)
    missing = [term for term in expected if term not in hits]
    feedback = (
        f"Score {score:.2f}. Include missing benchmark guidance: {missing}. "
        "The prompt should be concise, action-oriented, and specifically tell the agent to inspect, edit, "
        "run targeted tests, preserve cancellation cleanup, and report verification."
    )
    return dspy.Prediction(score=score, feedback=feedback)


def compile_prompt(*, max_metric_calls: int, output_dir: Path, max_tokens: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    lm = _gateway_lm(max_tokens=max_tokens)
    dspy.configure(lm=lm)

    seed_prompt = build_seed_agent_prompt(AgentPromptSpec(task="Solve Terminal-Bench coding tasks with dedicated tools."))
    program = AgentPromptProgram(seed_prompt=seed_prompt)
    optimizer = dspy.GEPA(
        metric=_metric,
        reflection_lm=lm,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=1,
        num_threads=1,
        track_stats=True,
        log_dir=output_dir.as_posix(),
        skip_perfect_score=False,
    )
    trainset = _trainset()
    compiled = optimizer.compile(program, trainset=trainset, valset=trainset)
    instructions = compiled.generate.signature.instructions.strip()
    prompt_path = output_dir / "compiled-agent-prompt.txt"
    prompt_path.write_text(instructions + "\n", encoding="utf-8")
    metadata = {
        "compiled_prompt_path": prompt_path.as_posix(),
        "model": "qwen3.5-35b-a3b-ud-iq3_s",
        "train_examples": len(trainset),
        "max_metric_calls": max_metric_calls,
        "instruction_chars": len(instructions),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print("---PROMPT---")
    print(instructions)
    return prompt_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile the TinyHarness agent prompt with DSPy GEPA.")
    parser.add_argument("--max-metric-calls", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GEPA_PROMPT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compile_prompt(max_metric_calls=args.max_metric_calls, output_dir=args.output_dir, max_tokens=args.max_tokens)
    return 0
