from __future__ import annotations

from types import SimpleNamespace

from tinyharness import dspy_prompt
from tinyharness.dspy_prompt import AgentPromptSpec, build_agent_prompt_config, compile_gepa_agent_prompt


def test_build_agent_prompt_config_returns_custom_prompt_and_tools() -> None:
    config = build_agent_prompt_config("fix the failing benchmark task")

    assert config.source == "dspy-gepa-seed"
    assert config.tools == dspy_prompt.DEFAULT_AGENT_TOOLS
    assert "TinyHarness benchmark agent" in config.system_prompt
    assert "fix the failing benchmark task" in config.system_prompt
    assert "Bash, Read, Edit, Write, Grep, Glob, LS" in config.system_prompt
    assert "Use tool calls deliberately" in config.system_prompt


def test_compile_gepa_agent_prompt_returns_optimized_instruction(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _FakeGEPA:
        def __init__(self, **kwargs) -> None:
            calls["gepa_kwargs"] = kwargs

        def compile(self, student, *, trainset, valset=None):
            calls["student"] = student
            calls["trainset"] = trainset
            calls["valset"] = valset
            student.generate.signature.instructions = "Optimized GEPA instruction."
            return student

    monkeypatch.setattr(dspy_prompt.dspy, "GEPA", _FakeGEPA)
    monkeypatch.setattr(dspy_prompt.dspy, "LM", lambda name: f"lm:{name}")
    monkeypatch.setattr(dspy_prompt.dspy, "configure", lambda **kwargs: calls.setdefault("configure", kwargs))

    prompt = compile_gepa_agent_prompt(
        AgentPromptSpec(task="solve one terminal benchmark task"),
        trainset=[SimpleNamespace(input="task", expected="done")],
        task_lm="openai/gpt-4o-mini",
        reflection_lm="openai/gpt-5",
        metric=lambda example, pred, trace=None: True,
    )

    assert prompt == "Optimized GEPA instruction."
    assert calls["configure"] == {"lm": "lm:openai/gpt-4o-mini"}
    assert calls["gepa_kwargs"] == {
        "metric": calls["gepa_kwargs"]["metric"],
        "reflection_lm": "lm:openai/gpt-5",
        "auto": "light",
        "num_threads": 1,
        "track_stats": True,
    }
    assert calls["trainset"] == [SimpleNamespace(input="task", expected="done")]
