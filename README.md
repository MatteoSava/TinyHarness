# TinyHarness

TinyHarness is a small benchmark harness for coding-agent experiments on Terminal-Bench 2.0.
It runs a Claude-Code-style SDK agent against a Modal-hosted Qwen gateway, records benchmark
artifacts locally, and logs structured metrics to MLflow.

The current agent path is designed for repeatable experiments:

- Qwen inference is routed through a local OpenAI-compatible gateway on Modal.
- Agent execution uses dedicated tool permissions instead of reading local Claude settings.
- DSPy/GEPA can compile the system prompt used by the benchmark agent.
- Runs write their Harbor config, gateway config, summaries, traces, and MLflow metadata under `artifacts/`.

## Status

What is working now:

- `uv run pytest -q` passes locally.
- `uv run tinyharness agent-prompt "fix the task"` prints the DSPy-generated agent prompt config.
- `uv run tinyharness compile-gepa-prompt --max-metric-calls 8` produces a compiled prompt under `state/dspy/gepa-agent-prompt/`.
- The latest local GEPA micro-run improved the internal prompt metric from `0.175` to `0.625`.
- A Terminal-Bench sample run executes through Harbor and Modal.

Current limitation:

- The available Terminal-Bench sample run is not a passing benchmark result yet. The latest sampled task executed and ran verifier tests, but failed the task-specific cleanup expectation. Treat the GEPA score as a prompt-quality optimization signal, not as a benchmark pass rate.

## Quick Start

```bash
uv sync --group dev
cp .env.example .env
uv run pytest -q
```

Deploy or update the Qwen gateway:

```bash
uv run tinyharness serve-qwen
```

Inspect the seed DSPy prompt:

```bash
uv run tinyharness agent-prompt "Solve one Terminal-Bench task with minimal edits."
```

Compile a GEPA prompt:

```bash
uv run tinyharness compile-gepa-prompt --max-metric-calls 8
```

Run a focused benchmark sample with the compiled prompt:

```bash
TINYHARNESS_DSPY_COMPILED_PROMPT_PATH=state/dspy/gepa-agent-prompt/compiled-agent-prompt.txt \
uv run tinyharness run-benchmark --task-set gepa-v0 --tasks cancel-async-tasks --mode lean
```

Print a saved run summary:

```bash
uv run tinyharness fetch-results <run-id>
```

## Architecture

- `src/tinyharness/modal_server.py` defines the Modal Qwen gateway.
- `src/tinyharness/harbor_agents.py` installs and runs the SDK-based benchmark agent.
- `src/tinyharness/sdk_runner.py` executes the agent and captures traces/tool calls.
- `src/tinyharness/dspy_prompt.py` builds the agent prompt and dedicated tool list.
- `src/tinyharness/gepa_prompt_compiler.py` compiles the prompt with DSPy/GEPA.
- `src/tinyharness/mlflow_tracking.py` records run metrics and artifacts.

## Artifacts

Generated files are intentionally ignored by git:

- `artifacts/runs/<run-id>/` for benchmark outputs.
- `artifacts/mlflow/` and `state/mlflow/` for local MLflow storage.
- `state/dspy/gepa-agent-prompt/` for compiled prompt outputs.
- `state/modal/` for deployed Modal endpoint metadata.

## Notes

- Python dependencies are pinned in `pyproject.toml` and `uv.lock`.
- Modal runtime Python packages are pinned in `src/tinyharness/modal_server.py`.
- The Modal gateway expects `TINYHARNESS_PROXY_TOKEN` in `.env` or the shell.
- Remote MLflow is optional; local MLflow works with the default state paths.
