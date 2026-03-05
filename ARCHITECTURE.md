# Architecture

## Summary
TinyHarness has four moving parts:

1. A local CLI that resolves config, deploys the Modal apps, launches Harbor jobs, and logs MLflow runs.
2. A Modal app that downloads the GGUF, starts `llama-server`, and fronts it with a LiteLLM gateway that Claude Code can target through `ANTHROPIC_BASE_URL`.
3. A custom Harbor installed agent that installs Claude Code plus the Anthropic Agent SDK inside each benchmark sandbox and runs the task with a DSPy/GEPA-backed system prompt plus a dedicated Claude Code tool allowlist.
4. A local MLflow tracking store with one parent run per benchmark invocation and one live child run plus trace per task. A remote Modal-hosted MLflow server remains optional.

## Runtime Layout
- `src/tinyharness/config.py`: typed config objects and env resolution
- `src/tinyharness/modal_server.py`: Modal deployment spec and launch script
- `src/tinyharness/harbor_agents.py`: Harbor installed agent wrapper
- `src/tinyharness/sdk_runner.py`: in-container SDK execution logic
- `src/tinyharness/dspy_prompt.py`: DSPy prompt program, GEPA compilation helper, and agent prompt config
- `src/tinyharness/benchmark.py`: Harbor job generation and orchestration
- `src/tinyharness/mlflow_tracking.py`: MLflow experiment logging
- `src/tinyharness/results.py`: run parsing and summaries
- `src/tinyharness/mlflow_server.py`: optional Modal MLflow server spec and bootstrap launcher

## Persistent State
- `artifacts/runs/<job-name>/`: Harbor job output, summaries, and captured subprocess logs
- `state/modal/qwen-server.json`: last deployed gateway metadata
- `state/mlflow/mlflow.db`: local SQLite backend store
- `state/modal/mlflow-server.json`: optional deployed MLflow server metadata

## Tracking Modes
- Default: local MLflow via `sqlite:///state/mlflow/mlflow.db`
- Optional remote mode: deploy `serve-mlflow`, then set `MLFLOW_TRACKING_URI` to the remote server URL before running benchmarks

## Defaults
- Dataset: `terminal-bench@2.0`
- Smoke tasks: `cancel-async-tasks`, `filter-js-from-html`, `sqlite-db-truncate`
- GPU: `L4`
- Model alias: `qwen3.5-35b-a3b-ud-iq3_s`
- Context window: `65536`
- Agent prompt: DSPy/GEPA seed prompt by default, optionally replaced by `TINYHARNESS_DSPY_COMPILED_PROMPT_PATH`
- Claude Code settings source: programmatic only
- MLflow default backend: local SQLite
