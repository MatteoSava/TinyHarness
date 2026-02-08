# Architecture

## Summary
TinyHarness has four moving parts:

1. A local CLI that resolves config, deploys the Modal app, launches Harbor jobs, and logs MLflow runs.
2. A Modal app that downloads the GGUF, starts `llama-server`, and fronts it with a LiteLLM gateway that Claude Code can target through `ANTHROPIC_BASE_URL`.
3. A custom Harbor installed agent that installs Claude Code plus the Anthropic Agent SDK inside each benchmark sandbox and runs the task with the `claude_code` preset.
4. A local MLflow tracking store that records one parent run per benchmark invocation and one child run per task.

## Runtime Layout
- `src/tinyharness/config.py`: typed config objects and env resolution
- `src/tinyharness/modal_server.py`: Modal deployment spec and launch script
- `src/tinyharness/harbor_agents.py`: Harbor installed agent wrapper
- `src/tinyharness/sdk_runner.py`: in-container SDK execution logic
- `src/tinyharness/benchmark.py`: Harbor job generation and orchestration
- `src/tinyharness/mlflow_tracking.py`: MLflow experiment logging
- `src/tinyharness/results.py`: run parsing and summaries

## Persistent State
- `artifacts/runs/<job-name>/`: Harbor job output, summaries, and captured subprocess logs
- `artifacts/mlflow/`: MLflow artifact store
- `state/mlflow/mlflow.db`: SQLite backend store
- `state/modal/qwen-server.json`: last deployed gateway metadata

## Defaults
- Dataset: `terminal-bench@2.0`
- Smoke tasks: `cancel-async-tasks`, `filter-js-from-html`, `sqlite-db-truncate`
- GPU: `L4`
- Model alias: `qwen3.5-35b-a3b-ud-iq3_s`
- Context window: `65536`
- Claude Code settings source: programmatic only
