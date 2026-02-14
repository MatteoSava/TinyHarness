# AGENTS.md

## Project Overview
TinyHarness benchmarks coding-agent behavior on Terminal-Bench 2.0. The default path runs a Claude-Code-style agent through Anthropic's Python Agent SDK, routes inference to a Modal-hosted Qwen gateway backed by `llama.cpp` and LiteLLM, and logs runs plus traces to a local MLflow store.

## Quick Start
- `uv sync --group dev`
- `cp .env.example .env`
- `uv run tinyharness serve-qwen`
- `uv run tinyharness run-smoke`
- `uv run tinyharness mlflow-ui`

## Local Commands
- Install deps: `uv sync --group dev`
- Deploy the Qwen gateway on Modal: `uv run tinyharness serve-qwen`
- Run the Terminal-Bench smoke subset: `uv run tinyharness run-smoke`
- Print the summary for one saved run: `uv run tinyharness fetch-results smoke-v0-20260312-110505`
- Launch the local MLflow UI: `uv run tinyharness mlflow-ui`
- Deploy the optional remote MLflow server on Modal: `uv run tinyharness serve-mlflow`
- Open the deployed remote MLflow UI: `uv run tinyharness mlflow-ui --remote`
- Run the test suite locally: `uv run pytest -q`

### Notes
- `serve-qwen` uses your authenticated `modal` CLI session if present.
- `serve-qwen` also uses the Modal secret `huggingface-secret` if it exists.
- `run-smoke` expects `TINYHARNESS_PROXY_TOKEN` to be available in `.env` or the shell.
- Benchmark artifacts land under `artifacts/runs/<run-id>/`.
- Local MLflow uses `state/mlflow/mlflow.db` with artifacts in `artifacts/mlflow/`.
- Remote MLflow is optional and requires `TINYHARNESS_MLFLOW_BACKEND_STORE_URI`, `TINYHARNESS_MLFLOW_ADMIN_PASSWORD`, and `TINYHARNESS_MLFLOW_FLASK_SECRET_KEY`.
- Remote MLflow metadata is cached in `state/modal/mlflow-server.json` when you deploy it.
- Remote benchmark logging is opt-in. Set `MLFLOW_TRACKING_URI` to the remote server URL before `run-smoke` if you want runs to log there instead of the local store.

## Architecture
- Runtime config lives in [`ARCHITECTURE.md`](/Users/matteo/Developer/TinyHarness/ARCHITECTURE.md).
- Python package: `src/tinyharness/`
- Tests: `tests/`
- Generated run artifacts: `artifacts/runs/`
- MLflow state: `state/mlflow/` by default and `state/modal/mlflow-server.json` only when remote MLflow is deployed

## Code Conventions
- Use `uv` for Python execution and dependency management.
- Keep the CLI thin and push logic into importable modules.
- Do not add broad abstraction layers for one-off integrations.
- Prefer typed, explicit config objects over loose dict plumbing.
- Keep Modal and Harbor wiring deterministic and testable.

## Forbidden Patterns
- Do not call Anthropic-hosted models in the v0 smoke path.
- Do not silently fall back from `L4` to another GPU.
- Do not read `.claude` settings in the SDK-based agent baseline.
- Do not write benchmark outputs outside `artifacts/` and `state/`.

## Testing
- Run all tests: `uv run pytest`
- Run one file: `uv run pytest tests/test_modal_server.py`
- The tests are unit-level only; real Modal and Harbor execution still needs credentials.

## CI/CD
- No CI is configured yet.
- The local acceptance path is `uv sync`, `uv run pytest`, then manual Modal and benchmark execution.
- If Modal is already authenticated via `modal` CLI profile/config, you do not need `MODAL_TOKEN_ID` or `MODAL_TOKEN_SECRET` in `.env`.
- If the Modal workspace already has a secret named `huggingface-secret`, you do not need a local `HF_TOKEN` in `.env`.
