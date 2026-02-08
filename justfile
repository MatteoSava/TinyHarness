set dotenv-load := true

default:
    @just --list

sync:
    uv sync --group dev

test:
    uv run pytest -q

serve:
    uv run tinyharness serve-qwen

serve-dev:
    uv run tinyharness serve-qwen --dev

smoke:
    uv run tinyharness run-smoke

fetch run_id:
    uv run tinyharness fetch-results {{run_id}}

mlflow:
    uv run tinyharness mlflow-ui

status:
    uv run tinyharness --help

