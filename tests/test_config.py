from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinyharness.config import (
    AppConfig,
    ConfigError,
    resolve_gateway_base_url,
    resolve_proxy_token,
    resolve_tracking_uri,
)


def test_config_from_env_uses_explicit_values() -> None:
    env = {
        "TINYHARNESS_MODEL_ALIAS": "custom-model",
        "TINYHARNESS_CONTEXT_WINDOW": "8192",
        "TINYHARNESS_TASKS": "task-a,task-b",
        "TINYHARNESS_N_TASKS": "10",
        "TINYHARNESS_MLFLOW_DB_PATH": "tmp/mlflow.db",
    }

    config = AppConfig.from_env(env)

    assert config.model.model_alias == "custom-model"
    assert config.model.context_window == 8192
    assert config.benchmark.tasks == ("task-a", "task-b")
    assert config.benchmark.n_tasks == 10
    assert config.tracking.backend_store_path == Path("tmp/mlflow.db")


def test_resolve_gateway_base_url_prefers_env(tmp_path: Path) -> None:
    state_path = tmp_path / "modal.json"
    state_path.write_text(json.dumps({"web_url": "https://state.example"}), encoding="utf-8")

    assert (
        resolve_gateway_base_url(path=state_path, env={"ANTHROPIC_BASE_URL": "https://env.example"})
        == "https://env.example"
    )


def test_resolve_gateway_base_url_uses_state_file(tmp_path: Path) -> None:
    state_path = tmp_path / "modal.json"
    state_path.write_text(json.dumps({"web_url": "https://state.example"}), encoding="utf-8")

    assert resolve_gateway_base_url(path=state_path, env={}) == "https://state.example"


def test_resolve_proxy_token_raises_when_missing() -> None:
    config = AppConfig.from_env({})

    with pytest.raises(ConfigError):
        resolve_proxy_token(config.agent, env={})


def test_resolve_tracking_uri_prefers_explicit_env() -> None:
    config = AppConfig.from_env({"MLFLOW_TRACKING_URI": "https://mlflow.example"})

    assert resolve_tracking_uri(config.tracking, env={"MLFLOW_TRACKING_URI": "https://override.example"}) == "https://override.example"


def test_resolve_tracking_uri_defaults_to_local_sqlite_even_when_remote_state_exists(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "mlflow-state.json"
    state_path.write_text(json.dumps({"web_url": "https://modal-mlflow.example"}), encoding="utf-8")
    config = AppConfig.from_env({})
    from tinyharness import config as config_module

    monkeypatch.setattr(config_module, "MLFLOW_MODAL_STATE_PATH", state_path)

    expected = f"sqlite:///{config.tracking.backend_store_path.resolve()}"
    assert resolve_tracking_uri(config.tracking, env={}) == expected
    assert resolve_tracking_uri(config.tracking, env={"MLFLOW_TRACKING_URI": ""}) == expected
