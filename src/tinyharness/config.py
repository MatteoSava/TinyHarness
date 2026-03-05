from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tinyharness.constants import (
    DEFAULT_BENCHMARK_MODE,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_CACHE_PROMPT,
    DEFAULT_GPU_TYPE,
    DEFAULT_GATEWAY_DEBUG,
    DEFAULT_HARBOR_DATASET,
    DEFAULT_LITELLM_PORT,
    DEFAULT_MLFLOW_ALLOWED_HOSTS,
    DEFAULT_MLFLOW_ARTIFACT_MOUNT_PATH,
    DEFAULT_MLFLOW_ARTIFACT_VOLUME_NAME,
    DEFAULT_MLFLOW_DB_PATH,
    DEFAULT_MLFLOW_EXPERIMENT,
    DEFAULT_MLFLOW_MODAL_APP_NAME,
    DEFAULT_MLFLOW_MODAL_FUNCTION_NAME,
    DEFAULT_MLFLOW_PORT,
    DEFAULT_MODAL_APP_NAME,
    DEFAULT_MODAL_FUNCTION_NAME,
    DEFAULT_MODAL_MAX_CONTAINERS,
    DEFAULT_MODAL_SCALEDOWN_WINDOW_SEC,
    DEFAULT_MODAL_VOLUME_NAME,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_REPO,
    DEFAULT_PARALLEL_REQUESTS,
    DEFAULT_RUNNER,
    DEFAULT_SEED,
    DEFAULT_SERVER_PORT,
    DEFAULT_TASKS,
    DEFAULT_TASK_SET,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_MODAL_STATE_PATH,
    MODAL_STATE_PATH,
    RUNS_DIR,
)


class ConfigError(ValueError):
    pass


class BenchmarkMode(StrEnum):
    DEBUG = "debug"
    LEAN = "lean"

    @classmethod
    def from_value(cls, value: str | None, *, default: "BenchmarkMode") -> "BenchmarkMode":
        if value is None:
            return default
        return cls(value.strip().lower())


def _env(name: str, default: str | None = None, env: dict[str, str] | None = None) -> str | None:
    source = env if env is not None else os.environ
    value = source.get(name)
    return value if value not in (None, "") else default


def _int_env(name: str, default: int, env: dict[str, str] | None = None) -> int:
    value = _env(name, env=env)
    if value is None:
        return default
    return int(value)


def _optional_int_env(name: str, env: dict[str, str] | None = None) -> int | None:
    value = _env(name, env=env)
    if value is None:
        return None
    return int(value)


def _float_env(name: str, default: float, env: dict[str, str] | None = None) -> float:
    value = _env(name, env=env)
    if value is None:
        return default
    return float(value)


def _csv_env(name: str, default: tuple[str, ...], env: dict[str, str] | None = None) -> tuple[str, ...]:
    value = _env(name, env=env)
    if value is None:
        return default
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or default


def _bool_env(name: str, default: bool, env: dict[str, str] | None = None) -> bool:
    value = _env(name, env=env)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelConfig:
    modal_app_name: str = DEFAULT_MODAL_APP_NAME
    modal_function_name: str = DEFAULT_MODAL_FUNCTION_NAME
    modal_volume_name: str = DEFAULT_MODAL_VOLUME_NAME
    modal_hf_secret_name: str | None = "huggingface-secret"
    hf_repo_id: str = DEFAULT_MODEL_REPO
    hf_filename: str = DEFAULT_MODEL_FILE
    model_alias: str = DEFAULT_MODEL_ALIAS
    gpu: str = DEFAULT_GPU_TYPE
    context_window: int = DEFAULT_CONTEXT_WINDOW
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    seed: int = DEFAULT_SEED
    cache_prompt: bool = DEFAULT_CACHE_PROMPT
    gateway_debug: bool = DEFAULT_GATEWAY_DEBUG
    parallel_requests: int = DEFAULT_PARALLEL_REQUESTS
    server_port: int = DEFAULT_SERVER_PORT
    litellm_port: int = DEFAULT_LITELLM_PORT
    llama_port: int = 8001
    max_containers: int = DEFAULT_MODAL_MAX_CONTAINERS
    scaledown_window_sec: int = DEFAULT_MODAL_SCALEDOWN_WINDOW_SEC

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ModelConfig":
        return cls(
            modal_app_name=_env("TINYHARNESS_MODAL_APP_NAME", DEFAULT_MODAL_APP_NAME, env) or DEFAULT_MODAL_APP_NAME,
            modal_function_name=_env("TINYHARNESS_MODAL_FUNCTION_NAME", DEFAULT_MODAL_FUNCTION_NAME, env)
            or DEFAULT_MODAL_FUNCTION_NAME,
            modal_volume_name=_env("TINYHARNESS_MODAL_VOLUME_NAME", DEFAULT_MODAL_VOLUME_NAME, env)
            or DEFAULT_MODAL_VOLUME_NAME,
            modal_hf_secret_name=_env("TINYHARNESS_MODAL_HF_SECRET_NAME", "huggingface-secret", env),
            hf_repo_id=_env("TINYHARNESS_MODEL_REPO", DEFAULT_MODEL_REPO, env) or DEFAULT_MODEL_REPO,
            hf_filename=_env("TINYHARNESS_MODEL_FILE", DEFAULT_MODEL_FILE, env) or DEFAULT_MODEL_FILE,
            model_alias=_env("TINYHARNESS_MODEL_ALIAS", DEFAULT_MODEL_ALIAS, env) or DEFAULT_MODEL_ALIAS,
            gpu=_env("TINYHARNESS_GPU_TYPE", DEFAULT_GPU_TYPE, env) or DEFAULT_GPU_TYPE,
            context_window=_int_env("TINYHARNESS_CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW, env),
            temperature=_float_env("TINYHARNESS_TEMPERATURE", DEFAULT_TEMPERATURE, env),
            top_p=_float_env("TINYHARNESS_TOP_P", DEFAULT_TOP_P, env),
            top_k=_int_env("TINYHARNESS_TOP_K", DEFAULT_TOP_K, env),
            seed=_int_env("TINYHARNESS_SEED", DEFAULT_SEED, env),
            cache_prompt=_bool_env("TINYHARNESS_CACHE_PROMPT", DEFAULT_CACHE_PROMPT, env),
            gateway_debug=_bool_env("TINYHARNESS_GATEWAY_DEBUG", DEFAULT_GATEWAY_DEBUG, env),
            parallel_requests=_int_env("TINYHARNESS_PARALLEL_REQUESTS", DEFAULT_PARALLEL_REQUESTS, env),
            server_port=_int_env("TINYHARNESS_SERVER_PORT", DEFAULT_SERVER_PORT, env),
            litellm_port=_int_env("TINYHARNESS_LITELLM_PORT", DEFAULT_LITELLM_PORT, env),
            llama_port=_int_env("TINYHARNESS_LLAMA_PORT", 8001, env),
            max_containers=_int_env("TINYHARNESS_MODAL_MAX_CONTAINERS", DEFAULT_MODAL_MAX_CONTAINERS, env),
            scaledown_window_sec=_int_env(
                "TINYHARNESS_MODAL_SCALEDOWN_WINDOW_SEC",
                DEFAULT_MODAL_SCALEDOWN_WINDOW_SEC,
                env,
            ),
        )


@dataclass(frozen=True)
class AgentConfig:
    import_path: str = "tinyharness.harbor_agents:QwenClaudeSDKAgent"
    workspace_cwd: str = "/app"
    max_turns: int = 20
    max_thinking_tokens: int = 8192
    settings_sources: tuple[str, ...] = ()
    proxy_token_env: str = "TINYHARNESS_PROXY_TOKEN"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "AgentConfig":
        settings_sources = _csv_env("TINYHARNESS_SETTING_SOURCES", (), env)
        return cls(
            import_path=_env("TINYHARNESS_AGENT_IMPORT_PATH", "tinyharness.harbor_agents:QwenClaudeSDKAgent", env)
            or "tinyharness.harbor_agents:QwenClaudeSDKAgent",
            workspace_cwd=_env("TINYHARNESS_WORKSPACE_CWD", "/app", env) or "/app",
            max_turns=_int_env("TINYHARNESS_MAX_TURNS", 20, env),
            max_thinking_tokens=_int_env("TINYHARNESS_MAX_THINKING_TOKENS", 8192, env),
            settings_sources=settings_sources,
            proxy_token_env=_env("TINYHARNESS_PROXY_TOKEN_ENV", "TINYHARNESS_PROXY_TOKEN", env)
            or "TINYHARNESS_PROXY_TOKEN",
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str = DEFAULT_HARBOR_DATASET
    task_set_name: str = DEFAULT_TASK_SET
    mode: BenchmarkMode = BenchmarkMode.DEBUG
    tasks: tuple[str, ...] | None = DEFAULT_TASKS
    n_tasks: int | None = None
    jobs_dir: Path = RUNS_DIR
    runner: str = DEFAULT_RUNNER
    sandbox_timeout_secs: int = 60 * 60 * 4
    sandbox_idle_timeout_secs: int = 60 * 20

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "BenchmarkConfig":
        return cls(
            dataset=_env("TINYHARNESS_DATASET", DEFAULT_HARBOR_DATASET, env) or DEFAULT_HARBOR_DATASET,
            task_set_name=_env("TINYHARNESS_TASK_SET", DEFAULT_TASK_SET, env) or DEFAULT_TASK_SET,
            mode=BenchmarkMode.from_value(
                _env("TINYHARNESS_BENCHMARK_MODE", DEFAULT_BENCHMARK_MODE, env),
                default=BenchmarkMode.DEBUG,
            ),
            tasks=_csv_env("TINYHARNESS_TASKS", DEFAULT_TASKS, env),
            n_tasks=_optional_int_env("TINYHARNESS_N_TASKS", env),
            jobs_dir=Path(_env("TINYHARNESS_JOBS_DIR", RUNS_DIR.as_posix(), env) or RUNS_DIR.as_posix()),
            runner=_env("TINYHARNESS_RUNNER", DEFAULT_RUNNER, env) or DEFAULT_RUNNER,
            sandbox_timeout_secs=_int_env("TINYHARNESS_MODAL_SANDBOX_TIMEOUT", 60 * 60 * 4, env),
            sandbox_idle_timeout_secs=_int_env("TINYHARNESS_MODAL_IDLE_TIMEOUT", 60 * 20, env),
        )


@dataclass(frozen=True)
class TrackingConfig:
    experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT
    tracking_uri: str | None = None
    backend_store_path: Path = DEFAULT_MLFLOW_DB_PATH
    artifact_root: Path = MLFLOW_ARTIFACTS_DIR
    modal_app_name: str = DEFAULT_MLFLOW_MODAL_APP_NAME
    modal_function_name: str = DEFAULT_MLFLOW_MODAL_FUNCTION_NAME
    backend_store_uri: str | None = None
    artifact_volume_name: str = DEFAULT_MLFLOW_ARTIFACT_VOLUME_NAME
    artifact_mount_path: str = DEFAULT_MLFLOW_ARTIFACT_MOUNT_PATH
    admin_username: str = "admin"
    admin_password_env: str = "TINYHARNESS_MLFLOW_ADMIN_PASSWORD"
    flask_secret_key_env: str = "TINYHARNESS_MLFLOW_FLASK_SECRET_KEY"
    server_max_containers: int = 1
    server_scaledown_window_sec: int = 60
    allowed_hosts: str = DEFAULT_MLFLOW_ALLOWED_HOSTS
    port: int = DEFAULT_MLFLOW_PORT

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "TrackingConfig":
        backend_store = Path(
            _env("TINYHARNESS_MLFLOW_DB_PATH", DEFAULT_MLFLOW_DB_PATH.as_posix(), env)
            or DEFAULT_MLFLOW_DB_PATH.as_posix()
        )
        artifact_root = Path(
            _env("TINYHARNESS_MLFLOW_ARTIFACT_ROOT", MLFLOW_ARTIFACTS_DIR.as_posix(), env)
            or MLFLOW_ARTIFACTS_DIR.as_posix()
        )
        return cls(
            experiment_name=_env("TINYHARNESS_MLFLOW_EXPERIMENT", DEFAULT_MLFLOW_EXPERIMENT, env)
            or DEFAULT_MLFLOW_EXPERIMENT,
            tracking_uri=_env("MLFLOW_TRACKING_URI", env=env),
            backend_store_path=backend_store,
            artifact_root=artifact_root,
            modal_app_name=_env("TINYHARNESS_MLFLOW_MODAL_APP_NAME", DEFAULT_MLFLOW_MODAL_APP_NAME, env)
            or DEFAULT_MLFLOW_MODAL_APP_NAME,
            modal_function_name=_env("TINYHARNESS_MLFLOW_MODAL_FUNCTION_NAME", DEFAULT_MLFLOW_MODAL_FUNCTION_NAME, env)
            or DEFAULT_MLFLOW_MODAL_FUNCTION_NAME,
            backend_store_uri=_env("TINYHARNESS_MLFLOW_BACKEND_STORE_URI", env=env),
            artifact_volume_name=_env(
                "TINYHARNESS_MLFLOW_ARTIFACT_VOLUME_NAME",
                DEFAULT_MLFLOW_ARTIFACT_VOLUME_NAME,
                env,
            )
            or DEFAULT_MLFLOW_ARTIFACT_VOLUME_NAME,
            artifact_mount_path=_env(
                "TINYHARNESS_MLFLOW_ARTIFACT_MOUNT_PATH",
                DEFAULT_MLFLOW_ARTIFACT_MOUNT_PATH,
                env,
            )
            or DEFAULT_MLFLOW_ARTIFACT_MOUNT_PATH,
            admin_username=_env("TINYHARNESS_MLFLOW_ADMIN_USERNAME", "admin", env) or "admin",
            admin_password_env=_env(
                "TINYHARNESS_MLFLOW_ADMIN_PASSWORD_ENV",
                "TINYHARNESS_MLFLOW_ADMIN_PASSWORD",
                env,
            )
            or "TINYHARNESS_MLFLOW_ADMIN_PASSWORD",
            flask_secret_key_env=_env(
                "TINYHARNESS_MLFLOW_FLASK_SECRET_KEY_ENV",
                "TINYHARNESS_MLFLOW_FLASK_SECRET_KEY",
                env,
            )
            or "TINYHARNESS_MLFLOW_FLASK_SECRET_KEY",
            server_max_containers=_int_env("TINYHARNESS_MLFLOW_MAX_CONTAINERS", 1, env),
            server_scaledown_window_sec=_int_env("TINYHARNESS_MLFLOW_SCALEDOWN_WINDOW_SEC", 60, env),
            allowed_hosts=_env("TINYHARNESS_MLFLOW_ALLOWED_HOSTS", DEFAULT_MLFLOW_ALLOWED_HOSTS, env)
            or DEFAULT_MLFLOW_ALLOWED_HOSTS,
            port=_int_env("TINYHARNESS_MLFLOW_PORT", DEFAULT_MLFLOW_PORT, env),
        )


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    agent: AgentConfig
    benchmark: BenchmarkConfig
    tracking: TrackingConfig

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "AppConfig":
        return cls(
            model=ModelConfig.from_env(env),
            agent=AgentConfig.from_env(env),
            benchmark=BenchmarkConfig.from_env(env),
            tracking=TrackingConfig.from_env(env),
        )


def ensure_state_dirs(config: AppConfig) -> None:
    config.benchmark.jobs_dir.mkdir(parents=True, exist_ok=True)
    config.tracking.backend_store_path.parent.mkdir(parents=True, exist_ok=True)
    config.tracking.artifact_root.mkdir(parents=True, exist_ok=True)
    MODAL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def require_env_vars(*names: str, env: dict[str, str] | None = None) -> dict[str, str]:
    source = env if env is not None else os.environ
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        value = source.get(name)
        if value:
            resolved[name] = value
        else:
            missing.append(name)

    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    return resolved


def load_modal_state(path: Path = MODAL_STATE_PATH) -> dict[str, object]:
    if not path.exists():
        raise ConfigError(
            f"Modal gateway metadata not found at {path}. Run `uv run tinyharness serve-qwen` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_mlflow_modal_state(path: Path = MLFLOW_MODAL_STATE_PATH) -> dict[str, object]:
    if not path.exists():
        raise ConfigError(
            f"Modal MLflow metadata not found at {path}. Run `uv run tinyharness serve-mlflow` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_gateway_base_url(path: Path = MODAL_STATE_PATH, env: dict[str, str] | None = None) -> str:
    explicit = _env("ANTHROPIC_BASE_URL", env=env) or _env("TINYHARNESS_GATEWAY_URL", env=env)
    if explicit:
        return explicit

    state = load_modal_state(path)
    web_url = state.get("web_url")
    if not isinstance(web_url, str) or not web_url:
        raise ConfigError(f"Modal state file {path} does not contain a usable web_url.")
    return web_url


def resolve_proxy_token(config: AgentConfig, env: dict[str, str] | None = None) -> str:
    source = env if env is not None else os.environ
    value = source.get(config.proxy_token_env)
    if not value:
        raise ConfigError(
            f"Missing proxy auth token env var {config.proxy_token_env}. Check .env.example."
        )
    return value


def resolve_tracking_uri(config: TrackingConfig, env: dict[str, str] | None = None) -> str:
    explicit = _env("MLFLOW_TRACKING_URI", env=env) or config.tracking_uri
    if explicit:
        return explicit
    return local_tracking_uri(config)


def local_tracking_uri(config: TrackingConfig) -> str:
    return f"sqlite:///{config.backend_store_path.resolve()}"


def resolve_remote_tracking_uri(path: Path = MLFLOW_MODAL_STATE_PATH) -> str:
    state = load_mlflow_modal_state(path)
    web_url = state.get("web_url")
    if not isinstance(web_url, str) or not web_url:
        raise ConfigError(f"Modal MLflow state file {path} does not contain a usable web_url.")
    return web_url


def resolve_mlflow_password(config: TrackingConfig, env: dict[str, str] | None = None) -> str:
    source = env if env is not None else os.environ
    value = source.get(config.admin_password_env)
    if not value:
        raise ConfigError(
            f"Missing MLflow admin password env var {config.admin_password_env}. Check .env.example."
        )
    return value


def resolve_mlflow_flask_secret(config: TrackingConfig, env: dict[str, str] | None = None) -> str:
    source = env if env is not None else os.environ
    value = source.get(config.flask_secret_key_env)
    if not value:
        raise ConfigError(
            f"Missing MLflow Flask secret env var {config.flask_secret_key_env}. Check .env.example."
        )
    return value
