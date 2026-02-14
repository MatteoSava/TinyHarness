from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass

import modal

from tinyharness.config import TrackingConfig


@dataclass(frozen=True)
class ModalMlflowServerSpec:
    image_tag: str
    volume_name: str
    mount_path: str
    web_port: int
    allowed_hosts: str
    app_name: str
    function_name: str
    max_containers: int
    scaledown_window_sec: int
    launch_script: str


def _secret_env(config: TrackingConfig) -> dict[str, str]:
    values: dict[str, str] = {}
    backend_store_uri = os.environ.get("TINYHARNESS_MLFLOW_BACKEND_STORE_URI")
    if backend_store_uri:
        values["TINYHARNESS_MLFLOW_BACKEND_STORE_URI"] = backend_store_uri

    admin_password = os.environ.get(config.admin_password_env)
    if admin_password:
        values["TINYHARNESS_MLFLOW_ADMIN_PASSWORD"] = admin_password

    flask_secret = os.environ.get(config.flask_secret_key_env)
    if flask_secret:
        values["TINYHARNESS_MLFLOW_FLASK_SECRET_KEY"] = flask_secret

    values["TINYHARNESS_MLFLOW_ADMIN_USERNAME"] = config.admin_username
    return values


def _build_secrets(config: TrackingConfig) -> list[modal.Secret]:
    env_values = _secret_env(config)
    if not env_values:
        return []
    return [modal.Secret.from_dict(env_values)]


def build_launch_script(config: TrackingConfig) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

mkdir -p {config.artifact_mount_path}

cat > /tmp/mlflow-basic-auth.ini <<INI
[mlflow]
default_permission = READ
database_uri = ${{TINYHARNESS_MLFLOW_BACKEND_STORE_URI}}
admin_username = ${{TINYHARNESS_MLFLOW_ADMIN_USERNAME}}
admin_password = ${{TINYHARNESS_MLFLOW_ADMIN_PASSWORD}}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
grant_default_workspace_access = false
INI

export MLFLOW_AUTH_CONFIG_PATH=/tmp/mlflow-basic-auth.ini
export MLFLOW_FLASK_SERVER_SECRET_KEY="${{TINYHARNESS_MLFLOW_FLASK_SECRET_KEY}}"

mlflow db upgrade "${{TINYHARNESS_MLFLOW_BACKEND_STORE_URI}}"
python -m mlflow.server.auth db upgrade --url "${{TINYHARNESS_MLFLOW_BACKEND_STORE_URI}}"

exec mlflow server \
  --backend-store-uri "${{TINYHARNESS_MLFLOW_BACKEND_STORE_URI}}" \
  --serve-artifacts \
  --artifacts-destination "file://{config.artifact_mount_path}" \
  --host 0.0.0.0 \
  --port {config.port} \
  --workers 1 \
  --app-name basic-auth \
  --allowed-hosts "{config.allowed_hosts}"
"""


def build_server_spec(config: TrackingConfig) -> ModalMlflowServerSpec:
    return ModalMlflowServerSpec(
        image_tag="python:3.12-slim",
        volume_name=config.artifact_volume_name,
        mount_path=config.artifact_mount_path,
        web_port=config.port,
        allowed_hosts=config.allowed_hosts,
        app_name=config.modal_app_name,
        function_name=config.modal_function_name,
        max_containers=config.server_max_containers,
        scaledown_window_sec=config.server_scaledown_window_sec,
        launch_script=build_launch_script(config),
    )


def _build_image() -> modal.Image:
    return modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "mlflow>=3.10.1",
        "psycopg[binary]>=3.2.0",
    )


_CONFIG = TrackingConfig.from_env()
_SPEC = build_server_spec(_CONFIG)
_VOLUME = modal.Volume.from_name(_SPEC.volume_name, create_if_missing=True)
_SECRETS = _build_secrets(_CONFIG)

app = modal.App(name=_CONFIG.modal_app_name)


def _wait_for_port(port: int, timeout_sec: float = 120.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(1.0)
    raise TimeoutError(f"Port {port} did not become ready within {timeout_sec} seconds.")


@app.function(
    image=_build_image(),
    volumes={_SPEC.mount_path: _VOLUME},
    secrets=_SECRETS,
    timeout=60 * 60 * 12,
    max_containers=_SPEC.max_containers,
    scaledown_window=_SPEC.scaledown_window_sec,
    cpu=2,
    name=_SPEC.function_name,
)
@modal.web_server(port=_SPEC.web_port, startup_timeout=60 * 5)
def serve_mlflow() -> None:
    subprocess.Popen(["bash", "-lc", _SPEC.launch_script])
    _wait_for_port(_SPEC.web_port, timeout_sec=60.0 * 5.0)


def resolve_web_url(config: TrackingConfig | None = None) -> str:
    resolved = config or _CONFIG
    function = modal.Function.from_name(resolved.modal_app_name, resolved.modal_function_name)
    return function.get_web_url()


def serialize_spec(spec: ModalMlflowServerSpec) -> str:
    return json.dumps(
        {
            "image_tag": spec.image_tag,
            "volume_name": spec.volume_name,
            "mount_path": spec.mount_path,
            "web_port": spec.web_port,
            "allowed_hosts": spec.allowed_hosts,
            "app_name": spec.app_name,
            "function_name": spec.function_name,
            "max_containers": spec.max_containers,
            "scaledown_window_sec": spec.scaledown_window_sec,
            "launch_script": spec.launch_script,
        },
        indent=2,
    )
