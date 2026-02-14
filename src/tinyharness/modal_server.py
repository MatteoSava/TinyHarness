from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass

import modal

from tinyharness.config import ModelConfig
from tinyharness.constants import DEFAULT_MODAL_FUNCTION_NAME


@dataclass(frozen=True)
class ModalServerSpec:
    image_tag: str
    volume_name: str
    mount_path: str
    gpu: str
    web_port: int
    llama_port: int
    model_repo: str
    model_filename: str
    model_alias: str
    launch_script: str


def _secret_env() -> dict[str, str]:
    values: dict[str, str] = {}
    value = os.environ.get("TINYHARNESS_PROXY_TOKEN")
    if value:
        values["TINYHARNESS_PROXY_TOKEN"] = value
    return values


def _build_secrets(config: ModelConfig) -> list[modal.Secret]:
    secrets: list[modal.Secret] = []
    if config.modal_hf_secret_name:
        secrets.append(modal.Secret.from_name(config.modal_hf_secret_name))

    env_values = _secret_env()
    if env_values:
        secrets.append(modal.Secret.from_dict(env_values))

    return secrets


def build_litellm_config(config: ModelConfig) -> dict[str, object]:
    return {
        "model_list": [
            {
                "model_name": config.model_alias,
                "litellm_params": {
                    "model": f"openai/{config.model_alias}",
                    "api_base": f"http://127.0.0.1:{config.llama_port}/v1",
                    "api_key": os.environ.get("TINYHARNESS_PROXY_TOKEN", "tinyharness"),
                },
            }
        ],
        "general_settings": {
            "master_key": os.environ.get("TINYHARNESS_PROXY_TOKEN", "tinyharness"),
            "disable_spend_logs": True,
        },
    }


def build_launch_script(config: ModelConfig) -> str:
    litellm_config = json.dumps(build_litellm_config(config))
    return f"""#!/usr/bin/env bash
set -euo pipefail

llama_bin="/app/llama-server"
if [ ! -x "$llama_bin" ]; then
  echo "llama-server binary not found at $llama_bin" >&2
  exit 1
fi

MODEL_PATH="/models/{config.hf_filename}"
if [ ! -f "$MODEL_PATH" ]; then
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
token = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)
if not token:
    raise RuntimeError(
        "No Hugging Face token found. Set HF_TOKEN locally or provide it via the Modal secret."
    )
hf_hub_download(
    repo_id="{config.hf_repo_id}",
    filename="{config.hf_filename}",
    token=token,
    local_dir="/models",
    local_dir_use_symlinks=False,
)
PY
fi

cat > /tmp/litellm-config.json <<'JSON'
{litellm_config}
JSON

"$llama_bin" \
  --host 127.0.0.1 \
  --port {config.llama_port} \
  --model "$MODEL_PATH" \
  --ctx-size {config.context_window} \
  --parallel {config.parallel_requests} \
  --alias {config.model_alias} \
  --api-key "$TINYHARNESS_PROXY_TOKEN" \
  >/tmp/llama-server.log 2>&1 &

litellm --config /tmp/litellm-config.json --host 0.0.0.0 --port {config.server_port} \
  >/tmp/litellm.log 2>&1 &
"""


def build_server_spec(config: ModelConfig) -> ModalServerSpec:
    return ModalServerSpec(
        image_tag="ghcr.io/ggml-org/llama.cpp:server-cuda",
        volume_name=config.modal_volume_name,
        mount_path="/models",
        gpu=config.gpu,
        web_port=config.server_port,
        llama_port=config.llama_port,
        model_repo=config.hf_repo_id,
        model_filename=config.hf_filename,
        model_alias=config.model_alias,
        launch_script=build_launch_script(config),
    )


def _build_image() -> modal.Image:
    return (
        modal.Image.from_registry(
            "ghcr.io/ggml-org/llama.cpp:server-cuda",
            add_python="3.12",
            setup_dockerfile_commands=["ENTRYPOINT []"],
        )
        .uv_pip_install("huggingface_hub>=0.31.4", "litellm[proxy]>=1.76.0")
    )


_CONFIG = ModelConfig.from_env()
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
    gpu=_CONFIG.gpu,
    volumes={_SPEC.mount_path: _VOLUME},
    secrets=_SECRETS,
    timeout=60 * 60 * 12,
    max_containers=_CONFIG.max_containers,
    scaledown_window=_CONFIG.scaledown_window_sec,
    name=DEFAULT_MODAL_FUNCTION_NAME,
)
@modal.concurrent(max_inputs=1, target_inputs=1)
@modal.web_server(port=_SPEC.web_port, startup_timeout=60 * 20)
def serve_openai_gateway() -> None:
    subprocess.Popen(["bash", "-lc", _SPEC.launch_script])
    _wait_for_port(_SPEC.llama_port, timeout_sec=60.0 * 20.0)
    _wait_for_port(_SPEC.web_port, timeout_sec=60.0 * 5.0)


def resolve_web_url(config: ModelConfig | None = None) -> str:
    resolved = config or _CONFIG
    function = modal.Function.from_name(
        resolved.modal_app_name,
        DEFAULT_MODAL_FUNCTION_NAME,
    )
    return function.get_web_url()
