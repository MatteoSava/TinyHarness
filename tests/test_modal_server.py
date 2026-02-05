from __future__ import annotations

from tinyharness.config import ModelConfig
from tinyharness.modal_server import build_server_spec


def test_modal_server_spec_pins_l4_and_expected_components() -> None:
    spec = build_server_spec(ModelConfig())

    assert spec.gpu == "L4"
    assert spec.mount_path == "/models"
    assert spec.model_repo == "unsloth/Qwen3.5-35B-A3B-GGUF"
    assert spec.model_filename == "Qwen3.5-35B-A3B-UD-IQ3_S.gguf"
    assert 'llama_bin="/app/llama-server"' in spec.launch_script
    assert "--ctx-size 65536" in spec.launch_script
    assert "litellm" in spec.launch_script
    assert "--api-key \"$TINYHARNESS_PROXY_TOKEN\"" in spec.launch_script
