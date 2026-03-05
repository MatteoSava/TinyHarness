from __future__ import annotations

from tinyharness.config import ModelConfig
from tinyharness.modal_server import build_litellm_config, build_server_spec, wait_for_gateway_ports


def test_modal_server_spec_pins_l4_and_expected_components() -> None:
    spec = build_server_spec(ModelConfig())

    assert spec.gpu == "L4"
    assert spec.mount_path == "/models"
    assert spec.model_repo == "unsloth/Qwen3.5-35B-A3B-GGUF"
    assert spec.model_filename == "Qwen3.5-35B-A3B-UD-IQ3_S.gguf"
    assert 'llama_bin="/app/llama-server"' in spec.launch_script
    assert "--ctx-size 65536" in spec.launch_script
    assert "litellm" in spec.launch_script
    assert "python -m tinyharness.gateway_debug" in spec.launch_script
    assert "--api-key \"$TINYHARNESS_PROXY_TOKEN\"" in spec.launch_script


def test_modal_server_spec_uses_deterministic_sampling_defaults() -> None:
    spec = build_server_spec(ModelConfig())

    assert "--seed 42" in spec.launch_script
    assert "--temp 0.0" in spec.launch_script
    assert "--top-k 1" in spec.launch_script
    assert "--top-p 1.0" in spec.launch_script


def test_litellm_config_pins_request_sampling_and_disables_prompt_cache() -> None:
    config = build_litellm_config(ModelConfig())

    model = config["model_list"][0]["litellm_params"]
    general = config["general_settings"]

    assert model["api_base"] == "http://127.0.0.1:8000/openai-proxy/v1"
    assert model["temperature"] == 0.0
    assert model["top_p"] == 1.0
    assert model["seed"] == 42
    assert model["extra_body"]["top_k"] == 1
    assert model["extra_body"]["cache_prompt"] is False
    assert general["forward_client_headers_to_llm_api"] is True


def test_wait_for_gateway_ports_checks_llama_litellm_and_gateway(monkeypatch) -> None:
    calls: list[tuple[int, float]] = []

    def fake_wait_for_port(port: int, timeout_sec: float = 120.0) -> None:
        calls.append((port, timeout_sec))

    monkeypatch.setattr("tinyharness.modal_server._wait_for_port", fake_wait_for_port)

    wait_for_gateway_ports(ModelConfig())

    assert calls == [
        (8001, 60.0 * 20.0),
        (8002, 60.0 * 5.0),
        (8000, 60.0 * 5.0),
    ]
