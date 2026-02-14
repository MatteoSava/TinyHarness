from __future__ import annotations

from tinyharness.config import TrackingConfig
from tinyharness.mlflow_server import build_server_spec


def test_mlflow_server_spec_uses_volume_and_basic_auth() -> None:
    spec = build_server_spec(TrackingConfig(backend_store_uri="postgresql://example"))

    assert spec.volume_name == "tinyharness-mlflow-artifacts"
    assert spec.mount_path == "/mlartifacts"
    assert spec.max_containers == 1
    assert "mlflow db upgrade" in spec.launch_script
    assert "python -m mlflow.server.auth db upgrade" in spec.launch_script
    assert "--app-name basic-auth" in spec.launch_script
    assert '--artifacts-destination "file:///mlartifacts"' in spec.launch_script
