from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from harbor.environments.modal import ModalEnvironment

from tinyharness.harbor_runner import guarded_modal_stop


class _FakeSandbox:
    def __init__(self) -> None:
        self.terminate_called = False
        self.wait_called = False
        self.terminate = SimpleNamespace(aio=self._terminate)
        self.wait = SimpleNamespace(aio=self._wait)

    async def _terminate(self) -> None:
        self.terminate_called = True

    async def _wait(self, *, raise_on_termination: bool = False) -> None:
        del raise_on_termination
        self.wait_called = True
        await asyncio.sleep(60)


def test_guarded_modal_stop_times_out_wait_and_clears_state() -> None:
    sandbox = _FakeSandbox()
    warnings: list[str] = []

    environment = object.__new__(ModalEnvironment)
    environment._sandbox = sandbox
    environment._app = object()
    environment._image = object()
    environment.logger = SimpleNamespace(
        warning=lambda message: warnings.append(str(message)),
    )

    started_at = time.perf_counter()
    asyncio.run(guarded_modal_stop(environment, timeout_sec=0.01))
    elapsed = time.perf_counter() - started_at

    assert elapsed < 0.5
    assert sandbox.terminate_called is True
    assert sandbox.wait_called is True
    assert environment._sandbox is None
    assert environment._app is None
    assert environment._image is None
    assert any("timeout" in message.lower() for message in warnings)
