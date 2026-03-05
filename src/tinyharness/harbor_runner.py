from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from harbor.environments.modal import ModalEnvironment
from harbor.job import Job
from harbor.models.job.config import JobConfig

DEFAULT_MODAL_STOP_TIMEOUT_SEC = 30.0


async def guarded_modal_stop(
    environment: ModalEnvironment,
    *,
    timeout_sec: float = DEFAULT_MODAL_STOP_TIMEOUT_SEC,
) -> None:
    if not environment._sandbox:
        return

    try:
        try:
            await asyncio.wait_for(environment._terminate_sandbox(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            environment.logger.warning(
                f"Modal sandbox terminate timeout after {timeout_sec:.1f}s"
            )

        try:
            await asyncio.wait_for(
                environment._sandbox.wait.aio(raise_on_termination=False),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            environment.logger.warning(
                f"Modal sandbox wait timeout after {timeout_sec:.1f}s"
            )
    except Exception as exc:
        environment.logger.warning(f"Error terminating Modal sandbox: {exc}")
    finally:
        environment._sandbox = None
        environment._app = None
        environment._image = None


def install_modal_stop_guard(
    *,
    timeout_sec: float = DEFAULT_MODAL_STOP_TIMEOUT_SEC,
) -> None:
    if getattr(ModalEnvironment.stop, "__tinyharness_guarded__", False):
        return

    async def _guarded_stop(self: ModalEnvironment, delete: bool) -> None:
        del delete
        await guarded_modal_stop(self, timeout_sec=timeout_sec)

    setattr(_guarded_stop, "__tinyharness_guarded__", True)
    ModalEnvironment.stop = _guarded_stop


def run_job(config_path: Path) -> int:
    install_modal_stop_guard()
    config = JobConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    asyncio.run(Job(config).run())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinyharness.harbor_runner")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    return run_job(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
