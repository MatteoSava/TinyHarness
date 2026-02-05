from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

CLAUDE_CODE_PRESET = {"type": "preset", "preset": "claude_code"}


def build_sdk_options(
    *,
    cwd: str,
    max_turns: int | None,
    max_thinking_tokens: int | None,
    model: str | None,
) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        cwd=cwd,
        model=model,
        permission_mode="bypassPermissions",
        setting_sources=[],
        system_prompt=CLAUDE_CODE_PRESET,
        tools=CLAUDE_CODE_PRESET,
        max_turns=max_turns,
        max_thinking_tokens=max_thinking_tokens,
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in vars(value).items()}
    return value


def _message_to_dict(message: Any) -> dict[str, Any]:
    payload = _jsonable(message)
    payload["message_type"] = message.__class__.__name__
    return payload


def _assistant_text(messages: Iterable[Any]) -> str:
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                chunks.append(block.text.rstrip())
    return "\n\n".join(chunks)


def _usage_tokens(result: ResultMessage | None) -> tuple[int | None, int | None, int | None]:
    if result is None or not result.usage:
        return None, None, None

    usage = result.usage
    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")
    cache = usage.get("cache_read_input_tokens")

    return (
        int(prompt) if prompt is not None else None,
        int(completion) if completion is not None else None,
        int(cache) if cache is not None else None,
    )


async def _run_sdk(
    *,
    instruction: str,
    cwd: str,
    logs_dir: Path,
    max_turns: int | None,
    max_thinking_tokens: int | None,
    model: str | None,
) -> int:
    logs_dir.mkdir(parents=True, exist_ok=True)
    options = build_sdk_options(
        cwd=cwd,
        max_turns=max_turns,
        max_thinking_tokens=max_thinking_tokens,
        model=model,
    )

    raw_messages: list[Any] = []
    result_message: ResultMessage | None = None

    client = ClaudeSDKClient(options=options)
    await client.connect()
    try:
        await client.query(instruction)
        with (logs_dir / "sdk-messages.jsonl").open("w", encoding="utf-8") as handle:
            async for message in client.receive_response():
                raw_messages.append(message)
                handle.write(json.dumps(_message_to_dict(message), ensure_ascii=False) + "\n")
                handle.flush()
                if isinstance(message, ResultMessage):
                    result_message = message
    finally:
        await client.disconnect()

    assistant_text = _assistant_text(raw_messages)
    if assistant_text:
        print(assistant_text)

    prompt_tokens, completion_tokens, cache_tokens = _usage_tokens(result_message)
    metadata = {
        "duration_ms": result_message.duration_ms if result_message else None,
        "duration_api_ms": result_message.duration_api_ms if result_message else None,
        "num_turns": result_message.num_turns if result_message else None,
        "stop_reason": result_message.stop_reason if result_message else None,
        "session_id": result_message.session_id if result_message else None,
        "result": result_message.result if result_message else None,
        "gateway_base_url": os.environ.get("ANTHROPIC_BASE_URL"),
        "model": model,
    }
    summary = {
        "n_input_tokens": prompt_tokens,
        "n_cache_tokens": cache_tokens,
        "n_output_tokens": completion_tokens,
        "cost_usd": result_message.total_cost_usd if result_message else None,
        "is_error": result_message.is_error if result_message else True,
        "metadata": metadata,
    }

    (logs_dir / "assistant.txt").write_text(assistant_text, encoding="utf-8")
    (logs_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if result_message and result_message.is_error:
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Claude Agent SDK against one instruction.")
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-thinking-tokens", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("instruction")
    args = parser.parse_args(argv)

    exit_code = asyncio.run(
        _run_sdk(
            instruction=args.instruction,
            cwd=args.cwd,
            logs_dir=Path(args.logs_dir),
            max_turns=args.max_turns,
            max_thinking_tokens=args.max_thinking_tokens,
            model=args.model,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

