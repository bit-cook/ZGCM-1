"""AReaL-compatible reward exports and dispatcher for the released domains."""

from __future__ import annotations

import asyncio
from typing import Any

from rewards.ifeval_reward import ifeval_reward_fn
from rewards.math_reward import math_reward_fn
from rewards.sandbox_client import code_reward_fn


def reward_router_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int] | None = None,
    completion_ids: list[int] | None = None,
    **kwargs: Any,
) -> float:
    """Dispatch one row to its domain handler using the standard RLVR signature."""

    domain = kwargs.get("domain")
    handlers = {
        "math": math_reward_fn,
        "code": code_reward_fn,
        "if": ifeval_reward_fn,
    }
    try:
        handler = handlers[str(domain)]
    except KeyError as exc:
        raise ValueError(f"unsupported reward domain: {domain!r}") from exc
    return float(handler(prompt, completions, prompt_ids, completion_ids, **kwargs))


async def async_reward_router_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int] | None = None,
    completion_ids: list[int] | None = None,
    **kwargs: Any,
) -> float:
    """Non-blocking variant for custom async rollout workflows."""

    return await asyncio.to_thread(
        reward_router_fn,
        prompt,
        completions,
        prompt_ids,
        completion_ids,
        **kwargs,
    )


__all__ = [
    "async_reward_router_fn",
    "code_reward_fn",
    "ifeval_reward_fn",
    "math_reward_fn",
    "reward_router_fn",
]
