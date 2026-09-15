"""Completion parsing shared by reward handlers."""

from __future__ import annotations

import re


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def visible_answer(completion: str) -> str | None:
    """Return the answer after a naturally closed thinking section.

    The target chat template opens ``<think>`` in the generation prompt, so the
    generated token stream normally contains only the closing tag.  Treating an
    unclosed or empty answer as wrong avoids assigning a verifier score to a
    truncated chain of thought.
    """

    if not isinstance(completion, str) or "</think>" not in completion:
        return None
    answer = completion.rsplit("</think>", 1)[-1]
    answer = answer.replace("<answer>", "").replace("</answer>", "").strip()
    return answer or None


def extract_python_code(answer: str) -> str:
    """Extract the last fenced Python block, falling back to the visible text."""

    matches = _CODE_BLOCK_RE.findall(answer)
    return (matches[-1] if matches else answer).strip()
