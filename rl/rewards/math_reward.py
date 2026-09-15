"""Math label normalization and verifier-backed reward."""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from rewards.completion import visible_answer


_ASSIGNMENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*\s*=\s*(.+)$", re.DOTALL)
_SIMPLE_FRAC_RE = re.compile(r"\\frac\s*([A-Za-z0-9])\s*([A-Za-z0-9])")


def canonical_math_label(label: str) -> str:
    """Canonicalize harmless answer-format variants before equivalence tests."""

    text = str(label).strip()
    assignment = _ASSIGNMENT_RE.fullmatch(text)
    if assignment:
        text = assignment.group(1)
    text = _SIMPLE_FRAC_RE.sub(r"\\frac{\1}{\2}", text)
    return re.sub(r"\s+", "", text)


@lru_cache(maxsize=1)
def _math_worker():
    from areal.reward import MathVerifyWorker

    return MathVerifyWorker(try_extract_without_anchor=True, precision=6, timeout=8.0)


def labels_equivalent(left: str, right: str) -> bool:
    """Compare two gold labels without treating units/spacing as conflicts."""

    if canonical_math_label(left) == canonical_math_label(right):
        return True
    worker = _math_worker()
    return bool(worker.verify(str(left), str(right)) or worker.verify(str(right), str(left)))


def collapse_equivalent_labels(labels: Iterable[str]) -> tuple[list[str], bool]:
    """Return unique variants and whether they form one equivalence class."""

    variants = sorted({str(label).strip() for label in labels})
    if len(variants) <= 1:
        return variants, True
    representative = variants[0]
    return variants, all(labels_equivalent(representative, value) for value in variants[1:])


def math_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int] | None = None,
    completion_ids: list[int] | None = None,
    **kwargs: Any,
) -> float:
    """AReaL-compatible binary mathematical equivalence reward."""

    del prompt, prompt_ids, completion_ids
    answer = visible_answer(completions)
    labels = kwargs.get("answers") or []
    if answer is None or not isinstance(labels, list) or not labels:
        return 0.0
    worker = _math_worker()
    for label in labels:
        if worker.verify(answer, str(label)):
            return 1.0
    return 0.0
