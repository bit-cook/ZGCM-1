"""Instruction-following reward backed by Open-Instruct's 54-checker registry."""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from rewards.completion import visible_answer


class IFEvalConfigurationError(RuntimeError):
    """Raised when the authoritative IFEvalG implementation is unavailable."""


class IFEvalRuntimeError(RuntimeError):
    """Raised when an installed checker fails instead of returning a verdict."""


def _append_import_path(env_name: str, *, prepend: bool) -> None:
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise IFEvalConfigurationError(f"missing required environment variable {env_name}")
    path = str(Path(value).resolve())
    if not Path(path).exists():
        raise IFEvalConfigurationError(f"{env_name} does not exist")
    if path in sys.path:
        return
    if prepend:
        sys.path.insert(0, path)
    else:
        sys.path.append(path)


@lru_cache(maxsize=1)
def _instruction_registry():
    nltk_data = os.environ.get("NLTK_DATA", "").strip()
    if not nltk_data:
        raise IFEvalConfigurationError("missing required environment variable NLTK_DATA")
    nltk_root = Path(nltk_data).resolve()
    if not (nltk_root / "tokenizers" / "punkt").is_dir() or not (
        nltk_root / "tokenizers" / "punkt_tab" / "english"
    ).is_dir():
        raise IFEvalConfigurationError("NLTK_DATA is missing punkt or punkt_tab/english")
    _append_import_path("ZGCM_OPEN_INSTRUCT_ROOT", prepend=True)
    _append_import_path("ZGCM_OPEN_INSTRUCT_SITE_PACKAGES", prepend=False)
    try:
        from open_instruct.IFEvalG import instructions_registry
    except Exception as exc:
        raise IFEvalConfigurationError("failed to import Open-Instruct IFEvalG registry") from exc
    return instructions_registry.INSTRUCTION_DICT


def validate_ifeval_registry(instruction_ids: set[str]) -> None:
    """Fail before launch if any dataset checker is absent."""

    missing = sorted(instruction_ids.difference(_instruction_registry()))
    if missing:
        raise IFEvalConfigurationError(f"IFEvalG registry is missing {len(missing)} instruction ids")


def ifeval_preflight(jsonl_paths: list[str]) -> dict[str, int]:
    """Full-scan dataset IDs, kwargs construction, and checker runtime deps."""

    registry = _instruction_registry()
    rows = 0
    occurrences = 0
    seen_ids: set[str] = set()
    for path_value in jsonl_paths:
        path = Path(path_value).resolve()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows += 1
                for spec in row.get("ifeval_spec") or []:
                    instruction_id = spec["instruction_id"]
                    checker_class = registry.get(instruction_id)
                    if checker_class is None:
                        raise IFEvalConfigurationError(f"unknown IFEvalG checker: {instruction_id}")
                    checker_kwargs = {key: value for key, value in (spec.get("kwargs") or {}).items() if value is not None}
                    checker = checker_class(instruction_id)
                    checker.build_description(**checker_kwargs)
                    checker.check_following("Health check response.")
                    occurrences += 1
                    seen_ids.add(instruction_id)
    if not rows or not occurrences:
        raise IFEvalConfigurationError("IFEval preflight scanned no rows or constraints")
    return {
        "registry_instruction_ids": len(registry),
        "dataset_instruction_ids": len(seen_ids),
        "rows": rows,
        "constraint_occurrences": occurrences,
    }


def ifeval_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int] | None = None,
    completion_ids: list[int] | None = None,
    **kwargs: Any,
) -> float:
    """Return the fraction of explicit instruction constraints that are satisfied."""

    del prompt_ids, completion_ids
    answer = visible_answer(completions)
    specs = kwargs.get("ifeval_spec") or []
    if answer is None or not isinstance(specs, list) or not specs:
        return 0.0

    registry = _instruction_registry()
    passed = 0
    for spec in specs:
        if not isinstance(spec, dict):
            return 0.0
        instruction_id = spec.get("instruction_id")
        checker_class = registry.get(instruction_id)
        if checker_class is None:
            return 0.0
        checker_kwargs = spec.get("kwargs") or {}
        if not isinstance(checker_kwargs, dict):
            return 0.0
        checker_kwargs = {key: value for key, value in checker_kwargs.items() if value is not None}
        try:
            checker = checker_class(instruction_id)
            checker.build_description(**checker_kwargs)
            verdict = checker.check_following(answer)
            # Some IFEvalG checkers use None as their ordinary negative verdict.
            # Checker exceptions still fail the atomic group through the handler below.
            if verdict is None:
                verdict = False
            passed += int(bool(answer.strip()) and bool(verdict))
        except Exception as exc:
            # A checker exception is infrastructure failure, not evidence that
            # the model violated the instruction. Let the workflow reject the
            # whole atomic GRPO group and surface the backend fault.
            raise IFEvalRuntimeError(f"IFEvalG checker failed: {instruction_id}") from exc
    return float(passed / len(specs))
