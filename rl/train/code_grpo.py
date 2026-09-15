#!/usr/bin/env python3
"""ZGCM-1 code GRPO with sandbox-verified rewards and audited evaluation.

This is the training entry used for the code GRPO stage of the ZGCM-1
mixed-RL experiments. It implements:

- a sandbox-verified fraction-of-tests reward (see ``rewards/code_reward.py``
  and ``rewards/sandbox_client.py``);
- retry of transient sandbox infrastructure failures with bounded attempts;
- zero reward for responses truncated by the length cap;
- group filtering by reward variance (dynamic sampling);
- baseline and final evaluations plus a post-training audit of the
  periodically dumped evaluation rollouts.

Run it on top of AReaL v2.0.0 with ``PYTHONPATH`` pointing at the ``rl/``
directory of this repository.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from areal import PPOTrainer, workflow_context
from areal.api import ModelResponse
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.utils import stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.perf_tracer import trace_session
from areal.workflow.rlvr import RLVRWorkflow, default_get_input_ids_fn
from rewards.sandbox_client import SandboxExecutionError


REQUIRED_FIELDS = {
    "sample_id",
    "domain",
    "task_type",
    "messages",
    "answers",
    "code_asset_hash",
}
GROUP_SIZE = 8
TRAIN_BATCH = 384
TOTAL_STEPS = 153
MAX_PROMPT_TOKENS = 4096
MAX_NEW_TOKENS = 65536
MAX_TOTAL_TOKENS = 98304
UPDATE_MAX_SEQLEN = 69632
EVAL_FREQ_STEPS = 51
REWARD_TIMEOUT_SECONDS = 120.0
SANDBOX_INFRA_MAX_ATTEMPTS = 3
SANDBOX_INFRA_RETRY_DELAYS = (1.0, 3.0)

STATUS_ROOT = Path(os.environ.get("ZGCM_RL_STATUS_ROOT", "./status"))
EXPECTED_TRAIN_ROWS = int(os.environ["ZGCM_EXPECTED_TRAIN_ROWS"]) if os.environ.get("ZGCM_EXPECTED_TRAIN_ROWS") else None
EXPECTED_VALID_ROWS = int(os.environ["ZGCM_EXPECTED_VALID_ROWS"]) if os.environ.get("ZGCM_EXPECTED_VALID_ROWS") else None


def is_retryable_sandbox_failure(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, SandboxExecutionError) and str(exc) in {
        "sandbox HTTP status 503",
        "sandbox request failed",
    }


def normalize_stop_reason(resp: ModelResponse) -> str:
    reason = str(getattr(resp, "stop_reason", "unknown"))
    if reason not in {"stop", "length", "abort"}:
        raise RuntimeError(f"unsupported generation stop reason: {reason!r}")
    return reason


class StrictAsyncCodeWorkflow(RLVRWorkflow):
    """Propagate service failures and score length-capped responses as zero."""

    def __init__(
        self,
        reward_fn: str,
        async_reward_fn: str,
        reward_timeout_seconds: float,
        eval_all_tests_pass1: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(reward_fn=reward_fn, **kwargs)
        self.reward_fn = import_from_string(reward_fn)
        self._async_reward_fn = import_from_string(async_reward_fn)
        self._reward_timeout_seconds = float(reward_timeout_seconds)
        self._eval_all_tests_pass1 = bool(eval_all_tests_pass1)
        if self._reward_timeout_seconds <= 0:
            raise ValueError("reward_timeout_seconds must be positive")
        self.async_reward_fn = self._call_reward

    async def _call_reward(self, *args: Any, **kwargs: Any) -> float:
        kwargs["code_eval_all_tests_pass1"] = self._eval_all_tests_pass1
        result: Any = None
        for attempt in range(1, SANDBOX_INFRA_MAX_ATTEMPTS + 1):
            try:
                result = await asyncio.wait_for(
                    self._async_reward_fn(*args, **kwargs),
                    timeout=self._reward_timeout_seconds,
                )
                break
            except (TimeoutError, SandboxExecutionError) as exc:
                if (
                    not is_retryable_sandbox_failure(exc)
                    or attempt == SANDBOX_INFRA_MAX_ATTEMPTS
                ):
                    raise
                delay = SANDBOX_INFRA_RETRY_DELAYS[attempt - 1]
                print(
                    "ZGCM_SANDBOX_INFRA_RETRY "
                    + json.dumps(
                        {
                            "attempt": attempt,
                            "delay_seconds": delay,
                            "error": str(exc),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                await asyncio.sleep(delay)
        reward = float(result)
        if not math.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward!r}")
        return reward

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> float:
        output_len = len(resp.output_tokens)
        if output_len > MAX_NEW_TOKENS:
            raise RuntimeError(
                f"vLLM returned {output_len} output tokens above {MAX_NEW_TOKENS}"
            )
        stop_reason = normalize_stop_reason(resp)
        if stop_reason == "abort":
            raise RuntimeError("vLLM aborted generation; refusing to score infra failure")

        completion = self.tokenizer.decode(resp.output_tokens)
        capped = stop_reason == "length"
        if capped:
            reward = 0.0
        else:
            reward = await self.async_reward_fn(
                prompt_str,
                completion,
                resp.input_tokens,
                resp.output_tokens,
                **task_data,
                finish_reason=stop_reason,
                stop_reason=stop_reason,
                response_was_truncated=False,
            )

        try:
            stats_tracker.get(workflow_context.stat_scope()).scalar(
                response_truncated=float(capped),
                response_normal_stop=float(stop_reason == "stop"),
                response_output_tokens=float(output_len),
                length_cap_forced_zero=float(capped),
                reward_after_cap_override=float(reward),
            )
        except Exception:
            pass
        return reward


def _validate_messages(messages: Any, sample_id: str) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{sample_id}: messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"{sample_id}: every message must be an object")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"{sample_id}: invalid role {message.get('role')!r}")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"{sample_id}: message content must be a string")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def load_code_jsonl(
    path: str,
    tokenizer: Any,
    *,
    max_prompt_tokens: int,
    expected_rows: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            sample_id = str(row["sample_id"])
            if sample_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate {sample_id}")
            seen_ids.add(sample_id)
            if row["domain"] != "code":
                raise ValueError(f"{path}:{line_number}: expected code domain")
            if row["task_type"] not in {"code", "code_stdio"}:
                raise ValueError(
                    f"{path}:{line_number}: invalid code task type "
                    f"{row['task_type']!r}"
                )
            if not isinstance(row["answers"], list):
                raise ValueError(f"{path}:{line_number}: answers must be a list")
            if not _is_sha256(row["code_asset_hash"]):
                raise ValueError(f"{path}:{line_number}: invalid code_asset_hash")
            _validate_messages(row["messages"], sample_id)
            prompt_tokens = len(
                default_get_input_ids_fn(
                    row["messages"], tokenizer, enable_thinking=True
                )
            )
            if prompt_tokens > max_prompt_tokens:
                raise ValueError(
                    f"{path}:{line_number}: prompt has {prompt_tokens} tokens, "
                    f"above {max_prompt_tokens}"
                )
            row["rendered_prompt_tokens"] = prompt_tokens
            rows.append(row)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {len(rows)}")
    return rows


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def accept_code_group(trajectory: dict[str, Any]) -> bool:
    """Keep complete, finite, mixed-reward groups within the update cap."""

    attention_mask = trajectory.get("attention_mask")
    rewards = trajectory.get("rewards")
    reason = "accepted"
    max_seen = 0
    if attention_mask is None or rewards is None:
        reason = "missing_required_tensor"
    else:
        lengths = attention_mask.sum(-1).reshape(-1)
        reward_values = rewards.float().reshape(-1)
        max_seen = _as_int(lengths.max())
        if reward_values.numel() != GROUP_SIZE:
            reason = "bad_group_size"
        elif not bool(torch.isfinite(reward_values).all().item()):
            reason = "nonfinite_reward"
        elif max_seen > UPDATE_MAX_SEQLEN:
            reason = "over_update_cap"
        elif float(reward_values.std(unbiased=False).item()) < 1e-6:
            reason = "zero_variance_group"

    accepted = reason == "accepted"
    try:
        stats_tracker.get("rollout").scalar(
            update_group_accepted=float(accepted),
            update_group_rejected=float(not accepted),
            update_group_max_seqlen=float(max_seen),
            update_group_over_cap=float(reason == "over_update_cap"),
            update_group_zero_variance=float(reason == "zero_variance_group"),
        )
    except Exception:
        pass
    if not accepted:
        print(
            "ZGCM_GROUP_REJECT "
            + json.dumps({"reason": reason, "max_seen": max_seen}, sort_keys=True),
            flush=True,
        )
    return accepted


def assert_contract(config: GRPOConfig) -> None:
    """Fail fast when the algorithmic settings deviate from the recipe."""

    checks = {
        "total_train_epochs": (config.total_train_epochs, 3),
        "total_train_steps": (config.total_train_steps, TOTAL_STEPS),
        "train_batch": (config.train_dataset.batch_size, TRAIN_BATCH),
        "group_size": (config.gconfig.n_samples, GROUP_SIZE),
        "max_new_tokens": (config.gconfig.max_new_tokens, MAX_NEW_TOKENS),
        "max_tokens": (config.gconfig.max_tokens, MAX_TOTAL_TOKENS),
        "max_tokens_per_mb": (
            config.actor.mb_spec.max_tokens_per_mb,
            UPDATE_MAX_SEQLEN,
        ),
        "mask_no_eos_with_zero": (config.actor.mask_no_eos_with_zero, False),
        "ppo_n_minibatches": (config.actor.ppo_n_minibatches, 12),
        "lr": (config.actor.optimizer.lr, 1.0e-6),
        "kl_ctl": (config.actor.kl_ctl, 0.0),
        "vllm_max_model_len": (config.vllm.max_model_len, MAX_TOTAL_TOKENS),
        "eval_freq_steps": (config.evaluator.freq_steps, EVAL_FREQ_STEPS),
        "save_freq_steps": (config.saver.freq_steps, EVAL_FREQ_STEPS),
        "rollout_temperature": (config.gconfig.temperature, 1.0),
        "eval_greedy": (config.eval_gconfig.greedy, False),
        "eval_temperature": (config.eval_gconfig.temperature, 1.0),
        "eval_samples": (config.eval_gconfig.n_samples, 1),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if config.critic is not None or config.ref is not None or config.teacher is not None:
        mismatches["GRPO_roles"] = {
            "actual": "critic/ref/teacher configured",
            "expected": "all null",
        }
    if mismatches:
        raise ValueError("invalid training contract: " + json.dumps(mismatches))


def _scalar_stats(stats: dict[str, Any]) -> dict[str, float]:
    result = {}
    for key, value in stats.items():
        if key.endswith("__count"):
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[key] = number
    return result


def audit_eval_dump(
    fileroot: Path,
    *,
    experiment_name: str,
    trial_name: str,
    expected_version: int,
    expected_rows: int,
) -> dict[str, Any]:
    pattern = (
        f"*/{experiment_name}/{trial_name}/eval-rollout/{expected_version}"
    )
    matches = sorted((fileroot / "logs").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one eval dump directory for version {expected_version}, "
            f"found {len(matches)}"
        )

    records: list[dict[str, Any]] = []
    for path in sorted(matches[0].glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"{path}:{line_number}: expected object")
                records.append(value)

    task_ids = [int(record["task_id"]) for record in records]
    unique_task_ids = set(task_ids)
    expected_task_ids = set(range(expected_rows))
    if len(records) != expected_rows or unique_task_ids != expected_task_ids:
        missing = sorted(expected_task_ids - unique_task_ids)
        duplicates = sorted(
            task_id for task_id in unique_task_ids if task_ids.count(task_id) > 1
        )
        raise RuntimeError(
            "incomplete code evaluation dump: "
            + json.dumps(
                {
                    "rows": len(records),
                    "unique_task_ids": len(unique_task_ids),
                    "missing_task_ids": missing,
                    "duplicate_task_ids": duplicates,
                },
                sort_keys=True,
            )
        )

    correct = 0
    for record in records:
        task_id = int(record["task_id"])
        if int(record.get("sample_idx", -1)) != 0:
            raise RuntimeError(f"task {task_id}: expected sample_idx=0")
        if int(record.get("head_version", -1)) != expected_version:
            raise RuntimeError(f"task {task_id}: wrong head_version")
        if int(record.get("tail_version", -1)) != expected_version:
            raise RuntimeError(f"task {task_id}: wrong tail_version")
        reward = float(record["reward"])
        if reward not in {0.0, 1.0}:
            raise RuntimeError(
                f"task {task_id}: eval reward must be all-tests binary"
            )
        correct += int(reward)
    return {
        "rows": len(records),
        "unique_task_ids": len(unique_task_ids),
        "correct": correct,
        "score": correct / expected_rows,
        "dump_dir": str(matches[0]),
    }


def write_eval_marker(
    marker_path: Path,
    *,
    phase: str,
    version: int,
    audit: dict[str, Any],
    stats: dict[str, float] | None = None,
) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(
            {
                "schema": "zgcm-code-eval-v1",
                "phase": phase,
                "model_version": version,
                "rows": audit["rows"],
                "unique_task_ids": audit["unique_task_ids"],
                "correct": audit["correct"],
                "score": audit["score"],
                "dump_dir": audit["dump_dir"],
                "valid": True,
                "metric": "all_tests_pass_at_1",
                "max_tests_per_problem": 64,
                "temperature": 1.0,
                "stats": stats or {},
                "completed_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_explicit_eval(
    trainer: Any,
    *,
    config: GRPOConfig,
    workflow: str,
    workflow_kwargs: dict[str, Any],
    phase: str,
    marker_path: Path,
    expected_version: int,
    expected_rows: int,
) -> None:
    if trainer.eval_rollout is None or trainer.valid_dataloader is None:
        raise RuntimeError("explicit evaluation requires valid_dataset")
    trainer._evaluate_fn(workflow, workflow_kwargs)
    stats = _scalar_stats(trainer.eval_rollout.export_stats())
    version = int(trainer.eval_rollout.get_version())
    if version != expected_version:
        raise RuntimeError(f"{phase} model version {version} != {expected_version}")
    if not stats:
        raise RuntimeError(f"{phase} evaluation produced no finite statistics")
    audit = audit_eval_dump(
        Path(config.rollout.fileroot),
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        expected_version=expected_version,
        expected_rows=expected_rows,
    )
    write_eval_marker(
        marker_path,
        phase=phase,
        version=version,
        audit=audit,
        stats=stats,
    )
    print(
        f"ZGCM_CODE_EVAL phase={phase} version={version} "
        f"marker={marker_path}",
        flush=True,
    )


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GRPOConfig)
    assert_contract(config)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = load_code_jsonl(
        config.train_dataset.path,
        tokenizer,
        max_prompt_tokens=config.train_dataset.max_length or MAX_PROMPT_TOKENS,
        expected_rows=EXPECTED_TRAIN_ROWS,
    )
    valid_dataset = load_code_jsonl(
        config.valid_dataset.path,
        tokenizer,
        max_prompt_tokens=config.valid_dataset.max_length or MAX_PROMPT_TOKENS,
        expected_rows=EXPECTED_VALID_ROWS,
    )
    valid_rows = len(valid_dataset)
    workflow = "train.code_grpo.StrictAsyncCodeWorkflow"
    workflow_kwargs = {
        "reward_fn": "rewards.code_reward_fn",
        "async_reward_fn": "rewards.async_reward_router_fn",
        "reward_timeout_seconds": REWARD_TIMEOUT_SECONDS,
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "enable_thinking": True,
    }
    eval_workflow_kwargs = dict(workflow_kwargs)
    eval_workflow_kwargs["gconfig"] = config.eval_gconfig
    eval_workflow_kwargs["eval_all_tests_pass1"] = True

    print(
        "ZGCM_CODE_GRPO_CONTRACT "
        + json.dumps(
            {
                "algorithm": "GRPO",
                "train_rows": len(train_dataset),
                "valid_rows": valid_rows,
                "eval_metric": "all_tests_pass_at_1",
                "eval_max_tests_per_problem": 64,
                "prompt_batch": TRAIN_BATCH,
                "group_size": GROUP_SIZE,
                "trajectories_per_update": TRAIN_BATCH * GROUP_SIZE,
                "steps": TOTAL_STEPS,
                "rollout_max_new_tokens": MAX_NEW_TOKENS,
                "learner_max_total_tokens": UPDATE_MAX_SEQLEN,
                "truncated_reward": 0.0,
                "mask_no_eos_with_zero": False,
                "sandbox_infra_max_attempts": SANDBOX_INFRA_MAX_ATTEMPTS,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    baseline_marker = STATUS_ROOT / "code_eval_baseline.json"
    final_marker = STATUS_ROOT / "code_eval_final.json"
    with PPOTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        if trainer.recover_info is None:
            run_explicit_eval(
                trainer,
                config=config,
                workflow=workflow,
                workflow_kwargs=eval_workflow_kwargs,
                phase="baseline",
                marker_path=baseline_marker,
                expected_version=0,
                expected_rows=valid_rows,
            )
        elif not baseline_marker.is_file():
            raise RuntimeError("recovering run is missing the baseline eval marker")

        trainer.train(
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn="train.code_grpo.accept_code_group",
        )
        for version in range(EVAL_FREQ_STEPS, TOTAL_STEPS, EVAL_FREQ_STEPS):
            audit = audit_eval_dump(
                Path(config.rollout.fileroot),
                experiment_name=config.experiment_name,
                trial_name=config.trial_name,
                expected_version=version,
                expected_rows=valid_rows,
            )
            write_eval_marker(
                STATUS_ROOT / f"code_eval_step{version}.json",
                phase=f"step{version}",
                version=version,
                audit=audit,
            )
        run_explicit_eval(
            trainer,
            config=config,
            workflow=workflow,
            workflow_kwargs=eval_workflow_kwargs,
            phase="final",
            marker_path=final_marker,
            expected_version=TOTAL_STEPS,
            expected_rows=valid_rows,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
