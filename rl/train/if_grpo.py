#!/usr/bin/env python3
"""ZGCM-1 instruction-following GRPO with exact-version evals.

This is the training entry used for the instruction-following GRPO stage
of the ZGCM-1 mixed-RL experiments. It implements:

- a fraction-of-satisfied-constraints reward backed by Open-Instruct's
  IFEvalG checker registry (see ``rewards/ifeval_reward.py``);
- a registry preflight over the prompt files before training starts;
- zero reward for responses truncated by the length cap;
- group filtering by fractional-reward variance (dynamic sampling);
- deterministic version-anchored evaluation with on-disk markers.

Run it on top of AReaL v2.0.0 with ``PYTHONPATH`` pointing at the ``rl/``
directory of this repository.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from areal import PPOTrainer, workflow_context
from areal.api import ModelResponse
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.infra import current_platform
from areal.utils import stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.environ import is_single_controller
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.perf_tracer import trace_session
from areal.workflow.rlvr import RLVRWorkflow, default_get_input_ids_fn
from rewards.ifeval_reward import ifeval_preflight


REQUIRED_FIELDS = {
    "sample_id",
    "domain",
    "task_type",
    "messages",
    "answers",
    "ifeval_spec",
}
GROUP_SIZE = 8
TRAIN_BATCH = 384
TOTAL_STEPS = 219
MAX_PROMPT_TOKENS = 4096
MAX_NEW_TOKENS = 65536
MAX_TOTAL_TOKENS = 98304
UPDATE_MAX_SEQLEN = 69632
ROLLOUT_REQUEST_TIMEOUT = 14400
EVAL_SAMPLES = 8
EVAL_FREQ_STEPS = 10
EVAL_VERSIONS = frozenset({0, *range(EVAL_FREQ_STEPS, TOTAL_STEPS, EVAL_FREQ_STEPS), TOTAL_STEPS})
REWARD_TIMEOUT_SECONDS = 120.0

STATUS_ROOT = Path(os.environ.get("ZGCM_RL_STATUS_ROOT", "./status"))
EXPECTED_TRAIN_ROWS = int(os.environ["ZGCM_EXPECTED_TRAIN_ROWS"]) if os.environ.get("ZGCM_EXPECTED_TRAIN_ROWS") else None
EXPECTED_VALID_ROWS = int(os.environ["ZGCM_EXPECTED_VALID_ROWS"]) if os.environ.get("ZGCM_EXPECTED_VALID_ROWS") else None


def normalize_stop_reason(resp: ModelResponse) -> str:
    reason = str(getattr(resp, "stop_reason", "unknown"))
    if reason not in {"stop", "length", "abort"}:
        raise RuntimeError(f"unsupported generation stop reason: {reason!r}")
    return reason


class IFEvalWorkflow(RLVRWorkflow):
    """Score the fraction of explicit IF constraints satisfied by the answer."""

    def __init__(
        self,
        reward_fn: str,
        async_reward_fn: str,
        reward_timeout_seconds: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(reward_fn=reward_fn, **kwargs)
        self.reward_fn = import_from_string(reward_fn)
        self._async_reward_fn = import_from_string(async_reward_fn)
        self._reward_timeout_seconds = float(reward_timeout_seconds)
        if self._reward_timeout_seconds <= 0:
            raise ValueError("reward_timeout_seconds must be positive")
        self.async_reward_fn = self._call_reward

    async def _call_reward(self, *args: Any, **kwargs: Any) -> float:
        result = await asyncio.wait_for(
            self._async_reward_fn(*args, **kwargs),
            timeout=self._reward_timeout_seconds,
        )
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
        reward = float(reward)
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError(f"IFEval reward must be finite in [0,1], got {reward!r}")

        try:
            metrics = {
                "response_truncated": float(capped),
                "response_normal_stop": float(stop_reason == "stop"),
                "response_output_tokens": float(output_len),
                "length_cap_forced_zero": float(capped),
                "ifeval_reward": float(reward),
            }
            scope = workflow_context.stat_scope()
            stats_tracker.get(scope).scalar(**metrics)
            benchmark = task_data.get("benchmark")
            if benchmark:
                stats_tracker.get(f"{scope}/{benchmark}").scalar(**metrics)
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


def load_if_jsonl(
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
            if row["domain"] != "if":
                raise ValueError(f"{path}:{line_number}: expected if domain")
            if not isinstance(row["answers"], list):
                raise ValueError(f"{path}:{line_number}: answers must be a list")
            if not isinstance(row["ifeval_spec"], list) or not row["ifeval_spec"]:
                raise ValueError(f"{path}:{line_number}: ifeval_spec must be non-empty")
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


def accept_if_group(trajectory: dict[str, Any]) -> bool:
    """Filter IF groups by finite fractional reward variance and length safety."""

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
        elif bool(((reward_values < 0.0) | (reward_values > 1.0)).any().item()):
            reason = "reward_out_of_range"
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
        "rollout_request_timeout": (
            config.rollout.request_timeout,
            ROLLOUT_REQUEST_TIMEOUT,
        ),
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
        "eval_samples": (config.eval_gconfig.n_samples, EVAL_SAMPLES),
        "eval_temperature": (config.eval_gconfig.temperature, 1.0),
        "eval_greedy": (config.eval_gconfig.greedy, False),
        "valid_batch": (config.valid_dataset.batch_size, 153),
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


def run_explicit_eval(
    trainer: Any,
    *,
    workflow: str,
    workflow_kwargs: dict[str, Any],
    phase: str,
    marker_path: Path,
    expected_version: int,
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
    stats["eval/model_version"] = float(version)
    trainer._pending_eval_stats = stats
    payload = {
        "schema": "zgcm-if-eval-v1",
        "domain": "if",
        "phase": phase,
        "model_version": version,
        "samples_per_row": EVAL_SAMPLES,
        "reward_semantics": "fraction_of_explicit_constraints_satisfied",
        "stats": stats,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker_path)
    print(
        f"ZGCM_IF_EVAL phase={phase} "
        f"version={version} marker={marker_path}",
        flush=True,
    )


def eval_marker(status_root: Path, version: int) -> Path:
    return status_root / "evaluations" / f"version_{version:03d}.json"


def run_initial_weight_sync_gate(trainer: Any, marker_path: Path) -> None:
    """Prove the weight-sync path before evaluation can consume memory."""

    if marker_path.exists():
        raise RuntimeError(f"fresh weight-sync gate marker already exists: {marker_path}")
    versions_before = {
        "actor": int(trainer.actor.get_version()),
        "rollout": int(trainer.rollout.get_version()),
        "eval_rollout": int(trainer.eval_rollout.get_version()),
    }
    if set(versions_before.values()) != {0}:
        raise RuntimeError(
            "initial weight-sync gate requires version 0: "
            + json.dumps(versions_before, sort_keys=True)
        )

    started = time.monotonic()
    trainer.actor.update_weights(trainer.weight_update_meta.with_version(0))
    elapsed_seconds = time.monotonic() - started

    versions_after = {
        "actor": int(trainer.actor.get_version()),
        "rollout": int(trainer.rollout.get_version()),
        "eval_rollout": int(trainer.eval_rollout.get_version()),
    }
    if versions_after != versions_before:
        raise RuntimeError(
            "weight-sync gate changed model versions: "
            + json.dumps(
                {"before": versions_before, "after": versions_after},
                sort_keys=True,
            )
        )

    payload = {
        "schema": "zgcm-weight-sync-gate-v1",
        "status": "passed",
        "model_version": 0,
        "transport": str(trainer.weight_update_meta.type),
        "full_actor_weight_broadcast": True,
        "optimizer_step": False,
        "elapsed_seconds": elapsed_seconds,
        "versions_before": versions_before,
        "versions_after": versions_after,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker_path)
    print(
        "ZGCM_WEIGHT_SYNC_GATE "
        f"status=passed version=0 elapsed_seconds={elapsed_seconds:.3f} "
        f"marker={marker_path}",
        flush=True,
    )


def run_resume_weight_sync_gate(
    trainer: Any, marker_path: Path, expected_version: int
) -> None:
    """Catch up the fresh eval engine to the recovered train/rollout version.

    RecoverHandler restores the actor and training rollout to
    ``last_step.global_step + 1`` but intentionally does not touch the
    separately-created evaluation rollout.  A resumed run therefore starts
    with versions ``{actor: N, rollout: N, eval_rollout: 0}``.  Broadcast the
    recovered actor weights once to the eval engine, then reconnect the actor
    to the training rollout before entering PPO.
    """

    if marker_path.exists():
        raise RuntimeError(f"resume weight-sync gate marker already exists: {marker_path}")
    if trainer.eval_rollout is None:
        raise RuntimeError("resume weight-sync gate requires eval_rollout")

    versions_before = {
        "actor": int(trainer.actor.get_version()),
        "rollout": int(trainer.rollout.get_version()),
        "eval_rollout": int(trainer.eval_rollout.get_version()),
    }
    expected_before = {
        "actor": expected_version,
        "rollout": expected_version,
        "eval_rollout": 0,
    }
    if versions_before != expected_before:
        raise RuntimeError(
            "resume weight-sync gate has unexpected versions: "
            + json.dumps(
                {"expected": expected_before, "actual": versions_before},
                sort_keys=True,
            )
        )

    started = time.monotonic()
    meta = trainer.weight_update_meta.with_version(expected_version)
    trainer.actor.connect_engine(trainer.eval_rollout, meta)
    trainer.eval_rollout.pause()
    try:
        trainer.actor.update_weights(meta)
    finally:
        trainer.eval_rollout.resume()
        trainer.actor.connect_engine(trainer.rollout, meta)
    trainer.eval_rollout.set_version(expected_version)
    elapsed_seconds = time.monotonic() - started

    versions_after = {
        "actor": int(trainer.actor.get_version()),
        "rollout": int(trainer.rollout.get_version()),
        "eval_rollout": int(trainer.eval_rollout.get_version()),
    }
    expected_after = {
        "actor": expected_version,
        "rollout": expected_version,
        "eval_rollout": expected_version,
    }
    if versions_after != expected_after:
        raise RuntimeError(
            "resume weight-sync gate produced unexpected versions: "
            + json.dumps(
                {"expected": expected_after, "actual": versions_after},
                sort_keys=True,
            )
        )

    payload = {
        "schema": "zgcm-resume-weight-sync-gate-v1",
        "status": "passed",
        "model_version": expected_version,
        "transport": str(meta.type),
        "full_actor_weight_broadcast": True,
        "optimizer_step": False,
        "recovery_version": expected_version,
        "elapsed_seconds": elapsed_seconds,
        "versions_before": versions_before,
        "versions_after": versions_after,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker_path)
    print(
        "ZGCM_RESUME_WEIGHT_SYNC_GATE "
        f"status=passed version={expected_version} "
        f"elapsed_seconds={elapsed_seconds:.3f} marker={marker_path}",
        flush=True,
    )


class ExactVersionPPOTrainer(PPOTrainer):
    """Use AReaL evaluation engines with model-version based triggering."""

    def _evaluate(
        self,
        eval_workflow: Any,
        eval_workflow_kwargs: dict[str, Any],
        epoch: int,
        epoch_step: int,
        global_step: int,
    ) -> None:
        del epoch, epoch_step
        version = global_step + 1
        if version not in EVAL_VERSIONS:
            return
        marker = eval_marker(STATUS_ROOT, version)
        if marker.is_file():
            print(
                f"ZGCM_IF_EVAL_SKIP version={version} marker={marker}",
                flush=True,
            )
            return
        run_explicit_eval(
            self,
            workflow=eval_workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
            phase="periodic" if version != TOTAL_STEPS else "final",
            marker_path=marker,
            expected_version=version,
        )

    def _export_and_commit_stats(
        self,
        epoch: int,
        epoch_step: int,
        global_step: int,
    ) -> None:
        stats = self.actor.export_stats()
        stats.update(self.rollout.export_stats())
        pending_eval_stats = getattr(self, "_pending_eval_stats", None)
        if pending_eval_stats is not None:
            stats.update(pending_eval_stats)
            self._pending_eval_stats = None
        elif self.eval_rollout is not None:
            stats.update(self.eval_rollout.export_stats())
        self.stats_logger.commit(epoch, epoch_step, global_step, stats)

        if not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GRPOConfig)
    assert_contract(config)
    if not is_single_controller():
        raise RuntimeError("exact-version evaluation requires single-controller mode")

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = load_if_jsonl(
        config.train_dataset.path,
        tokenizer,
        max_prompt_tokens=config.train_dataset.max_length or MAX_PROMPT_TOKENS,
        expected_rows=EXPECTED_TRAIN_ROWS,
    )
    valid_dataset = load_if_jsonl(
        config.valid_dataset.path,
        tokenizer,
        max_prompt_tokens=config.valid_dataset.max_length or MAX_PROMPT_TOKENS,
        expected_rows=EXPECTED_VALID_ROWS,
    )
    preflight = ifeval_preflight([config.train_dataset.path, config.valid_dataset.path])
    print("ZGCM_IFEVAL_PREFLIGHT " + json.dumps(preflight, sort_keys=True), flush=True)

    workflow = "train.if_grpo.IFEvalWorkflow"
    workflow_kwargs = {
        "reward_fn": "rewards.ifeval_reward_fn",
        "async_reward_fn": "rewards.async_reward_router_fn",
        "reward_timeout_seconds": REWARD_TIMEOUT_SECONDS,
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "enable_thinking": True,
    }
    eval_workflow_kwargs = dict(workflow_kwargs)
    eval_workflow_kwargs["gconfig"] = config.eval_gconfig

    print(
        "ZGCM_IF_GRPO_CONTRACT "
        + json.dumps(
            {
                "algorithm": "GRPO",
                "train_rows": len(train_dataset),
                "valid_rows": len(valid_dataset),
                "eval_samples_per_row": EVAL_SAMPLES,
                "eval_versions": sorted(EVAL_VERSIONS),
                "prompt_batch": TRAIN_BATCH,
                "group_size": GROUP_SIZE,
                "trajectories_per_update": TRAIN_BATCH * GROUP_SIZE,
                "steps": TOTAL_STEPS,
                "rollout_max_new_tokens": MAX_NEW_TOKENS,
                "learner_max_total_tokens": UPDATE_MAX_SEQLEN,
                "truncated_reward": 0.0,
                "reward_semantics": "fraction_of_explicit_constraints_satisfied",
                "mask_no_eos_with_zero": False,
                "rollout_request_timeout": ROLLOUT_REQUEST_TIMEOUT,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    with ExactVersionPPOTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        if trainer.recover_info is None:
            run_initial_weight_sync_gate(
                trainer,
                STATUS_ROOT / "preflight" / "weight_sync_gate_version_000.json",
            )
        else:
            recovery_version = trainer.recover_info.last_step_info.global_step + 1
            run_resume_weight_sync_gate(
                trainer,
                STATUS_ROOT
                / "preflight"
                / f"weight_sync_gate_resume_version_{recovery_version:03d}.json",
                expected_version=recovery_version,
            )
        current_version = int(trainer.eval_rollout.get_version())
        if current_version in EVAL_VERSIONS:
            marker = eval_marker(STATUS_ROOT, current_version)
            if not marker.is_file():
                phase = "baseline" if current_version == 0 else "resume_catchup"
                run_explicit_eval(
                    trainer,
                    workflow=workflow,
                    workflow_kwargs=eval_workflow_kwargs,
                    phase=phase,
                    marker_path=marker,
                    expected_version=current_version,
                )
        if current_version == 0 and trainer.recover_info is not None:
            raise RuntimeError("fresh version 0 unexpectedly has recover state")

        trainer.train(
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn="train.if_grpo.accept_if_group",
        )
        final_marker = eval_marker(STATUS_ROOT, TOTAL_STEPS)
        if not final_marker.is_file():
            current_version = int(trainer.eval_rollout.get_version())
            if current_version != TOTAL_STEPS:
                raise RuntimeError(
                    f"training ended at model version {current_version}, "
                    f"expected {TOTAL_STEPS}"
                )
            run_explicit_eval(
                trainer,
                workflow=workflow,
                workflow_kwargs=eval_workflow_kwargs,
                phase="final_catchup",
                marker_path=final_marker,
                expected_version=TOTAL_STEPS,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
