"""HTTP-only client for isolated Code reward execution."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from rewards.completion import extract_python_code, visible_answer


class SandboxConfigurationError(RuntimeError):
    """Raised when the isolated sandbox contract is not configured."""


class SandboxExecutionError(RuntimeError):
    """Raised for backend errors; these are not incorrect model answers."""


class InvalidSandboxRequestError(ValueError):
    """Typed per-sample 400/422 rejection; quarantine without pausing."""


_FAILURE_LOCK = threading.Lock()
_CONSECUTIVE_FAILURES = 0
_CONSECUTIVE_SCHEMA_FAILURES = 0
_IDENTITY_LOCK = threading.Lock()
_LAST_IDENTITY_CHECK = 0.0
_EXPECTED_IDENTITY: dict[str, str] | None = None
_IDENTITY_FIELDS = (
    "service_instance_id",
    "started_at",
    "sandbox_code_sha256",
    "asset_manifest_sha256",
    "token_sha256",
)


def record_code_infrastructure_failure(error_class: str) -> None:
    """Write a metadata-only pause marker for watchdog-visible failures."""

    global _CONSECUTIVE_FAILURES
    path_value = os.environ.get("ZGCM_CODE_PAUSE_FILE", "").strip()
    if not path_value:
        return
    with _FAILURE_LOCK:
        _CONSECUTIVE_FAILURES += 1
        payload = {
            "error_class": error_class,
            "time": datetime.now(UTC).isoformat(),
            "consecutive_failures": _CONSECUTIVE_FAILURES,
        }
        path = Path(path_value).resolve()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)


def _record_schema_failure() -> None:
    global _CONSECUTIVE_SCHEMA_FAILURES
    with _FAILURE_LOCK:
        _CONSECUTIVE_SCHEMA_FAILURES += 1
        failures = _CONSECUTIVE_SCHEMA_FAILURES
    threshold = int(os.environ.get("ZGCM_CODE_SCHEMA_FAILURE_THRESHOLD", "3"))
    if failures >= threshold:
        record_code_infrastructure_failure("sandbox_schema")


def _reset_failure_counters() -> None:
    global _CONSECUTIVE_FAILURES, _CONSECUTIVE_SCHEMA_FAILURES
    with _FAILURE_LOCK:
        _CONSECUTIVE_FAILURES = 0
        _CONSECUTIVE_SCHEMA_FAILURES = 0


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        record_code_infrastructure_failure("sandbox_configuration")
        raise SandboxConfigurationError(f"missing required environment variable {name}")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@lru_cache(maxsize=4)
def load_code_asset(asset_hash: str) -> list[Any]:
    """Load and content-verify a test asset without executing any code."""

    if not isinstance(asset_hash, str) or not re_full_hash(asset_hash):
        raise SandboxConfigurationError("invalid code asset hash")
    root = Path(_required_env("ZGCM_CODE_ASSET_ROOT")).resolve()
    path = root / asset_hash[:2] / f"{asset_hash}.json.gz"
    try:
        with gzip.open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise SandboxConfigurationError("code asset is missing or unreadable") from exc
    if hashlib.sha256(raw).hexdigest() != asset_hash:
        raise SandboxConfigurationError("code asset content hash mismatch")
    try:
        tests = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SandboxConfigurationError("code asset is invalid JSON") from exc
    if not isinstance(tests, list) or not tests:
        raise SandboxConfigurationError("code asset is not a non-empty test list")
    return tests


def re_full_hash(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _endpoint(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith(suffix) else f"{base}{suffix}"


def _load_expected_identity() -> dict[str, str]:
    global _EXPECTED_IDENTITY
    if _EXPECTED_IDENTITY is not None:
        return _EXPECTED_IDENTITY
    path = Path(_required_env("ZGCM_CODE_EXPECTED_IDENTITY_FILE")).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = {field: str(value[field]) for field in _IDENTITY_FIELDS}
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        record_code_infrastructure_failure("sandbox_identity_config")
        raise SandboxConfigurationError("Code identity marker is missing or invalid") from exc
    _EXPECTED_IDENTITY = expected
    return expected


def verify_live_sandbox_identity(base_url: str, token: str, timeout: float, *, force: bool = False) -> None:
    """Refresh attested service identity with a bounded per-process TTL."""

    global _LAST_IDENTITY_CHECK
    ttl = float(os.environ.get("ZGCM_CODE_IDENTITY_TTL_SECONDS", "60"))
    now = time.monotonic()
    with _IDENTITY_LOCK:
        if not force and _LAST_IDENTITY_CHECK and now - _LAST_IDENTITY_CHECK < ttl:
            return
        expected = _load_expected_identity()
        try:
            import requests

            response = requests.get(
                _endpoint(base_url, "/health"),
                headers={"Authorization": f"Bearer {token}"},
                timeout=min(timeout, 30.0),
            )
            if response.status_code in {401, 403}:
                raise SandboxExecutionError("sandbox health authentication failed")
            response.raise_for_status()
            live = response.json()
        except Exception as exc:
            record_code_infrastructure_failure("sandbox_identity_connectivity")
            raise SandboxExecutionError("sandbox live identity refresh failed") from exc
        if live.get("status") != "healthy" or live.get("sandbox_mode") != "attested-http":
            record_code_infrastructure_failure("sandbox_identity")
            raise SandboxExecutionError("sandbox live identity status mismatch")
        if any(str(live.get(field)) != expected[field] for field in _IDENTITY_FIELDS):
            record_code_infrastructure_failure("sandbox_identity")
            raise SandboxExecutionError("sandbox live identity mismatch")
        _LAST_IDENTITY_CHECK = now


def _request_json(url: str, token: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise SandboxConfigurationError("requests is required for the sandbox client") from exc
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        if response.status_code in {401, 403}:
            record_code_infrastructure_failure("sandbox_auth")
            raise SandboxExecutionError(f"sandbox authentication HTTP status {response.status_code}")
        if response.status_code in {400, 422}:
            raise InvalidSandboxRequestError(f"sandbox rejected sample with HTTP {response.status_code}")
        if response.status_code >= 500:
            record_code_infrastructure_failure("sandbox_upstream")
            raise SandboxExecutionError(f"sandbox HTTP status {response.status_code}")
        if response.status_code >= 400:
            record_code_infrastructure_failure("sandbox_protocol")
        response.raise_for_status()
        value = response.json()
    except (SandboxExecutionError, InvalidSandboxRequestError):
        raise
    except Exception as exc:
        record_code_infrastructure_failure("sandbox_connectivity")
        raise SandboxExecutionError("sandbox request failed") from exc
    if not isinstance(value, dict):
        _record_schema_failure()
        raise SandboxExecutionError("sandbox response is not an object")
    return value


def _score_response(value: dict[str, Any]) -> float:
    if "score" in value:
        try:
            score = float(value["score"])
        except (TypeError, ValueError, OverflowError) as exc:
            _record_schema_failure()
            raise SandboxExecutionError("sandbox score is not numeric") from exc
    else:
        results = value.get("results")
        if not isinstance(results, list) or not results:
            _record_schema_failure()
            raise SandboxExecutionError("sandbox response has no test results")
        if not all(isinstance(item, bool) for item in results):
            _record_schema_failure()
            raise SandboxExecutionError("sandbox test results are not booleans")
        score = sum(float(item) for item in results) / len(results)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        _record_schema_failure()
        raise SandboxExecutionError("sandbox score is not finite within [0, 1]")
    _reset_failure_counters()
    return score


def code_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int] | None = None,
    completion_ids: list[int] | None = None,
    **kwargs: Any,
) -> float:
    """AReaL-compatible Code reward; execution always occurs behind HTTP."""

    del prompt, prompt_ids, completion_ids
    answer = visible_answer(completions)
    if answer is None:
        return 0.0
    asset_hash = kwargs.get("code_asset_hash")
    task_type = kwargs.get("task_type")
    prompt_hash = kwargs.get("prompt_hash")
    if not re_full_hash(asset_hash or "") or task_type not in {"code", "code_stdio"}:
        return 0.0
    mode = _required_env("ZGCM_CODE_SANDBOX_MODE")
    base_url = _required_env("ZGCM_CODE_SANDBOX_URL")
    token = _required_env("ZGCM_CODE_SANDBOX_TOKEN")
    timeout = float(os.environ.get("ZGCM_CODE_HTTP_TIMEOUT_SECONDS", "180"))
    max_tests = int(os.environ.get("ZGCM_CODE_MAX_TESTS", "8"))
    max_execution_time = float(os.environ.get("ZGCM_CODE_TEST_TIMEOUT_SECONDS", "2"))
    program = extract_python_code(answer)
    if not program:
        return 0.0

    if mode in {"namespace-http", "attested-http"}:
        if mode == "attested-http":
            verify_live_sandbox_identity(base_url, token, timeout)
        payload = {
            "program": program,
            "asset_hash": asset_hash,
            "task_type": task_type,
            "selection_seed": str(prompt_hash or asset_hash),
            "max_tests": max_tests,
            "max_execution_time": max_execution_time,
        }
        value = _request_json(_endpoint(base_url, "/v1/verify"), token, payload, timeout)
    elif mode == "open-instruct-http":
        tests = load_code_asset(asset_hash)
        if max_tests > 0 and len(tests) > max_tests:
            ordered = sorted(
                range(len(tests)),
                key=lambda index: hashlib.sha256(f"{prompt_hash}:{index}".encode()).digest(),
            )
            tests = [tests[index] for index in ordered[:max_tests]]
        suffix = "/test_program_stdio" if task_type == "code_stdio" else "/test_program"
        payload = {"program": program, "tests": tests, "max_execution_time": max_execution_time}
        value = _request_json(_endpoint(base_url, suffix), token, payload, timeout)
    else:
        raise SandboxConfigurationError(f"unsupported ZGCM_CODE_SANDBOX_MODE: {mode}")
    return _score_response(value)
