"""Trusted inner runner copied into the minimal Code sandbox rootfs."""

from __future__ import annotations

import json
import math
import os
import resource
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


def _write_result(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
    sys.stdout.flush()


def _normalize_stdio_value(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value) + "\n"
    return str(value)


def _stdio_equal(actual: str, expected_value: Any) -> bool:
    expected = _normalize_stdio_value(expected_value)
    actual_tokens = actual.strip().split()
    expected_tokens = expected.strip().split()
    if len(actual_tokens) != len(expected_tokens):
        return False
    for actual_token, expected_token in zip(actual_tokens, expected_tokens, strict=True):
        if actual_token == expected_token:
            continue
        try:
            left = float(actual_token)
            right = float(expected_token)
        except ValueError:
            return False
        if not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-8):
            return False
    return True


def _run_function_test(payload: dict[str, Any], workdir: Path) -> bool:
    script_path = workdir / "function_test.py"
    marker_path = workdir / "function_test.passed"
    sink_path = workdir / "candidate-output.bin"
    script = (
        "from pathlib import Path\n"
        "ns = {'__name__': '__main__', '__builtins__': __builtins__}\n"
        "try:\n"
        f"    exec(compile({payload['program']!r}, '<candidate>', 'exec'), ns)\n"
        f"    exec(compile({payload['test']!r}, '<test>', 'exec'), ns)\n"
        "except BaseException:\n"
        "    raise SystemExit(1)\n"
        f"Path({str(marker_path)!r}).write_text({payload['nonce']!r}, encoding='utf-8')\n"
    )
    script_path.write_text(script, encoding="utf-8")
    try:
        with sink_path.open("wb") as sink:
            process = subprocess.run(
                [sys.executable, str(script_path)],
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=sink,
                cwd=workdir,
                timeout=float(payload["max_execution_time"]),
                check=False,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"},
            )
        return (
            process.returncode == 0
            and marker_path.is_file()
            and marker_path.read_text(encoding="utf-8", errors="replace") == payload["nonce"]
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _run_stdio_test(payload: dict[str, Any], workdir: Path) -> bool:
    candidate_path = workdir / "candidate.py"
    stdout_path = workdir / "candidate.stdout"
    stderr_path = workdir / "candidate.stderr"
    candidate_path.write_text(payload["program"], encoding="utf-8")
    stdin = _normalize_stdio_value(payload["test"]["input"]).encode()
    timeout = float(payload["max_execution_time"])
    try:
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.run(
                [sys.executable, str(candidate_path)],
                input=stdin,
                stdout=stdout_handle,
                stderr=stderr_handle,
                cwd=workdir,
                timeout=timeout,
                check=False,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"},
            )
        if process.returncode != 0:
            return False
        actual = stdout_path.read_text(encoding="utf-8", errors="replace")
        return _stdio_equal(actual, payload["test"]["output"])
    except (OSError, subprocess.TimeoutExpired):
        return False


def _probe(payload: dict[str, Any], workdir: Path) -> None:
    writable_root = True
    try:
        Path("/etc/zgcm-sandbox-write-probe").write_text("unsafe", encoding="utf-8")
    except OSError:
        writable_root = False
    work_probe = workdir / "work-write-probe"
    work_probe.write_text("ok", encoding="utf-8")
    # The minimal chroot built by sandbox_rootfs contains no /mnt, so reaching
    # it would mean the host filesystem leaked into the sandbox.
    shared_root_unreachable = not Path("/mnt").exists()
    token_unavailable = all("TOKEN" not in key.upper() and "AUTHORIZATION" not in key.upper() for key in os.environ)
    network_blocked = False
    probe_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe_socket.settimeout(0.5)
    try:
        network_blocked = probe_socket.connect_ex(("1.1.1.1", 53)) != 0
    finally:
        probe_socket.close()
    limits = {
        "cpu": list(resource.getrlimit(resource.RLIMIT_CPU)),
        "as": list(resource.getrlimit(resource.RLIMIT_AS)),
        "nproc": list(resource.getrlimit(resource.RLIMIT_NPROC)),
        "fsize": list(resource.getrlimit(resource.RLIMIT_FSIZE)),
        "nofile": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
    }
    _write_result(
        {
            "nonce": payload["nonce"],
            "probe": {
                "chroot_root": Path("/sandbox_runner.py").is_file(),
                "root_write_blocked": not writable_root,
                "work_write_allowed": work_probe.exists(),
                "shared_root_unreachable": shared_root_unreachable,
                "token_unavailable": token_unavailable,
                "network_blocked": network_blocked,
                "limits": limits,
            },
        }
    )


def main() -> None:
    payload_path = Path(sys.argv[1]).resolve()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    workdir = payload_path.parent
    os.chdir(workdir)
    if payload.get("mode") == "probe":
        _probe(payload, workdir)
        return
    passed = False
    if payload.get("task_type") == "code":
        passed = _run_function_test(payload, workdir)
    elif payload.get("task_type") == "code_stdio":
        passed = _run_stdio_test(payload, workdir)
    _write_result({"nonce": payload["nonce"], "passed": bool(passed)})


if __name__ == "__main__":
    main()
