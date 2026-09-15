"""Loopback HTTP service for namespace-isolated Code reward execution."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import hmac
import ipaddress
import json
import math
import os
import secrets
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from rewards.sandbox_client import load_code_asset, re_full_hash
from rewards.sandbox_rootfs import _is_tmpfs, build_rootfs


HEALTH_ATTESTATION_KEYS = (
    "user_namespace",
    "network_namespace",
    "pid_namespace",
    "chroot_root",
    "ro_root",
    "tmpfs_workdir",
    "shared_root_unreachable",
    "token_unavailable",
    "no_new_privs",
    "rlimit_cpu",
    "rlimit_as",
    "rlimit_nproc",
    "rlimit_fsize",
    "rlimit_nofile",
    "stdout_cap",
    "network_blocked",
)


class SandboxServerError(RuntimeError):
    """Raised when a request cannot be executed with full isolation."""


class _UidPool:
    def __init__(self, first_uid: int = 61000, last_uid: int = 64999) -> None:
        self._available = list(range(first_uid, last_uid + 1))
        self._lock = threading.Condition()

    def acquire(self) -> int:
        with self._lock:
            while not self._available:
                self._lock.wait()
            return self._available.pop()

    def release(self, uid: int) -> None:
        with self._lock:
            self._available.append(uid)
            self._lock.notify()


class NamespaceSandbox:
    def __init__(self, rootfs: Path, max_processes: int, test_parallelism: int) -> None:
        if os.geteuid() != 0:
            raise SandboxServerError("namespace sandbox server must run as outer root")
        self.rootfs = rootfs.resolve()
        manifest_path = self.rootfs / "rootfs_manifest.json"
        if not manifest_path.is_file():
            raise SandboxServerError("sandbox rootfs manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.python = str(manifest["python"])
        self.uid_pool = _UidPool()
        self.process_slots = threading.BoundedSemaphore(max_processes)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_processes)
        self.test_parallelism = max(1, min(test_parallelism, max_processes))
        self.stdout_cap = int(os.environ.get("ZGCM_SANDBOX_STDOUT_CAP_BYTES", "65536"))
        self.memory_limit = int(os.environ.get("ZGCM_SANDBOX_MEMORY_BYTES", str(1024 * 1024 * 1024)))
        self.nproc_limit = int(os.environ.get("ZGCM_SANDBOX_NPROC", "32"))
        self.nofile_limit = int(os.environ.get("ZGCM_SANDBOX_NOFILE", "64"))
        self.attestation = self._attest()
        if not all(self.attestation.get(key) is True for key in HEALTH_ATTESTATION_KEYS):
            raise SandboxServerError("namespace sandbox attestation failed closed")

    @staticmethod
    def _base_isolation_command(uid: int) -> list[str]:
        return [
            "setpriv",
            f"--reuid={uid}",
            f"--regid={uid}",
            "--clear-groups",
            "--no-new-privs",
            "unshare",
            "--propagation",
            "unchanged",
            "--user",
            "--map-root-user",
            "--net",
            "--pid",
            "--fork",
            "--kill-child",
        ]

    def _runner_command(self, uid: int, inner_payload_path: str, timeout: float) -> list[str]:
        cpu_limit = max(2, int(math.ceil(timeout)) + 1)
        return [
            "timeout",
            "--signal=KILL",
            f"{max(3.0, timeout + 2.0):.3f}",
            *self._base_isolation_command(uid),
            "prlimit",
            f"--cpu={cpu_limit}:{cpu_limit}",
            f"--as={self.memory_limit}:{self.memory_limit}",
            f"--nproc={self.nproc_limit}:{self.nproc_limit}",
            f"--fsize={self.stdout_cap}:{self.stdout_cap}",
            f"--nofile={self.nofile_limit}:{self.nofile_limit}",
            "--core=0:0",
            "chroot",
            str(self.rootfs),
            self.python,
            "/sandbox_runner.py",
            inner_payload_path,
        ]

    def _run_payload(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        uid = self.uid_pool.acquire()
        request_id = uuid.uuid4().hex
        outer_dir = self.rootfs / "work" / request_id
        outer_payload = outer_dir / "payload.json"
        inner_payload = f"/work/{request_id}/payload.json"
        try:
            outer_dir.mkdir(mode=0o700)
            outer_payload.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
            os.chown(outer_dir, uid, uid)
            os.chown(outer_payload, uid, uid)
            os.chmod(outer_payload, 0o400)
            with self.process_slots:
                with tempfile.TemporaryFile(dir="/dev/shm") as stdout_file, tempfile.TemporaryFile(
                    dir="/dev/shm"
                ) as stderr_file:
                    process = subprocess.run(
                        self._runner_command(uid, inner_payload, timeout),
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        timeout=max(5.0, timeout + 4.0),
                        check=False,
                        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8"},
                    )
                    stdout_file.seek(0, os.SEEK_END)
                    output_size = stdout_file.tell()
                    if output_size > self.stdout_cap or process.returncode != 0:
                        raise SandboxServerError("isolated runner failed or exceeded output limit")
                    stdout_file.seek(0)
                    output = stdout_file.read(self.stdout_cap).decode("utf-8", errors="replace")
            lines = [line for line in output.splitlines() if line.strip()]
            if not lines:
                raise SandboxServerError("isolated runner returned no attested result")
            value = json.loads(lines[-1])
            if value.get("nonce") != payload.get("nonce"):
                raise SandboxServerError("isolated runner nonce mismatch")
            return value
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise SandboxServerError("isolated runner invocation failed") from exc
        finally:
            shutil.rmtree(outer_dir, ignore_errors=True)
            self.uid_pool.release(uid)

    def _namespace_probe(self) -> dict[str, Any]:
        uid = self.uid_pool.acquire()
        try:
            script = (
                "for n in user net pid; do printf '%s=' \"$n\"; readlink /proc/self/ns/$n; done; "
                "awk '/^NoNewPrivs:/ {print \"no_new_privs=\" $2}' /proc/self/status"
            )
            command = [*self._base_isolation_command(uid), "/bin/sh", "-c", script]
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
            parsed = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            return parsed
        finally:
            self.uid_pool.release(uid)

    def _attest(self) -> dict[str, bool]:
        for executable in ("setpriv", "unshare", "prlimit", "timeout", "chroot"):
            if shutil.which(executable) is None:
                raise SandboxServerError(f"required isolation executable is missing: {executable}")
        parent_namespaces = {name: os.readlink(f"/proc/self/ns/{name}") for name in ("user", "net", "pid")}
        child = self._namespace_probe()
        nonce = secrets.token_hex(16)
        probe = self._run_payload({"mode": "probe", "nonce": nonce}, timeout=2.0)["probe"]
        limits = probe["limits"]
        return {
            "user_namespace": child.get("user") != parent_namespaces["user"],
            "network_namespace": child.get("net") != parent_namespaces["net"],
            "pid_namespace": child.get("pid") != parent_namespaces["pid"],
            "chroot_root": bool(probe.get("chroot_root")),
            "ro_root": bool(probe.get("root_write_blocked")),
            "tmpfs_workdir": _is_tmpfs(self.rootfs),
            "shared_root_unreachable": bool(probe.get("shared_root_unreachable")),
            "token_unavailable": bool(probe.get("token_unavailable")),
            "no_new_privs": child.get("no_new_privs") == "1",
            "rlimit_cpu": 0 < limits["cpu"][0] < 60,
            "rlimit_as": limits["as"][0] == self.memory_limit,
            "rlimit_nproc": limits["nproc"][0] == self.nproc_limit,
            "rlimit_fsize": limits["fsize"][0] == self.stdout_cap,
            "rlimit_nofile": limits["nofile"][0] == self.nofile_limit,
            "stdout_cap": self.stdout_cap <= 1024 * 1024,
            "network_blocked": bool(probe.get("network_blocked")),
        }

    def run_test(self, program: str, task_type: str, test: Any, timeout: float) -> tuple[int, float]:
        nonce = secrets.token_hex(24)
        payload = {
            "mode": "test",
            "nonce": nonce,
            "program": program,
            "task_type": task_type,
            "test": test,
            "max_execution_time": timeout,
        }
        started = time.monotonic()
        result = self._run_payload(payload, timeout=timeout)
        passed = int(result.get("passed") is True)
        return passed, time.monotonic() - started

    def verify(self, payload: dict[str, Any]) -> dict[str, Any]:
        program = payload.get("program")
        asset_hash = payload.get("asset_hash")
        task_type = payload.get("task_type")
        selection_seed = payload.get("selection_seed")
        max_tests = int(payload.get("max_tests", 8))
        timeout = float(payload.get("max_execution_time", 2.0))
        if not isinstance(program, str) or not program.strip() or len(program.encode()) > 512 * 1024:
            raise ValueError("program must be a non-empty string no larger than 512 KiB")
        if not isinstance(asset_hash, str) or not re_full_hash(asset_hash):
            raise ValueError("asset_hash must be 64 lowercase hex characters")
        if task_type not in {"code", "code_stdio"}:
            raise ValueError("task_type must be code or code_stdio")
        if not isinstance(selection_seed, str) or not selection_seed:
            raise ValueError("selection_seed must be non-empty")
        if not 1 <= max_tests <= 64 or not 0.1 <= timeout <= 10.0:
            raise ValueError("max_tests or max_execution_time is outside the allowed range")
        tests = load_code_asset(asset_hash)
        order = sorted(
            range(len(tests)),
            key=lambda index: hashlib.sha256(f"{selection_seed}:{index}".encode()).digest(),
        )
        selected = [tests[index] for index in order[: min(max_tests, len(tests))]]
        outcomes: list[tuple[int, float]] = []
        for offset in range(0, len(selected), self.test_parallelism):
            futures = [
                self.executor.submit(self.run_test, program, task_type, test, timeout)
                for test in selected[offset : offset + self.test_parallelism]
            ]
            outcomes.extend(future.result() for future in futures)
        results = [result for result, _runtime in outcomes]
        runtimes = [runtime for _result, runtime in outcomes]
        score = sum(results) / len(results) if results else 0.0
        return {
            "asset_hash": asset_hash,
            "results": results,
            "runtimes": runtimes,
            "score": score,
            "test_count": len(results),
        }


class SandboxApplication:
    def __init__(self, token: str, sandbox: NamespaceSandbox, asset_root: Path) -> None:
        self.token = token
        self.sandbox = sandbox
        self.identity = self._build_identity(token, asset_root)

    @staticmethod
    def _build_identity(token: str, asset_root: Path) -> dict[str, str]:
        code_digest = hashlib.sha256()
        for name in ("sandbox_server.py", "sandbox_runner.py", "sandbox_rootfs.py", "sandbox_client.py"):
            path = Path(__file__).with_name(name)
            code_digest.update(name.encode() + b"\0" + path.read_bytes())
        asset_manifest = asset_root.resolve() / "manifest.json"
        if not asset_manifest.is_file():
            raise SandboxServerError("Code asset manifest is missing")
        return {
            "service_instance_id": uuid.uuid4().hex,
            "started_at": datetime.now(UTC).isoformat(),
            "sandbox_code_sha256": code_digest.hexdigest(),
            "asset_manifest_sha256": hashlib.sha256(asset_manifest.read_bytes()).hexdigest(),
            "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        }


class SandboxRequestHandler(BaseHTTPRequestHandler):
    server_version = "ZGCMNamespaceSandbox/1.0"

    @property
    def app(self) -> SandboxApplication:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.app.token}"
        return hmac.compare_digest(header, expected)

    def _send_json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=True, sort_keys=True).encode()
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self._authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if self.path != "/health":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "healthy",
                "sandbox_mode": "attested-http",
                "attestation": self.app.sandbox.attestation,
                **self.app.identity,
            },
        )

    def do_POST(self) -> None:
        if not self._authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if self.path != "/v1/verify":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 < content_length <= 1024 * 1024:
                raise ValueError("request body is empty or too large")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("request body is not an object")
            result = self.app.sandbox.verify(payload)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "sandbox_backend_error"})
            return
        self._send_json(HTTPStatus.OK, result)


class SandboxHTTPServer(ThreadingHTTPServer):
    request_queue_size = 512
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--rootfs", type=Path, required=True)
    parser.add_argument("--build-rootfs", action="store_true")
    parser.add_argument("--token-env", default="ZGCM_CODE_SANDBOX_TOKEN")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--max-processes", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--test-parallelism", type=int, default=4)
    parser.add_argument("--allow-private-bind", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        try:
            address = ipaddress.ip_address(args.host)
        except ValueError as exc:
            raise SystemExit("sandbox host must be a literal private address") from exc
        shared_range = ipaddress.ip_network("100.64.0.0/10")
        if not args.allow_private_bind or (not address.is_private and address not in shared_range):
            raise SystemExit("non-loopback bind requires --allow-private-bind and a private address")
    if args.token_file is not None:
        token_path = args.token_file.resolve()
        token_stat = token_path.stat()
        if stat.S_IMODE(token_stat.st_mode) != 0o600:
            raise SystemExit("--token-file must have mode 0600")
        token = token_path.read_text(encoding="utf-8").strip()
    else:
        token = os.environ.get(args.token_env, "")
    if len(token) < 32:
        raise SystemExit(f"{args.token_env} must contain at least 32 characters")
    os.environ["ZGCM_CODE_ASSET_ROOT"] = str(args.asset_root.resolve())
    if args.build_rootfs:
        build_rootfs(args.rootfs)
    elif not (args.rootfs / "rootfs_manifest.json").is_file():
        raise SystemExit("sandbox rootfs is missing; pass --build-rootfs")
    sandbox = NamespaceSandbox(args.rootfs, args.max_processes, args.test_parallelism)
    app = SandboxApplication(token, sandbox, args.asset_root)
    server = SandboxHTTPServer((args.host, args.port), SandboxRequestHandler)
    server.app = app  # type: ignore[attr-defined]
    server.serve_forever(poll_interval=0.25)


if __name__ == "__main__":
    main()
