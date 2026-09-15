"""Build a minimal, permission-read-only Python rootfs on local tmpfs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import sysconfig
import uuid
from pathlib import Path


def _copy_into_root(source: Path, rootfs: Path) -> None:
    destination = rootfs / source.relative_to("/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination, follow_symlinks=True)


def _ldd_paths(binary: Path) -> set[Path]:
    result = subprocess.run(["ldd", str(binary)], check=True, capture_output=True, text=True)
    paths: set[Path] = set()
    for line in result.stdout.splitlines():
        for token in line.replace("=>", " ").split():
            if token.startswith("/") and Path(token).exists():
                # Preserve the loader-visible path (for example /lib64/...) in
                # addition to its resolved target. The chroot cannot follow a
                # host-side compatibility symlink that was never copied.
                paths.add(Path(token))
    return paths


def _is_tmpfs(path: Path) -> bool:
    resolved = path.resolve()
    best_length = -1
    best_fstype = ""
    for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mountpoint = Path(fields[1].replace("\\040", " ")).resolve()
        try:
            resolved.relative_to(mountpoint)
        except ValueError:
            continue
        if len(str(mountpoint)) > best_length:
            best_length = len(str(mountpoint))
            best_fstype = fields[2]
    return best_fstype in {"tmpfs", "ramfs"}


def build_rootfs(output: Path) -> dict[str, str | bool]:
    output = output.resolve()
    if not _is_tmpfs(output.parent):
        raise RuntimeError("sandbox rootfs must be built below a tmpfs mount such as /dev/shm")
    python = Path(sys.executable).resolve()
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    if not python.is_absolute() or not stdlib.is_absolute():
        raise RuntimeError("Python runtime paths must be absolute")

    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, mode=0o755)
    _copy_into_root(python, temporary)
    _copy_into_root(stdlib, temporary)

    dependency_paths = _ldd_paths(python)
    lib_dynload = stdlib / "lib-dynload"
    if lib_dynload.exists():
        for extension in lib_dynload.glob("*.so"):
            dependency_paths.update(_ldd_paths(extension))
    for dependency in sorted(dependency_paths):
        _copy_into_root(dependency, temporary)

    runner_source = Path(__file__).with_name("sandbox_runner.py").resolve()
    shutil.copy2(runner_source, temporary / "sandbox_runner.py")
    (temporary / "etc").mkdir(exist_ok=True)
    (temporary / "dev").mkdir(exist_ok=True)
    null_device = temporary / "dev" / "null"
    os.mknod(null_device, stat.S_IFCHR | 0o666, os.makedev(1, 3))
    (temporary / "work").mkdir(exist_ok=True)

    manifest = {
        "python": f"/{python.relative_to('/')}",
        "stdlib": f"/{stdlib.relative_to('/')}",
        "tmpfs": True,
    }
    (temporary / "rootfs_manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    # Candidate processes map a non-root outer UID to namespace root. Keeping
    # runtime files owned by outer root makes the chroot effectively read-only;
    # only per-request directories below /work are chowned writable.
    for path in sorted(temporary.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            continue
        os.chown(path, 0, 0)
        os.chmod(path, 0o555 if path.is_dir() or os.access(path, os.X_OK) else 0o444)
    os.chown(temporary, 0, 0)
    os.chmod(temporary, 0o555)
    os.chmod(temporary / "work", 0o711)
    os.chmod(temporary / "sandbox_runner.py", 0o444)
    os.chmod(null_device, 0o666)
    if output.exists():
        shutil.rmtree(output)
    os.replace(temporary, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_rootfs(args.output)


if __name__ == "__main__":
    main()
