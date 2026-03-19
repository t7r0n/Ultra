"""Sandbox profiles and execution helpers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
import shlex
import shutil
import subprocess
import threading
import time
from typing import Callable, Sequence

from ..config import SandboxName


@dataclass(slots=True)
class SandboxProfile:
    mode: SandboxName
    allow_network: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SandboxResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_s: float | None = None
    streamed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_sandbox_profile(mode: SandboxName, *, allow_network: bool = False) -> SandboxProfile:
    return SandboxProfile(mode=mode, allow_network=allow_network)


def run_in_sandbox(
    command: Sequence[str],
    *,
    profile: SandboxProfile,
    cwd: Path,
    timeout: int = 600,
    stream: bool = False,
    stream_sink: Callable[[str, str], None] | None = None,
) -> SandboxResult:
    cwd = cwd.resolve()
    sandbox_command = build_sandbox_command(command, profile=profile, cwd=cwd)
    start = time.perf_counter()
    if stream or stream_sink is not None:
        process = subprocess.Popen(
            sandbox_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=sandbox_env(),
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def consume(pipe, stream_name: str, sink: list[str]) -> None:
            if pipe is None:
                return
            for line in iter(pipe.readline, ""):
                sink.append(line)
                if stream_sink is not None:
                    stream_sink(stream_name, line)
            pipe.close()

        stdout_thread = threading.Thread(target=consume, args=(process.stdout, "stdout", stdout_chunks), daemon=True)
        stderr_thread = threading.Thread(target=consume, args=(process.stderr, "stderr", stderr_chunks), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        return SandboxResult(
            command=list(sandbox_command),
            returncode=returncode,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            timed_out=timed_out,
            duration_s=time.perf_counter() - start,
            streamed=True,
        )
    result = subprocess.run(
        sandbox_command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=sandbox_env(),
    )
    return SandboxResult(
        command=list(sandbox_command),
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_s=time.perf_counter() - start,
        streamed=False,
    )


def build_sandbox_command(command: Sequence[str], *, profile: SandboxProfile, cwd: Path) -> list[str]:
    shell_command = shell_join(command)
    if profile.mode == "docker":
        executable = shutil.which("docker")
        if executable is None:
            raise RuntimeError("docker sandbox requested but `docker` is not available.")
        docker_command = [
            executable,
            "run",
            "--rm",
            "-v",
            f"{cwd}:/workspace:rw",
            "-w",
            "/workspace",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
        ]
        if not profile.allow_network:
            docker_command.extend(["--network", "none"])
        docker_command.extend(["python:3.11-slim", "bash", "-lc", shell_command])
        return docker_command
    if profile.mode == "firejail":
        executable = shutil.which("firejail")
        if executable is None:
            raise RuntimeError("firejail sandbox requested but `firejail` is not available.")
        firejail_command = [
            executable,
            "--quiet",
            "--private=" + str(cwd),
            "--private-tmp",
            "--caps.drop=all",
            "--nonewprivs",
            "--nogroups",
        ]
        if not profile.allow_network:
            firejail_command.append("--net=none")
        firejail_command.extend(["bash", "-lc", shell_command])
        return firejail_command
    raise RuntimeError(f"Unsupported sandbox mode: {profile.mode}")


def shell_join(command: Sequence[str]) -> str:
    if len(command) == 1:
        return str(command[0])
    return " ".join(shlex.quote(str(part)) for part in command)


def sandbox_env() -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
    }
    for key in ("TMPDIR", "TEMP", "TMP"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env
