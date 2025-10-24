from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(slots=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str


class Sandbox:
    def __init__(self, profile: str, workdir: Optional[Path] = None) -> None:
        self.profile = profile
        self.workdir = workdir or Path(tempfile.mkdtemp(prefix="ultra_sandbox_"))

    def run(self, command: Iterable[str], timeout: int = 60) -> SandboxResult:
        raise NotImplementedError


class DockerSandbox(Sandbox):
    def run(self, command: Iterable[str], timeout: int = 60) -> SandboxResult:
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("Docker is required for docker sandbox profile")
        full_command: List[str] = [
            docker,
            "run",
            "--rm",
            "--network=none",
            "-v",
            f"{self.workdir}:{self.workdir}",
            "-w",
            str(self.workdir),
            "python:3.10",
            *command,
        ]
        completed = subprocess.run(  # pragma: no cover - external process
            full_command,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        return SandboxResult(completed.returncode, completed.stdout, completed.stderr)


class FirejailSandbox(Sandbox):
    def run(self, command: Iterable[str], timeout: int = 60) -> SandboxResult:
        firejail = shutil.which("firejail")
        if firejail is None:
            raise RuntimeError("firejail is required for firejail sandbox profile")
        full_command = [
            firejail,
            "--quiet",
            "--net=none",
            "--private={}".format(self.workdir),
            "bash",
            "-lc",
            " ".join(command),
        ]
        completed = subprocess.run(  # pragma: no cover - external process
            full_command,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        return SandboxResult(completed.returncode, completed.stdout, completed.stderr)


class LocalSandbox(Sandbox):
    def run(self, command: Iterable[str], timeout: int = 60) -> SandboxResult:
        completed = subprocess.run(  # pragma: no cover - external process
            list(command),
            cwd=str(self.workdir),
            timeout=timeout,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return SandboxResult(completed.returncode, completed.stdout, completed.stderr)


def create_sandbox(profile: str, workdir: Optional[Path] = None) -> Sandbox:
    if profile == "docker":
        return DockerSandbox(profile, workdir)
    if profile == "firejail":
        return FirejailSandbox(profile, workdir)
    return LocalSandbox(profile, workdir)

