"""Agentic coding loop for ULTRA."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

from .config import BackendName, GlobalConfig
from .runtime import run_ultra_turn
from .tools.logging import RunArtifacts, append_jsonl
from .tools.sandbox import SandboxProfile, SandboxResult, run_in_sandbox


AGENT_ACTION_SCHEMA = {
    "type": "object",
    "required": ["action"],
    "properties": {
        "action": {"type": "string"},
        "tool": {"type": "string"},
        "verify_kind": {"type": "string"},
        "path": {"type": "string"},
        "content": {"type": "string"},
        "command": {"type": "string"},
        "code": {"type": "string"},
        "final": {"type": "string"},
    },
}


@dataclass(slots=True)
class AgentStep:
    index: int
    action: dict[str, Any]
    observation: str
    verification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRunResult:
    steps: list[AgentStep]
    final: str
    workspace: Path
    verification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["workspace"] = str(self.workspace)
        return payload


@dataclass(slots=True)
class VerificationCheck:
    name: str
    result: SandboxResult

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "result": self.result.to_dict()}


@dataclass(slots=True)
class VerificationSummary:
    passed: bool
    checks: list[VerificationCheck]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [check.to_dict() for check in self.checks]}

    def to_text(self) -> str:
        if not self.checks:
            return "No verification checks were available."
        lines = [f"verification_passed={self.passed}"]
        for check in self.checks:
            lines.append(f"[{check.name}] returncode={check.result.returncode}")
            if check.result.stdout.strip():
                lines.append("stdout:")
                lines.append(check.result.stdout.strip())
            if check.result.stderr.strip():
                lines.append("stderr:")
                lines.append(check.result.stderr.strip())
        return "\n".join(lines)


def run_agent(
    *,
    model_ref: str,
    backend: BackendName,
    config: GlobalConfig,
    goal: str,
    artifacts: RunArtifacts,
    sandbox_profile: SandboxProfile,
    open_terminal: bool,
    max_steps: int = 6,
    max_reflexive_retries: int = 2,
) -> AgentRunResult:
    workspace = artifacts.root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    steps: list[AgentStep] = []
    transcript = [
        {
            "role": "user",
            "content": (
                "You are operating as an agentic coding assistant.\n"
                "Use JSON only. Available actions: read_file, write_file, shell, python, verify, final.\n"
                "Map actions to explicit tools: filesystem.read_file, filesystem.write_file, shell.run, python.run, verify.run.\n"
                "All writes must stay inside the workspace.\n"
                "Do not reveal secrets. Redact tokens, API keys, passwords, and private credentials.\n"
                f"Workspace: {workspace}\n"
                f"Goal: {goal}"
            ),
        }
    ]
    final_message = ""
    last_verification: VerificationSummary | None = None
    retries_used = 0
    terminal_log = artifacts.root / "terminal.jsonl"
    terminal_events = 0
    for index in range(max_steps):
        agent_config = deepcopy(config)
        agent_config.structured_output.enabled = True
        agent_config.structured_output.json_schema = AGENT_ACTION_SCHEMA
        turn = run_ultra_turn(
            model_ref=model_ref,
            backend=backend,
            config=agent_config,
            ultra_profile="coding",
            conversation=transcript,
            artifacts=artifacts,
            seed_base=index * 1000,
        )
        action = turn.final_payload if isinstance(turn.final_payload, dict) else parse_action_json(turn.final_text)
        if not isinstance(action, dict):
            final_message = "Agent failed to produce a valid action."
            break
        observation, streamed_events = execute_action(
            action,
            workspace=workspace,
            sandbox_profile=sandbox_profile,
            open_terminal=open_terminal,
            terminal_log=terminal_log,
            step_index=index,
            emit_live_terminal=open_terminal,
        )
        terminal_events += streamed_events
        verification = maybe_verify_action(action, workspace=workspace, sandbox_profile=sandbox_profile)
        last_verification = verification or last_verification
        step = AgentStep(
            index=index,
            action=action,
            observation=observation,
            verification=verification.to_dict() if verification is not None else None,
        )
        steps.append(step)
        append_jsonl(
            artifacts.candidates_jsonl,
            {
                "agent_step": index,
                "action": action,
                "tool": classify_tool(action),
                "observation": observation,
                "terminal_streamed": streamed_events > 0,
                "verification": verification.to_dict() if verification is not None else None,
            },
        )
        if action.get("action") == "final":
            if verification is not None and not verification.passed and retries_used < max_reflexive_retries:
                retries_used += 1
                transcript.append({"role": "assistant", "content": json.dumps(action, sort_keys=True)})
                transcript.append(
                    {
                        "role": "user",
                        "content": (
                            "Verification failed. Fix the solution and continue.\n"
                            f"{verification.to_text()}"
                        ),
                    }
                )
                continue
            final_message = str(action.get("final") or observation)
            break
        transcript.append({"role": "assistant", "content": json.dumps(action, sort_keys=True)})
        follow_up = f"Observation:\n{observation}"
        if verification is not None:
            follow_up += "\n\nVerification:\n" + verification.to_text()
        transcript.append({"role": "user", "content": follow_up})
    else:
        final_message = "Agent reached the maximum step budget without producing a final answer."
    artifacts.final_txt.write_text(final_message + ("\n" if not final_message.endswith("\n") else ""), encoding="utf-8")
    artifacts.metrics_json.write_text(
        json.dumps(
            {
                "steps": len(steps),
                "workspace": str(workspace),
                "open_terminal": open_terminal,
                "terminal_log": str(terminal_log) if terminal_log.exists() else None,
                "terminal_events": terminal_events,
                "verification": last_verification.to_dict() if last_verification is not None else None,
                "reflexive_retries_used": retries_used,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return AgentRunResult(
        steps=steps,
        final=final_message,
        workspace=workspace,
        verification=last_verification.to_dict() if last_verification is not None else None,
    )


def parse_action_json(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def execute_action(
    action: dict[str, Any],
    *,
    workspace: Path,
    sandbox_profile: SandboxProfile,
    open_terminal: bool,
    terminal_log: Path | None = None,
    step_index: int | None = None,
    emit_live_terminal: bool = False,
) -> tuple[str, int]:
    action_name = str(action.get("action", ""))
    if action_name == "read_file":
        path = resolve_workspace_path(workspace, str(action.get("path", "")))
        if not path.exists():
            return f"File not found: {path}", 0
        return redact_secrets(path.read_text(encoding="utf-8")), 0
    if action_name == "write_file":
        path = resolve_workspace_path(workspace, str(action.get("path", "")))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(action.get("content", "")), encoding="utf-8")
        return f"Wrote {path}", 0
    if action_name == "shell":
        command = str(action.get("command", "")).strip()
        if not command:
            return "Missing shell command.", 0
        enforce_shell_policy(command, sandbox_profile=sandbox_profile)
        sink, counter = build_terminal_sink(
            terminal_log=terminal_log,
            step_index=step_index,
            emit_live=emit_live_terminal,
        )
        result = run_in_sandbox(
            [command],
            profile=sandbox_profile,
            cwd=workspace,
            stream=open_terminal,
            stream_sink=sink if open_terminal else None,
        )
        if open_terminal:
            return redact_secrets(result.stdout + ("\n" + result.stderr if result.stderr else "")), counter["count"]
        return redact_secrets(f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"), 0
    if action_name == "python":
        code = str(action.get("code", "")).strip()
        if not code:
            return "Missing python code.", 0
        script = workspace / f".agent_step_{len(list(workspace.glob('.agent_step_*')))}.py"
        script.write_text(code, encoding="utf-8")
        sink, counter = build_terminal_sink(
            terminal_log=terminal_log,
            step_index=step_index,
            emit_live=emit_live_terminal,
        )
        result = run_in_sandbox(
            [f"python {script.name}"],
            profile=sandbox_profile,
            cwd=workspace,
            stream=open_terminal,
            stream_sink=sink if open_terminal else None,
        )
        if open_terminal:
            return redact_secrets(result.stdout + ("\n" + result.stderr if result.stderr else "")), counter["count"]
        return redact_secrets(f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"), 0
    if action_name == "verify":
        summary = run_workspace_verification(workspace=workspace, sandbox_profile=sandbox_profile, verify_kind=str(action.get("verify_kind") or "auto"))
        return summary.to_text(), 0
    if action_name == "final":
        return redact_secrets(str(action.get("final", ""))), 0
    return f"Unsupported action: {action_name}", 0


def resolve_workspace_path(workspace: Path, relative_path: str) -> Path:
    candidate = (workspace / relative_path).resolve()
    if workspace.resolve() not in candidate.parents and candidate != workspace.resolve():
        raise RuntimeError("Agent attempted to escape the workspace.")
    return candidate


def maybe_verify_action(
    action: dict[str, Any],
    *,
    workspace: Path,
    sandbox_profile: SandboxProfile,
) -> VerificationSummary | None:
    action_name = str(action.get("action", ""))
    if action_name in {"write_file", "python", "shell", "verify", "final"}:
        verify_kind = str(action.get("verify_kind") or "auto")
        return run_workspace_verification(workspace=workspace, sandbox_profile=sandbox_profile, verify_kind=verify_kind)
    return None


def run_workspace_verification(
    *,
    workspace: Path,
    sandbox_profile: SandboxProfile,
    verify_kind: str = "auto",
) -> VerificationSummary:
    checks: list[VerificationCheck] = []
    for name, command in verification_commands(workspace=workspace, verify_kind=verify_kind, sandbox_profile=sandbox_profile):
        result = run_in_sandbox([command], profile=sandbox_profile, cwd=workspace)
        checks.append(VerificationCheck(name=name, result=result))
    passed = all(check.result.returncode == 0 for check in checks) if checks else True
    return VerificationSummary(passed=passed, checks=checks)


def verification_commands(
    *,
    workspace: Path,
    verify_kind: str,
    sandbox_profile: SandboxProfile,
) -> list[tuple[str, str]]:
    python_files = sorted(workspace.rglob("*.py"))
    commands: list[tuple[str, str]] = []
    if verify_kind in {"auto", "compile"} and python_files:
        commands.append(("compile", "python -m compileall -q ."))
    if sandbox_profile.mode != "docker" and verify_kind in {"auto", "lint"} and python_files and shutil.which("ruff"):
        commands.append(("lint", "ruff check ."))
    has_pytest_suite = (workspace / "tests").exists() or (workspace / "pytest.ini").exists() or (workspace / "pyproject.toml").exists()
    if verify_kind in {"auto", "tests"} and has_pytest_suite:
        if sandbox_profile.mode != "docker" and shutil.which("pytest"):
            commands.append(("tests", "pytest -q"))
        elif python_files:
            commands.append(("tests", "python -m unittest discover -s tests -v"))
    if verify_kind == "lint" and not any(name == "lint" for name, _ in commands) and python_files:
        commands.append(("lint", "python -m compileall -q ."))
    if verify_kind == "tests" and not any(name == "tests" for name, _ in commands) and python_files:
        commands.append(("tests", "python -m compileall -q ."))
    return commands


def classify_tool(action: dict[str, Any]) -> str:
    action_name = str(action.get("action", ""))
    return {
        "read_file": "filesystem.read_file",
        "write_file": "filesystem.write_file",
        "shell": "shell.run",
        "python": "python.run",
        "verify": "verify.run",
        "final": "final",
    }.get(action_name, action_name)


def enforce_shell_policy(command: str, *, sandbox_profile: SandboxProfile) -> None:
    lowered = command.lower()
    escape_attempts = ("docker ", "podman ", "firejail ", "sudo ", "su ", "mount ", "umount ", "unshare ", "nsenter ")
    if any(token in lowered for token in escape_attempts):
        raise RuntimeError("Sandbox escape-style shell command blocked by policy.")
    if sandbox_profile.allow_network:
        return
    blocked = ("curl ", "wget ", "ssh ", "scp ", "nc ", "ncat ", "git clone", "pip install http")
    if any(token in lowered for token in blocked):
        raise RuntimeError("Network-style shell command blocked by sandbox policy.")


def redact_secrets(text: str) -> str:
    redacted = text
    grouped_patterns = (
        r"(?i)(api[_-]?key\s*[=:]\s*)([^\s]+)",
        r"(?i)(token\s*[=:]\s*)([^\s]+)",
        r"(?i)(password\s*[=:]\s*)([^\s]+)",
        r"(?i)(secret\s*[=:]\s*)([^\s]+)",
    )
    for pattern in grouped_patterns:
        redacted = re.sub(pattern, lambda match: f"{match.group(1)}[REDACTED]", redacted)
    redacted = re.sub(r"\bsk-[A-Za-z0-9]{10,}\b", "[REDACTED]", redacted)
    return redacted


def build_terminal_sink(
    *,
    terminal_log: Path | None,
    step_index: int | None,
    emit_live: bool = False,
) -> tuple[Any, dict[str, int]]:
    counter = {"count": 0}

    def sink(stream_name: str, chunk: str) -> None:
        counter["count"] += 1
        redacted = redact_secrets(chunk)
        if emit_live:
            target = sys.stderr if stream_name == "stderr" else sys.stdout
            target.write(redacted)
            target.flush()
        if terminal_log is None:
            return
        append_jsonl(
            terminal_log,
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "step_index": step_index,
                "stream": stream_name,
                "chunk": redacted,
            },
        )

    return sink, counter
