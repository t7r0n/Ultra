from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultra.agent_loop import enforce_shell_policy, redact_secrets, resolve_workspace_path, run_agent, run_workspace_verification
from ultra.config import GlobalConfig
from ultra.tools.logging import initialize_run_artifacts
from ultra.tools.sandbox import SandboxResult, build_sandbox_profile


class AgentLoopTests(unittest.TestCase):
    def test_resolve_workspace_path_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            with self.assertRaises(RuntimeError):
                resolve_workspace_path(workspace, "../outside.txt")

    def test_workspace_verification_passes_when_no_checks_are_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_workspace_verification(
                workspace=Path(tmpdir),
                sandbox_profile=build_sandbox_profile("docker"),
            )

        self.assertTrue(summary.passed)
        self.assertEqual(summary.checks, [])

    def test_agent_retries_once_after_failed_final_verification(self) -> None:
        config = GlobalConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "agent-run"
            artifacts = initialize_run_artifacts(
                command="agent",
                model_ref="fake/model",
                backend="hf",
                config=config.to_dict(),
                hardware=None,
                logdir=run_dir,
                extra={},
                structured_output=False,
            )
            fake_turns = [
                types.SimpleNamespace(
                    final_payload={"action": "write_file", "path": "main.py", "content": "print('hi')\n"},
                    final_text='{"action":"write_file"}',
                ),
                types.SimpleNamespace(
                    final_payload={"action": "final", "final": "done", "verify_kind": "compile"},
                    final_text='{"action":"final"}',
                ),
                types.SimpleNamespace(
                    final_payload={"action": "final", "final": "fixed", "verify_kind": "compile"},
                    final_text='{"action":"final"}',
                ),
            ]
            sandbox_results = [
                SandboxResult(command=["python -m compileall -q ."], returncode=0, stdout="", stderr=""),
                SandboxResult(command=["python -m compileall -q ."], returncode=1, stdout="", stderr="compile failed"),
                SandboxResult(command=["python -m compileall -q ."], returncode=0, stdout="", stderr=""),
            ]
            with patch("ultra.agent_loop.run_ultra_turn", side_effect=fake_turns), patch(
                "ultra.agent_loop.run_in_sandbox",
                side_effect=sandbox_results,
            ):
                result = run_agent(
                    model_ref="fake/model",
                    backend="hf",
                    config=config,
                    goal="fix the file",
                    artifacts=artifacts,
                    sandbox_profile=build_sandbox_profile("docker"),
                    open_terminal=False,
                    max_steps=4,
                )

            metrics = (run_dir / "metrics.json").read_text(encoding="utf-8")

        self.assertEqual(result.final, "fixed")
        self.assertIn('"reflexive_retries_used": 1', metrics)

    def test_shell_policy_blocks_network_commands_without_network_access(self) -> None:
        with self.assertRaises(RuntimeError):
            enforce_shell_policy("curl https://example.com", sandbox_profile=build_sandbox_profile("docker"))

    def test_shell_policy_blocks_escape_commands_even_with_network_allowed(self) -> None:
        with self.assertRaises(RuntimeError):
            enforce_shell_policy("sudo bash -lc 'id'", sandbox_profile=build_sandbox_profile("docker", allow_network=True))

    def test_redact_secrets_masks_common_secret_patterns(self) -> None:
        redacted = redact_secrets("api_key=abcd1234 token: secretvalue sk-abcdefghijklmnop")
        self.assertIn("api_key=[REDACTED]", redacted)
        self.assertIn("token: [REDACTED]", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_agent_open_terminal_writes_redacted_terminal_log(self) -> None:
        config = GlobalConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "agent-run"
            artifacts = initialize_run_artifacts(
                command="agent",
                model_ref="fake/model",
                backend="hf",
                config=config.to_dict(),
                hardware=None,
                logdir=run_dir,
                extra={},
                structured_output=False,
            )
            fake_turns = [
                types.SimpleNamespace(
                    final_payload={"action": "shell", "command": "printf hi"},
                    final_text='{"action":"shell"}',
                ),
                types.SimpleNamespace(
                    final_payload={"action": "final", "final": "done"},
                    final_text='{"action":"final"}',
                ),
            ]

            def fake_run_in_sandbox(*args, **kwargs):
                del args
                sink = kwargs.get("stream_sink")
                if sink is not None:
                    sink("stdout", "hello from shell\n")
                    sink("stderr", "token=supersecret\n")
                return SandboxResult(
                    command=["printf hi"],
                    returncode=0,
                    stdout="hello from shell\n",
                    stderr="token=supersecret\n",
                    streamed=True,
                )

            with patch("ultra.agent_loop.run_ultra_turn", side_effect=fake_turns), patch(
                "ultra.agent_loop.run_in_sandbox",
                side_effect=fake_run_in_sandbox,
            ):
                result = run_agent(
                    model_ref="fake/model",
                    backend="hf",
                    config=config,
                    goal="run a command",
                    artifacts=artifacts,
                    sandbox_profile=build_sandbox_profile("docker"),
                    open_terminal=True,
                    max_steps=3,
                )

            terminal_entries = [
                json.loads(line)
                for line in (run_dir / "terminal.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(result.final, "done")
        self.assertEqual(len(terminal_entries), 2)
        self.assertEqual(metrics_payload["terminal_events"], 2)
        self.assertEqual(metrics_payload["terminal_log"], str(run_dir / "terminal.jsonl"))
        self.assertEqual(terminal_entries[0]["stream"], "stdout")
        self.assertIn("[REDACTED]", terminal_entries[1]["chunk"])
