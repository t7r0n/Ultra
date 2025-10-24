from __future__ import annotations

import json
import pathlib
from typing import Optional

import typer

from . import __version__
from .mcd.inspector import InspectionResult, inspect_model
from .tools import estimator as estimator_module
from .tools import hwcheck
from .tools import logging as run_logging
from .ultra_mode import fanout, refine, selection, structured

app = typer.Typer(name="ultra", help="Ultra Mode inference and evaluation orchestrator")


def _print_json(data: object) -> None:
    typer.echo(json.dumps(data, indent=2, sort_keys=True, default=str))


@app.callback()
def main_callback(version: Optional[bool] = typer.Option(
    None,
    "--version",
    help="Show the ULTRA CLI version and exit.",
    is_flag=True,
    callback=lambda value: typer.echo(__version__) if value else None,
)) -> None:
    """Callback to expose version flag."""
    if version:
        raise typer.Exit()


@app.command()
def doctor(json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of text.")) -> None:
    """Inspect the current machine for Ultra Mode compatibility."""
    report = hwcheck.collect_system_report()
    if json_output:
        _print_json(report)
    else:
        typer.echo(hwcheck.format_report(report))


@app.command()
def fetch(
    model_ref: str = typer.Argument(..., help="HF repo id or local GGUF path."),
    revision: Optional[str] = typer.Option(None, "--revision", help="Specific hub revision."),
    local_dir: Optional[pathlib.Path] = typer.Option(None, "--local-dir", file_okay=True, dir_okay=True, exists=False),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code", help="Allow execution of remote code."),
) -> None:
    """Download model weights and configs to a local directory."""
    resolved = hwcheck.fetch_model_snapshot(
        model_ref=model_ref,
        revision=revision,
        local_dir=local_dir,
        trust_remote_code=trust_remote_code,
    )
    typer.echo(str(resolved))


@app.command()
def inspect(
    model_ref: str = typer.Argument(..., help="HF repo id or local directory."),
    backend: str = typer.Option("auto", "--backend", help="Backend hint."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Perform model discovery and introspection."""
    result = inspect_model(model_ref=model_ref, preferred_backend=backend)
    if json_output:
        _print_json(result.to_dict())
    else:
        typer.echo(result.pretty())


@app.command()
def estimate(
    model_ref: str = typer.Argument(..., help="Model reference (local config directory)."),
    ctx: int = typer.Option(8192, "--ctx", help="Total context length (prompt + generated)."),
    batch: int = typer.Option(1, "--batch", help="Batch size."),
    precision: str = typer.Option("auto", "--precision", help="Precision override."),
    backend: str = typer.Option("auto", "--backend", help="Backend hint."),
) -> None:
    """Estimate resource requirements for a model."""
    inspection = inspect_model(model_ref=model_ref, preferred_backend=backend)
    estimate_result = estimator_module.estimate_memory(
        inspection=inspection,
        context=ctx,
        batch_size=batch,
        precision=precision,
        backend=backend,
    )
    typer.echo(estimator_module.format_estimate(estimate_result))


@app.command()
def chat(
    model_ref: str = typer.Argument(..., help="Model reference for chat."),
    backend: str = typer.Option("auto", "--backend", help="Backend hint."),
    ultra_profile: str = typer.Option("default", "--ultra-profile", help="Ultra profile preset."),
    config_path: Optional[pathlib.Path] = typer.Option(None, "--config", exists=True, file_okay=True, dir_okay=False),
    logdir: Optional[pathlib.Path] = typer.Option(None, "--logdir", file_okay=False, dir_okay=True),
) -> None:
    """Run the Ultra Mode inference pipeline interactively."""
    inspection = inspect_model(model_ref=model_ref, preferred_backend=backend)
    profile = fanout.load_ultra_profile(config_path=config_path, profile_name=ultra_profile)
    prompts = structured.collect_interactive_messages()
    candidates = fanout.generate_candidates(inspection, profile, backend)
    ranked = selection.select_candidate(candidates, profile, inspection)
    final = refine.refine_candidate(ranked.best_candidate, inspection, profile)
    run_logging.persist_chat_run(
        logdir=logdir,
        inspection=inspection,
        profile=profile,
        prompts=prompts,
        candidates=candidates,
        selection=ranked,
        final=final,
    )
    typer.echo(final.content)


@app.command()
def agent(
    model_ref: str = typer.Argument(..., help="Model reference for agent mode."),
    backend: str = typer.Option("auto", "--backend"),
    open_terminal: bool = typer.Option(False, "--open-terminal"),
    sandbox: str = typer.Option("docker", "--sandbox", help="Sandbox implementation."),
) -> None:
    """Launch the agentic coding loop."""
    inspection = inspect_model(model_ref=model_ref, preferred_backend=backend)
    profile = fanout.load_ultra_profile(config_path=None, profile_name="coding")
    sandbox_controller = run_logging.prepare_agent_run(open_terminal=open_terminal, sandbox=sandbox)
    typer.echo(
        selection.bootstrap_agentic_session(
            inspection=inspection,
            profile=profile,
            sandbox=sandbox_controller,
        )
    )


@app.command()
def eval(
    model_ref: str = typer.Argument(..., help="Model reference."),
    suite: str = typer.Option("harness", "--suite", help="Evaluation suite."),
    tasks: Optional[str] = typer.Option(None, "--tasks", help="Comma separated tasks."),
    backend: str = typer.Option("auto", "--backend"),
    out: Optional[pathlib.Path] = typer.Option(None, "--out", help="Output path for metrics."),
) -> None:
    """Run evaluation harnesses through the Ultra pipeline."""
    inspection = inspect_model(model_ref=model_ref, preferred_backend=backend)
    runner = selection.build_evaluation_runner(suite)
    results = runner(
        inspection=inspection,
        tasks=[task.strip() for task in tasks.split(",")] if tasks else None,
        backend=backend,
    )
    if out:
        out.write_text(json.dumps(results, indent=2, sort_keys=True))
    _print_json(results)
