from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .mcd.inspector import ModelDiscoveryError, inspect_model
from .tools import estimator, hwcheck, logging as run_logging
from .ultra_mode import fanout, refine, selection, structured

app = typer.Typer(name="ultra", help="Ultra Mode inference and evaluation orchestrator")


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@app.callback()
def main_callback(version: Optional[bool] = typer.Option(None, "--version", is_flag=True)) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def doctor(json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of text.")) -> None:
    report = hwcheck.collect_system_report()
    if json_output:
        _echo_json(report)
    else:
        typer.echo(hwcheck.format_report(report))


@app.command()
def fetch(
    model_ref: str = typer.Argument(..., help="HF repo id or local GGUF path."),
    revision: Optional[str] = typer.Option(None, "--revision"),
    local_dir: Optional[Path] = typer.Option(None, "--local-dir", file_okay=True, dir_okay=True),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    payload = hwcheck.fetch_model_snapshot(
        model_ref=model_ref,
        revision=revision,
        local_dir=local_dir,
        trust_remote_code=trust_remote_code,
    )
    if json_output:
        _echo_json(payload)
    else:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def inspect(
    model_ref: str = typer.Argument(..., help="HF repo id or local directory."),
    backend: str = typer.Option("auto", "--backend", help="Backend hint."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    try:
        result = inspect_model(model_ref=model_ref, preferred_backend=backend)
    except ModelDiscoveryError as exc:
        raise typer.Exit(code=1) from exc
    if json_output:
        _echo_json(result.to_dict())
    else:
        typer.echo(result.pretty())


@app.command()
def estimate(
    model_ref: str = typer.Argument(..., help="Model reference (local config directory)."),
    ctx: int = typer.Option(8192, "--ctx", help="Total context length."),
    batch: int = typer.Option(1, "--batch", help="Batch size."),
    precision: str = typer.Option("auto", "--precision", help="Precision override."),
    backend: str = typer.Option("auto", "--backend", help="Backend hint."),
) -> None:
    inspection = inspect_model(model_ref=model_ref, preferred_backend=backend)
    breakdown = estimator.estimate_memory(
        inspection=inspection,
        context=ctx,
        batch_size=batch,
        precision=precision,
        backend=backend,
    )
    typer.echo(estimator.format_estimate(breakdown))


@app.command()
def chat(
    model_ref: str = typer.Argument(..., help="Model reference for chat."),
    backend: str = typer.Option("auto", "--backend", help="Backend hint."),
    ultra_profile: str = typer.Option("default", "--ultra-profile", help="Ultra profile preset."),
    config_path: Optional[Path] = typer.Option(None, "--config", exists=True, file_okay=True, dir_okay=False),
    logdir: Optional[Path] = typer.Option(None, "--logdir", file_okay=False, dir_okay=True),
) -> None:
    inspection = inspect_model(model_ref=model_ref, preferred_backend=backend)
    profile = fanout.load_ultra_profile(config_path=config_path, profile_name=ultra_profile)
    messages = structured.collect_interactive_messages()
    candidates = fanout.generate_candidates(
        inspection=inspection,
        profile=profile,
        backend=backend,
        messages=messages,
    )
    selection_result = selection.select_candidate(
        candidates=candidates,
        profile=profile,
        inspection=inspection,
    )
    refined = refine.refine_candidate(selection_result.candidate, inspection, profile)
    run_logging.persist_chat_run(
        logdir=logdir,
        inspection=inspection,
        profile=profile,
        prompts=messages,
        candidates=candidates,
        selection=selection_result,
        final=refined,
    )
    payload = structured.enforce_structure(inspection=inspection, profile=profile, candidate=selection_result.candidate)
    if profile.structured_output.enabled:
        _echo_json(payload)
    else:
        typer.echo(refined.content)


@app.command()
def agent(
    model_ref: str = typer.Argument(..., help="Model reference for agent mode."),
    backend: str = typer.Option("auto", "--backend"),
    open_terminal: bool = typer.Option(False, "--open-terminal"),
    sandbox: str = typer.Option("docker", "--sandbox", help="Sandbox implementation."),
) -> None:
    inspection = inspect_model(model_ref=model_ref, preferred_backend=backend)
    profile = fanout.load_ultra_profile(config_path=None, profile_name="coding")
    sandbox_env = run_logging.prepare_agent_run(open_terminal=open_terminal, sandbox=sandbox)
    typer.echo(
        selection.bootstrap_agentic_session(
            inspection=inspection,
            profile=profile,
            sandbox=sandbox_env,
        )
    )


@app.command()
def eval(
    model_ref: str = typer.Argument(..., help="Model reference."),
    suite: str = typer.Option("harness", "--suite", help="Evaluation suite."),
    tasks: Optional[str] = typer.Option(None, "--tasks", help="Comma separated tasks."),
    backend: str = typer.Option("auto", "--backend"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output path for metrics."),
) -> None:
    inspection = inspect_model(model_ref=model_ref, preferred_backend=backend)
    runner = selection.build_evaluation_runner(suite)
    task_list = [task.strip() for task in tasks.split(",")] if tasks else None
    results = runner(inspection=inspection, tasks=task_list, backend=backend)
    if out:
        out.write_text(json.dumps(results, indent=2, sort_keys=True))
    _echo_json(results)

