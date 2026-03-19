"""
A very small subset of the Typer API.

This module mimics enough of Typer for the ULTRA CLI without relying on the
real external dependency. It supports defining commands via decorators,
positional arguments, list-valued arguments, boolean flags, and Literal-backed
choices.
"""

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, get_args, get_origin, get_type_hints


__all__ = [
    "Typer",
    "Option",
    "Argument",
    "Exit",
    "echo",
]


def echo(message: str) -> None:
    """Print helper used by the CLI."""
    print(message)


class Exit(SystemExit):
    """Custom exit exception used to interrupt CLI execution."""


class Option:
    """Metadata wrapper used to mark keyword-style CLI options."""

    def __init__(
        self,
        default: Any = None,
        *param_decls: str,
        help: str | None = None,
        metavar: str | None = None,
    ) -> None:
        self.default = default
        self.param_decls = tuple(param_decls)
        self.help = help
        self.metavar = metavar


class Argument:
    """Metadata wrapper for positional arguments."""

    def __init__(
        self,
        default: Any = inspect._empty,
        *param_decls: str,
        help: str | None = None,
        metavar: str | None = None,
    ) -> None:
        self.default = inspect._empty if default is Ellipsis else default
        self.param_decls = tuple(param_decls)
        self.help = help
        self.metavar = metavar


@dataclass
class _ParamSpec:
    name: str
    dest: str
    kind: str


class Typer:
    """A thin decorator-based CLI inspired by Typer."""

    def __init__(self, *, help: str | None = None) -> None:
        self._parser = argparse.ArgumentParser(prog="ultra", description=help)
        self._subparsers = self._parser.add_subparsers(dest="command")
        self._commands: Dict[str, tuple[Callable[..., Any], List[_ParamSpec]]] = {}

    # decorator
    def command(self, name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            cmd_name = name or func.__name__.replace("_", "-")
            parser = self._subparsers.add_parser(cmd_name, help=func.__doc__)
            param_specs = self._register_params(parser, func)
            self._commands[cmd_name] = (func, param_specs)
            return func

        return decorator

    def _register_params(self, parser: argparse.ArgumentParser, func: Callable[..., Any]) -> List[_ParamSpec]:
        specs: List[_ParamSpec] = []
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        for param in signature.parameters.values():
            annotation = type_hints.get(param.name, param.annotation)
            param_type, choices, is_multi = self._resolve_annotation(annotation)
            option_meta: Option | Argument | None = None
            default = param.default
            if isinstance(default, (Option, Argument)):
                option_meta = default
                default = default.default

            dest = param.name.replace("-", "_")
            if default is inspect._empty and not isinstance(option_meta, Option):
                arg_names = list(option_meta.param_decls) if isinstance(option_meta, Argument) and option_meta.param_decls else [param.name]
                kwargs: Dict[str, Any] = {
                    "help": option_meta.help if isinstance(option_meta, Argument) else None,
                    "metavar": option_meta.metavar if isinstance(option_meta, Argument) else None,
                }
                if param_type is not bool:
                    kwargs["type"] = param_type
                if choices is not None:
                    kwargs["choices"] = choices
                if is_multi:
                    kwargs["nargs"] = "+"
                parser.add_argument(*arg_names, **{key: value for key, value in kwargs.items() if value is not None})
                specs.append(_ParamSpec(name=param.name, dest=param.name, kind="arg"))
                continue

            flag_names = list(option_meta.param_decls) if isinstance(option_meta, Option) and option_meta.param_decls else [f"--{param.name.replace('_', '-')}"]
            kwargs: Dict[str, Any] = {
                "help": option_meta.help if isinstance(option_meta, Option) and option_meta.help else None,
                "metavar": option_meta.metavar if isinstance(option_meta, Option) else None,
            }
            if param_type is bool:
                if default in (True, False):
                    kwargs["action"] = "store_false" if default else "store_true"
                else:
                    kwargs["type"] = bool
            else:
                kwargs["type"] = param_type
            if choices is not None:
                kwargs["choices"] = choices
            if is_multi:
                kwargs["nargs"] = "+"
            if default is not inspect._empty:
                kwargs["default"] = default
            parser.add_argument(*flag_names, **{k: v for k, v in kwargs.items() if v is not None})
            specs.append(_ParamSpec(name=param.name, dest=param.name.replace("-", "_"), kind="option"))
        return specs

    def _resolve_annotation(self, annotation: Any) -> tuple[type, list[Any] | None, bool]:
        if annotation in (inspect._empty, None):
            return str, None, False
        origin = get_origin(annotation)
        if origin is not None and str(origin) not in {"<class 'list'>", "typing.Literal"}:
            union_members = [candidate for candidate in get_args(annotation) if candidate is not type(None)]
            if len(union_members) == 1:
                return self._resolve_annotation(union_members[0])
        if origin in {list, List}:
            item_type, choices, _ = self._resolve_annotation(get_args(annotation)[0])
            return item_type, choices, True
        if origin is not None and str(origin).endswith("Literal"):
            values = list(get_args(annotation))
            if not values:
                return str, None, False
            return type(values[0]), values, False
        if annotation is bool:
            return bool, None, False
        return annotation, None, False

    def __call__(self, args: Sequence[str] | None = None) -> None:
        namespace = self._parser.parse_args(args=args)
        command = getattr(namespace, "command", None)
        if not command:
            self._parser.print_help()
            raise Exit(0)
        func, specs = self._commands[command]
        kwargs: Dict[str, Any] = {}
        for spec in specs:
            kwargs[spec.name] = getattr(namespace, spec.dest)
        func(**kwargs)
