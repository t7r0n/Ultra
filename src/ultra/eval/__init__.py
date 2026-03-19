"""Evaluation suite scaffolds."""

from .arena import SUITE_NAME as ARENA_SUITE_NAME, describe_suite as describe_arena_suite
from .code import SUITE_NAME as CODE_SUITE_NAME, describe_suite as describe_code_suite
from .harness import SUITE_NAME as HARNESS_SUITE_NAME, describe_suite as describe_harness_suite
from .longctx import SUITE_NAME as LONG_SUITE_NAME, describe_suite as describe_long_suite
from .rag import SUITE_NAME as RAG_SUITE_NAME, describe_suite as describe_rag_suite

__all__ = [
    "ARENA_SUITE_NAME",
    "CODE_SUITE_NAME",
    "HARNESS_SUITE_NAME",
    "LONG_SUITE_NAME",
    "RAG_SUITE_NAME",
    "describe_arena_suite",
    "describe_code_suite",
    "describe_harness_suite",
    "describe_long_suite",
    "describe_rag_suite",
]
