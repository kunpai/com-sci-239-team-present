"""Parsing Claude's replies into candidates and evaluating them."""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.sandbox import RunResult

_FENCE = re.compile(r"```[A-Za-z]*[ \t]*\n(.*?)```", re.DOTALL)
_FLAGS = re.compile(r"^[ \t]*FLAGS:[ \t]*(.*?)[ \t]*$", re.MULTILINE)
_RAW_KEEP = 2000


class ParseError(ValueError):
    """Raised when a reply has no usable code block or FLAGS line."""


@dataclass(frozen=True)
class Candidate:
    """A parsed reply: source code plus compiler flags."""

    code: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class Step:
    """One evaluated attempt, plus the critique written about it (Self-Refine)."""

    code: str
    flags: tuple[str, ...]
    result: RunResult
    feedback: str | None = None


RunFn = Callable[[str, tuple[str, ...]], RunResult]


def parse_candidate(text: str) -> Candidate:
    """Take the last fenced block that defines matmul and the last FLAGS line."""
    blocks = [b for b in _FENCE.findall(text) if "matmul" in b]
    if not blocks:
        raise ParseError("no fenced code block defining matmul found in the reply")
    flag_lines = _FLAGS.findall(text)
    if not flag_lines:
        raise ParseError("no 'FLAGS:' line found in the reply")
    try:
        flags = tuple(shlex.split(flag_lines[-1]))
    except ValueError as exc:
        raise ParseError(f"could not parse the FLAGS line: {exc}") from exc
    return Candidate(blocks[-1].strip("\n") + "\n", flags)


def evaluate_reply(text: str, run_fn: RunFn) -> Step:
    """Parse a reply and run it; parse failures become failed steps, never exceptions."""
    try:
        cand = parse_candidate(text)
    except ParseError as exc:
        failed = RunResult("parse", False, False, None, None, str(exc))
        return Step(code=text[:_RAW_KEEP], flags=(), result=failed)
    return Step(cand.code, cand.flags, run_fn(cand.code, cand.flags))


def best_step(steps: Sequence[Step]) -> Step | None:
    """Fastest correct step (highest speedup), or None when nothing is correct."""
    correct = [s for s in steps if s.result.success]
    if not correct:
        return None
    return max(correct, key=lambda s: s.result.speedup if s.result.speedup is not None else 0.0)
