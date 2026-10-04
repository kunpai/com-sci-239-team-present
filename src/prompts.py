"""All prompt text. The architecture is never named anywhere in here."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.candidate import Step
from src.sandbox import RunResult

TASK_PATH = Path(__file__).resolve().parent.parent / "task" / "prompt_task.md"

_LABELS = {
    "parse": "Your reply could not be parsed",
    "flags": "Compiler flags rejected",
    "compile": "Compilation failed",
    "run": "The program crashed or exited abnormally",
    "timeout": "Timed out",
    "incorrect": "Wrong result",
}


def load_task(path: Path | None = None) -> str:
    """Return the shared task statement (the default file, or `path` for an ablation)."""
    return (path or TASK_PATH).read_text().strip() + "\n"


def format_observation(result: RunResult) -> str:
    """Render a measured result the way the model sees it."""
    if result.stage == "ok":
        parts = [
            "Compiled and passed the correctness checks.",
            f"Runtime: {result.time_ms:.1f} ms.",
        ]
        if result.speedup is not None:
            parts.append(f"Speedup versus the naive triple loop: {result.speedup:.2f}x.")
        if result.stderr:
            parts.append(f"Compiler output:\n{result.stderr}")
    else:
        parts = [f"{_LABELS.get(result.stage, result.stage)}.", result.stderr]
    return "\n".join(p for p in parts if p)


def scalar_observation(result: RunResult) -> str:
    """Scalar feedback only: compiled and correct flags plus the speedup (n/a if it failed)."""
    speedup = f"{result.speedup:.2f}x" if result.success and result.speedup is not None else "n/a"
    compiled, correct = ("yes" if flag else "no" for flag in (result.compiled, result.correct))
    return f"Compiled: {compiled}. Correct: {correct}. Speedup: {speedup}."


def _render_step(index: int, step: Step, observation: str | None = None) -> str:
    lines = [
        f"### Attempt {index}",
        f"FLAGS: {' '.join(step.flags)}",
        "```c",
        step.code.rstrip("\n"),
        "```",
        f"Observation: {observation or format_observation(step.result)}",
    ]
    if step.feedback:
        lines.append(f"Feedback: {step.feedback}")
    return "\n".join(lines)


def _render_history(history: Sequence[Step]) -> str:
    return "\n\n".join(_render_step(i, s) for i, s in enumerate(history))


def generation_prompt(task: str, memory: Sequence[str]) -> str:
    """Task statement, plus a memory block when memory is non-empty."""
    if not memory:
        return task
    notes = "\n".join(f"- {entry.strip()}" for entry in memory)
    return f"{task}\nMemory from earlier sessions:\n{notes}\n"


def feedback_prompt(task: str, history: Sequence[Step]) -> str:
    """Ask for specific feedback on the latest attempt (Self-Refine FEEDBACK step)."""
    return (
        f"{task}\nBelow are your attempts so far, oldest first. Each was compiled and run; "
        f"the observations are measured.\n\n{_render_history(history)}\n\n"
        "Write specific, actionable feedback on the latest attempt: what limits its speed "
        "(memory access pattern, cache blocking, vectorization, threading, instruction choice, "
        "compiler flags) and, if it failed, why. Do not write new code. "
        "Reply with the feedback only, in under 200 words.\n"
    )


def refine_prompt(task: str, history: Sequence[Step]) -> str:
    """Ask for an improved attempt using the feedback (Self-Refine REFINE step)."""
    return (
        f"{task}\nBelow are your attempts so far, oldest first, with feedback on each.\n\n"
        f"{_render_history(history)}\n\n"
        "Using the feedback, write an improved version. Reply in the format required by the "
        "task statement: one fenced C block, then a line starting with FLAGS:.\n"
    )


def reflection_prompt(task: str, history: Sequence[Step], cap_words: int) -> str:
    """Ask for notes to a future self, distilled from a finished refinement run."""
    return (
        f"{task}\nYou just finished the series of attempts below.\n\n"
        f"{_render_history(history)}\n\n"
        "Write notes for your future self on this task: which optimizations helped and by how "
        "much, why they helped, what did not help, and what you would try next. Plain prose or "
        f"a short bullet list, at most {cap_words} words, no code blocks.\n"
    )


def shorten_prompt(text: str, cap_words: int) -> str:
    """Ask for a shorter version of over-long notes."""
    return (
        f"Shorten the following notes to at most {cap_words} words, keeping the most useful "
        f"facts. Reply with the shortened notes only.\n\n{text}\n"
    )


def failure_lesson_prompt(
    task: str, step: Step, memory: Sequence[str], cap_words: int, feedback: str = "diagnostic"
) -> str:
    """Ask for a short lesson after a failed trial (Reflexion self-reflection).

    With `feedback="scalar"` the model sees only compiled/correct flags and the speedup, not
    the failure stage or the compiler output.
    """
    observation = scalar_observation(step.result) if feedback == "scalar" else None
    notes = "\n".join(f"- {m.strip()}" for m in memory) or "(none)"
    return (
        f"{task}\nMemory you were given for this attempt:\n{notes}\n\n"
        f"Your attempt failed.\n{_render_step(0, step, observation)}\n\n"
        f"Write a short lesson, at most {cap_words} words and no code, saying what went wrong "
        "and what to do differently next time. Reply with the lesson only.\n"
    )
