"""Verbal episodic memory: writing it from a refinement run and from failed trials."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from src.candidate import Step
from src.config import Config
from src.llm import LLM
from src.prompts import failure_lesson_prompt, reflection_prompt, shorten_prompt
from src.traces import TraceScope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodicMemory:
    """Sliding window of verbal notes (Reflexion's Omega)."""

    entries: tuple[str, ...] = ()
    window: int = 3

    def append(self, entry: str) -> EpisodicMemory:
        """Return a new memory with `entry` added and the oldest entries dropped."""
        return EpisodicMemory((*self.entries, entry)[-self.window :], self.window)


def clip_words(text: str, cap: int) -> str:
    """Trim whitespace and keep at most `cap` words."""
    words = text.split()
    return text.strip() if len(words) <= cap else " ".join(words[:cap])


def write_memory(
    llm: LLM, task: str, history: Sequence[Step], cfg: Config, trace: TraceScope
) -> str:
    """Distill a refinement run into notes; one shorten retry, then a hard clip."""
    prompt = reflection_prompt(task, history, cfg.memory_cap_words)
    reply = llm(prompt)
    trace.log_llm("memory_reflection", prompt, reply)
    text = reply.text.strip()
    if len(text.split()) > cfg.memory_cap_words:
        prompt = shorten_prompt(text, cfg.memory_cap_words)
        reply = llm(prompt)
        trace.log_llm("memory_shorten", prompt, reply)
        text = reply.text.strip()
    if len(text.split()) > cfg.memory_cap_words:
        trace.log("memory_truncated", words=len(text.split()), cap=cfg.memory_cap_words)
        text = clip_words(text, cfg.memory_cap_words)
    return text


def reflect_on_failure(
    llm: LLM, task: str, step: Step, memory: EpisodicMemory, cfg: Config, trace: TraceScope
) -> str:
    """Write a short lesson about a failed trial (Reflexion self-reflection)."""
    prompt = failure_lesson_prompt(
        task, step, memory.entries, cfg.lesson_cap_words, cfg.feedback
    )
    reply = llm(prompt)
    trace.log_llm("lesson", prompt, reply)
    return clip_words(reply.text, cfg.lesson_cap_words)
