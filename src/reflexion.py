"""Reflexion-style trials: attempt, evaluate, reflect on failure, retry with updated memory."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.candidate import RunFn, Step, evaluate_reply
from src.config import Config
from src.llm import LLM
from src.memory import EpisodicMemory, reflect_on_failure
from src.prompts import generation_prompt
from src.traces import TraceScope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrialOutcome:
    """All trials, the final memory, and the 1-based index of the first success (or None)."""

    trials: tuple[Step, ...]
    memory: EpisodicMemory
    first_success_trial: int | None


def reflexion_trials(
    task: str,
    llm: LLM,
    run_fn: RunFn,
    memory: EpisodicMemory,
    cfg: Config,
    trace: TraceScope,
) -> TrialOutcome:
    """Run up to `cfg.trials` trials; success means compiled and correct."""
    steps: list[Step] = []
    first_success: int | None = None
    for trial in range(1, cfg.trials + 1):
        trace.set_memory(memory.entries)
        prompt = generation_prompt(task, memory.entries)
        reply = llm(prompt)
        trace.log_llm("generate", prompt, reply, trial=trial)
        step = evaluate_reply(reply.text, run_fn)
        steps.append(step)
        trace.log_step(step, trial=trial)
        logger.info("trial %d/%d: stage=%s", trial, cfg.trials, step.result.stage)
        if step.result.success:
            first_success = trial
            break
        if trial < cfg.trials:
            lesson = reflect_on_failure(llm, task, step, memory, cfg, trace)
            memory = memory.append(lesson)
            trace.log("memory_update", trial=trial, lesson=lesson)
    trace.set_memory(memory.entries)
    return TrialOutcome(tuple(steps), memory, first_success)
