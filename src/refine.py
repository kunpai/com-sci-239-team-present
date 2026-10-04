"""Self-Refine: generate, then repeat (feedback call, refine call) for a fixed number of rounds."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from src.candidate import RunFn, Step, best_step, evaluate_reply
from src.config import Config
from src.llm import LLM
from src.prompts import feedback_prompt, generation_prompt, refine_prompt
from src.traces import TraceScope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefineOutcome:
    """Full attempt history and the fastest correct attempt (None if nothing was correct)."""

    history: tuple[Step, ...]
    best: Step | None


def self_refine(
    task: str, llm: LLM, run_fn: RunFn, cfg: Config, trace: TraceScope
) -> RefineOutcome:
    """Run one Self-Refine episode; every call and run is traced."""
    prompt = generation_prompt(task, ())
    reply = llm(prompt)
    trace.log_llm("generate", prompt, reply, round_no=0)
    history = [evaluate_reply(reply.text, run_fn)]
    trace.log_step(history[-1], round_no=0)

    for round_no in range(1, cfg.rounds + 1):
        prompt = feedback_prompt(task, history)
        reply = llm(prompt)
        trace.log_llm("feedback", prompt, reply, round_no=round_no)
        history[-1] = replace(history[-1], feedback=reply.text.strip())

        prompt = refine_prompt(task, history)
        reply = llm(prompt)
        trace.log_llm("refine", prompt, reply, round_no=round_no)
        history.append(evaluate_reply(reply.text, run_fn))
        trace.log_step(history[-1], round_no=round_no)
        logger.info(
            "round %d/%d: stage=%s speedup=%s",
            round_no, cfg.rounds, history[-1].result.stage, history[-1].result.speedup,
        )  # fmt: skip

    return RefineOutcome(tuple(history), best_step(history))
