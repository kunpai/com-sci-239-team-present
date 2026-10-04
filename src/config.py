"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace

FEEDBACK_MODES = ("diagnostic", "scalar")


@dataclass(frozen=True)
class Config:
    """Immutable settings for one experiment run."""

    rounds: int = 10  # Self-Refine feedback/refine rounds in arm A
    repeats: int = 3  # independent A -> B/C repeats
    trials: int = 3  # max Reflexion trials in arms B and C
    window: int = 3  # episodic memory sliding window (Reflexion's Omega)
    memory_cap_words: int = 200
    lesson_cap_words: int = 80
    n: int = 1024  # matrix size
    reps: int = 3  # timing repetitions inside the harness (best-of)
    cpus: int = 4
    mem_limit: str = "4g"
    llm_timeout_s: int = 300
    compile_timeout_s: int = 300
    run_timeout_s: int = 1800
    max_naive_s: float = 120.0  # preflight aborts if the naive run exceeds this
    model: str | None = None
    # what Reflexion's failure reflection sees: "diagnostic" = stage + compiler output,
    # "scalar" = only compiled yes/no, correct yes/no and the speedup (no stage, no stderr)
    feedback: str = "diagnostic"

    def __post_init__(self) -> None:
        if self.feedback not in FEEDBACK_MODES:
            raise ValueError(f"feedback must be one of {FEEDBACK_MODES}, got {self.feedback!r}")

    def quick(self) -> Config:
        """Return a copy sized for a fast end-to-end check."""
        return replace(self, rounds=3, repeats=1)
