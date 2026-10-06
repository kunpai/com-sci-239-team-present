"""Aggregate records into results.json, results.md, a per-round plot and a traces README."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Config
from src.refine import RefineOutcome
from src.reflexion import TrialOutcome


@dataclass(frozen=True)
class RefineRecord:
    """Arm A summary for one repeat; `speedups[i]` is attempt i (None if it failed)."""

    repeat: int
    speedups: tuple[float | None, ...]
    best_speedup: float | None
    memory: str | None
    note: str = ""


@dataclass(frozen=True)
class ArmRecord:
    """Arm B, C or D summary for one repeat."""

    arm: str
    repeat: int
    first_try_success: bool
    trials_to_success: int | None
    final_speedup: float | None
    first_failure_stage: str | None
    n_steps: int
    skipped: bool = False
    note: str = ""


def summarize_refine(repeat: int, outcome: RefineOutcome, memory: str | None) -> RefineRecord:
    """Summarize an arm A run."""
    speedups = tuple(s.result.speedup if s.result.success else None for s in outcome.history)
    best = outcome.best.result.speedup if outcome.best else None
    note = "" if memory else "no correct candidate on amd64; no memory written"
    return RefineRecord(repeat, speedups, best, memory, note)


def summarize_trials(arm: str, repeat: int, outcome: TrialOutcome) -> ArmRecord:
    """Summarize an arm B, C or D run."""
    first = outcome.trials[0]
    idx = outcome.first_success_trial
    winner = outcome.trials[idx - 1] if idx else None
    return ArmRecord(
        arm=arm,
        repeat=repeat,
        first_try_success=first.result.success,
        trials_to_success=idx,
        final_speedup=winner.result.speedup if winner else None,
        first_failure_stage=None if first.result.success else first.result.stage,
        n_steps=len(outcome.trials),
    )


def skipped_arm(arm: str, repeat: int, note: str) -> ArmRecord:
    """Placeholder for an arm that could not run (for example C without a memory)."""
    return ArmRecord(arm, repeat, False, None, None, None, 0, skipped=True, note=note)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _rate(records: Sequence[ArmRecord]) -> str:
    done = [r for r in records if not r.skipped]
    if not done:
        return "n/a"
    hits = sum(r.first_try_success for r in done)
    return f"{hits}/{len(done)}"


def _mean_trials(records: Sequence[ArmRecord]) -> str:
    """Mean trials over repeats that succeeded, always stating how many repeats that was."""
    done = [r for r in records if not r.skipped]
    wins = [r.trials_to_success for r in done if r.trials_to_success]
    if not done:
        return "n/a"
    if not wins:
        return f"never (0/{len(done)} succeeded)"
    text = f"{sum(wins) / len(wins):.2f}"
    return text if len(wins) == len(done) else f"{text} ({len(wins)}/{len(done)} succeeded)"


def _plot(path: Path, refine: Sequence[RefineRecord]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for rec in refine:
        ys = [math.nan if v is None else v for v in rec.speedups]
        ax.plot(range(len(ys)), ys, marker="o", label=f"repeat {rec.repeat}")
    ax.set_xlabel("Self-Refine round (0 = first draft)")
    ax.set_ylabel("speedup vs naive (amd64)")
    ax.set_title("Arm A: speedup per round (gaps = failed attempt)")
    if refine:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _results_md(
    cfg: Config,
    naive_ms: Mapping[str, float],
    refine: Sequence[RefineRecord],
    arms: Sequence[ArmRecord],
) -> str:
    naive = ", ".join(f"{k}={v:.1f}" for k, v in naive_ms.items())
    lines = [
        "# Results",
        "",
        (
            f"Config: n={cfg.n}, rounds={cfg.rounds}, repeats={cfg.repeats}, "
            f"trials={cfg.trials}, window={cfg.window}. Naive baseline (ms): {naive}"
        ),
        "",
        "## Arm A: Self-Refine on amd64",
        "",
        "| repeat | best speedup | speedup per attempt | note |",
        "|---|---|---|---|",
    ]
    for r in refine:
        per = ", ".join(_fmt(v) for v in r.speedups)
        lines.append(f"| {r.repeat} | {_fmt(r.best_speedup)} | {per} | {r.note} |")
    lines += ["", "## Arms B (no memory) and C (x86 memory) on arm/v7; D (x86 memory) on amd64", ""]
    lines += ["| arm | First-try success | mean trials to success |", "|---|---|---|"]
    for arm in (name for name in ("B", "C", "D") if any(a.arm == name for a in arms)):
        recs = [a for a in arms if a.arm == arm]
        lines.append(f"| {arm} | {_rate(recs)} | {_mean_trials(recs)} |")
    lines += [
        "",
        "### Per repeat",
        "",
        (
            "| arm | repeat | first try | trials to success | final speedup | "
            "first failure stage | note |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        lines.append(
            f"| {a.arm} | {a.repeat} | {'yes' if a.first_try_success else 'no'} | "
            f"{a.trials_to_success or '-'} | {_fmt(a.final_speedup)} | "
            f"{a.first_failure_stage or '-'} | {a.note} |"
        )
    return "\n".join(lines) + "\n"


def _arm_line(a: ArmRecord) -> str:
    if a.skipped:
        return f"- {a.arm}/r{a.repeat}: skipped. {a.note}"
    if a.first_try_success:
        return f"- {a.arm}/r{a.repeat}: succeeded on the first try."
    outcome = f"first try failed at stage '{a.first_failure_stage}'"
    if a.trials_to_success:
        tail = f"; succeeded at trial {a.trials_to_success} (speedup {_fmt(a.final_speedup)}x)"
    else:
        tail = "; never succeeded"
    return f"- {a.arm}/r{a.repeat}: {outcome}{tail}."


def _readme(run_dir: Path, refine: Sequence[RefineRecord], arms: Sequence[ArmRecord]) -> str:
    lines = [
        f"# Traces for {run_dir.name}",
        "",
        "Layout: `<arm>/r<repeat>/<NN>-<kind>.json` (kinds: `llm`, `sandbox`, `memory_update`, ...).",
        "`A` = Self-Refine on amd64 (its `memory.md` is the distilled memory), `B` = arm/v7 trials",
        "without memory, `C` = arm/v7 trials with A's memory, `D` = amd64 trials with A's memory.",
        "",
        "## Scenarios",
        "",
    ]
    for r in refine:
        lines.append(
            f"- A/r{r.repeat}: best speedup {_fmt(r.best_speedup)}x over "
            f"{len(r.speedups)} attempts. {r.note}".rstrip()
        )
    lines += [_arm_line(a) for a in arms]
    return "\n".join(lines) + "\n"


def write_report(
    run_dir: Path,
    cfg: Config,
    naive_ms: Mapping[str, float],
    refine: Sequence[RefineRecord],
    arms: Sequence[ArmRecord],
) -> None:
    """Write results.json, results.md, speedup_by_round.png and README.md into `run_dir`."""
    payload = {
        "config": asdict(cfg),
        "naive_ms": dict(naive_ms),
        "refine": [asdict(r) for r in refine],
        "arms": [asdict(a) for a in arms],
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (run_dir / "results.md").write_text(_results_md(cfg, naive_ms, refine, arms))
    (run_dir / "README.md").write_text(_readme(run_dir, refine, arms))
    _plot(run_dir / "speedup_by_round.png", refine)
