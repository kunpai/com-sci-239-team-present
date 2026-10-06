"""Run the Self-Refine -> memory -> Reflexion transfer experiment end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform as pyplatform
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

from src.candidate import RunFn
from src.config import FEEDBACK_MODES, Config
from src.llm import LLM, ClaudeCodeLLM, LLMError
from src.memory import EpisodicMemory, write_memory
from src.prompts import TASK_PATH, load_task
from src.refine import self_refine
from src.reflexion import reflexion_trials
from src.report import (
    ArmRecord,
    RefineRecord,
    skipped_arm,
    summarize_refine,
    summarize_trials,
    write_report,
)
from src.sandbox import (
    AMD64,
    ARMV7,
    Platform,
    SandboxError,
    build_image,
    docker_available,
    measure_naive,
    run_candidate,
)
from src.traces import TraceWriter

logger = logging.getLogger("run")

RunFnFor = Callable[[Platform], RunFn]


def _version(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    return proc.stdout.strip() or proc.stderr.strip() or "unavailable"


def _repo_relative(path: Path) -> str:
    """Path relative to this repo when inside it, so traces do not record a home directory."""
    try:
        return str(path.resolve().relative_to(Path(__file__).resolve().parent))
    except ValueError:
        return str(path)


def write_run_meta(
    run_dir: Path,
    cfg: Config,
    naive_ms: dict[str, float],
    task_path: Path,
    task: str,
    memory_source: Path | None = None,
    arm_d: bool = False,
) -> None:
    """Record config, task statement and environment for reproducibility."""
    meta = {
        "config": asdict(cfg),
        "naive_ms": naive_ms,
        "task_file": _repo_relative(task_path),
        "task_sha256": hashlib.sha256(task.encode()).hexdigest(),
        "memory_source": _repo_relative(memory_source) if memory_source else None,
        "target": (AMD64 if arm_d else ARMV7).name,
        "arms": ["D"] if arm_d else ["B", "C"],
        "python": pyplatform.python_version(),
        "claude": _version(["claude", "--version"]),
        "docker": _version(["docker", "--version"]),
        "host": pyplatform.platform(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


def run_repeat(
    repeat: int,
    cfg: Config,
    llm: LLM,
    writer: TraceWriter,
    run_fn_for: RunFnFor,
    task: str,
    memory_source: Path | None = None,
    arm_d: bool = False,
) -> tuple[RefineRecord | None, list[ArmRecord]]:
    """One repeat: arm A, then arm B (no memory) and arm C (A's memory) on arm/v7.

    With `memory_source` arm A is not run: its memory is read from that earlier run instead.
    With `arm_d` only arm D runs: A's memory on amd64, the platform it came from.
    """
    a_scope = writer.scope("A", repeat, AMD64.name)
    memory_text: str | None = None
    refine_record: RefineRecord | None = None
    if memory_source is not None:
        memory_text = (memory_source / "A" / f"r{repeat}" / "memory.md").read_text().strip()
        (a_scope.directory / "memory.md").write_text(memory_text + "\n")
        a_scope.log("memory_reused", source=str(memory_source))
    else:
        outcome = self_refine(task, llm, run_fn_for(AMD64), cfg, a_scope)
        if outcome.best is not None:
            memory_text = write_memory(llm, task, outcome.history, cfg, a_scope)
            (a_scope.directory / "memory.md").write_text(memory_text + "\n")
        refine_record = summarize_refine(repeat, outcome, memory_text)

    arms: list[ArmRecord] = []
    if arm_d:
        arms.append(_memory_arm("D", repeat, task, llm, run_fn_for(AMD64), memory_text, cfg, writer))
        return refine_record, arms
    b_out = reflexion_trials(
        task, llm, run_fn_for(ARMV7), EpisodicMemory((), cfg.window), cfg,
        writer.scope("B", repeat, ARMV7.name),
    )  # fmt: skip
    arms.append(summarize_trials("B", repeat, b_out))
    arms.append(_memory_arm("C", repeat, task, llm, run_fn_for(ARMV7), memory_text, cfg, writer))
    return refine_record, arms


def _memory_arm(
    arm: str,
    repeat: int,
    task: str,
    llm: LLM,
    run_fn: RunFn,
    memory_text: str | None,
    cfg: Config,
    writer: TraceWriter,
) -> ArmRecord:
    """Reflexion trials seeded with A's memory (arm C on arm/v7, arm D on amd64)."""
    if memory_text is None:
        return skipped_arm(arm, repeat, "no correct candidate on amd64, so no memory to transfer")
    platform = AMD64 if arm == "D" else ARMV7
    out = reflexion_trials(
        task, llm, run_fn, EpisodicMemory((memory_text,), cfg.window), cfg,
        writer.scope(arm, repeat, platform.name),
    )  # fmt: skip
    return summarize_trials(arm, repeat, out)


def _source_repeats(source: Path, only: Sequence[int] | None = None) -> list[int]:
    """Repeat numbers for which `source` stored an arm A memory (optionally just `only`)."""
    found = sorted(int(p.parent.name[1:]) for p in source.glob("A/r[0-9]*/memory.md"))
    if not found:
        raise FileNotFoundError(f"no A/r*/memory.md under {source}: no memory to reuse")
    if only is None:
        return found
    missing = sorted(set(only) - set(found))
    if missing:
        raise FileNotFoundError(f"repeat {missing} has no stored memory under {source}")
    return sorted(set(only))


def run_experiment(
    cfg: Config,
    llm: LLM,
    run_fn_for: RunFnFor,
    naive_ms: dict[str, float],
    out_dir: Path,
    run_id: str,
    task_path: Path | None = None,
    memory_source: Path | None = None,
    arm_d: bool = False,
    only_repeats: Sequence[int] | None = None,
) -> Path:
    """Run all repeats, tolerating LLM failures per repeat, and write the report.

    With `memory_source` (an earlier run directory) the repeats are the ones that run stored a
    memory for (or `only_repeats` among them), and arm A is not re-run. With `arm_d` the
    only arm run is D: A's memory on amd64.
    """
    task = load_task(task_path)
    repeats = (
        _source_repeats(memory_source, only_repeats)
        if memory_source
        else list(range(1, cfg.repeats + 1))
    )
    writer = TraceWriter(out_dir, run_id)
    write_run_meta(
        writer.run_dir, cfg, naive_ms, task_path or TASK_PATH, task, memory_source, arm_d
    )
    refine_records: list[RefineRecord] = []
    arm_records: list[ArmRecord] = []
    for repeat in repeats:
        logger.info("repeat %d (%d total)", repeat, len(repeats))
        try:
            refine_record, arms = run_repeat(
                repeat, cfg, llm, writer, run_fn_for, task, memory_source, arm_d
            )
        except (LLMError, SandboxError) as exc:
            kind = "llm_error" if isinstance(exc, LLMError) else "infra_error"
            logger.error("repeat %d aborted (%s): %s", repeat, kind, exc)
            writer.scope("run", repeat).log(kind, error=str(exc))
            continue
        if refine_record is not None:
            refine_records.append(refine_record)
        arm_records.extend(arms)
    write_report(writer.run_dir, cfg, naive_ms, refine_records, arm_records)
    return writer.run_dir


def preflight(cfg: Config) -> dict[str, float]:
    """Fail fast on a missing daemon/CLI or a broken harness; return naive baseline times (ms)."""
    if not docker_available():
        raise SystemExit("Docker daemon is not running")
    if shutil.which("claude") is None:
        raise SystemExit("`claude` CLI not found on PATH")
    naive: dict[str, float] = {}
    for platform in (AMD64, ARMV7):
        logger.info("building %s", platform.image)
        build_image(platform)
        result = measure_naive(platform, cfg)
        if not result.success or result.time_ms is None:
            raise SystemExit(
                f"naive reference failed on {platform.name}: {result.stage}\n{result.stderr}"
            )
        total_s = result.time_ms * cfg.reps / 1000.0
        logger.info(
            "naive on %s: %.1f ms per run (%.1f s for %d reps)",
            platform.name, result.time_ms, total_s, cfg.reps,
        )  # fmt: skip
        if total_s > cfg.max_naive_s:
            raise SystemExit(
                f"naive run on {platform.name} takes {total_s:.0f} s "
                f"(> {cfg.max_naive_s:.0f} s); lower --n or raise max_naive_s"
            )
        naive[platform.name] = result.time_ms
    return naive


def build_config(args: argparse.Namespace) -> Config:
    """Apply CLI overrides to the default config."""
    cfg = Config().quick() if args.quick else Config()
    overrides = {
        k: v
        for k, v in {
            "n": args.n, "repeats": args.repeats, "rounds": args.rounds,
            "trials": args.trials, "model": args.model, "feedback": args.feedback,
        }.items()
        if v is not None
    }  # fmt: skip
    return replace(cfg, **overrides)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="1 repeat, 3 rounds")
    parser.add_argument("--n", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--model")
    parser.add_argument(
        "--feedback", choices=FEEDBACK_MODES,
        help="Reflexion failure signal: diagnostic (stage + compiler output) or scalar (compiled, correct, speedup)",
    )  # fmt: skip
    parser.add_argument(
        "--reuse-memory-from", type=Path,
        help="earlier run directory whose arm A memories are reused (skips arm A)",
    )  # fmt: skip
    parser.add_argument(
        "--only-repeats", type=int, nargs="+",
        help="with --reuse-memory-from: only these repeat numbers",
    )  # fmt: skip
    parser.add_argument(
        "--arm-d", action="store_true",
        help="run only arm D (A's memory on amd64, the platform it came from) instead of B and C",
    )  # fmt: skip
    parser.add_argument("--out", default="traces")
    parser.add_argument("--run-id")
    parser.add_argument("--task-file", type=Path, help="task statement (default task/prompt_task.md)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = build_config(args)
    naive = preflight(cfg)

    def run_fn_for(platform: Platform) -> RunFn:
        return lambda code, flags: run_candidate(platform, code, flags, cfg, naive[platform.name])

    run_id = args.run_id or time.strftime("matxfer_%Y%m%d_%H%M%S")
    llm = ClaudeCodeLLM(cfg.llm_timeout_s, cfg.model)
    run_dir = run_experiment(
        cfg, llm, run_fn_for, naive, Path(args.out), run_id,
        task_path=args.task_file, memory_source=args.reuse_memory_from,
        arm_d=args.arm_d, only_repeats=args.only_repeats,
    )  # fmt: skip
    logger.info("done: %s", run_dir / "results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
