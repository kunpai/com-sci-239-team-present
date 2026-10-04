"""Compile and run candidate matmul code inside Docker for a given platform."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.config import Config

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
WORK_ROOT = ROOT / ".work"
STDERR_CLIP = 4000

_FLAG_OK = re.compile(r"^-[A-Za-z0-9_=+.,/:-]+$")
_FLAG_DENY = (
    "-o", "-wrapper", "-specs", "-fplugin", "-B", "-x", "-include", "-imacros",
    "-Xlinker", "-Xassembler", "-Wl,", "-Wa,", "-Wp,", "-l", "-L", "-I", "-iquote",
    "-isystem", "-save-temps",
)  # fmt: skip


@dataclass(frozen=True)
class Platform:
    """A Docker platform and the image tag built for it."""

    name: str
    docker_platform: str
    image: str


AMD64 = Platform("amd64", "linux/amd64", "matxfer:amd64")
ARMV7 = Platform("armv7", "linux/arm/v7", "matxfer:armv7")


@dataclass(frozen=True)
class RunResult:
    """Outcome of compiling and running one candidate."""

    stage: str  # ok | parse | flags | compile | run | timeout | incorrect
    compiled: bool
    correct: bool
    time_ms: float | None
    speedup: float | None
    stderr: str

    @property
    def success(self) -> bool:
        """True when the candidate compiled and produced correct results."""
        return self.compiled and self.correct


class FlagError(ValueError):
    """Raised when a compiler flag is not allowed."""


class SandboxError(RuntimeError):
    """Raised when Docker itself fails (daemon, image, resources), not the candidate."""


def check_flags(flags: Sequence[str]) -> None:
    """Reject flags with odd characters or ones that change outputs, paths or the linker."""
    for tok in flags:
        if not _FLAG_OK.match(tok):
            raise FlagError(f"unsupported characters in flag {tok!r}")
        if tok.startswith(_FLAG_DENY):
            raise FlagError(f"flag {tok!r} is not allowed")


def _clip(text: str, limit: int = STDERR_CLIP) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def docker_available() -> bool:
    """Return True when the Docker daemon answers."""
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def build_image(platform: Platform) -> None:
    """Build (or reuse the cached) image for `platform`."""
    cmd = [
        "docker", "build", "--platform", platform.docker_platform,
        "-t", platform.image, "-f", str(ROOT / "Dockerfile"), str(ROOT),
    ]  # fmt: skip
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )


def _docker_run(
    platform: Platform, workdir: Path, argv: list[str], cfg: Config, timeout_s: int
) -> tuple[int, str, str, bool]:
    """Run `argv` in a throwaway container; returns (rc, stdout, stderr, timed_out)."""
    name = f"matxfer-{uuid.uuid4().hex[:12]}"
    cmd = [
        "docker", "run", "--rm", "--name", name,
        "--platform", platform.docker_platform,
        "--network", "none", "--cpus", str(cfg.cpus),
        "--cpuset-cpus", f"0-{cfg.cpus - 1}",  # --cpus alone is only a quota; nproc stays 10
        "--memory", cfg.mem_limit,
        "--pids-limit", "512", "-v", f"{workdir}:/work", "-w", "/work",
        platform.image, *argv,
    ]  # fmt: skip
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
        return 124, "", "timed out", True
    # `docker run` exits 125 for its own failures and 126/127 when it cannot start the
    # command; those are the harness's fault and must never be shown to the model.
    docker_failed = proc.returncode == 125 or (
        proc.returncode in (126, 127) and proc.stderr.lstrip().startswith("docker:")
    )
    if docker_failed:
        raise SandboxError(f"docker run failed (exit {proc.returncode}): {proc.stderr[:500]}")
    return proc.returncode, proc.stdout, proc.stderr, False


def _grab(out: str, key: str) -> float | None:
    match = re.search(rf"^{key} (\S+)$", out, re.MULTILINE)
    return float(match.group(1)) if match else None


def _parse_run(out: str, err: str, rc: int, cfg: Config, naive_ms: float | None) -> RunResult:
    small = _grab(out, "SMALL_OK")
    freivalds = _grab(out, "FREIVALDS_OK")
    odd = _grab(out, "ODD_OK")
    time_ms = _grab(out, "TIME_MS")
    if rc != 0 or small is None:
        return RunResult("run", True, False, None, None, f"exit code {rc}\n{_clip(err)}")
    if small == 0:
        msg = (
            "wrong result at n=37 compared with a full reference "
            "(check odd sizes and that all of C is written)"
        )
        return RunResult("incorrect", True, False, None, None, msg)
    if odd != 1:  # printed before the timing runs, so it is the first n-sized check to fail
        msg = (
            f"wrong result at the odd size n={cfg.n | 1} "
            "(handle every n, including tails and sizes your fast path skips)"
        )
        return RunResult("incorrect", True, False, None, None, msg)
    if freivalds != 1 or time_ms is None:
        msg = f"wrong result at n={cfg.n} (Freivalds check C*x == A*(B*x) failed)"
        return RunResult("incorrect", True, False, None, None, msg)
    speedup = naive_ms / time_ms if naive_ms and time_ms > 0 else None
    return RunResult("ok", True, True, time_ms, speedup, "")


def run_candidate(
    platform: Platform,
    code: str,
    flags: Sequence[str],
    cfg: Config,
    naive_ms: float | None = None,
) -> RunResult:
    """Compile `code` with `flags` and run the harness on `platform`."""
    try:
        check_flags(flags)
    except FlagError as exc:
        return RunResult("flags", False, False, None, None, str(exc))
    WORK_ROOT.mkdir(exist_ok=True)
    workdir = Path(tempfile.mkdtemp(dir=WORK_ROOT))
    try:
        (workdir / "matmul.c").write_text(code)
        compile_argv = [
            "gcc", *flags, "-o", "bench", "/opt/harness.o", "matmul.c", "-lm", "-lpthread",
        ]  # fmt: skip
        rc, _out, err, timed_out = _docker_run(
            platform, workdir, compile_argv, cfg, cfg.compile_timeout_s
        )
        if timed_out:
            return RunResult("timeout", False, False, None, None, "compilation timed out")
        if rc != 0:
            return RunResult("compile", False, False, None, None, _clip(err))
        rc, out, err, timed_out = _docker_run(
            platform, workdir, ["./bench", str(cfg.n), str(cfg.reps)], cfg, cfg.run_timeout_s
        )
        if timed_out:
            msg = f"run timed out after {cfg.run_timeout_s} s"
            return RunResult("timeout", True, False, None, None, msg)
        return _parse_run(out, err, rc, cfg, naive_ms)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def measure_naive(platform: Platform, cfg: Config) -> RunResult:
    """Run the naive triple loop (flags `-O2`) to get the per-platform baseline time."""
    code = (ROOT / "task" / "naive.c").read_text()
    return run_candidate(platform, code, ["-O2"], cfg)
