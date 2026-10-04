# Memory Transfer Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a harness that refines a C matmul with Claude Code on `linux/amd64` (Self-Refine), distills the result into a verbal memory, then tests that memory on `linux/arm/v7` (Reflexion-style trials), saving a full trace of every step.

**Architecture:** A Python orchestrator (`run.py`) drives three arms (A: Self-Refine on amd64, B: arm32 trials without memory, C: arm32 trials with A's memory). `src/llm.py` is the only code that calls Claude Code; `src/sandbox.py` is the only code that calls Docker. Each arm writes numbered JSON traces; `src/report.py` aggregates results.

**Tech Stack:** Python >= 3.11 with `uv`, pytest, ruff, mypy, matplotlib; Docker (`gcc:13`, platforms `linux/amd64` and `linux/arm/v7`); `claude -p` headless as the LLM; C harness compiled in the container.

**Spec:** `plan/2026-10-01-memory-transfer-demo-design.md` (approved). Read it first.

## Spec adjustments made in this plan

The spec's file list is illustrative; this plan makes four small, non-behavioral changes:
1. Adds `src/candidate.py` (reply parsing, `Step`, `RunFn`) and `src/prompts.py` (all prompt text) so `refine.py` and `memory.py` stay small.
2. Adds `src/reflexion.py` for the arm B/C trial loop (spec put it under `run.py`).
3. Adds `src/report.py` for `results.md`, `results.json`, the per-round plot and `traces/README.md`.
4. Trace path gets a repeat directory: `traces/<run_id>/<arm>/r<repeat>/<NN>-<kind>.json`.

## Environment findings (verified 2026-10-01 before writing this plan)

- Docker 29.4.3 on Apple Silicon, 10 CPUs, 12.5 GB. `gcc:13` runs on both `linux/amd64` (x86_64, 64-bit) and `linux/arm/v7` (armv7l, 32-bit, gcc 13.4.0).
- `claude -p ... --tools ""` disables only built-in tools; MCP servers (Gmail, Drive, Docs) stay loaded, and the user's global CLAUDE.md and hooks would leak into every prompt and contaminate the experiment.
- `--bare` cannot authenticate (it skips the keychain), so it is unusable.
- Verified isolated call: `--tools "" --strict-mcp-config --disable-slash-commands --setting-sources "" --no-session-persistence --output-format json` with stdin from `/dev/null` gives 0 tools, no MCP servers, no skills, and a leakage probe about CLAUDE.md/hooks answered `NONE`. Cost about $0.005 per trivial call.
- `--output-format json` prints either one result object or a list of events; parse the last object whose `type` is `"result"`.
- This directory is not a git repository, so no commit steps appear below. To get version control, say so and Task 1 gains a `git init` plus `.gitignore` for `.venv/`, `.work/` and `traces/`.

## Global Constraints

Copied from the spec; every task's requirements include these.

- Platforms: `linux/amd64` and `linux/arm/v7`; both run under emulation on the MacBook.
- The prompts never name the architecture; arms differ only in the memory block.
- Defaults: rounds=10, repeats=3, trials<=3, memory window Omega=3, memory cap about 200 words, `n=1024`.
- Correctness: O(n^2) Freivalds-style check at size `n`, plus a full-reference check at n=37 for odd sizes.
- Candidate output is one `matmul.c` plus a `FLAGS:` line; the harness uses Claude's flags as given.
- Untrusted code runs only in the container: no network, CPU/memory/time limits, argv list never a shell.
- Metrics: first-try success, trials to first success, final speedup versus naive on the same platform.
- Traces: LLM steps record prompt, response, session_id, usage, cost, duration_s; sandbox steps record candidate_code, flags, platform, compiled, correct, time_ms, speedup, stderr; all steps record arm, repeat, step number, memory entries in effect.
- Python style (user rules): type hints, docstrings, `logging` not `print`, frozen dataclasses for config, `__all__` in package `__init__.py`, no bare `except`, files under about 400 lines.
- `uv` for packaging. Spec and plan documents live in `/plan`.

## Review Focus

Failure modes the spec implies but a straightforward implementation tends to miss; each has a pinning test in the named task.

1. Replies with several code blocks, a `FLAGS:` line inside a code comment, repeated `FLAGS:` lines, or an empty reply (Task 5).
2. A candidate that accumulates into `C` assuming it is zero-initialized (Task 3, `accum.c`).
3. A candidate that is correct only when `n` is a multiple of 4 (Task 3, `notail.c`).
4. A container left running after a timeout (Task 3, `loop.c`).
5. Arm A produces no correct candidate (so no memory, arm C skipped), and an LLM failure mid-repeat must not lose the other repeats (Task 9).

---

### Task 1: Project scaffold, config, trace writer

**Files:**
- Create: `pyproject.toml`, `src/__init__.py`, `src/config.py`, `src/traces.py`
- Test: `tests/test_config.py`, `tests/test_traces.py`

**Interfaces:**
- Produces: `Config` (frozen dataclass, `Config.quick()`); `TraceWriter(root: Path, run_id: str)` with `.run_dir: Path` and `.scope(arm: str, repeat: int) -> TraceScope`; `TraceScope` with `.directory: Path`, `.set_memory(entries: Sequence[str]) -> None`, `.log(kind: str, **fields) -> Path`, `.log_llm(role: str, prompt: str, reply: Any, **ctx) -> Path`, `.log_step(step: Any, **ctx) -> Path`. `reply` and `step` must be dataclass instances.

- [ ] **Step 1: Write `pyproject.toml` and the package marker**

```toml
[project]
name = "matxfer"
version = "0.1.0"
description = "Self-Refine on x86, Reflexion-style memory transfer to arm32"
requires-python = ">=3.11"
dependencies = ["matplotlib>=3.8"]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6", "mypy>=1.11"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
markers = [
    "docker: needs a running Docker daemon",
    "llm: needs the claude CLI and spends tokens",
]

[tool.ruff]
line-length = 100

[tool.mypy]
ignore_missing_imports = true
```

`src/__init__.py`:

```python
"""Memory-transfer demo: Self-Refine on x86, Reflexion-style memory on arm32."""

__all__: list[str] = []
```

Create an empty `tests/__init__.py` (so tests can `from tests.conftest import ...`).

Run: `cd comsci239_team && uv sync`
Expected: creates `.venv` and `uv.lock`, installs matplotlib, pytest, ruff, mypy.

- [ ] **Step 2: Write the failing tests**

`tests/test_config.py`:

```python
from dataclasses import FrozenInstanceError

import pytest

from src.config import Config


def test_defaults_match_spec() -> None:
    cfg = Config()
    assert (cfg.rounds, cfg.repeats, cfg.trials, cfg.window) == (10, 3, 3, 3)
    assert cfg.memory_cap_words == 200
    assert cfg.n == 1024


def test_config_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        Config().rounds = 5  # type: ignore[misc]


def test_quick_shrinks_rounds_and_repeats_only() -> None:
    cfg = Config().quick()
    assert (cfg.rounds, cfg.repeats) == (3, 1)
    assert cfg.n == Config().n
```

`tests/test_traces.py`:

```python
import json
from dataclasses import dataclass
from pathlib import Path

from src.traces import TraceWriter


@dataclass(frozen=True)
class _Reply:
    text: str
    cost_usd: float


@dataclass(frozen=True)
class _Step:
    code: str


def test_scope_writes_numbered_files_with_context(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path, "run1")
    scope = writer.scope("C", 2)
    scope.set_memory(["note one"])
    p1 = scope.log("llm", role="generate")
    p2 = scope.log("sandbox", compiled=False)
    assert p1.name == "01-llm.json" and p2.name == "02-sandbox.json"
    assert p1.parent == tmp_path / "run1" / "C" / "r2"
    data = json.loads(p2.read_text())
    assert data["arm"] == "C" and data["repeat"] == 2 and data["step"] == 2
    assert data["memory"] == ["note one"] and data["compiled"] is False


def test_log_llm_and_log_step_flatten_dataclasses(tmp_path: Path) -> None:
    scope = TraceWriter(tmp_path, "run1").scope("A", 1)
    llm_path = scope.log_llm("feedback", "PROMPT", _Reply("hi", 0.01), round_no=3)
    step_path = scope.log_step(_Step("int x;"), trial=1)
    llm = json.loads(llm_path.read_text())
    assert llm["role"] == "feedback" and llm["prompt"] == "PROMPT"
    assert llm["text"] == "hi" and llm["round_no"] == 3
    assert json.loads(step_path.read_text())["code"] == "int x;"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py tests/test_traces.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`.

- [ ] **Step 4: Write the implementation**

`src/config.py`:

```python
"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace


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
    llm_timeout_s: int = 600
    compile_timeout_s: int = 300
    run_timeout_s: int = 1800
    max_naive_s: float = 120.0  # preflight aborts if the naive run exceeds this
    model: str | None = None

    def quick(self) -> Config:
        """Return a copy sized for a fast end-to-end check."""
        return replace(self, rounds=3, repeats=1)
```

`src/traces.py`:

```python
"""Per-step JSON trace files."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


class TraceScope:
    """Writes numbered JSON files into one (arm, repeat) directory."""

    def __init__(self, directory: Path, arm: str, repeat: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._arm = arm
        self._repeat = repeat
        self._step = 0
        self._memory: list[str] = []

    def set_memory(self, entries: Sequence[str]) -> None:
        """Record the memory entries in effect for subsequent steps."""
        self._memory = list(entries)

    def log(self, kind: str, **fields: Any) -> Path:
        """Write the next numbered trace file and return its path."""
        self._step += 1
        payload: dict[str, Any] = {
            "arm": self._arm,
            "repeat": self._repeat,
            "step": self._step,
            "kind": kind,
            "memory": list(self._memory),
            **fields,
        }
        path = self.directory / f"{self._step:02d}-{kind}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def log_llm(self, role: str, prompt: str, reply: Any, **ctx: Any) -> Path:
        """Log one LLM call; `reply` is a dataclass (LLMReply)."""
        return self.log("llm", role=role, prompt=prompt, **asdict(reply), **ctx)

    def log_step(self, step: Any, **ctx: Any) -> Path:
        """Log one sandbox evaluation; `step` is a dataclass (Step)."""
        return self.log("sandbox", **asdict(step), **ctx)


class TraceWriter:
    """Owns `<root>/<run_id>/` and hands out per-arm scopes."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def scope(self, arm: str, repeat: int) -> TraceScope:
        """Return the scope for `<run_dir>/<arm>/r<repeat>/`."""
        return TraceScope(self.run_dir / arm / f"r{repeat}", arm, repeat)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_traces.py -v`
Expected: 5 passed.

---

### Task 2: C harness, naive reference, Docker image

**Files:**
- Create: `task/harness.c`, `task/naive.c`, `Dockerfile`, `.dockerignore`

**Interfaces:**
- Produces: image `matxfer:<amd64|armv7>` containing `/opt/harness.o` (compiled `-O2`). Candidates link as `gcc <FLAGS> -o bench /opt/harness.o matmul.c -lm -lpthread`. `./bench N REPS` prints `SMALL_OK 0|1` (n=37 against a full double-precision reference), then, only if that passed, `FREIVALDS_OK 0|1` and `TIME_MS <best-of-REPS>`. `C` is filled with `0xFF` bytes (NaN) before every call.

- [ ] **Step 1: Write `task/naive.c`**

```c
#include <stddef.h>

void matmul(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++) {
                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            }
            C[(size_t)i * n + j] = s;
        }
    }
}
```

- [ ] **Step 2: Write `task/harness.c`**

```c
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void matmul(const float *A, const float *B, float *C, int n);

static uint32_t rng_state = 12345u;

static float next_float(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / 8388608.0f) - 1.0f; /* [-1, 1) */
}

static float *alloc_matrix(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * n * sizeof(float)) != 0) return NULL;
    return (float *)p;
}

static void fill(float *m, size_t n) {
    for (size_t i = 0; i < n * n; i++) m[i] = next_float();
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* Full O(n^3) reference in double precision; only used for the tiny size. */
static void reference(const float *A, const float *B, float *C, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double s = 0.0;
            for (size_t k = 0; k < n; k++) s += (double)A[i * n + k] * (double)B[k * n + j];
            C[i * n + j] = (float)s;
        }
    }
}

/* 1 if `got` matches `want` in max-norm relative error <= 1e-3 and has no NaN/Inf. */
static int close_enough(const double *got, const double *want, size_t len) {
    double max_diff = 0.0, max_want = 0.0;
    for (size_t i = 0; i < len; i++) {
        if (!isfinite(got[i])) return 0;
        double d = fabs(got[i] - want[i]);
        if (d > max_diff) max_diff = d;
        if (fabs(want[i]) > max_want) max_want = fabs(want[i]);
    }
    return max_diff <= 1e-3 * (max_want + 1e-9);
}

static int check_small(void) {
    const size_t n = 37;
    float *A = alloc_matrix(n), *B = alloc_matrix(n), *C = alloc_matrix(n), *R = alloc_matrix(n);
    double *got = malloc(n * n * sizeof(double)), *want = malloc(n * n * sizeof(double));
    if (!A || !B || !C || !R || !got || !want) return 0;
    fill(A, n);
    fill(B, n);
    memset(C, 0xFF, n * n * sizeof(float));
    matmul(A, B, C, (int)n);
    reference(A, B, R, n);
    for (size_t i = 0; i < n * n; i++) {
        got[i] = (double)C[i];
        want[i] = (double)R[i];
    }
    int ok = close_enough(got, want, n * n);
    free(A); free(B); free(C); free(R); free(got); free(want);
    return ok;
}

/* Freivalds-style O(n^2) check: compare C*x with A*(B*x) for a fixed random x. */
static int check_freivalds(const float *A, const float *B, const float *C, size_t n) {
    double *x = malloc(n * sizeof(double)), *y = malloc(n * sizeof(double));
    double *z = malloc(n * sizeof(double)), *w = malloc(n * sizeof(double));
    if (!x || !y || !z || !w) return 0;
    for (size_t j = 0; j < n; j++) x[j] = (double)next_float();
    for (size_t k = 0; k < n; k++) {
        double s = 0.0;
        for (size_t j = 0; j < n; j++) s += (double)B[k * n + j] * x[j];
        y[k] = s;
    }
    for (size_t i = 0; i < n; i++) {
        double s = 0.0;
        for (size_t k = 0; k < n; k++) s += (double)A[i * n + k] * y[k];
        z[i] = s;
        double t = 0.0;
        for (size_t j = 0; j < n; j++) t += (double)C[i * n + j] * x[j];
        w[i] = t;
    }
    int ok = close_enough(w, z, n);
    free(x); free(y); free(z); free(w);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: bench N REPS\n");
        return 2;
    }
    size_t n = (size_t)strtoull(argv[1], NULL, 10);
    int reps = atoi(argv[2]);
    if (n == 0 || reps < 1) {
        fprintf(stderr, "bad arguments\n");
        return 2;
    }

    int small_ok = check_small();
    printf("SMALL_OK %d\n", small_ok);
    fflush(stdout);
    if (!small_ok) return 0;

    float *A = alloc_matrix(n), *B = alloc_matrix(n), *C = alloc_matrix(n);
    if (!A || !B || !C) {
        fprintf(stderr, "allocation failed for n=%zu\n", n);
        return 2;
    }
    fill(A, n);
    fill(B, n);

    double best = -1.0;
    int freivalds_ok = 0;
    for (int r = 0; r < reps; r++) {
        memset(C, 0xFF, n * n * sizeof(float));
        double t0 = now_ms();
        matmul(A, B, C, (int)n);
        double ms = now_ms() - t0;
        if (best < 0.0 || ms < best) best = ms;
        if (r == 0) freivalds_ok = check_freivalds(A, B, C, n);
    }
    printf("FREIVALDS_OK %d\n", freivalds_ok);
    printf("TIME_MS %.3f\n", best);
    return 0;
}
```

- [ ] **Step 3: Write `Dockerfile` and `.dockerignore`**

`Dockerfile`:

```dockerfile
FROM gcc:13
COPY task/harness.c /opt/harness.c
RUN gcc -O2 -c /opt/harness.c -o /opt/harness.o
WORKDIR /work
```

`.dockerignore`:

```
*
!task/harness.c
```

- [ ] **Step 4: Build both images**

Run:
```bash
docker build --platform linux/amd64 -t matxfer:amd64 .
docker build --platform linux/arm/v7 -t matxfer:armv7 .
```
Expected: both builds finish without error (the arm/v7 build runs gcc under emulation and is slower).

- [ ] **Step 5: Verify the harness on both platforms with the naive reference**

Run:
```bash
docker run --rm --platform linux/amd64 -v "$PWD/task:/t" matxfer:amd64 sh -c 'gcc -O2 -o /tmp/bench /opt/harness.o /t/naive.c -lm -lpthread && /tmp/bench 64 1'
docker run --rm --platform linux/arm/v7 -v "$PWD/task:/t" matxfer:armv7 sh -c 'gcc -O2 -o /tmp/bench /opt/harness.o /t/naive.c -lm -lpthread && /tmp/bench 64 1'
```
Expected, for each platform: `SMALL_OK 1`, `FREIVALDS_OK 1`, `TIME_MS <number>`.

---

### Task 3: Sandbox (`src/sandbox.py`)

**Files:**
- Create: `src/sandbox.py`, `tests/canned/wrong.c`, `tests/canned/loop.c`, `tests/canned/accum.c`, `tests/canned/notail.c`, `tests/canned/avx2.c`
- Test: `tests/test_sandbox.py`

**Interfaces:**
- Consumes: `Config` (Task 1); image tags and harness protocol (Task 2).
- Produces: `Platform(name, docker_platform, image)`; constants `AMD64`, `ARMV7`; `RunResult(stage, compiled, correct, time_ms, speedup, stderr)` with `.success -> bool` (`compiled and correct`); `stage` is one of `"ok" | "flags" | "compile" | "run" | "timeout" | "incorrect"` (`"parse"` is used by Task 5); `FlagError`; `check_flags(flags: Sequence[str]) -> None`; `docker_available() -> bool`; `build_image(platform) -> None`; `run_candidate(platform, code, flags, cfg, naive_ms=None) -> RunResult`; `measure_naive(platform, cfg) -> RunResult`.

- [ ] **Step 1: Write the canned C files**

`tests/canned/wrong.c`:

```c
#include <stddef.h>

void matmul(const float *A, const float *B, float *C, int n) {
    (void)A; (void)B;
    for (size_t i = 0; i < (size_t)n * n; i++) C[i] = 0.0f;
}
```

`tests/canned/loop.c`:

```c
void matmul(const float *A, const float *B, float *C, int n) {
    (void)A; (void)B; (void)C; (void)n;
    for (;;) {
    }
}
```

`tests/canned/accum.c` (assumes `C` starts at zero):

```c
#include <stddef.h>

void matmul(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float a = A[(size_t)i * n + k];
            for (int j = 0; j < n; j++) C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
        }
    }
}
```

`tests/canned/notail.c` (correct only when `n % 4 == 0`):

```c
#include <stddef.h>

void matmul(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j + 4 <= n; j += 4) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int k = 0; k < n; k++) {
                float a = A[(size_t)i * n + k];
                const float *b = B + (size_t)k * n + j;
                s0 += a * b[0]; s1 += a * b[1]; s2 += a * b[2]; s3 += a * b[3];
            }
            float *c = C + (size_t)i * n + j;
            c[0] = s0; c[1] = s1; c[2] = s2; c[3] = s3;
        }
    }
}
```

`tests/canned/avx2.c` (the x86-only candidate; used with flags `-O3 -mavx2 -mfma`):

```c
#include <immintrin.h>
#include <stddef.h>
#include <string.h>

void matmul(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        float *c = C + (size_t)i * n;
        memset(c, 0, (size_t)n * sizeof(float));
        for (int k = 0; k < n; k++) {
            float a_s = A[(size_t)i * n + k];
            __m256 a = _mm256_set1_ps(a_s);
            const float *b = B + (size_t)k * n;
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 cv = _mm256_loadu_ps(c + j);
                cv = _mm256_fmadd_ps(a, _mm256_loadu_ps(b + j), cv);
                _mm256_storeu_ps(c + j, cv);
            }
            for (; j < n; j++) c[j] += a_s * b[j];
        }
    }
}
```

- [ ] **Step 2: Write the failing tests**

`tests/test_sandbox.py`:

```python
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from src.config import Config
from src.sandbox import (
    AMD64,
    ARMV7,
    FlagError,
    build_image,
    check_flags,
    docker_available,
    measure_naive,
    run_candidate,
)

CANNED = Path(__file__).parent / "canned"
NAIVE = Path(__file__).parent.parent / "task" / "naive.c"
CFG = Config(n=64, reps=1, cpus=2, compile_timeout_s=120, run_timeout_s=60)


def test_check_flags_accepts_normal_flags() -> None:
    check_flags(["-O3", "-march=native", "-fopenmp", "-ffast-math", "-funroll-loops"])


@pytest.mark.parametrize(
    "bad", [["-o", "x"], ["-wrapper", "sh"], ["-O3;rm"], ["-lfoo"], ["-I/etc"], ["-Wl,-z"], ["@f"]]
)
def test_check_flags_rejects_dangerous_flags(bad: list[str]) -> None:
    with pytest.raises(FlagError):
        check_flags(bad)


@pytest.fixture(scope="module")
def images() -> None:
    if not docker_available():
        pytest.skip("Docker daemon not available")
    build_image(AMD64)
    build_image(ARMV7)


@pytest.mark.docker
@pytest.mark.parametrize("platform", [AMD64, ARMV7])
def test_naive_is_correct_on_both_platforms(images: None, platform) -> None:
    result = run_candidate(platform, NAIVE.read_text(), ["-O2"], CFG, naive_ms=100.0)
    assert result.stage == "ok" and result.success
    assert result.time_ms is not None and result.speedup is not None


@pytest.mark.docker
def test_wrong_result_is_incorrect(images: None) -> None:
    result = run_candidate(AMD64, (CANNED / "wrong.c").read_text(), ["-O2"], CFG)
    assert result.stage == "incorrect" and result.compiled and not result.correct


@pytest.mark.docker
def test_accumulating_into_c_is_caught(images: None) -> None:
    result = run_candidate(AMD64, (CANNED / "accum.c").read_text(), ["-O2"], CFG)
    assert result.stage == "incorrect"


@pytest.mark.docker
def test_candidate_correct_only_for_multiples_of_four_is_caught(images: None) -> None:
    result = run_candidate(AMD64, (CANNED / "notail.c").read_text(), ["-O2"], CFG)
    assert result.stage == "incorrect"
    assert "37" in result.stderr


@pytest.mark.docker
def test_infinite_loop_times_out_and_leaves_no_container(images: None) -> None:
    cfg = replace(CFG, run_timeout_s=5)
    result = run_candidate(AMD64, (CANNED / "loop.c").read_text(), ["-O2"], cfg)
    assert result.stage == "timeout" and result.compiled
    leftover = subprocess.run(
        ["docker", "ps", "-q", "--filter", "name=matxfer-"], capture_output=True, text=True
    )
    assert leftover.stdout.strip() == ""


@pytest.mark.docker
def test_avx2_compiles_on_amd64_and_fails_to_compile_on_arm32(images: None) -> None:
    code = (CANNED / "avx2.c").read_text()
    flags = ["-O3", "-mavx2", "-mfma"]
    on_x86 = run_candidate(AMD64, code, flags, CFG)
    on_arm = run_candidate(ARMV7, code, flags, CFG)
    assert on_x86.compiled
    assert on_arm.stage == "compile" and not on_arm.compiled
    assert on_arm.stderr  # the compiler error is what the Reflexion step reads


@pytest.mark.docker
def test_rejected_flags_never_reach_docker(images: None) -> None:
    result = run_candidate(AMD64, NAIVE.read_text(), ["-o", "evil"], CFG)
    assert result.stage == "flags"


@pytest.mark.docker
def test_measure_naive_returns_a_time(images: None) -> None:
    result = measure_naive(AMD64, CFG)
    assert result.success and result.time_ms and result.time_ms > 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_sandbox.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.sandbox'`.

- [ ] **Step 4: Write `src/sandbox.py`**

```python
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
    subprocess.run(
        [
            "docker", "build", "--platform", platform.docker_platform,
            "-t", platform.image, "-f", str(ROOT / "Dockerfile"), str(ROOT),
        ],  # fmt: skip
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
        "--network", "none", "--cpus", str(cfg.cpus), "--memory", cfg.mem_limit,
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
    return proc.returncode, proc.stdout, proc.stderr, False


def _grab(out: str, key: str) -> float | None:
    match = re.search(rf"^{key} (\S+)$", out, re.M)
    return float(match.group(1)) if match else None


def _parse_run(out: str, err: str, rc: int, cfg: Config, naive_ms: float | None) -> RunResult:
    small, freivalds, time_ms = _grab(out, "SMALL_OK"), _grab(out, "FREIVALDS_OK"), _grab(out, "TIME_MS")
    if rc != 0 or small is None:
        return RunResult("run", True, False, None, None, f"exit code {rc}\n{_clip(err)}")
    if small == 0:
        msg = "wrong result at n=37 compared with a full reference (check odd sizes and that all of C is written)"
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_sandbox.py -v`
Expected: all pass (the docker tests take a few minutes the first time because of the emulated builds). If `test_avx2_compiles_on_amd64...` fails because gcc rejects `-mavx2` on amd64, report it; do not weaken the assertion.

---

### Task 4: Claude Code wrapper (`src/llm.py`)

**Files:**
- Create: `src/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `LLMError`; `LLMReply(text, session_id, cost_usd, duration_s, usage, model_usage)` (frozen dataclass); `LLM` Protocol (`__call__(prompt: str) -> LLMReply`); `parse_cli_output(stdout: str) -> LLMReply`; `ClaudeCodeLLM(timeout_s: int, model: str | None = None, retries: int = 1)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_llm.py`:

```python
import json
import shutil
import subprocess

import pytest

from src.llm import ClaudeCodeLLM, LLMError, parse_cli_output

RESULT = {
    "type": "result",
    "result": "OK",
    "session_id": "s1",
    "total_cost_usd": 0.01,
    "duration_ms": 1200,
    "usage": {"input_tokens": 5},
    "modelUsage": {"some-model": {}},
}


def test_parse_single_object() -> None:
    reply = parse_cli_output(json.dumps(RESULT))
    assert reply.text == "OK" and reply.session_id == "s1"
    assert reply.cost_usd == 0.01 and reply.duration_s == 1.2


def test_parse_event_list_takes_last_result() -> None:
    events = [{"type": "system", "subtype": "init"}, RESULT]
    assert parse_cli_output(json.dumps(events)).text == "OK"


def test_parse_error_result_raises() -> None:
    with pytest.raises(LLMError):
        parse_cli_output(json.dumps({**RESULT, "is_error": True, "result": "Not logged in"}))


def test_parse_garbage_raises() -> None:
    with pytest.raises(LLMError):
        parse_cli_output("not json")


def _completed(stdout: str, rc: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], rc, stdout=stdout, stderr="")


def test_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(cmd, 1)
        return _completed(json.dumps(RESULT))

    monkeypatch.setattr("src.llm.subprocess.run", fake_run)
    reply = ClaudeCodeLLM(timeout_s=5)("hello")
    assert reply.text == "OK" and len(calls) == 2


def test_gives_up_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.llm.subprocess.run", lambda cmd, **kw: _completed("", rc=1))
    with pytest.raises(LLMError):
        ClaudeCodeLLM(timeout_s=5)("hello")


def test_command_is_isolated_from_user_config(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return _completed(json.dumps(RESULT))

    monkeypatch.setattr("src.llm.subprocess.run", fake_run)
    ClaudeCodeLLM(timeout_s=5, model="some-model")("hello")
    cmd = seen[0]
    assert cmd[:3] == ["claude", "-p", "hello"]
    assert cmd[cmd.index("--tools") + 1] == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "some-model"


@pytest.mark.llm
@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not installed")
def test_real_claude_smoke() -> None:
    reply = ClaudeCodeLLM(timeout_s=120)("Reply with exactly: OK")
    assert "OK" in reply.text and reply.session_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -v -m "not llm"`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.llm'`.

- [ ] **Step 3: Write `src/llm.py`**

```python
"""Claude Code (`claude -p`) as the LLM, isolated from the user's config."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Built-in tools off, no MCP servers, no skills, no user/project settings (CLAUDE.md, hooks),
# no session files. Verified on 2026-10-01; `--bare` is NOT used because it cannot authenticate.
_ISOLATION_ARGS = [
    "--output-format", "json",
    "--tools", "",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--setting-sources", "",
    "--no-session-persistence",
]  # fmt: skip


class LLMError(RuntimeError):
    """Raised when a Claude Code call fails after retries."""


@dataclass(frozen=True)
class LLMReply:
    """One reply plus the metadata recorded in traces."""

    text: str
    session_id: str
    cost_usd: float
    duration_s: float
    usage: dict[str, Any]
    model_usage: dict[str, Any]


class LLM(Protocol):
    """Anything that turns a prompt into a reply (real or scripted)."""

    def __call__(self, prompt: str) -> LLMReply: ...


def parse_cli_output(stdout: str) -> LLMReply:
    """Parse `claude -p --output-format json` output (one object or a list of events)."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise LLMError(f"claude output is not JSON: {stdout[:200]!r}") from exc
    items = data if isinstance(data, list) else [data]
    results = [i for i in items if isinstance(i, dict) and i.get("type") == "result"]
    if not results:
        raise LLMError("claude output has no result object")
    result = results[-1]
    if result.get("is_error"):
        raise LLMError(f"claude reported an error: {result.get('result')!r}")
    return LLMReply(
        text=str(result.get("result", "")),
        session_id=str(result.get("session_id", "")),
        cost_usd=float(result.get("total_cost_usd") or 0.0),
        duration_s=float(result.get("duration_ms") or 0) / 1000.0,
        usage=dict(result.get("usage") or {}),
        model_usage=dict(result.get("modelUsage") or {}),
    )


class ClaudeCodeLLM:
    """Calls `claude -p` once per prompt, retrying on failure."""

    def __init__(self, timeout_s: int, model: str | None = None, retries: int = 1) -> None:
        self._timeout_s = timeout_s
        self._model = model
        self._retries = retries
        self._cwd = Path(tempfile.mkdtemp(prefix="matxfer-llm-"))  # empty: no CLAUDE.md to find

    def _command(self, prompt: str) -> list[str]:
        cmd = ["claude", "-p", prompt, *_ISOLATION_ARGS]
        if self._model:
            cmd += ["--model", self._model]
        return cmd

    def __call__(self, prompt: str) -> LLMReply:
        last: Exception | None = None
        for attempt in range(1 + self._retries):
            try:
                proc = subprocess.run(
                    self._command(prompt),
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=self._timeout_s,
                    stdin=subprocess.DEVNULL,
                    cwd=self._cwd,
                    check=False,
                )
                if proc.returncode != 0:
                    raise LLMError(f"claude exited {proc.returncode}: {proc.stderr[:300]!r}")
                return parse_cli_output(proc.stdout)
            except (subprocess.TimeoutExpired, OSError, LLMError) as exc:
                last = exc
                logger.warning("claude call failed (attempt %d): %s", attempt + 1, exc)
        raise LLMError(f"claude failed after {1 + self._retries} attempts: {last}") from last
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v -m "not llm"` then `uv run pytest tests/test_llm.py -v -m llm`
Expected: first run all pass; second run (real call, about $0.005) passes. If the real call fails with an authentication error, stop and report: the isolation flags are the verified ones, so the cause is the login state.

---

### Task 5: Candidate parsing, `Step`, prompts

**Files:**
- Create: `src/candidate.py`, `src/prompts.py`, `task/prompt_task.md`
- Test: `tests/conftest.py`, `tests/test_candidate.py`, `tests/test_prompts.py`

**Interfaces:**
- Consumes: `RunResult` (Task 3).
- Produces in `src/candidate.py`: `ParseError`; `Candidate(code: str, flags: tuple[str, ...])`; `parse_candidate(text: str) -> Candidate`; `Step(code, flags, result, feedback=None)` (frozen dataclass); `RunFn = Callable[[str, tuple[str, ...]], RunResult]`; `evaluate_reply(text: str, run_fn: RunFn) -> Step`; `best_step(steps: Sequence[Step]) -> Step | None`.
- Produces in `src/prompts.py`: `load_task() -> str`; `format_observation(result: RunResult) -> str`; `generation_prompt(task: str, memory: Sequence[str]) -> str`; `feedback_prompt(task: str, history: Sequence[Step]) -> str`; `refine_prompt(task: str, history: Sequence[Step]) -> str` (last step has `feedback` set); `reflection_prompt(task: str, history: Sequence[Step], cap_words: int) -> str`; `shorten_prompt(text: str, cap_words: int) -> str`; `failure_lesson_prompt(task: str, step: Step, memory: Sequence[str], cap_words: int) -> str`.
- Produces in `tests/conftest.py` (used by Tasks 6 to 9): `FakeLLM(replies)` with `.prompts`, `code_reply(tag, flags="-O3")`, `ok(speedup)`, `fail(stage, msg="boom")`, `scripted_run(table)`.

- [ ] **Step 1: Write `task/prompt_task.md`**

````markdown
You are optimizing a C function for speed.

Write a single C source file that defines

    void matmul(const float *A, const float *B, float *C, int n);

which computes C = A * B for square n x n matrices of single-precision (float) values stored in row-major order, so element (i, j) of A is A[i*n + j]. The three buffers do not overlap. C is not initialized: your function must write every element of C.

Requirements:
- The result must be correct for every positive n, not only for convenient sizes. Results are checked against a reference with a relative tolerance of about 1e-3, so reordering floating-point sums is fine.
- The benchmark runs large matrices (around a thousand rows and columns) and reports the best of several runs, so optimize for large n.
- The file is compiled with gcc and linked, together with a fixed benchmark harness that calls matmul, against the C math library and pthreads. Do not define main. Only the C standard library, libm, pthreads and OpenMP are available.
- You choose the compiler flags. They are passed to gcc as given (for example -O3 or -fopenmp). Flags that set the output file, add include or library paths, or pass options through to the linker are rejected.
- Optimize for speed on the machine this runs on.

Reply format. Reply with the complete source file in exactly one fenced block:

```c
...
```

followed on its own line by the compiler flags, for example:

FLAGS: -O3 -march=native

Do not write any other code block or any other line starting with FLAGS:.
````

(The file content is exactly the text between the outer four-backtick fences, including the inner triple-backtick block.)

- [ ] **Step 2: Write `tests/conftest.py`**

```python
from collections.abc import Mapping

from src.candidate import RunFn
from src.llm import LLMReply
from src.sandbox import RunResult


class FakeLLM:
    """Scripted LLM: returns queued replies and records every prompt."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> LLMReply:
        self.prompts.append(prompt)
        if not self._replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        return LLMReply(self._replies.pop(0), "sid", 0.0, 0.0, {}, {})


def code_reply(tag: str, flags: str = "-O3") -> str:
    """A well-formed candidate reply; the first code line is `// <tag>`."""
    return (
        "Here you go.\n```c\n"
        f"// {tag}\n"
        "void matmul(const float *A, const float *B, float *C, int n) {}\n"
        f"```\nFLAGS: {flags}\n"
    )


def ok(speedup: float) -> RunResult:
    return RunResult("ok", True, True, 100.0 / speedup, speedup, "")


def fail(stage: str, msg: str = "boom") -> RunResult:
    return RunResult(stage, stage != "compile", False, None, None, msg)


def scripted_run(table: Mapping[str, RunResult]) -> RunFn:
    """RunFn that looks up the result by the candidate's `// <tag>` first line."""

    def run(code: str, flags: tuple[str, ...]) -> RunResult:
        return table[code.splitlines()[0].removeprefix("// ")]

    return run
```

- [ ] **Step 3: Write the failing tests**

`tests/test_candidate.py`:

```python
import pytest

from src.candidate import ParseError, Step, best_step, evaluate_reply, parse_candidate
from tests.conftest import code_reply, fail, ok, scripted_run


def test_parse_basic() -> None:
    cand = parse_candidate(code_reply("v0", "-O3 -fopenmp"))
    assert "matmul" in cand.code and cand.flags == ("-O3", "-fopenmp")


def test_parse_takes_last_matmul_block_and_last_flags_line() -> None:
    text = (
        "```c\nint helper(void) { return 1; }\n```\nthinking...\n"
        "```c\n// FLAGS: -O0\nvoid matmul(const float *A, const float *B, float *C, int n) {}\n```\n"
        "FLAGS: -O2\nFLAGS: -O3 -fopenmp\n"
    )
    cand = parse_candidate(text)
    assert "helper" not in cand.code
    assert cand.flags == ("-O3", "-fopenmp")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no code here\nFLAGS: -O3",
        "```c\nint helper(void) { return 1; }\n```\nFLAGS: -O3",
        "```c\nvoid matmul(){}\n```\nno flags line",
        '```c\nvoid matmul(){}\n```\nFLAGS: -O3 "unterminated',
    ],
)
def test_parse_rejects_bad_replies(text: str) -> None:
    with pytest.raises(ParseError):
        parse_candidate(text)


def test_evaluate_reply_turns_parse_errors_into_failed_steps() -> None:
    step = evaluate_reply("garbage", scripted_run({}))
    assert step.result.stage == "parse" and not step.result.success
    assert step.result.stderr


def test_evaluate_reply_runs_good_candidates() -> None:
    step = evaluate_reply(code_reply("v0"), scripted_run({"v0": ok(2.0)}))
    assert step.result.success and step.flags == ("-O3",)


def test_best_step_is_fastest_correct_not_last() -> None:
    steps = [
        Step("a", (), ok(1.0)),
        Step("b", (), ok(3.0)),
        Step("c", (), fail("incorrect")),
        Step("d", (), ok(2.0)),
    ]
    assert best_step(steps) is steps[1]
    assert best_step([Step("x", (), fail("compile"))]) is None
```

`tests/test_prompts.py`:

```python
from src.candidate import Step
from src.prompts import (
    failure_lesson_prompt,
    feedback_prompt,
    format_observation,
    generation_prompt,
    load_task,
    refine_prompt,
    reflection_prompt,
    shorten_prompt,
)
from tests.conftest import fail, ok

ARCH_WORDS = ("x86", "amd64", "arm", "avx", "neon", "sse", "32-bit", "64-bit")


def test_task_statement_never_names_the_architecture() -> None:
    text = load_task().lower()
    assert not any(word in text for word in ARCH_WORDS)


def test_generation_prompt_memory_block_is_the_only_difference() -> None:
    task = load_task()
    assert generation_prompt(task, ()) == task
    with_memory = generation_prompt(task, ["use blocking"])
    assert with_memory.startswith(task) and "use blocking" in with_memory


def test_format_observation_ok_and_failures() -> None:
    assert "3.00x" in format_observation(ok(3.0))
    assert "Compilation failed" in format_observation(fail("compile", "bad flag"))
    assert "bad flag" in format_observation(fail("compile", "bad flag"))


def test_history_prompts_carry_code_observations_and_feedback() -> None:
    history = [Step("// v0\nint a;", ("-O2",), ok(1.5), "tile the loops"), Step("// v1", (), ok(2.0))]
    fb = feedback_prompt("TASK", history)
    assert "// v0" in fb and "1.50x" in fb and "tile the loops" in fb
    rf = refine_prompt("TASK", history)
    assert "Feedback" in rf and "FLAGS" in rf


def test_reflection_and_lesson_prompts_state_the_word_cap() -> None:
    step = Step("// v0", ("-O3",), fail("compile", "unknown option"))
    assert "200 words" in reflection_prompt("TASK", [step], 200)
    lesson = failure_lesson_prompt("TASK", step, ["old note"], 80)
    assert "80 words" in lesson and "unknown option" in lesson and "old note" in lesson
    assert "50 words" in shorten_prompt("long text", 50)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_candidate.py tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.candidate'`.

- [ ] **Step 5: Write `src/candidate.py`**

```python
"""Parsing Claude's replies into candidates and evaluating them."""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.sandbox import RunResult

_FENCE = re.compile(r"```[A-Za-z]*[ \t]*\n(.*?)```", re.S)
_FLAGS = re.compile(r"^[ \t]*FLAGS:[ \t]*(.*?)[ \t]*$", re.M)
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
```

- [ ] **Step 6: Write `src/prompts.py`**

```python
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


def load_task() -> str:
    """Return the shared task statement."""
    return TASK_PATH.read_text().strip() + "\n"


def format_observation(result: RunResult) -> str:
    """Render a measured result the way the model sees it."""
    if result.stage == "ok":
        parts = ["Compiled and passed the correctness checks.", f"Runtime: {result.time_ms:.1f} ms."]
        if result.speedup is not None:
            parts.append(f"Speedup versus the naive triple loop: {result.speedup:.2f}x.")
        if result.stderr:
            parts.append(f"Compiler output:\n{result.stderr}")
    else:
        parts = [f"{_LABELS.get(result.stage, result.stage)}.", result.stderr]
    return "\n".join(p for p in parts if p)


def _render_step(index: int, step: Step) -> str:
    lines = [
        f"### Attempt {index}",
        f"FLAGS: {' '.join(step.flags)}",
        "```c",
        step.code.rstrip("\n"),
        "```",
        f"Observation: {format_observation(step.result)}",
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
        f"{task}\nYou just finished the series of attempts below.\n\n{_render_history(history)}\n\n"
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


def failure_lesson_prompt(task: str, step: Step, memory: Sequence[str], cap_words: int) -> str:
    """Ask for a short lesson after a failed trial (Reflexion self-reflection)."""
    notes = "\n".join(f"- {m.strip()}" for m in memory) or "(none)"
    return (
        f"{task}\nMemory you were given for this attempt:\n{notes}\n\n"
        f"Your attempt failed.\n{_render_step(0, step)}\n\n"
        f"Write a short lesson, at most {cap_words} words and no code, saying what went wrong "
        "and what to do differently next time. Reply with the lesson only.\n"
    )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_candidate.py tests/test_prompts.py -v`
Expected: all pass. If `test_task_statement_never_names_the_architecture` fails on a substring (for example "arm" inside "alarm"), fix the task text wording, not the word list.

---

### Task 6: Self-Refine loop (`src/refine.py`)

**Files:**
- Create: `src/refine.py`
- Test: `tests/test_refine.py`

**Interfaces:**
- Consumes: `LLM`, `Config`, `TraceScope`, `Step`, `RunFn`, `evaluate_reply`, `best_step`, `generation_prompt`, `feedback_prompt`, `refine_prompt`, `load_task` (Tasks 1, 4, 5).
- Produces: `RefineOutcome(history: tuple[Step, ...], best: Step | None)`; `self_refine(task: str, llm: LLM, run_fn: RunFn, cfg: Config, trace: TraceScope) -> RefineOutcome`. History has `rounds + 1` steps; every step except the last has `feedback` set.

- [ ] **Step 1: Write the failing tests**

`tests/test_refine.py`:

```python
from pathlib import Path

from src.config import Config
from src.refine import self_refine
from src.traces import TraceScope
from tests.conftest import FakeLLM, code_reply, fail, ok, scripted_run


def test_self_refine_runs_rounds_and_keeps_history(tmp_path: Path) -> None:
    cfg = Config(rounds=2)
    llm = FakeLLM([code_reply("v0"), "fb0", code_reply("v1"), "fb1", code_reply("v2")])
    run_fn = scripted_run({"v0": ok(1.0), "v1": ok(3.0), "v2": ok(2.0)})
    out = self_refine("TASK", llm, run_fn, cfg, TraceScope(tmp_path, "A", 1))

    assert len(out.history) == 3
    assert [s.feedback for s in out.history] == ["fb0", "fb1", None]
    assert out.best == out.history[1]  # fastest correct, not the last
    assert "fb0" in llm.prompts[2]  # refine prompt carries the feedback
    assert "// v0" in llm.prompts[3]  # round-2 feedback prompt carries earlier attempts
    assert len(list(tmp_path.glob("*.json"))) == 8  # 5 llm calls + 3 sandbox runs


def test_unparseable_reply_is_a_failed_step_and_the_loop_continues(tmp_path: Path) -> None:
    cfg = Config(rounds=2)
    llm = FakeLLM([code_reply("v0"), "fb0", "I refuse to write code", "fb1", code_reply("v2")])
    run_fn = scripted_run({"v0": ok(1.0), "v2": ok(2.0)})
    out = self_refine("TASK", llm, run_fn, cfg, TraceScope(tmp_path, "A", 1))
    assert len(out.history) == 3
    assert out.history[1].result.stage == "parse"
    assert out.best == out.history[2]


def test_best_is_none_when_nothing_is_correct(tmp_path: Path) -> None:
    cfg = Config(rounds=1)
    llm = FakeLLM([code_reply("v0"), "fb0", code_reply("v1")])
    run_fn = scripted_run({"v0": fail("compile"), "v1": fail("incorrect")})
    out = self_refine("TASK", llm, run_fn, cfg, TraceScope(tmp_path, "A", 1))
    assert out.best is None and len(out.history) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_refine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.refine'`.

- [ ] **Step 3: Write `src/refine.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_refine.py -v`
Expected: 3 passed.

---

### Task 7: Memory and Reflexion trials (`src/memory.py`, `src/reflexion.py`)

**Files:**
- Create: `src/memory.py`, `src/reflexion.py`
- Test: `tests/test_memory.py`, `tests/test_reflexion.py`

**Interfaces:**
- Consumes: Tasks 1, 4, 5, 6 interfaces.
- Produces in `src/memory.py`: `EpisodicMemory(entries: tuple[str, ...] = (), window: int = 3)` with `.append(entry: str) -> EpisodicMemory` (drops the oldest beyond `window`); `clip_words(text: str, cap: int) -> str`; `write_memory(llm, task, history, cfg, trace) -> str` (reflection call, one shorten retry if over `cfg.memory_cap_words`, then hard clip); `reflect_on_failure(llm, task, step, memory, cfg, trace) -> str` (clipped to `cfg.lesson_cap_words`).
- Produces in `src/reflexion.py`: `TrialOutcome(trials: tuple[Step, ...], memory: EpisodicMemory, first_success_trial: int | None)`; `reflexion_trials(task, llm, run_fn, memory, cfg, trace) -> TrialOutcome` (up to `cfg.trials` trials; success means compiled and correct; a lesson is written and appended after every failed trial except the last).

- [ ] **Step 1: Write the failing tests**

`tests/test_memory.py`:

```python
from pathlib import Path

from src.candidate import Step
from src.config import Config
from src.memory import EpisodicMemory, clip_words, reflect_on_failure, write_memory
from src.traces import TraceScope
from tests.conftest import FakeLLM, fail, ok


def test_window_drops_oldest() -> None:
    mem = EpisodicMemory(window=3)
    for entry in ("a", "b", "c", "d"):
        mem = mem.append(entry)
    assert mem.entries == ("b", "c", "d")


def test_memory_is_immutable_value() -> None:
    base = EpisodicMemory(("x",), 3)
    assert base.append("y").entries == ("x", "y") and base.entries == ("x",)


def test_clip_words() -> None:
    assert clip_words("one two three four", 2) == "one two"
    assert clip_words("  short  ", 5) == "short"


def test_write_memory_retries_once_when_over_cap(tmp_path: Path) -> None:
    cfg = Config(memory_cap_words=5)
    llm = FakeLLM(["one two three four five six seven", "one two three"])
    text = write_memory(llm, "TASK", [Step("c", (), ok(2.0))], cfg, TraceScope(tmp_path, "A", 1))
    assert text == "one two three" and len(llm.prompts) == 2


def test_write_memory_hard_clips_when_retry_is_still_too_long(tmp_path: Path) -> None:
    cfg = Config(memory_cap_words=3)
    llm = FakeLLM(["a b c d e", "a b c d"])
    text = write_memory(llm, "TASK", [Step("c", (), ok(2.0))], cfg, TraceScope(tmp_path, "A", 1))
    assert text == "a b c"


def test_reflect_on_failure_clips_to_lesson_cap(tmp_path: Path) -> None:
    cfg = Config(lesson_cap_words=3)
    llm = FakeLLM(["w1 w2 w3 w4 w5"])
    step = Step("c", ("-O3",), fail("compile", "unknown option"))
    lesson = reflect_on_failure(
        llm, "TASK", step, EpisodicMemory(("old",), 3), cfg, TraceScope(tmp_path, "C", 1)
    )
    assert lesson == "w1 w2 w3" and "old" in llm.prompts[0]
```

`tests/test_reflexion.py`:

```python
from pathlib import Path

from src.config import Config
from src.memory import EpisodicMemory
from src.reflexion import reflexion_trials
from src.traces import TraceScope
from tests.conftest import FakeLLM, code_reply, fail, ok, scripted_run


def _scope(tmp_path: Path) -> TraceScope:
    return TraceScope(tmp_path, "C", 1)


def test_first_try_success_makes_one_call(tmp_path: Path) -> None:
    llm = FakeLLM([code_reply("t1")])
    out = reflexion_trials(
        "TASK", llm, scripted_run({"t1": ok(2.0)}), EpisodicMemory(("x86 note",), 3),
        Config(), _scope(tmp_path),
    )  # fmt: skip
    assert out.first_success_trial == 1 and len(out.trials) == 1 and len(llm.prompts) == 1
    assert "x86 note" in llm.prompts[0]  # memory is appended to the first prompt


def test_failures_write_lessons_and_memory_grows_with_original_first(tmp_path: Path) -> None:
    llm = FakeLLM([code_reply("t1"), "lesson one", code_reply("t2"), "lesson two", code_reply("t3")])
    run_fn = scripted_run({"t1": fail("compile"), "t2": fail("compile"), "t3": ok(1.5)})
    out = reflexion_trials(
        "TASK", llm, run_fn, EpisodicMemory(("original",), 3), Config(), _scope(tmp_path)
    )
    assert out.first_success_trial == 3
    assert out.memory.entries == ("original", "lesson one", "lesson two")
    assert "lesson one" in llm.prompts[2] and "lesson two" in llm.prompts[4]


def test_all_trials_fail_and_no_lesson_after_the_last(tmp_path: Path) -> None:
    llm = FakeLLM([code_reply("t1"), "l1", code_reply("t2"), "l2", code_reply("t3")])
    run_fn = scripted_run({"t1": fail("compile"), "t2": fail("run"), "t3": fail("incorrect")})
    out = reflexion_trials("TASK", llm, run_fn, EpisodicMemory((), 3), Config(), _scope(tmp_path))
    assert out.first_success_trial is None and len(out.trials) == 3
    assert len(llm.prompts) == 5  # 3 generations + 2 lessons, none after the final failure
    assert out.memory.entries == ("l1", "l2")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_memory.py tests/test_reflexion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.memory'`.

- [ ] **Step 3: Write `src/memory.py`**

```python
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
    prompt = failure_lesson_prompt(task, step, memory.entries, cfg.lesson_cap_words)
    reply = llm(prompt)
    trace.log_llm("lesson", prompt, reply)
    return clip_words(reply.text, cfg.lesson_cap_words)
```

- [ ] **Step 4: Write `src/reflexion.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory.py tests/test_reflexion.py -v`
Expected: 9 passed.

---

### Task 8: Report (`src/report.py`)

**Files:**
- Create: `src/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `RefineOutcome` (Task 6), `TrialOutcome` (Task 7), `Config`.
- Produces: `RefineRecord(repeat, speedups: tuple[float | None, ...], best_speedup, memory: str | None, note: str = "")`; `ArmRecord(arm, repeat, first_try_success, trials_to_success, final_speedup, first_failure_stage, n_steps, skipped: bool = False, note: str = "")`; `summarize_refine(repeat, outcome, memory) -> RefineRecord`; `summarize_trials(arm, repeat, outcome) -> ArmRecord`; `skipped_arm(arm, repeat, note) -> ArmRecord`; `write_report(run_dir: Path, cfg: Config, naive_ms: Mapping[str, float], refine: Sequence[RefineRecord], arms: Sequence[ArmRecord]) -> None` writing `results.json`, `results.md`, `speedup_by_round.png`, `README.md` into `run_dir`.

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:

```python
import json
from pathlib import Path

from src.candidate import Step
from src.config import Config
from src.memory import EpisodicMemory
from src.reflexion import TrialOutcome
from src.refine import RefineOutcome
from src.report import (
    ArmRecord,
    RefineRecord,
    skipped_arm,
    summarize_refine,
    summarize_trials,
    write_report,
)
from tests.conftest import fail, ok


def test_summarize_refine_uses_none_for_failed_attempts() -> None:
    history = (Step("a", (), ok(1.0)), Step("b", (), fail("compile")), Step("c", (), ok(3.0)))
    rec = summarize_refine(1, RefineOutcome(history, history[2]), "memory text")
    assert rec.speedups == (1.0, None, 3.0) and rec.best_speedup == 3.0 and rec.memory


def test_summarize_refine_without_a_correct_candidate_has_a_note() -> None:
    history = (Step("a", (), fail("compile")),)
    rec = summarize_refine(2, RefineOutcome(history, None), None)
    assert rec.best_speedup is None and "no correct candidate" in rec.note


def test_summarize_trials_first_try_and_later_success() -> None:
    first = TrialOutcome((Step("a", (), ok(2.0)),), EpisodicMemory(), 1)
    later = TrialOutcome(
        (Step("a", (), fail("compile")), Step("b", (), ok(1.5))), EpisodicMemory(), 2
    )
    never = TrialOutcome((Step("a", (), fail("run")),), EpisodicMemory(), None)
    a, b, c = (summarize_trials("C", 1, o) for o in (first, later, never))
    assert a.first_try_success and a.trials_to_success == 1 and a.final_speedup == 2.0
    assert not b.first_try_success and b.trials_to_success == 2 and b.first_failure_stage == "compile"
    assert c.trials_to_success is None and c.final_speedup is None


def test_write_report_creates_all_outputs(tmp_path: Path) -> None:
    refine = [RefineRecord(1, (1.0, 2.0, None, 3.0), 3.0, "use blocking")]
    arms = [
        ArmRecord("B", 1, True, 1, 1.2, None, 1),
        ArmRecord("C", 1, False, 2, 1.1, "compile", 2),
        skipped_arm("C", 2, "no memory"),
    ]
    write_report(tmp_path, Config(), {"amd64": 100.0, "armv7": 900.0}, refine, arms)
    for name in ("results.json", "results.md", "speedup_by_round.png", "README.md"):
        assert (tmp_path / name).exists(), name
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["naive_ms"]["armv7"] == 900.0 and len(data["arms"]) == 3
    md = (tmp_path / "results.md").read_text()
    assert "First-try success" in md and "compile" in md
    assert "A/r1" in (tmp_path / "README.md").read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.report'`.

- [ ] **Step 3: Write `src/report.py`**

```python
"""Aggregate records into results.json, results.md, a per-round plot and a traces README."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import Config  # noqa: E402
from src.reflexion import TrialOutcome  # noqa: E402
from src.refine import RefineOutcome  # noqa: E402


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
    """Arm B or C summary for one repeat."""

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
    """Summarize an arm B or C run."""
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
    vals = [r.trials_to_success for r in records if not r.skipped and r.trials_to_success]
    return f"{sum(vals) / len(vals):.2f}" if vals else "-"


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
    cfg: Config, naive_ms: Mapping[str, float], refine: Sequence[RefineRecord], arms: Sequence[ArmRecord]
) -> str:
    lines = [
        "# Results",
        "",
        f"Config: n={cfg.n}, rounds={cfg.rounds}, repeats={cfg.repeats}, trials={cfg.trials}, "
        f"window={cfg.window}. Naive baseline (ms): "
        + ", ".join(f"{k}={v:.1f}" for k, v in naive_ms.items()),
        "",
        "## Arm A: Self-Refine on amd64",
        "",
        "| repeat | best speedup | speedup per attempt | note |",
        "|---|---|---|---|",
    ]
    for r in refine:
        per = ", ".join(_fmt(v) for v in r.speedups)
        lines.append(f"| {r.repeat} | {_fmt(r.best_speedup)} | {per} | {r.note} |")
    lines += ["", "## Arms B (no memory) and C (x86 memory) on arm/v7", ""]
    lines += ["| arm | First-try success | mean trials to success |", "|---|---|---|"]
    for arm in ("B", "C"):
        recs = [a for a in arms if a.arm == arm]
        lines.append(f"| {arm} | {_rate(recs)} | {_mean_trials(recs)} |")
    lines += ["", "### Per repeat", "", "| arm | repeat | first try | trials to success | final speedup | first failure stage | note |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        lines.append(
            f"| {a.arm} | {a.repeat} | {'yes' if a.first_try_success else 'no'} | "
            f"{a.trials_to_success or '-'} | {_fmt(a.final_speedup)} | {a.first_failure_stage or '-'} | {a.note} |"
        )
    return "\n".join(lines) + "\n"


def _readme(run_dir: Path, refine: Sequence[RefineRecord], arms: Sequence[ArmRecord]) -> str:
    lines = [
        f"# Traces for {run_dir.name}",
        "",
        "Layout: `<arm>/r<repeat>/<NN>-<kind>.json` (kinds: `llm`, `sandbox`, `memory_update`, ...).",
        "`A` = Self-Refine on amd64 (its `memory.md` is the distilled memory), `B` = arm/v7 trials",
        "without memory, `C` = arm/v7 trials with A's memory.",
        "",
        "## Scenarios",
        "",
    ]
    for r in refine:
        lines.append(
            f"- A/r{r.repeat}: best speedup {_fmt(r.best_speedup)}x over {len(r.speedups)} attempts. {r.note}".rstrip()
        )
    for a in arms:
        if a.skipped:
            lines.append(f"- {a.arm}/r{a.repeat}: skipped. {a.note}")
            continue
        outcome = (
            "succeeded on the first try"
            if a.first_try_success
            else f"first try failed at stage '{a.first_failure_stage}'"
        )
        tail = (
            f"; succeeded at trial {a.trials_to_success} (speedup {_fmt(a.final_speedup)}x)"
            if a.trials_to_success and not a.first_try_success
            else ("" if a.first_try_success else "; never succeeded")
        )
        lines.append(f"- {a.arm}/r{a.repeat}: {outcome}{tail}.")
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: 4 passed.

---

### Task 9: Orchestrator (`run.py`)

**Files:**
- Create: `run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_repeat(repeat, cfg, llm, writer, run_fn_for, task) -> tuple[RefineRecord, list[ArmRecord]]` (`run_fn_for: Callable[[Platform], RunFn]`); `run_experiment(cfg, llm, run_fn_for, naive_ms, out_dir, run_id) -> Path`; `preflight(cfg) -> dict[str, float]`; `main(argv: Sequence[str] | None = None) -> int`. CLI flags: `--quick`, `--n`, `--repeats`, `--rounds`, `--trials`, `--model`, `--out` (default `traces`), `--run-id`.

- [ ] **Step 1: Write the failing tests**

`tests/test_run.py`:

```python
import json
from pathlib import Path

from run import run_experiment
from src.config import Config
from src.llm import LLMError, LLMReply
from src.sandbox import AMD64, ARMV7, Platform
from tests.conftest import FakeLLM, code_reply, fail, ok, scripted_run

NAIVE = {"amd64": 100.0, "armv7": 900.0}


def _run_fn_for(amd: dict, arm: dict):
    tables = {AMD64.name: amd, ARMV7.name: arm}

    def for_platform(platform: Platform):
        return scripted_run(tables[platform.name])

    return for_platform


def test_full_repeat_with_fakes_writes_traces_memory_and_report(tmp_path: Path) -> None:
    cfg = Config(rounds=1, repeats=1, trials=2)
    llm = FakeLLM(
        [
            code_reply("a0"), "fb", code_reply("a1"),  # arm A: generate, feedback, refine
            "memory: use avx2 intrinsics",  # memory reflection
            code_reply("b1"), "lesson b", code_reply("b2"),  # arm B: fail, lesson, succeed
            code_reply("c1"), "lesson c", code_reply("c2"),  # arm C: fail, lesson, succeed
        ]
    )
    run_fn_for = _run_fn_for(
        {"a0": ok(1.0), "a1": ok(4.0)},
        {"b1": fail("compile"), "b2": ok(1.2), "c1": fail("compile"), "c2": ok(1.1)},
    )
    run_dir = run_experiment(cfg, llm, run_fn_for, NAIVE, tmp_path, "t1")

    assert (run_dir / "A" / "r1" / "memory.md").read_text().strip() == "memory: use avx2 intrinsics"
    assert "memory: use avx2 intrinsics" in llm.prompts[4 + 3]  # arm C's first prompt (after B)
    assert json.loads((run_dir / "run_meta.json").read_text())["naive_ms"] == NAIVE
    data = json.loads((run_dir / "results.json").read_text())
    assert [a["arm"] for a in data["arms"]] == ["B", "C"]
    assert (run_dir / "results.md").exists() and (run_dir / "README.md").exists()


def test_no_correct_candidate_in_arm_a_skips_arm_c(tmp_path: Path) -> None:
    cfg = Config(rounds=1, repeats=1, trials=1)
    llm = FakeLLM([code_reply("a0"), "fb", code_reply("a1"), code_reply("b1")])
    run_fn_for = _run_fn_for(
        {"a0": fail("compile"), "a1": fail("incorrect")}, {"b1": ok(1.0)}
    )
    run_dir = run_experiment(cfg, llm, run_fn_for, NAIVE, tmp_path, "t2")
    data = json.loads((run_dir / "results.json").read_text())
    arms = {a["arm"]: a for a in data["arms"]}
    assert arms["B"]["first_try_success"] and arms["C"]["skipped"]
    assert data["refine"][0]["memory"] is None


class _DyingLLM(FakeLLM):
    """Raises LLMError on the first call of the first repeat, then behaves normally."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies)
        self._died = False

    def __call__(self, prompt: str) -> LLMReply:
        if not self._died:
            self._died = True
            raise LLMError("boom")
        return super().__call__(prompt)


def test_llm_failure_in_one_repeat_does_not_lose_the_others(tmp_path: Path) -> None:
    cfg = Config(rounds=0, repeats=2, trials=1)
    llm = _DyingLLM([code_reply("a0"), "memory", code_reply("b1"), code_reply("c1")])
    run_fn_for = _run_fn_for({"a0": ok(2.0)}, {"b1": ok(1.0), "c1": ok(1.0)})
    run_dir = run_experiment(cfg, llm, run_fn_for, NAIVE, tmp_path, "t3")
    data = json.loads((run_dir / "results.json").read_text())
    assert [r["repeat"] for r in data["refine"]] == [2]
    assert (run_dir / "run" / "r1" / "01-llm_error.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run'`.

- [ ] **Step 3: Write `run.py`**

```python
"""Run the Self-Refine -> memory -> Reflexion transfer experiment end to end."""

from __future__ import annotations

import argparse
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
from src.config import Config
from src.llm import LLM, ClaudeCodeLLM, LLMError
from src.memory import EpisodicMemory, write_memory
from src.prompts import load_task
from src.reflexion import reflexion_trials
from src.refine import self_refine
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


def write_run_meta(run_dir: Path, cfg: Config, naive_ms: dict[str, float]) -> None:
    """Record config and environment for reproducibility."""
    meta = {
        "config": asdict(cfg),
        "naive_ms": naive_ms,
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
) -> tuple[RefineRecord, list[ArmRecord]]:
    """One repeat: arm A, then arm B (no memory) and arm C (A's memory) on arm/v7."""
    a_scope = writer.scope("A", repeat)
    outcome = self_refine(task, llm, run_fn_for(AMD64), cfg, a_scope)
    memory_text: str | None = None
    if outcome.best is not None:
        memory_text = write_memory(llm, task, outcome.history, cfg, a_scope)
        (a_scope.directory / "memory.md").write_text(memory_text + "\n")
    refine_record = summarize_refine(repeat, outcome, memory_text)

    arms: list[ArmRecord] = []
    b_out = reflexion_trials(
        task, llm, run_fn_for(ARMV7), EpisodicMemory((), cfg.window), cfg, writer.scope("B", repeat)
    )
    arms.append(summarize_trials("B", repeat, b_out))
    if memory_text is None:
        arms.append(skipped_arm("C", repeat, "no correct candidate on amd64, so no memory to transfer"))
    else:
        c_out = reflexion_trials(
            task, llm, run_fn_for(ARMV7), EpisodicMemory((memory_text,), cfg.window), cfg,
            writer.scope("C", repeat),
        )  # fmt: skip
        arms.append(summarize_trials("C", repeat, c_out))
    return refine_record, arms


def run_experiment(
    cfg: Config,
    llm: LLM,
    run_fn_for: RunFnFor,
    naive_ms: dict[str, float],
    out_dir: Path,
    run_id: str,
) -> Path:
    """Run all repeats, tolerating LLM failures per repeat, and write the report."""
    task = load_task()
    writer = TraceWriter(out_dir, run_id)
    write_run_meta(writer.run_dir, cfg, naive_ms)
    refine_records: list[RefineRecord] = []
    arm_records: list[ArmRecord] = []
    for repeat in range(1, cfg.repeats + 1):
        logger.info("repeat %d/%d", repeat, cfg.repeats)
        try:
            refine_record, arms = run_repeat(repeat, cfg, llm, writer, run_fn_for, task)
        except LLMError as exc:
            logger.error("repeat %d aborted: %s", repeat, exc)
            writer.scope("run", repeat).log("llm_error", error=str(exc))
            continue
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
            raise SystemExit(f"naive reference failed on {platform.name}: {result.stage}\n{result.stderr}")
        total_s = result.time_ms * cfg.reps / 1000.0
        logger.info("naive on %s: %.1f ms per run (%.1f s for %d reps)", platform.name, result.time_ms, total_s, cfg.reps)
        if total_s > cfg.max_naive_s:
            raise SystemExit(
                f"naive run on {platform.name} takes {total_s:.0f} s (> {cfg.max_naive_s:.0f} s); "
                f"lower --n or raise max_naive_s"
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
            "trials": args.trials, "model": args.model,
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
    parser.add_argument("--out", default="traces")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = build_config(args)
    naive = preflight(cfg)

    def run_fn_for(platform: Platform) -> RunFn:
        return lambda code, flags: run_candidate(platform, code, flags, cfg, naive[platform.name])

    run_id = args.run_id or time.strftime("matxfer_%Y%m%d_%H%M%S")
    llm = ClaudeCodeLLM(cfg.llm_timeout_s, cfg.model)
    run_dir = run_experiment(cfg, llm, run_fn_for, naive, Path(args.out), run_id)
    logger.info("done: %s", run_dir / "results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_run.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the whole suite and static checks**

Run:
```bash
uv run pytest -q -m "not llm"
uv run ruff check . && uv run mypy src/ run.py
```
Expected: all tests pass; ruff and mypy report no errors. Fix reported issues in the owning file (unused imports, line length, missing annotations) without changing behavior.

---

### Task 10: Run end to end

**Files:**
- Create (by running): `traces/<run_id>/...`, `.work/` scratch (auto-removed per candidate)

**Interfaces:**
- Consumes: the finished `run.py`.
- Produces: `traces/matxfer_quick/` and `traces/matxfer_full/` with `results.md`, `results.json`, `speedup_by_round.png`, `README.md`, `run_meta.json` and per-step traces.

- [ ] **Step 1: Calibrate the matrix size**

Run: `uv run python run.py --quick --n 1024 --rounds 0 --trials 1 --run-id matxfer_calib`
Expected: the preflight log lines `naive on amd64: ... ms` and `naive on armv7: ... ms`. If preflight aborts because the naive run exceeds `max_naive_s` (120 s), lower `--n` (for example 512) and repeat. Note the naive times: they decide `n` for the full run. This run also spends a few Claude calls (one generation in A, B and C), which is intended as a first live check of the whole path.

- [ ] **Step 2: Quick end-to-end run**

Run (in the background; it logs to the terminal):
`uv run python run.py --quick --n <n from Step 1> --run-id matxfer_quick`
Expected: log lines for 3 rounds of arm A, a memory write, then trial lines for B and C; finishes with `done: traces/matxfer_quick/results.md`.

- [ ] **Step 3: Inspect the quick run**

Run: `ls traces/matxfer_quick/A/r1 traces/matxfer_quick/B/r1 traces/matxfer_quick/C/r1 && cat traces/matxfer_quick/A/r1/memory.md traces/matxfer_quick/results.md`
Expected: numbered `llm` and `sandbox` JSON files in every arm directory, a `memory.md` of at most about 200 words, and a results table. Open one `llm` trace and one `sandbox` trace and confirm every field listed in Global Constraints is present, and confirm the arm B/C prompts do not name the architecture. Report anything surprising (for example Claude writing portable code in arm A) before the full run.

- [ ] **Step 4: Full run**

Run (in the background): `uv run python run.py --n <n from Step 1> --run-id matxfer_full`
Expected: 3 repeats of 10 rounds, then B and C; about 100 Claude Code calls. Watch `traces/matxfer_full/` for progress. If a repeat aborts with `llm_error`, the other repeats still complete (the failed repeat appears as `run/r<k>/01-llm_error.json`).

- [ ] **Step 5: Collect the results**

Run: `cat traces/matxfer_full/results.md traces/matxfer_full/README.md`
Expected: the arm A speedup table and curve (`speedup_by_round.png`), and the B versus C comparison of first-try success, trials to success and final speedup. Report the numbers exactly as measured, including a null result if the memory transferred.

---

## Self-review (done while writing)

- **Spec coverage:** sections 1 and 2 (arms, memory, trials, disclosed deviations) are Tasks 5 to 9; section 3 (metrics, repeats) is Task 8; section 4 (components) is Tasks 1 to 9 plus the four adjustments listed at the top; section 5 (size, Freivalds, preflight, cpus) is Tasks 2, 3 and 9; section 6 (error handling) is Tasks 3, 4, 5, 9; section 7 (testing, canned AVX2) is Task 3 and the fakes in Tasks 5 to 9; section 8 (traces) is Tasks 1, 6, 7, 9; section 9 (cost, risks) is Task 10 Step 3; no spec requirement is without a task.
- **Placeholder scan:** none; every code step has full code and every run step has the expected result.
- **Type consistency:** `Step`, `RunFn`, `RunResult`, `LLMReply`, `TraceScope.log_llm/log_step`, `EpisodicMemory`, `TrialOutcome`, `RefineOutcome` use the same names and fields wherever they appear.
