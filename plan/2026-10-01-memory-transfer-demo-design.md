# Memory Transfer Demo: Self-Refine on x86, Reflexion-style Memory on arm32

CS 239 team presentation, Lecture 3 (Oct 6). Main paper: Reflexion. Optional paper: Self-Refine.
Status: design, awaiting review.

## 1. Purpose and hypothesis

Show that an optimization lesson learned in one environment and stored as episodic memory can
fail to transfer to a different environment.

- Self-Refine (within-episode): Claude Code refines a C matmul for performance on `linux/amd64`
  over 10 rounds.
- Reflexion-style memory (across-episode): the result is distilled into a verbal memory and
  appended to the context of the same task on `linux/arm/v7` (32-bit ARM).
- Hypothesis: the x86-derived memory (e.g. AVX2/FMA intrinsics) causes compile or correctness
  failures, or no gain, on arm32. A null result (Claude writes portable code and the memory
  transfers) is still reportable.
- The Reflexion paper does not evaluate memory transfer across environments (per the paper as
  summarized during brainstorming), so this probes a limitation rather than reproducing a result.

Everything runs on a MacBook: both platforms run under Docker emulation.

## 2. Experimental arms

All arms share one task statement: write a fast `matmul(const float*A, const float*B, float*C, int n)`
in C and a `FLAGS:` line with compile flags. The prompt never names the architecture. Arms differ
only in the memory block.

| Arm | Platform | Memory | Procedure |
|-----|----------|--------|-----------|
| A | linux/amd64 | none | Self-Refine: generate, then 10 x (feedback call, refine call). Pick the fastest correct candidate. A reflection call writes `memory.md` (<= ~200 words, one retry if over). |
| B | linux/arm/v7 | none | Reflexion trials, max 3 |
| C | linux/arm/v7 | `memory.md` from A | Reflexion trials, max 3 |

Reflexion trial: one generation, run in the arm32 sandbox. On failure, a reflection call reads the
compiler or correctness error and writes a short lesson, appended to memory. Memory is a sliding
window of at most 3 entries (Reflexion's Omega); in C the x86 memory is entry 1.

B and C use the identical loop so memory is the only variable.

Deviations from the papers (disclosed on slides):
- Self-Refine feedback calls see measured compile/correctness/timing results, which is closer to
  Reflexion's grounded evaluator than pure self-feedback.
- 10 rounds exceeds Self-Refine's 4; per-round speedup is plotted.
- Final candidate is the best correct one, not the last.
- Memory is written from a success, whereas Reflexion writes reflections from failures.

## 3. Metrics

Per repeat: first-try success on arm32, trials to first success, final speedup versus naive on the
same platform. Speedup is a same-platform ratio, so QEMU slowdown largely cancels. Repeats default
to 3; each repeat pairs one A run with its own memory feeding C.

## 4. Components

```
comsci239_team/
  pyproject.toml            # uv
  Dockerfile                # gcc base, built per --platform
  task/harness.c            # calls candidate matmul, correctness check, best-of-k timing
  task/prompt_task.md       # shared task statement
  src/llm.py                # `claude -p --output-format json`, tools disabled
  src/sandbox.py            # compile + run a candidate in a container for a platform
  src/refine.py             # Self-Refine loop
  src/memory.py             # reflection calls, memory.md, sliding window
  src/traces.py             # one JSON per step
  src/config.py             # frozen dataclass
  run.py                    # orchestrates arms, repeats, results table; --quick flag
  traces/<run_id>/<arm>/<NN>-<kind>.json, memory.md, results.md, README.md
```

- Candidate contract: one `matmul.c` plus a `FLAGS:` line. The harness uses Claude's flags as
  given, so failures are attributable to the model's output, not the harness.
- `sandbox.py` returns `compiled`, `correct`, `time_ms`, `speedup`, `stderr`; it is the only code
  that touches Docker. `llm.py` is the only code that touches Claude Code.
- Config (frozen dataclass): rounds=10, repeats=3, memory_cap_words=200, window=3, trials=3, n,
  cpus, per-call and per-run timeouts, max_naive_s.

## 5. Matrix size and correctness

- `n` is configurable, default 1024. 8000 was requested; naive matmul at n=8000 is about 1.0e12
  FLOPs with 768 MB for three float matrices, and under QEMU each run may take hours, with dozens
  of runs per repeat. Not measured yet. `--n 8000` is allowed.
- Correctness uses an O(n^2) Freivalds-style check (compare C*x with A*(B*x) for a random vector,
  relative tolerance) instead of a full O(n^3) reference.
- Preflight measures the naive time on each platform, prints it, and aborts if it exceeds
  `max_naive_s`.
- Containers get a fixed CPU count so threaded candidates are comparable.

## 6. Error handling

- Preflight: Docker daemon up, both images build, `claude` on PATH, naive reference passes the
  correctness check on both platforms.
- `claude -p` failure or timeout: retry once, then record `llm_error` and abort that repeat only.
- Unparseable reply (no C block or no `FLAGS:` line): treated as a failed candidate; the parse
  error is the next observation.
- Untrusted code runs only in the container: no network, CPU/memory/time limits, flags passed as
  an argv list, never through a shell.
- Compile and run timeouts count as failures. Resume after crash is out of scope for v1.

## 7. Testing (TDD)

- `sandbox.py`: canned C files (correct naive, wrong result, infinite loop, AVX2 intrinsics). The
  AVX2 file must compile on amd64 and fail on arm/v7. This proves the failure mode independent of
  the LLM and serves as a deterministic fallback for the live demo.
- `refine.py`, `memory.py`: fake `llm` with scripted replies; check history assembly, best-correct
  selection, memory length retry, sliding window.
- `llm.py`: one integration smoke test, skipped when `claude` is missing.
- Acceptance: `--quick` end-to-end (1 repeat, 3 rounds).

## 8. Traces

Every step writes JSON. LLM steps: prompt, response, session_id, usage, cost, duration_s.
Sandbox steps: candidate_code, flags, platform, compiled, correct, time_ms, speedup, stderr.
All steps: arm, repeat, round_or_trial, memory entries in effect. `traces/README.md` maps each
scenario to its outcome (also fits the course project's trace format).

## 9. Cost and risks

- About 22 Claude Code calls per A run and up to about 5 per B or C run; roughly 100 calls at 3
  repeats. Wall time unknown until the first real run.
- Risk: Claude writes portable code (`#ifdef __AVX2__`), so memory does not break anything.
  Mitigation: repeats, and the canned AVX2 fallback.
- Risk: QEMU timing noise. Mitigation: same-platform speedup ratios, best-of-k timing.

## 10. Out of scope

Resume after crash, other algorithms, real ARM hardware, statistics beyond reporting rates over
repeats.
