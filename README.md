# matxfer: Self-Refine on x86, Reflexion-style memory on arm32

`matxfer` is a small harness that asks one question: does an optimization lesson learned in one environment and stored as verbal memory transfer to a different one? We use Claude Code as the LLM, a C matrix multiply as the task, `linux/amd64` as the source environment, and `linux/arm/v7` (32-bit ARM) as the target. Everything runs on a laptop under Docker emulation. The repository holds the experiment pipeline (`run.py`, `src/`), the sandbox that compiles and measures candidates (`task/`, `Dockerfile`), the stored agent traces (`traces/`), and a single-file trace viewer (`viewer/`).

It was built for the CS 239 (Agentic Software Engineering, Fall 2026) team presentation on Reflexion (main paper) and Self-Refine (optional paper). The prompts never name the architecture, and every model call is stored, so a reader can check exactly what the model saw.

## Contributions

- **Three-arm experiment:** arm A refines a matmul with Self-Refine on amd64 for 10 rounds and distills the result into `memory.md`; arm B runs Reflexion trials on arm/v7 with no memory; arm C runs the identical loop with A's memory in the prompt. Memory is the only variable between B and C. An optional arm D runs the same loop with A's memory on amd64, the platform the memory came from, as a same-platform control.
- **Isolated Claude Code backend:** each call is a headless `claude -p` with built-in tools, MCP servers, slash commands, user settings, and session persistence all turned off, so the model sees the task text and nothing else.
- **Sandboxed measurement:** candidates are compiled and run in per-platform `gcc:13` containers (no network, pinned CPUs, memory and process limits), checked for correctness three ways, and timed best-of-3. Speedup is a same-platform ratio against a naive `-O2` triple loop.
- **Complete traces:** one JSON file per prompt, reply, sandbox result, and memory update, grouped by run, arm, and repeat.
- **Trace viewer:** one self-contained HTML file (10 MB for the current five runs) that walks every iteration of arms A, B, and C down to the exact memory each arm stored or was shown.

## Findings so far

Pooled over three runs (7 chains; the prompts and harness differ between runs, so treat this as informal): arm B succeeded on the first try in 7 of 7 chains, arm C in 4 of 7. All three C failures were x86-only constructs copied from the memory (`-mprefer-vector-width=512` once, an unconditional `#include <immintrin.h>` twice), and all three recovered on trial 2 through the Reflexion lesson. A one-sided Fisher test gives p of about 0.10, which is suggestive and nothing more.

Three scoping notes belong next to that number. The memory transfers more often than the hypothesis predicted, because Claude tends to write guarded or portable code. The container sentence in the original prompt (see `task/prompt_task.md`) nudges the model toward portability, which is why `task/prompt_task_generic.md` exists as an ablation. Timings under emulation are noisy (10 to 20 percent), so we report speedups as ratios and do not compare absolute times across platforms.

A later pair of runs (`matxfer_scalar_full2`, `matxfer_scalar_generic`) replaced the failure observation with scalar feedback only (compiled yes or no, correct yes or no, speedup) and reused the stored A memories. They are kept as a side note: B and C were both 3 of 4 on first try, and first-try rates cannot depend on feedback that arrives after the first trial.

## Repository layout

```
run.py                     orchestrator and CLI (preflight, repeats, per-repeat error handling)
src/
  config.py                frozen Config (rounds, repeats, trials, window, caps, size, limits, feedback mode)
  llm.py                   ClaudeCodeLLM: `claude -p` with isolation flags, retry with 30/60/120 s backoff
  prompts.py               every prompt builder; the architecture is never named here
  candidate.py             parse a reply into code + FLAGS, evaluate it, pick the best step
  refine.py                Self-Refine loop (generate, then feedback and refine per round)
  reflexion.py             Reflexion trials (attempt, evaluate, lesson on failure, retry)
  memory.py                EpisodicMemory window, write_memory (A), reflect_on_failure (B and C)
  sandbox.py               Docker build/run, flag allow and deny lists, result parsing
  traces.py                TraceWriter and TraceScope (numbered JSON files)
  report.py                results.json, results.md, README.md, speedup plot per run
task/
  harness.c                correctness checks, timing, and the output markers the sandbox parses
  naive.c                  baseline triple loop
  prompt_task.md           the original task prompt
  prompt_task_generic.md   the generic-prompt ablation
Dockerfile                 gcc:13 plus a prebuilt /opt/harness.o, built once per platform
viewer/
  build.py                 normalizes traces/ and embeds them in one HTML file
  template.html            the UI (vanilla JS and CSS, no dependencies)
  index.html               generated; rebuild after every new run
  browser_check.js         scripted browser regression (optional, needs puppeteer-core)
tests/                     127 tests (pytest); canned C sources live in tests/canned/
plan/                      design docs, implementation plans, and the progress ledger
traces/                    one directory per run (see "Trace layout")
```

## How the experiment works

**Arm A (Self-Refine, amd64).** The model writes a first attempt, then for each of 10 rounds writes feedback on the latest attempt and a refined version. Both calls see the full attempt history with the measured result of each attempt (compiled, correct, runtime, speedup, compiler output). The best correct attempt is distilled into at most 200 words of notes by a reflection call (one shorten retry, then a hard clip), saved as `A/r<k>/memory.md`.

**Arms B and C (Reflexion, arm/v7).** Up to 3 trials. Each trial is one generation, run in the arm/v7 sandbox. Success means compiled and correct. After a failure the model writes a lesson (at most 80 words), appended to a sliding memory window of 3 entries (Reflexion's Ω). B starts with an empty window. C starts with A's memory as entry 1, rendered under the heading `Memory from earlier sessions:`.

**Correctness and timing.** The harness checks a full reference product at n=37, a Freivalds check at the odd size `n|1`, and a Freivalds check at the benchmark size (O(n²) per probe), with C pre-filled with NaN so unwritten elements fail. It then times the candidate best-of-3 at n=1024 by default. Flags that set the output file, add include or library paths, or reach the linker are rejected before the container starts.

**Feedback modes.** The default is `diagnostic`: a failed attempt is shown to the lesson step with its failure stage and compiler output, a grounded evaluator in the spirit of Reflexion's heuristics. `--feedback scalar` shows only `Compiled: yes|no. Correct: yes|no. Speedup: X.XXx.` (speedup is `n/a` for a failed attempt).

## Prerequisites

- macOS or Linux with Docker running. On Apple silicon, `linux/arm/v7` runs under QEMU and `linux/amd64` under Rosetta or QEMU. Rosetta-backed amd64 has no AVX-512, which matters when reading A's memory.
- The `claude` CLI on `PATH`, logged in. Runs spend tokens (see the cost line below).
- Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync          # installs matplotlib plus the dev tools (pytest, ruff, mypy)
```

## Running the experiment

The first call builds both images (`matxfer:amd64`, `matxfer:armv7`) and measures the naive baseline on each platform. It aborts early if Docker or `claude` is missing, or if the naive run is too slow (over 120 s).

```bash
# smoke test: 1 repeat, 3 rounds
uv run python run.py --quick --run-id my_quick

# the full experiment: 3 repeats x (10 refine rounds, then B and C)
uv run python run.py --run-id my_full

# the generic-prompt ablation
uv run python run.py --task-file task/prompt_task_generic.md --run-id my_generic

# B and C only, reusing the memories stored by an earlier run (arm A is skipped)
uv run python run.py --reuse-memory-from traces/my_full --run-id my_full_again

# arm D only: A's memory from repeat 1 of an earlier run, on amd64 (no B, no C)
uv run python run.py --arm-d --reuse-memory-from traces/my_full --only-repeats 1 --run-id my_d
```

| Flag | Meaning | Default |
|---|---|---|
| `--quick` | 1 repeat, 3 rounds | off |
| `--n` | matrix size | 1024 |
| `--repeats` | independent A to B/C repeats | 3 |
| `--rounds` | Self-Refine rounds in arm A | 10 |
| `--trials` | maximum Reflexion trials in B and C | 3 |
| `--model` | model passed to `claude --model` | CLI default |
| `--feedback` | `diagnostic` or `scalar` | `diagnostic` |
| `--task-file` | task statement file | `task/prompt_task.md` |
| `--reuse-memory-from` | earlier run directory whose A memories are reused | none |
| `--only-repeats` | with `--reuse-memory-from`: only these repeat numbers | all stored |
| `--arm-d` | run only arm D (A's memory on amd64) instead of B and C | off |
| `--out`, `--run-id` | trace root and run name | `traces`, timestamped |

The full experiment made 76 Claude calls and cost about $17 in the last run (about 55 minutes of wall time, since runs are sequential to keep the CPU-pinned timings clean). A single failed repeat does not lose the others: an `LLMError` or `SandboxError` is logged as `llm_error` or `infra_error` for that repeat, and the run moves on.

## Trace layout

```
traces/<run>/
  run_meta.json            config, task file and sha256, naive baselines, claude and docker versions, host, memory source
  results.json / .md       per-repeat metrics and the B-versus-C table
  README.md                short run summary
  A/r<k>/NN-<kind>.json    arm A steps; memory.md is the stored memory
  B/r<k>/NN-<kind>.json    arm B trials
  C/r<k>/NN-<kind>.json    arm C trials
  D/r<k>/NN-<kind>.json    arm D trials (only in runs made with --arm-d)
  run/r<k>/NN-llm_error.json    aborted repeats
```

Step kinds are `llm` (prompt, reply, role, cost, duration), `sandbox` (code, flags, platform, result), `memory_update` (the lesson appended after a failed trial), `memory_truncated`, and `memory_reused`. Every step records the memory entries in effect when it ran.

## Trace viewer

The viewer is one HTML file with the data embedded as JSON. It needs no server, no network, and no build tools beyond Python's standard library.

**Open it.** Either double-click `viewer/index.html`, or run `open viewer/index.html` on macOS. It works from `file://`, so you can copy the file to the presenting laptop as is.

**Serve it (optional).** If you prefer a URL, or want a second device on a trusted network to read it:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory viewer
# then visit http://127.0.0.1:8765/index.html
```

Binding to `127.0.0.1` keeps it on your machine. The page contains every prompt and generated program, so do not bind it to `0.0.0.0` on a network you do not control. Serving `viewer/` alone also keeps the raw `traces/` directory unexposed.

**Rebuild it** after every new run (`index.html` is generated, not hand-edited):

```bash
uv run python viewer/build.py                    # reads traces/, writes viewer/index.html
uv run python viewer/build.py --traces /path --out /path/index.html
```

**Navigate it.** The overview shows the summary table and the three arm tiles. Each chain has three views (tabs at the top): **Timeline**, **Memory**, and **Transcript**. The timeline lays out A, B, and C side by side as step timelines; click a step to expand its prompt, reply, code (with a line diff against the previous attempt), flags, and measured result. The memory panel shows A's `memory.md` verbatim, the exact block that followed `Memory from earlier sessions:` in C's first prompt (the harness's double bullet and any hard clip are visible and flagged), and the memory in effect at each trial.

| Key | Action |
|---|---|
| `j` / `k` | next / previous step |
| `]` / `[` | next / previous chain (repeat) |
| `m` | toggle memory panel |
| `a` | toggle the transcript (all turns) |
| `f` | failures only (timeline) |
| `t` | toggle light and dark theme |
| `Esc` | close the inspector drawer |

Every view has a deep link, so a slide can point at an exact step. Calibration and quick runs are hidden by default (tick "dev runs" to show them).

```
#/<run>                                run overview
#/<run>/<repeat>/timeline              chain view
#/<run>/<repeat>/timeline/<arm>/<i>    one step, expanded (arm is A, B, or C; i counts from 0)
#/<run>/<repeat>/memory                memory panel
#/<run>/<repeat>/transcript            every trace step of the chain, in order
#/<run>/<repeat>/transcript/<arm>      the same, one arm only
#/<run>/<repeat>/transcript/<arm>/<n>  one step, expanded (n is the step number in the file name)
```

**Transcript.** The transcript is the complete record of one chain: every numbered trace file, in order, with nothing paired or summarized. A Claude call shows the full prompt and the full reply (with duration, cost, and tokens), a sandbox run shows the code (with a diff against the previous run), flags, and measured result, and a memory update shows the lesson as stored. Because it is the raw trace, it also contains what the timeline leaves out: the untrimmed lesson reply (before the word-cap clip), the memory-reflection and shorten calls, and any step kind the viewer does not know (shown as raw JSON). Controls: filter by arm, tick Claude calls, sandbox runs, or memory and other events on and off, search the text of the chain, expand or collapse all turns, and download what is shown as Markdown or JSON (`<run>_chain<k>_transcript.md`). Long prompts scroll in place, and the `open` button on each prompt or reply sends it to the inspector drawer.

For example, `index.html#/matxfer_full2/1/timeline/C/0` opens the first C trial of repeat 1 (the compile failure on `immintrin.h`), `index.html#/matxfer_full2/1/memory` shows the memory panel for that chain, and `index.html#/matxfer_full2/1/transcript/C` lists every turn of arm C in that chain. Query parameters go before the hash: `?theme=light` or `?theme=dark`, `?fail=1` for failures only, `?dev=1` to show dev runs.

**Safety.** Trace text is untrusted model output. The UI renders it only through DOM text nodes (never `innerHTML`), and the embedded JSON escapes `&`, `<`, `>`, U+2028, and U+2029, so a `</script>` inside a reply cannot break the page. `viewer/browser_check.js` includes a hostile-text case that guards this.

**Browser regression (optional).** `viewer/browser_check.js` drives Google Chrome through the views above (78 checks). It needs `puppeteer-core` and a hostile-text fixture page:

```bash
npm i puppeteer-core
NODE_PATH=$PWD/node_modules node viewer/browser_check.js "$PWD" /tmp/shots /path/to/evil.html
```

## Tests

```bash
uv run pytest -q -m "not llm"      # 127 tests; the sandbox tests need Docker
uv run pytest -q -m llm            # live Claude calls; spends tokens
uv run ruff check . && uv run mypy src run.py
```

The suite uses a scripted `FakeLLM` and canned C sources (wrong result, infinite loop, accumulate-without-clear, missing tail handling, AVX2-only, and a guarded fallback) so that the sandbox and the loops are tested without model calls. Tests marked `docker` need a running daemon.

## Scope and limitations

- **Statistics:** 3 chains per run and 7 pooled. The result is a demonstration, not a significance claim.
- **Prompt and harness drift:** `matxfer_full` predates the harness fixes (CPU pinning, the odd-size check, Docker error attribution) and is marked "older harness" in the viewer. `matxfer_full2` and `matxfer_generic` use the fixed harness.
- **Generic ablation:** `matxfer_generic` completed one of three repeats. Two aborted when a Claude call stalled past the 300 s timeout. The retry backoff added since then addresses this, but those traces predate it.
- **Memory text:** A's memories sometimes assert AVX-512 behavior that cannot occur under Rosetta, so the narrative inside a memory is not reliable evidence about the hardware. The harness prefixes each entry with `- `, which produces a double bullet in C's memory block; the viewer shows it as is.
- **Arm D:** `matxfer_d_full` and `matxfer_d_full2` ran D on repeat 1 of `matxfer_full` and `matxfer_full2`. Both passed on the first try (176x and 154x over the amd64 naive loop), and neither attempt used `immintrin.h`. Two chains is a sanity check on the memory in its home environment, not a comparison.
- **Runs that reuse memory:** in a run made with `--reuse-memory-from`, arm A was not re-run, so its column shows the reused `memory.md` and a "not re-run" note instead of attempts, and its transcript is the single `memory_reused` event.
- **Timing artifacts:** buffers are shared across timing repetitions, and emulated timings carry 10 to 20 percent noise.
- **Deviations from the papers:** A's feedback sees measured results (closer to a grounded evaluator than pure self-feedback), 10 rounds exceeds Self-Refine's 4, the final candidate is the best correct one rather than the last, and the memory is written from a success whereas Reflexion writes reflections from failures.

## Extending it

- **A new task prompt:** add a Markdown file under `task/` and pass it with `--task-file`. The task hash is recorded in `run_meta.json`, and the viewer shows it.
- **A new platform:** add a `Platform` in `src/sandbox.py`, then wire it into `run.py` (`preflight` and `run_repeat`). The viewer infers platform per arm, so add it to `PLATFORM_BY_ARM` in `viewer/build.py` if it is not recorded in the traces.
- **A different feedback signal:** `src/prompts.py` owns `format_observation` and `scalar_observation`; add a mode to `FEEDBACK_MODES` in `src/config.py` and a test beside `tests/test_scalar_feedback.py`.
- **More design detail:** `plan/` holds the design docs for the experiment and the viewer, the implementation plans, and `progress.md`, the ledger of every ruling and deferred minor.
