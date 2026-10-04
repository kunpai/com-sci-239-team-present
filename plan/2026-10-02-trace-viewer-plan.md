# Trace Viewer Implementation Plan

> For agentic workers: executed natively (inline) with TDD. Spec: `plan/2026-10-02-trace-viewer-design.md`.

**Goal:** `uv run python viewer/build.py` writes a self-contained `viewer/index.html` that browses every run in `traces/` down to exact prompts, replies, code, results and memory for arms A, B and C.

**Architecture:** `viewer/build.py` normalizes traces into one dict (pure functions, tested) and injects it as escaped JSON into `viewer/template.html` (vanilla JS/CSS, no dependencies).

**Tech stack:** Python stdlib, pytest; browser verification with screenshots.

## Global constraints (from the spec)

- Single self-contained file, no CDN/server; < 25 MB for the current runs.
- Trace text is untrusted: DOM text nodes only (never `innerHTML`); embedded JSON escapes `<`, `>`, `&`, U+2028/2029.
- Tolerate older traces: infer `platform` (A=amd64, B/C=armv7), hash the task text when `task_sha256` is absent, missing `run_meta.json`.
- Memory shown verbatim: A's `memory.md`; C's first-prompt memory block; per-trial memory in effect.
- Python style: type hints, logging not print, files < 400 lines, ruff and mypy clean.

## Trace facts the extractor relies on (verified from the harness code)

- A round r: llm `feedback` (round_no=r) critiques attempt r-1; llm `refine` (round_no=r) produces attempt r; sandbox (round_no=r) is attempt r; attempt 0 is llm `generate` + sandbox round_no=0. The sandbox step's own `feedback` field is always null, so the critique of attempt r is the `feedback` llm text with round_no=r+1.
- B/C trial t: llm `generate` (trial=t), sandbox (trial=t); after a failure, llm `lesson` (no trial key) immediately followed by `memory_update` (trial=t, lesson). Every step records `memory` (entries in effect).
- A: llm `memory_reflection`, optional llm `memory_shorten`, optional `memory_truncated`; `memory.md` in the A directory.
- C's memory block is the text after `Memory from earlier sessions:\n` in its first generate prompt; the task statement is A's first generate prompt.
- `run/r<k>/NN-llm_error|infra_error.json` mark aborted repeats.

## Review focus

1. A reply containing `</script>` or `<!--` must not break the page (build test + browser check).
2. A repeat with no A (aborted) or no C (skipped) must render, not crash.
3. Old runs without `platform`/`task_sha256`/`run_meta.json` must build.
4. Very large prompts (125 KB) must not freeze the page (lazy render on expand).
5. Deep-link hashes pointing at a missing run/step must fall back gracefully.

### Task 1: data extraction (`viewer/build.py`) — TDD

**Files:** create `viewer/__init__.py`, `viewer/build.py`; test `tests/test_viewer_build.py`.
**Interfaces (produces):** `load_run(run_dir: Path) -> dict`; `build_data(traces_root: Path) -> dict` (`{"runs": [...]}`, sorted by run id); `render_html(data: dict, template: str) -> str` (replaces `/*__DATA__*/null` with escaped JSON); `main(argv) -> int` (`--traces`, `--out`, `--template`).
Run dict: `id, dev, meta, task_text, task_sha256, cost_usd, llm_calls, repeats[]`. Repeat dict: `repeat, A, B, C, errors[]`. A: `attempts[]` (`round, gen_prompt, gen_text, code, flags, platform, result, critique, critique_prompt, files`), `memory_md, memory_reflection, memory_shorten, truncated`. B/C: `trials[]` (`trial, prompt, text, code, flags, platform, result, memory, lesson, lesson_prompt`), `memory_seen`.
Steps: write tests first (synthetic traces from `run_experiment` with `FakeLLM`; dying-LLM repeat; deleted `platform`/`run_meta.json`; `</script>` in a reply; determinism), watch them fail, implement, watch them pass.

### Task 2: UI (`viewer/template.html`)

Load the `frontend-design` skill first. Views per spec section 4: run picker, run overview, chain view (A | B | C), memory panel, step inspector drawer, failure filter, keyboard navigation, hash deep links, dark/light, lazy rendering of long text, line diff of code between consecutive attempts/trials. JS is authored at execution time (about 600 lines; duplicating it here would add nothing), checked against the acceptance list below.
Acceptance: opens from `file://`; every spec view works on `matxfer_full2` and `matxfer_generic`; the C failure with `immintrin.h` shows red stderr, the lesson and the memory in effect; aborted repeats in `matxfer_generic` render; no console errors.

### Task 3: real-data build + browser verification

`uv run python viewer/build.py` on the real `traces/` (assert < 25 MB, add as a test marked slow); screenshots of picker, chain view, memory panel, failure step, dark/light, narrow width; fix findings; add a one-line usage note to the ledger.

### Final: fresh-context review of `viewer/` (security: untrusted text; correctness: memory extraction vs. the harness code), then report.
