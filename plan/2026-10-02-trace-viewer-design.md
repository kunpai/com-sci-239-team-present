# Trace Viewer: Design

Status: design, awaiting review. Builds on `plan/2026-10-01-memory-transfer-demo-design.md` (the experiment this inspects).

## 1. Purpose

A web interface for browsing what the experiment did, down to exact text: every Self-Refine iteration
(arm A), every Reflexion trial (arms B and C), the prompts and replies, the code and its measured result,
and the exact memory stored by A and what B and C were shown. Used to prepare and give the Oct 6 team
presentation, and to answer "what exactly did the model see?" without opening JSON files.

## 2. Form

- `viewer/build.py` (Python standard library only) reads every run under `traces/` and writes one
  self-contained `viewer/index.html`: data embedded as JSON, vanilla JS and CSS inline, no CDN, no server.
  It opens by double-click (`file://`) and can be copied to any laptop.
- Local only. Nothing is published or uploaded. (Hosting it as a private claude.ai Artifact is a possible
  later step the user decides on; traces contain full prompts and generated code.)
- Size: about 10 MB of trace JSON today; the page must stay under 25 MB for all current runs.

## 3. Data model (what build.py extracts)

Trace layout (written by the harness): `traces/<run>/<arm>/r<k>/<NN>-<kind>.json`, plus
`A/r<k>/memory.md`, `run/r<k>/NN-{llm_error,infra_error}.json`, `run_meta.json`, `results.json`.

Normalized output, per run: `meta` (id, config, naive_ms, task hash, cost, call count, claude/docker versions),
`task_text` (the first A generate prompt, which is exactly the task statement), `repeats[]`. Per repeat:
- `A`: ordered `attempts[]` (round 0..N): feedback prompt+text, refine prompt+text, code, flags, platform,
  result (stage, compiled, correct, time_ms, speedup, stderr), and the stored `memory_md`; the raw
  reflection reply, any shorten reply, and a `truncated` flag when `memory_truncated` exists.
- `B`, `C`: ordered `trials[]`: generate prompt+text, code, flags, result, memory entries in effect, and the
  lesson (llm role `lesson`, plus `memory_update`) written after a failed trial. `C.memory_seen` is the exact
  block that followed `Memory from earlier sessions:` in C's first prompt; `B.memory_seen` is null.
- `errors[]`: aborted repeats (`llm_error` / `infra_error`) with the message.

Tolerance rules: older runs lack `platform` (infer amd64 for A, armv7 for B/C), lack `task_sha256` (hash the
task text instead), and calibration/quick runs have fewer steps. Missing optional fields never crash the
build; unknown kinds are kept in a raw list and shown in the inspector.

## 4. Views

1. **Run picker**: run id, prompt variant (task hash and a "task text" popover), config, calls, cost,
   aborted repeats flagged; dev runs (calib, quick) grouped and collapsed by default.
2. **Run overview**: repeats as rows, arms A/B/C as cards: first-try success or failure stage, trials to
   success, speedup, plus the run-level table (B vs C first-try rates).
3. **Chain view**: A | B | C columns of step timelines. A shows a speedup sparkline and attempts 0..N;
   B/C show trials with the lesson between them and the compiler error in red. Each step expands to prompt,
   reply, code with a line diff against the previous attempt/trial, flags, result.
4. **Memory panel** (always one click away from a chain): A's `memory.md` verbatim; what C's first prompt
   contained verbatim (with the harness's double-bullet and hard-clip artifacts visible and flagged);
   B = "no memory"; per trial, the memory entries in effect, with newly appended lessons highlighted.
5. **Step inspector**: drawer with tabs (Prompt, Reply, Code, Result, Raw JSON) and copy buttons.
6. Filters and navigation: show only failures, previous/next step with the keyboard, deep links by URL
   hash (run, repeat, arm, step) so a slide can point at an exact step.

## 5. Quality and safety

- All trace text (prompts, replies, code, stderr) is untrusted model output: rendered only with
  `textContent`/DOM text nodes, never `innerHTML`; embedded JSON escapes `<`, `>`, `&`, U+2028/2029 so a
  `</script>` inside a reply cannot break out.
- Dark and light theme via CSS tokens, readable at projector size, usable at phone width.
- Large texts (prompts up to about 125 KB) render lazily on expand; code in monospace with line numbers.

## 6. Testing

- pytest on `build.py` using synthetic traces made with the real `TraceWriter`/report code: chain
  structure and ordering, exact memory extraction for C, truncation flag, aborted repeats, missing-field
  tolerance, `</script>` escaping, deterministic output.
- A build over the real `traces/` must succeed and stay under 25 MB.
- Browser verification (screenshots) of: run picker, chain view, memory panel, a failure step with the
  compiler error, deep link, dark/light.

## 7. Out of scope

Live updating during a run, editing traces, hosting, comparing runs side by side beyond the overview
table, syntax highlighting beyond monospace and line numbers (can be added later).

## 8. Risks

- File size growth if more runs are added: mitigated by lazy rendering; if it passes 25 MB, switch to
  per-run data files behind a tiny local server (documented fallback, not built now).
- Old traces differ slightly in schema (see tolerance rules); the real-data build test guards this.
