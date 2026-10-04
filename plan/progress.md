# SDD ledger — plan: plan/2026-10-01-memory-transfer-demo-plan.md

Spec: plan/2026-10-01-memory-transfer-demo-design.md (read).

Pre-flight (shared interfaces, plan Interfaces blocks vs. each other):
- T1 -> T6/T7/T9: TraceScope.log_llm/log_step take dataclasses; LLMReply (T4) and Step (T5) are dataclasses. Consistent.
- T3 -> T5: RunResult(stage, compiled, correct, time_ms, speedup, stderr) used positionally by candidate.py ("parse" stage) and conftest.ok/fail. Consistent.
- T4 -> T5/T6/T7/T9: LLMReply positional order (text, session_id, cost_usd, duration_s, usage, model_usage) matches conftest.FakeLLM. Consistent.
- T5 -> T6/T7: Step(code, flags, result, feedback), RunFn, evaluate_reply, best_step, prompt function signatures. Consistent.
- T6/T7 -> T8/T9: RefineOutcome, TrialOutcome, EpisodicMemory names/fields match report.py and run.py. Consistent.
- T8 -> T9: ArmRecord/RefineRecord/skipped_arm/write_report signatures match run.py. Consistent.

Ruling: directory is not a git repo — no worktree, no commits, no sdd scripts; ledger lives here, tests are the per-task gate — spec/plan already note no commit steps — cost if wrong: no review package for the final review, so I review the files directly.
Task 1: complete (tests: uv run pytest -q → 5 passed)
Task 2: complete (harness on amd64+armv7 with naive.c n=64 → SMALL_OK 1, FREIVALDS_OK 1, TIME_MS 0.091 / 0.890)
Task 3: complete (uv run pytest tests/test_sandbox.py -q → 17 passed; avx2.c compiles on amd64, compile-fails on armv7)
Task 4: complete (uv run pytest tests/test_llm.py → 7 passed unit + 1 passed real claude smoke)
Task 5: Ruling: arch-word test matched 'sse' inside 'passed' — switched the test to whole-word matching instead of rewording the task text — cost if wrong: a real arch word embedded in a longer token would slip through
Task 5: complete (uv run pytest tests/test_candidate.py tests/test_prompts.py → 15 passed)
Task 6: complete (uv run pytest tests/test_refine.py → 3 passed)
Task 7: complete (uv run pytest tests/test_memory.py tests/test_reflexion.py → 9 passed)
Task 8: complete (uv run pytest tests/test_report.py → 4 passed)
Task 9: complete (uv run pytest tests/test_run.py → 3 passed; full suite 63 passed; ruff + mypy clean after fixes)
Task 10: Ruling: calibration run showed Claude Code always tells the model the host OS ("macOS (Darwin 25.3.0)", not removable: --system-prompt keeps it, --exclude-dynamic-system-prompt-sections only moves it), so the model guessed Apple-M NEON and arm C failed on an AArch64-only intrinsic — added one neutral sentence to task/prompt_task.md ("built and run inside a Linux container, not on the machine you are running on. The CPU is not disclosed.") for ALL arms, plus a pinning test — cost if wrong: the task text now differs slightly from the spec's wording and may nudge Claude toward portable code; revert by deleting the sentence
Task 10: Ruling: llm_timeout_s default 600 -> 300 — one stalled call cost 10 min, normal calls take 5-100 s — cost if wrong: a legitimately slow refine call (long history) is retried once then aborts its repeat
Task 10: calibrate run (n=1024, rounds 0, trials 1): naive amd64 2124 ms, armv7 4149 ms; A first draft 206x on amd64; B (no memory) ok 2.45x on armv7; C (memory) compile fail (AArch64 NEON intrinsic, pre-fix). ~$0.6 for 6 calls.
Task 10: quick run (n=1024, rounds 3, repeats 1): A 227->244x on amd64; B ok 2.62x; C ok first try 4.14x (memory transferred, did not break). 11 llm calls, $1.23.
Task 10: complete — full run matxfer_full (n=1024, 3 repeats x 10 rounds, trials<=3): 77 llm calls, $18.10, no aborted repeats. A best speedups 262x/354x/325x (amd64, emulated). B first-try 3/3. C first-try 2/3; C/r3 failed to compile on armv7 ('unrecognized command-line option -mprefer-vector-width=512', x86-only flag taken from memory), Reflexion lesson written, trial 2 succeeded (2.64x). C/r1, C/r2 succeeded first try (r1 with #ifdef-guarded AVX intrinsics + fallback, r2 portable GCC vector extensions).
Final: Ruling: keeping plan/progress.md as the record (no git history exists to replace it) — cost if wrong: none
Final review (fresh reviewer, model fable): no Critical; 4 Important + 1 re-graded to Important.
Final: fixed docker daemon/image errors were logged as compile/run failures and fed to the model — test_docker_infrastructure_errors_raise_instead_of_blaming_the_candidate + test_sandbox_infrastructure_failure_aborts_only_that_repeat, RED (SandboxError missing) -> GREEN, suite 71/71
Final: fixed --cpus is only a CFS quota (container saw nproc=10) — added --cpuset-cpus; test_container_sees_exactly_the_configured_cpus (old flags nproc=10 shown above) RED->GREEN, suite 71/71
Final: fixed odd-size check could be bypassed by a small-n fallback — harness ODD_OK Freivalds check at n|1 before timing; test_small_n_fallback_does_not_hide_a_broken_tail RED->GREEN, suite 71/71
Final: fixed mean-trials column silently dropped repeats that never succeeded — now "x.xx (k/N succeeded)" / "never (0/N succeeded)"; test_mean_trials_does_not_hide_repeats_that_never_succeeded RED->GREEN, suite 71/71
Final: fixed (re-graded Minor -> Important: spec section 8 lists it) sandbox traces lacked `platform` — test_sandbox_trace_records_the_platform + test_experiment_traces_name_the_platform_each_step_ran_on RED->GREEN, suite 71/71
Final: IMPORTANT consequence — traces/matxfer_full was produced BEFORE these fixes (quota-only CPU limit, no odd-size check, no platform field). B-vs-C comparison is unaffected (same harness for both) but arm A timings were noisier than intended. Re-run not done (another ~$18, ~55 min) — user decision.
Final: minor (deferred): unbounded stdout capture from the container (host-memory risk on a runaway printf)
Final: minor (deferred): only LLMError/SandboxError are caught per repeat; any other exception loses the report
Final: minor (deferred): --trials 0 / --rounds -1 / --repeats 0 not validated (IndexError after LLM spend)
Final: minor (deferred): empty memory-reflection reply is accepted as a valid memory (empty bullet) and mislabels the report note
Final: minor (deferred): FLAGS parsing rejects backticked flags and two-token `--param X` forms
Final: minor (deferred): compiler warnings from successful compiles are discarded, so the model never sees them (format_observation branch is dead)
Final: minor (deferred): prompt passed as one argv string (108 KB at round 10; fine on macOS, would hit the 128 KB per-arg limit on Linux)
Final: minor (deferred): `docker rm -f` has no timeout; matxfer-llm temp dirs never removed; root-owned .work files on native Linux; immediate LLM retry with no backoff; linker pass-through flags (-e/-T/-z/-u) not denied
Final: Ruling: AVX-512 narrative in the memories is unreliable — Rosetta amd64 reports __AVX2__/__FMA__ but not __AVX512F__, so "zmm" kernels ran their AVX2 fallback and `-mprefer-vector-width=512` was a no-op; the "~15% AVX-512 gain" in A/r3/memory.md is noise — results stand as measured, but slides must not present it as a measured AVX-512 lesson. The C/r3 compile failure is still a genuine x86-only-flag transfer failure — cost if wrong: an overclaim on a slide
Final: Ruling: harness lets best-of-3 reps share pointers, so persistent static buffers amortize page faults (all memories call this the biggest win) — within the task rules, left as is — cost if wrong: arm A speedups partly reflect a harness artifact
Final: Ruling: neutral container sentence in the task text is not neutral w.r.t. the hypothesis (nudges toward guarded/portable code, which is why 2 of 3 memories transferred) — kept, must be disclosed on slides as a deviation from the spec's task wording — cost if wrong: the null-ish result is partly caused by the sentence; a no-sentence run would tell
Final: Ruling: repeated FLAGS lines accepted (last wins); multi-line memory under one bullet; naive baseline measured once at preflight; EpisodicMemory(window=0) unreachable; trace numbering %02d past 99 steps; first-match _grab spoofable only adversarially — all left as is — cost if wrong: cosmetic or adversarial-only
Rerun (user-approved "yes please do this"): added --task-file + run_meta task_file/task_sha256 (3 tests, suite 74/74). Run 1 = matxfer_full2 (fixed harness, default prompt incl. container sentence). Run 2 = matxfer_generic (fixed harness, task/prompt_task_generic.md).
Rerun: Ruling: "very generic" read as: keep only the interface contract the harness needs (signature, row-major, C uninitialized, correct for every n, large benchmark, flag restrictions, reply format) and drop the container sentence, "machine this runs on", pthreads/OpenMP list and the -march=native example — cost if wrong: ablation answers a different question than intended; prompt file is read when run 2 starts so it can be edited before then
Rerun: Ruling: runs are sequential, not parallel — both use cpuset 0-3, so overlapping them would contaminate timings — cost if wrong: ~55 min longer wall time
Rerun results: matxfer_full2 (fixed harness, original prompt): 76 calls $17.44, B 3/3 first-try, C 2/3 (C/r1 failed: unconditional #include <immintrin.h> on armv7, recovered trial 2 via lesson). matxfer_generic (fixed harness, generic prompt sha b02333387252): 33 calls $9.76, ONLY 1 completed repeat — r1 and r3 aborted (llm_error: a claude call hung past the 300 s timeout and the single retry hung/exit-1 again). Completed r2: B first-try 1.09x; C first-try compile fail (immintrin.h) then trial 2 ok 5.15x.
Pooled over all runs (informal, prompts/harness differ): B first-try 7/7, C first-try 4/7, all 3 C failures were x86-only constructs from memory (-mprefer-vector-width=512, immintrin.h x2) and all recovered on trial 2.
Rerun: re-graded deferred minor "LLM retry has no backoff / only 1 retry" to Important by effect (cost 2 of 3 generic repeats). Not yet fixed: waiting on user decision about extra generic repeats; fix would be retries=3 with 30/60/120 s backoff, TDD.
--- Trace viewer (spec plan/2026-10-02-trace-viewer-design.md, plan plan/2026-10-02-trace-viewer-plan.md) ---
Viewer: Ruling: user said "go ahead" after I stated the sequence "write the plan, then build it natively" — treated as approval of the plan-then-native-build sequence instead of pausing for a separate plan review (they can read the plan file any time) — cost if wrong: a plan change would mean rework
Viewer: Ruling: the plan specifies the data layer fully (tests, interfaces) but leaves the ~600-line UI JS to execution time, checked against an acceptance list, to avoid writing it twice — cost if wrong: less review of UI code before it exists
Viewer: Task 1 complete (viewer/build.py, tests/test_viewer_build.py 9 passed; real traces: 5 runs, totals match results).
Viewer: Task 2/3 complete: viewer/template.html (~700 lines) + viewer/index.html (10.3 MB, built from traces/, regenerate with `uv run python viewer/build.py`). Browser checks (puppeteer-core + Chrome, copy in viewer/browser_check.js): 29/29 PASS incl. hostile </script><img onerror> text rendering as literal text, keyboard shortcuts, deep links, aborted repeats, phone width, light/dark. Fixed during verification: shortcuts dead after clicking a checkbox; wrong "skipped" message on aborted repeats; "memory 0 words" on repeats with no memory; unbounded error banner.
Viewer final review (fresh reviewer, fable): no Critical; security verdict clean (no sink beyond text nodes; _safe_json sound); every cross-checked number matched results.json.
Final: fixed aborted repeat's partial arm-A data shown as a completed chain (A summary row, "no memory" labelled as no correct candidate, aborted repeats counted in B/C stats) — browser checks "aborted repeat is labelled in the A summary row" / "A tile says it aborted before memory" / "memory panel blames the abort" RED->GREEN
Final: fixed "memory shown: N words" counted the harness bullet (201 vs cap 200) — checks "C tile word count excludes the harness bullet" / "memory panel chip" RED->GREEN
Final: fixed (re-graded Minor->Important: spec 4.4) memory panel did not separate A's carried-in entry from new lessons — check "carried-in A memory is not styled as a lesson" RED->GREEN
Final: fixed (re-graded) unknown run id in a deep link rendered another run's chain with no notice — check "unknown run in a deep link shows a notice" RED->GREEN
Final: fixed (re-graded) pre-fix harness runs looked identical to post-fix runs — run.platform_inferred + "older harness" chip; tests/test_viewer_platform.py (2) + browser checks RED->GREEN. Suite 87/87 + browser checks all PASS.
Final: minor (deferred): A tile blurb hardcodes "refines for 10 rounds, then writes memory.md" (wrong for calib/quick/aborted)
Final: minor (deferred): spec section 3 "unknown kinds kept raw" not implemented; raw lesson reply and memory_truncated word counts dropped by build.py
Final: minor (deferred): build.py robustness — llm cost loop and run_meta.json read without try/except, missing --traces dir gives a traceback, read_text/write_text use locale encoding (no encoding="utf-8"), lone surrogates would crash write_text, U+2028/9 written as literal characters in _safe_json, lesson prompt taken from step-1 without checking role
Final: minor (deferred): UX/a11y — failures-only toggle scrolls to top; drawer close does not restore focus; tile/card-head buttons contain divs; navigating to an identical hash is a no-op
Final: Ruling: partial arm-A data from aborted repeats stays visible but is labelled and excluded from B/C stats (matches results.md) — cost if wrong: a presenter may prefer to hide aborted chains entirely
Final: Ruling: browser_check.js is kept in viewer/ with a header comment (needs `npm i puppeteer-core` + Google Chrome) rather than adding node tooling to the Python project — cost if wrong: the UI regression check is manual to run
