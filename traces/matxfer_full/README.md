# Traces for matxfer_full

Layout: `<arm>/r<repeat>/<NN>-<kind>.json` (kinds: `llm`, `sandbox`, `memory_update`, ...).
`A` = Self-Refine on amd64 (its `memory.md` is the distilled memory), `B` = arm/v7 trials
without memory, `C` = arm/v7 trials with A's memory.

## Scenarios

- A/r1: best speedup 261.85x over 11 attempts.
- A/r2: best speedup 353.84x over 11 attempts.
- A/r3: best speedup 324.55x over 11 attempts.
- B/r1: succeeded on the first try.
- C/r1: succeeded on the first try.
- B/r2: succeeded on the first try.
- C/r2: succeeded on the first try.
- B/r3: succeeded on the first try.
- C/r3: first try failed at stage 'compile'; succeeded at trial 2 (speedup 2.64x).
