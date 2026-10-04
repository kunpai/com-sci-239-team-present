# Traces for matxfer_scalar_full2

Layout: `<arm>/r<repeat>/<NN>-<kind>.json` (kinds: `llm`, `sandbox`, `memory_update`, ...).
`A` = Self-Refine on amd64 (its `memory.md` is the distilled memory), `B` = arm/v7 trials
without memory, `C` = arm/v7 trials with A's memory.

## Scenarios

- B/r1: succeeded on the first try.
- C/r1: succeeded on the first try.
- B/r2: succeeded on the first try.
- C/r2: first try failed at stage 'compile'; succeeded at trial 2 (speedup 7.74x).
- B/r3: succeeded on the first try.
- C/r3: succeeded on the first try.
