# Traces for matxfer_full2

Layout: `<arm>/r<repeat>/<NN>-<kind>.json` (kinds: `llm`, `sandbox`, `memory_update`, ...).
`A` = Self-Refine on amd64 (its `memory.md` is the distilled memory), `B` = arm/v7 trials
without memory, `C` = arm/v7 trials with A's memory.

## Scenarios

- A/r1: best speedup 199.99x over 11 attempts.
- A/r2: best speedup 266.41x over 11 attempts.
- A/r3: best speedup 270.88x over 11 attempts.
- B/r1: succeeded on the first try.
- C/r1: first try failed at stage 'compile'; succeeded at trial 2 (speedup 5.61x).
- B/r2: succeeded on the first try.
- C/r2: succeeded on the first try.
- B/r3: succeeded on the first try.
- C/r3: succeeded on the first try.
