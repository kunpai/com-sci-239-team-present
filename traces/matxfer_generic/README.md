# Traces for matxfer_generic

Layout: `<arm>/r<repeat>/<NN>-<kind>.json` (kinds: `llm`, `sandbox`, `memory_update`, ...).
`A` = Self-Refine on amd64 (its `memory.md` is the distilled memory), `B` = arm/v7 trials
without memory, `C` = arm/v7 trials with A's memory.

## Scenarios

- A/r2: best speedup 205.59x over 11 attempts.
- B/r2: succeeded on the first try.
- C/r2: first try failed at stage 'compile'; succeeded at trial 2 (speedup 5.15x).
