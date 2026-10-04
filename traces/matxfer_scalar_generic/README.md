# Traces for matxfer_scalar_generic

Layout: `<arm>/r<repeat>/<NN>-<kind>.json` (kinds: `llm`, `sandbox`, `memory_update`, ...).
`A` = Self-Refine on amd64 (its `memory.md` is the distilled memory), `B` = arm/v7 trials
without memory, `C` = arm/v7 trials with A's memory.

## Scenarios

- B/r2: first try failed at stage 'parse'; succeeded at trial 2 (speedup 3.84x).
- C/r2: succeeded on the first try.
