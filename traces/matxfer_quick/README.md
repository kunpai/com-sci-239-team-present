# Traces for matxfer_quick

Layout: `<arm>/r<repeat>/<NN>-<kind>.json` (kinds: `llm`, `sandbox`, `memory_update`, ...).
`A` = Self-Refine on amd64 (its `memory.md` is the distilled memory), `B` = arm/v7 trials
without memory, `C` = arm/v7 trials with A's memory.

## Scenarios

- A/r1: best speedup 243.77x over 4 attempts.
- B/r1: succeeded on the first try.
- C/r1: succeeded on the first try.
