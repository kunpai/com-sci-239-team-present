# Traces for matxfer_d_full_r23

Layout: `<arm>/r<repeat>/<NN>-<kind>.json` (kinds: `llm`, `sandbox`, `memory_update`, ...).
`A` = Self-Refine on amd64 (its `memory.md` is the distilled memory), `B` = arm/v7 trials
without memory, `C` = arm/v7 trials with A's memory, `D` = amd64 trials with A's memory.

## Scenarios

- D/r2: succeeded on the first try.
- D/r3: succeeded on the first try.
