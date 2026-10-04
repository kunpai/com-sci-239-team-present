# Traces for matxfer_calib

Layout: `<arm>/r<repeat>/<NN>-<kind>.json` (kinds: `llm`, `sandbox`, `memory_update`, ...).
`A` = Self-Refine on amd64 (its `memory.md` is the distilled memory), `B` = arm/v7 trials
without memory, `C` = arm/v7 trials with A's memory.

## Scenarios

- A/r1: best speedup 206.48x over 1 attempts.
- B/r1: succeeded on the first try.
- C/r1: first try failed at stage 'compile'; never succeeded.
