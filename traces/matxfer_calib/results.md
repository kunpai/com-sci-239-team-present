# Results

Config: n=1024, rounds=0, repeats=1, trials=1, window=3. Naive baseline (ms): amd64=2124.3, armv7=4149.4

## Arm A: Self-Refine on amd64

| repeat | best speedup | speedup per attempt | note |
|---|---|---|---|
| 1 | 206.48 | 206.48 |  |

## Arms B (no memory) and C (x86 memory) on arm/v7

| arm | First-try success | mean trials to success |
|---|---|---|
| B | 1/1 | 1.00 |
| C | 0/1 | - |

### Per repeat

| arm | repeat | first try | trials to success | final speedup | first failure stage | note |
|---|---|---|---|---|---|---|
| B | 1 | yes | 1 | 2.45 | - |  |
| C | 1 | no | - | - | compile |  |
