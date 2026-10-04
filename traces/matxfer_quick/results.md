# Results

Config: n=1024, rounds=3, repeats=1, trials=3, window=3. Naive baseline (ms): amd64=2077.2, armv7=4149.8

## Arm A: Self-Refine on amd64

| repeat | best speedup | speedup per attempt | note |
|---|---|---|---|
| 1 | 243.77 | 227.26, 234.08, 225.83, 243.77 |  |

## Arms B (no memory) and C (x86 memory) on arm/v7

| arm | First-try success | mean trials to success |
|---|---|---|
| B | 1/1 | 1.00 |
| C | 1/1 | 1.00 |

### Per repeat

| arm | repeat | first try | trials to success | final speedup | first failure stage | note |
|---|---|---|---|---|---|---|
| B | 1 | yes | 1 | 2.62 | - |  |
| C | 1 | yes | 1 | 4.14 | - |  |
