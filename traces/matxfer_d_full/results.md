# Results

Config: n=1024, rounds=10, repeats=3, trials=3, window=3. Naive baseline (ms): amd64=2145.4, armv7=4048.9

## Arm A: Self-Refine on amd64

| repeat | best speedup | speedup per attempt | note |
|---|---|---|---|

## Arms B (no memory) and C (x86 memory) on arm/v7; D (x86 memory) on amd64

| arm | First-try success | mean trials to success |
|---|---|---|
| D | 1/1 | 1.00 |

### Per repeat

| arm | repeat | first try | trials to success | final speedup | first failure stage | note |
|---|---|---|---|---|---|---|
| D | 1 | yes | 1 | 175.83 | - |  |
