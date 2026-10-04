# Results

Config: n=1024, rounds=10, repeats=3, trials=3, window=3. Naive baseline (ms): amd64=2163.4, armv7=4059.5

## Arm A: Self-Refine on amd64

| repeat | best speedup | speedup per attempt | note |
|---|---|---|---|

## Arms B (no memory) and C (x86 memory) on arm/v7

| arm | First-try success | mean trials to success |
|---|---|---|
| B | 3/3 | 1.00 |
| C | 2/3 | 1.33 |

### Per repeat

| arm | repeat | first try | trials to success | final speedup | first failure stage | note |
|---|---|---|---|---|---|---|
| B | 1 | yes | 1 | 3.79 | - |  |
| C | 1 | yes | 1 | 5.37 | - |  |
| B | 2 | yes | 1 | 4.14 | - |  |
| C | 2 | no | 2 | 7.74 | compile |  |
| B | 3 | yes | 1 | 6.35 | - |  |
| C | 3 | yes | 1 | 6.40 | - |  |
