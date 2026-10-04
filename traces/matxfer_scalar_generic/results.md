# Results

Config: n=1024, rounds=10, repeats=3, trials=3, window=3. Naive baseline (ms): amd64=2168.0, armv7=4067.4

## Arm A: Self-Refine on amd64

| repeat | best speedup | speedup per attempt | note |
|---|---|---|---|

## Arms B (no memory) and C (x86 memory) on arm/v7

| arm | First-try success | mean trials to success |
|---|---|---|
| B | 0/1 | 2.00 |
| C | 1/1 | 1.00 |

### Per repeat

| arm | repeat | first try | trials to success | final speedup | first failure stage | note |
|---|---|---|---|---|---|---|
| B | 2 | no | 2 | 3.84 | parse |  |
| C | 2 | yes | 1 | 6.76 | - |  |
