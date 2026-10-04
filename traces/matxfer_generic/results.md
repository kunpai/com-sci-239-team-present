# Results

Config: n=1024, rounds=10, repeats=3, trials=3, window=3. Naive baseline (ms): amd64=2122.4, armv7=4030.9

## Arm A: Self-Refine on amd64

| repeat | best speedup | speedup per attempt | note |
|---|---|---|---|
| 2 | 205.59 | 179.78, 134.18, 168.32, 170.36, 162.27, 172.46, 125.70, 137.82, 153.05, 157.65, 205.59 |  |

## Arms B (no memory) and C (x86 memory) on arm/v7

| arm | First-try success | mean trials to success |
|---|---|---|
| B | 1/1 | 1.00 |
| C | 0/1 | 2.00 |

### Per repeat

| arm | repeat | first try | trials to success | final speedup | first failure stage | note |
|---|---|---|---|---|---|---|
| B | 2 | yes | 1 | 1.09 | - |  |
| C | 2 | no | 2 | 5.15 | compile |  |
