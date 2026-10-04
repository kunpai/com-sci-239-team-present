# Results

Config: n=1024, rounds=10, repeats=3, trials=3, window=3. Naive baseline (ms): amd64=2090.1, armv7=4163.1

## Arm A: Self-Refine on amd64

| repeat | best speedup | speedup per attempt | note |
|---|---|---|---|
| 1 | 261.85 | 212.39, 201.07, 261.85, 240.80, 204.41, 219.30, 247.03, 256.77, 224.38, 223.80, 239.66 |  |
| 2 | 353.84 | 292.98, 252.80, 270.99, 227.11, 314.73, 203.24, 310.24, 320.67, 316.20, 343.32, 353.84 |  |
| 3 | 324.55 | 278.42, 220.92, 246.21, 278.01, 275.27, 324.55, 304.95, 306.29, 266.49, 303.93, 312.28 |  |

## Arms B (no memory) and C (x86 memory) on arm/v7

| arm | First-try success | mean trials to success |
|---|---|---|
| B | 3/3 | 1.00 |
| C | 2/3 | 1.33 |

### Per repeat

| arm | repeat | first try | trials to success | final speedup | first failure stage | note |
|---|---|---|---|---|---|---|
| B | 1 | yes | 1 | 3.51 | - |  |
| C | 1 | yes | 1 | 2.79 | - |  |
| B | 2 | yes | 1 | 2.79 | - |  |
| C | 2 | yes | 1 | 3.80 | - |  |
| B | 3 | yes | 1 | 3.71 | - |  |
| C | 3 | no | 2 | 2.64 | compile |  |
