# Results

Config: n=1024, rounds=10, repeats=3, trials=3, window=3. Naive baseline (ms): amd64=2116.1, armv7=4068.2

## Arm A: Self-Refine on amd64

| repeat | best speedup | speedup per attempt | note |
|---|---|---|---|
| 1 | 199.99 | 193.55, 155.84, 173.08, 152.01, 156.82, 159.13, 167.06, 154.37, 199.99, 177.32, 193.89 |  |
| 2 | 266.41 | 191.56, 252.22, 233.26, 258.10, 266.41, 244.64, 242.81, 154.98, 260.22, 264.98, 256.22 |  |
| 3 | 270.88 | 193.61, 176.68, 262.38, 261.31, 246.52, 270.88, 195.85, 258.70, 268.07, 263.69, 246.78 |  |

## Arms B (no memory) and C (x86 memory) on arm/v7

| arm | First-try success | mean trials to success |
|---|---|---|
| B | 3/3 | 1.00 |
| C | 2/3 | 1.33 |

### Per repeat

| arm | repeat | first try | trials to success | final speedup | first failure stage | note |
|---|---|---|---|---|---|---|
| B | 1 | yes | 1 | 5.97 | - |  |
| C | 1 | no | 2 | 5.61 | compile |  |
| B | 2 | yes | 1 | 6.33 | - |  |
| C | 2 | yes | 1 | 6.95 | - |  |
| B | 3 | yes | 1 | 4.64 | - |  |
| C | 3 | yes | 1 | 7.67 | - |  |
