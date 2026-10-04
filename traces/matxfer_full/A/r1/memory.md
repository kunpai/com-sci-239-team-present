**Matmul notes (n≈1000, unknown CPU)**

**Best: attempt 2, 8.0 ms (~250 GFLOP/s)**
- Packed A and B in persistent static buffers (grow only when n increases)
- 12×32 AVX-512 / 6×16 AVX2 kernel, KC=256
- `collapse(2)` dynamic tasks, p-outer/q-inner
- Flags: `-O3 -march=native -fopenmp -ffast-math -funroll-loops`

**Helped:**
- Persistent buffers vs. per-call malloc: 10.4 → 8.0 ms (biggest win; avoids page faults)
- Packing A once
- KC≈256

**Hurt/neutral:**
- KC=192: 9.5; KC=504: 8.5; KC=n: 8.1 ms
- Thread-derived tiny tasks: 8.7 ms
- Static contiguous split + per-thread A pack: 10.2 ms
- Unpacked A or per-task redundant A pack: 9.3 ms
- Vectorized tr8 transpose/merged pack loop: 8.5–8.7 ms (packing isn't the bottleneck)
- Prefetch tweaks: neutral to slightly worse

**Untested:**
- If Zen4 (double-pumped 512-bit), try `-mprefer-vector-width=256` or force 6×16 AVX2
- One task per ib strip looping all jb, A packed once per kb
- Overlap packing next kb slice of B with compute
- 14×32 or 8×48 tile
- `proc_bind(close)` affinity

**Advice:** Change one variable at a time from attempt 2; noise ≈ ±0.5 ms.
