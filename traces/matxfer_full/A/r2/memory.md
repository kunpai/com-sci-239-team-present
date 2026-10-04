Notes (matmul, n≈1000, gcc, OpenMP, AVX-512):

- Baseline (Attempt 0): 6×16 ymm BLIS kernel, packed A and B, 7.1 ms.
- Unpacked A (read rows with stride n): 6.6 ms. Global A pre-pack (8.3) and per-task A repack (9.2) both hurt.
- KCMAX 384 slightly beat 320 (6.5 ms). 512 gave 10.3 ms because the B panel overflowed L1. KC=192 with 8×32 zmm gave no gain.
- AVX-512 only helped with a 12×32 zmm kernel: 24 accumulators, unpacked A, zero-padded ebuf for the row edge, NV=1 path for column edges. 6.1 ms.
- Best so far (Attempt 10), 5.9 ms: `ir` loop outer, `jp` inner, so A rows stay in L1.
- Neutral or harmful: `-mno-avx512f`, 16× manual unroll with prefetches, `-funroll-loops` with a big body, masked edge stores (neutral).
- Try next, one at a time:
  - KCMAX ~192–224 with 12×32 (B panel fits L1 vs. extra C pass).
  - Pack B per kb slice to drop the serial pre-pack barrier.
  - 8-column masked edge.
  - Tune MC and PPCMAX.
  - Pack the 12-row A strip contiguously per (task, kb), in case n=1024 aliasing matters.
