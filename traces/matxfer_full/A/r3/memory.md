Matmul task notes (n≈1000, float, unknown Linux CPU with AVX-512):

- **Baseline:** Goto-style AVX2 6×16 kernel, packed A/B, KC=256, dynamic OpenMP items: 7.5 ms (~270 GFLOP/s).
- **Best (Attempt 5): 6.4 ms.** 12×32 zmm kernel, `-mprefer-vector-width=512`, KC=256. A and B packed once, one barrier. Dynamic items are MC×NCG tiles over full K, with ≥3×nt items. Only the AVX-512 switch gave a real gain (~15%).
- **No help (7.5–7.6 ms):** static buffers, shared A packing, double buffering, 2D static thread grids, full-K items on AVX2. Blocking variations plateau, so the limit is likely kernel throughput or noise.
- **Regressions:** KC=384 with NR=32 (B panel overflows L1): 9.5 ms. KC=192: 6.9 ms. Unaligned kc=250: 6.8 ms. MR=8: 6.9 ms. Noinline edge kernel: 7.8 ms (probably noise).
- **Neutral:** physical-core threads via sysfs SMT width: 6.7 ms.
- **Noise:** ~±10%; differences under 0.5 ms aren't meaningful.
- **Next steps:**
  - Resubmit Attempt 5 verbatim to measure noise.
  - Try a 14×32 kernel; check for spills.
  - Pack B lazily per item to overlap with compute; replace the stride-n A gather with a vector transpose.
  - Add `-falign-loops=64`.
  - Prefetch next A/B micro-panels.
