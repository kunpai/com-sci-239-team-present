Notes for future self (matmul, n≈1000):

**Attempt 0: 10.3 ms (206x over naive)**
- B packed into 16-wide column panels.
- 6x16 register-blocked microkernel using GCC vector extensions (v4 floats, memcpy loads).
- pthreads with an atomic work queue over 48-row chunks.
- KC=256 K-blocking keeps A rows and the panel slice in L1/L2.
- -march=native may widen v4 to AVX/FMA; assembly not checked.
- Single attempt, no ablations, so per-piece contributions are unknown.

**Weaknesses**
- A is unpacked, so the kernel does strided scalar loads from 6 rows.
- Each chunk loops over all panels, giving poor A reuse across panels.
- Threads are spawned twice per call, which is noticeable overhead at 10 ms.

**Next steps**
- Pack A into MR-interleaved blocks.
- Use 8-wide vectors (AVX2) with an 8x8 or 6x16 kernel and explicit FMA. First check the architecture: darwin suggests possibly Apple M-series (NEON) rather than x86.
- Tune MC/KC/NC for L2/L3 using BLIS-style loop ordering.
- Merge packing and compute into one parallel region, or use a persistent OpenMP pool.
- Try -O3 -march=native -ffast-math -funroll-loops.
