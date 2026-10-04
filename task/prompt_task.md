You are optimizing a C function for speed.

Write a single C source file that defines

    void matmul(const float *A, const float *B, float *C, int n);

which computes C = A * B for square n x n matrices of single-precision (float) values stored in row-major order, so element (i, j) of A is A[i*n + j]. The three buffers do not overlap. C is not initialized: your function must write every element of C.

Requirements:
- The result must be correct for every positive n, not only for convenient sizes. Results are checked against a reference with a relative tolerance of about 1e-3, so reordering floating-point sums is fine.
- The benchmark runs large matrices (around a thousand rows and columns) and reports the best of several runs, so optimize for large n.
- The file is compiled with gcc and linked, together with a fixed benchmark harness that calls matmul, against the C math library and pthreads. Do not define main. Only the C standard library, libm, pthreads and OpenMP are available.
- You choose the compiler flags. They are passed to gcc as given (for example -O3 or -fopenmp). Flags that set the output file, add include or library paths, or pass options through to the linker are rejected.
- The program is built and run inside a Linux container, not on the machine you are running on. The CPU is not disclosed.
- Optimize for speed on the machine this runs on.

Reply format. Reply with the complete source file in exactly one fenced block:

```c
...
```

followed on its own line by the compiler flags, for example:

FLAGS: -O3 -march=native

Do not write any other code block or any other line starting with FLAGS:.
