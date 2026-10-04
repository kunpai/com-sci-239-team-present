Write a C function that multiplies two square matrices as fast as you can:

    void matmul(const float *A, const float *B, float *C, int n);

It computes C = A * B for n x n float matrices in row-major order, so element (i, j) of A is A[i*n + j]. The buffers do not overlap and C is uninitialized, so write every element. It must be correct for every positive n (small floating-point differences are fine). The benchmark uses large matrices.

Do not define main. The file is compiled with gcc together with a benchmark harness. Flags that set the output file, add include or library paths, or pass options to the linker are rejected.

Reply with the complete source file in one fenced code block, then on its own line the gcc flags you want, for example:

FLAGS: -O3
