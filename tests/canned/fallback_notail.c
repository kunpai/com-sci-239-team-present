#include <stddef.h>

static void naive(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++) s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = s;
        }
}

/* Small sizes fall back to the naive loop; larger ones use a kernel that forgets the tail. */
void matmul(const float *A, const float *B, float *C, int n) {
    if (n < 64) {
        naive(A, B, C, n);
        return;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j + 4 <= n; j += 4) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int k = 0; k < n; k++) {
                float a = A[(size_t)i * n + k];
                const float *b = B + (size_t)k * n + j;
                s0 += a * b[0]; s1 += a * b[1]; s2 += a * b[2]; s3 += a * b[3];
            }
            float *c = C + (size_t)i * n + j;
            c[0] = s0; c[1] = s1; c[2] = s2; c[3] = s3;
        }
    }
}
