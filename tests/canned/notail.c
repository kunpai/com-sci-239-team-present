#include <stddef.h>

void matmul(const float *A, const float *B, float *C, int n) {
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
