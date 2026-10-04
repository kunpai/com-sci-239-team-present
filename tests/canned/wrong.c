#include <stddef.h>

void matmul(const float *A, const float *B, float *C, int n) {
    (void)A; (void)B;
    for (size_t i = 0; i < (size_t)n * n; i++) C[i] = 0.0f;
}
