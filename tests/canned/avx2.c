#include <immintrin.h>
#include <stddef.h>
#include <string.h>

void matmul(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        float *c = C + (size_t)i * n;
        memset(c, 0, (size_t)n * sizeof(float));
        for (int k = 0; k < n; k++) {
            float a_s = A[(size_t)i * n + k];
            __m256 a = _mm256_set1_ps(a_s);
            const float *b = B + (size_t)k * n;
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 cv = _mm256_loadu_ps(c + j);
                cv = _mm256_fmadd_ps(a, _mm256_loadu_ps(b + j), cv);
                _mm256_storeu_ps(c + j, cv);
            }
            for (; j < n; j++) c[j] += a_s * b[j];
        }
    }
}
