#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void matmul(const float *A, const float *B, float *C, int n);

static uint32_t rng_state = 12345u;

static float next_float(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / 8388608.0f) - 1.0f; /* [-1, 1) */
}

static float *alloc_matrix(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * n * sizeof(float)) != 0) return NULL;
    return (float *)p;
}

static void fill(float *m, size_t n) {
    for (size_t i = 0; i < n * n; i++) m[i] = next_float();
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* Full O(n^3) reference in double precision; only used for the tiny size. */
static void reference(const float *A, const float *B, float *C, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double s = 0.0;
            for (size_t k = 0; k < n; k++) s += (double)A[i * n + k] * (double)B[k * n + j];
            C[i * n + j] = (float)s;
        }
    }
}

/* 1 if `got` matches `want` in max-norm relative error <= 1e-3 and has no NaN/Inf. */
static int close_enough(const double *got, const double *want, size_t len) {
    double max_diff = 0.0, max_want = 0.0;
    for (size_t i = 0; i < len; i++) {
        if (!isfinite(got[i])) return 0;
        double d = fabs(got[i] - want[i]);
        if (d > max_diff) max_diff = d;
        if (fabs(want[i]) > max_want) max_want = fabs(want[i]);
    }
    return max_diff <= 1e-3 * (max_want + 1e-9);
}

static int check_small(void) {
    const size_t n = 37;
    float *A = alloc_matrix(n), *B = alloc_matrix(n), *C = alloc_matrix(n), *R = alloc_matrix(n);
    double *got = malloc(n * n * sizeof(double)), *want = malloc(n * n * sizeof(double));
    if (!A || !B || !C || !R || !got || !want) return 0;
    fill(A, n);
    fill(B, n);
    memset(C, 0xFF, n * n * sizeof(float));
    matmul(A, B, C, (int)n);
    reference(A, B, R, n);
    for (size_t i = 0; i < n * n; i++) {
        got[i] = (double)C[i];
        want[i] = (double)R[i];
    }
    int ok = close_enough(got, want, n * n);
    free(A); free(B); free(C); free(R); free(got); free(want);
    return ok;
}

/* Freivalds-style O(n^2) check: compare C*x with A*(B*x) for a fixed random x. */
static int check_freivalds(const float *A, const float *B, const float *C, size_t n) {
    double *x = malloc(n * sizeof(double)), *y = malloc(n * sizeof(double));
    double *z = malloc(n * sizeof(double)), *w = malloc(n * sizeof(double));
    if (!x || !y || !z || !w) return 0;
    for (size_t j = 0; j < n; j++) x[j] = (double)next_float();
    for (size_t k = 0; k < n; k++) {
        double s = 0.0;
        for (size_t j = 0; j < n; j++) s += (double)B[k * n + j] * x[j];
        y[k] = s;
    }
    for (size_t i = 0; i < n; i++) {
        double s = 0.0;
        for (size_t k = 0; k < n; k++) s += (double)A[i * n + k] * y[k];
        z[i] = s;
        double t = 0.0;
        for (size_t j = 0; j < n; j++) t += (double)C[i * n + j] * x[j];
        w[i] = t;
    }
    int ok = close_enough(w, z, n);
    free(x); free(y); free(z); free(w);
    return ok;
}

/* Freivalds check at an odd size near n, so a fast path that is only right for "nice" sizes
 * (or a small-n fallback) cannot hide a broken tail. */
static int check_odd(size_t n) {
    float *A = alloc_matrix(n), *B = alloc_matrix(n), *C = alloc_matrix(n);
    if (!A || !B || !C) return 0;
    fill(A, n);
    fill(B, n);
    memset(C, 0xFF, n * n * sizeof(float));
    matmul(A, B, C, (int)n);
    int ok = check_freivalds(A, B, C, n);
    free(A); free(B); free(C);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: bench N REPS\n");
        return 2;
    }
    size_t n = (size_t)strtoull(argv[1], NULL, 10);
    int reps = atoi(argv[2]);
    if (n == 0 || reps < 1) {
        fprintf(stderr, "bad arguments\n");
        return 2;
    }

    int small_ok = check_small();
    printf("SMALL_OK %d\n", small_ok);
    fflush(stdout);
    if (!small_ok) return 0;

    int odd_ok = check_odd(n | 1);
    printf("ODD_OK %d\n", odd_ok);
    fflush(stdout);
    if (!odd_ok) return 0;

    float *A = alloc_matrix(n), *B = alloc_matrix(n), *C = alloc_matrix(n);
    if (!A || !B || !C) {
        fprintf(stderr, "allocation failed for n=%zu\n", n);
        return 2;
    }
    fill(A, n);
    fill(B, n);

    double best = -1.0;
    int freivalds_ok = 0;
    for (int r = 0; r < reps; r++) {
        memset(C, 0xFF, n * n * sizeof(float));
        double t0 = now_ms();
        matmul(A, B, C, (int)n);
        double ms = now_ms() - t0;
        if (best < 0.0 || ms < best) best = ms;
        if (r == 0) freivalds_ok = check_freivalds(A, B, C, n);
    }
    printf("FREIVALDS_OK %d\n", freivalds_ok);
    printf("TIME_MS %.3f\n", best);
    return 0;
}
