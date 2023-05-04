#include <cassert>
#include <iostream>

#include <blis/cblas.h>
#include <sys/time.h>

#include <immintrin.h>
#include <xmmintrin.h>

const int m = 128;
const int n = 256;
const int k = 1024;

const int iter = 10;

double now_in_sec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void myblas_sgemm_1(int m, int n, int k, const float *a, int lda,
                    const float *b, int ldb, float *c, int ldc) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      float sum = 0.0;
#pragma clang loop vectorize(enable)
      for (int l = 0; l < k; l++)
        sum += a[i * lda + l] * b[l * ldb + j];
      c[i * ldc + j] = sum;
    }
}

void myblas_sgemm_2(int m, int n, int k, const float *a, int lda,
                    const float *b, int ldb, float *c, int ldc) {
  assert(m % 8 == 0);
  assert(n % 8 == 0);
  assert(k % 8 == 0);
  for (int i = 0; i < m; i += 8)
    for (int j = 0; j < n; j += 8) {
      __m256 sum[8] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                       _mm256_setzero_ps(), _mm256_setzero_ps(),
                       _mm256_setzero_ps(), _mm256_setzero_ps(),
                       _mm256_setzero_ps(), _mm256_setzero_ps()};
      for (int l = 0; l < k; l++) {
        _mm_prefetch((const char *)(b + (l + 0) * ldb + j), _MM_HINT_T0);
        _mm_prefetch((const char *)(b + (l + 1) * ldb + j), _MM_HINT_T0);
        _mm_prefetch((const char *)(b + (l + 2) * ldb + j), _MM_HINT_T0);
        _mm_prefetch((const char *)(b + (l + 3) * ldb + j), _MM_HINT_T0);
        const __m256 a0 = _mm256_broadcast_ss(a + (i + 0) * lda + l);
        const __m256 a1 = _mm256_broadcast_ss(a + (i + 1) * lda + l);
        const __m256 a2 = _mm256_broadcast_ss(a + (i + 2) * lda + l);
        const __m256 a3 = _mm256_broadcast_ss(a + (i + 3) * lda + l);
        const __m256 a4 = _mm256_broadcast_ss(a + (i + 4) * lda + l);
        const __m256 a5 = _mm256_broadcast_ss(a + (i + 5) * lda + l);
        const __m256 a6 = _mm256_broadcast_ss(a + (i + 6) * lda + l);
        const __m256 a7 = _mm256_broadcast_ss(a + (i + 7) * lda + l);
        __m256 bs = _mm256_loadu_ps(b + l * ldb + j);
        sum[0] = _mm256_fmadd_ps(a0, bs, sum[0]);
        sum[1] = _mm256_fmadd_ps(a1, bs, sum[1]);
        sum[2] = _mm256_fmadd_ps(a2, bs, sum[2]);
        sum[3] = _mm256_fmadd_ps(a3, bs, sum[3]);
        sum[4] = _mm256_fmadd_ps(a4, bs, sum[4]);
        sum[5] = _mm256_fmadd_ps(a5, bs, sum[5]);
        sum[6] = _mm256_fmadd_ps(a6, bs, sum[6]);
        sum[7] = _mm256_fmadd_ps(a7, bs, sum[7]);
      }
      _mm256_storeu_ps(c + (i + 0) * ldc + j, sum[0]);
      _mm256_storeu_ps(c + (i + 1) * ldc + j, sum[1]);
      _mm256_storeu_ps(c + (i + 2) * ldc + j, sum[2]);
      _mm256_storeu_ps(c + (i + 3) * ldc + j, sum[3]);
      _mm256_storeu_ps(c + (i + 4) * ldc + j, sum[4]);
      _mm256_storeu_ps(c + (i + 5) * ldc + j, sum[5]);
      _mm256_storeu_ps(c + (i + 6) * ldc + j, sum[6]);
      _mm256_storeu_ps(c + (i + 7) * ldc + j, sum[7]);
    }
}

void fill_random(float *x, int n) {
  for (int i = 0; i < n; i++)
    x[i] = (float)rand() / (float)RAND_MAX;
}

bool allclose(const float *x, const float *y, int n) {
  for (int i = 0; i < n; i++) {
    // std::cout << x[i] << " vs " << y[i] << std::endl;
    // std::cout << fabs(x[i] - y[i]) << std::endl;
    if (fabs(x[i] - y[i]) > 1e-3)
      return false;
  }
  return true;
}

int main() {
  float *x = (float *)calloc(m * k, sizeof(float));
  float *y = (float *)calloc(k * n, sizeof(float));
  float *cblas_z = (float *)calloc(m * n, sizeof(float));
  float *myblas_z = (float *)calloc(m * n, sizeof(float));

  fill_random(x, m * k);
  fill_random(y, k * n);

  for (int i = 0; i < iter; i++) {
    const int ave = 30;
    double cblas_elapsed = 0.0;
    double myblas_elapsed = 0.0;

    for (int j = 0; j < ave; j++) {
      const double cblas_start = now_in_sec();
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, x, k,
                  y, n, 0.0, cblas_z, n);
      cblas_elapsed += now_in_sec() - cblas_start;

      const double myblas_start = now_in_sec();
      myblas_sgemm_2(m, n, k, x, k, y, n, myblas_z, n);
      myblas_elapsed += now_in_sec() - myblas_start;

      assert(allclose(cblas_z, myblas_z, m * n));
    }

    std::cout << "[blis] " << (cblas_elapsed * 1000.0 / ave) << " [ms]"
              << std::endl;
    std::cout << "[mine] " << (myblas_elapsed * 1000.0 / ave) << " [ms]"
              << std::endl;
  }

  return 0;
}
