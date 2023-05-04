#include <iostream>

#include <blis/cblas.h>
#include <sys/time.h>

const int m = 128;
const int n = 256;
const int k = 1024;

const int iter = 10;

double now_in_sec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
  float *x = (float *)calloc(m * k, sizeof(float));
  float *y = (float *)calloc(k * n, sizeof(float));
  float *z = (float *)calloc(m * n, sizeof(float));

  for (int i = 0; i < iter; i++) {
    const auto start = now_in_sec();
    const int ave = 100;
    for (int j = 0; j < ave; j++)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, x, k, y, n, 0.0, z, n);
    const auto elapsed = now_in_sec() - start;

    std::cout << "Elapsed: " << (elapsed * 1000.0 / ave) << " [ms]" << std::endl;
  }

  return 0;
}
