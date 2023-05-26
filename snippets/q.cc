#include <sys/time.h>

#include <iostream>

#define SZ 8 * 8 * 1024 * 1024

float input_0[SZ];
float input_1[SZ];
float output[SZ];

int16_t input_0_int16[SZ];
int16_t input_1_int16[SZ];
int16_t output_int16[SZ];

double now_in_sec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000.f / 1000.f;
}

void add(const float *a, const float *b, float *c, const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

void add_int16(const int16_t *a, const int16_t *b, int16_t *c, const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int attempts = 10;
  int count = 0;
  double sum = 0;
  for (int i = 0; i < attempts; i++) {
    const auto start = now_in_sec();
    { add(input_0, input_1, output, SZ); }
    const auto end = now_in_sec();
    if (i > 0) {
      sum += (end - start);
      count++;
    }
    std::cout << "float mean: " << sum / count << "s\t" << '\r';
    // std::cout << (end - start) << "s" << std::endl;
  }

  std::cout << std::endl;

  count = 0;
  sum = 0;
  for (int i = 0; i < attempts; i++) {
    const auto start = now_in_sec();
    { add_int16(input_0_int16, input_1_int16, output_int16, SZ); }
    const auto end = now_in_sec();
    if (i > 0) {
      sum += (end - start);
      count++;
    }
    std::cout << "int16_t mean: " << sum / count << "s\t" << '\r';
  }

  std::cout << std::endl;

  return 0;
}
