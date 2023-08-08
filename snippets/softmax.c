#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

double now_in_sec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000.f / 1000.f;
}

void softmax(float *input_name, float *output_name) {
  int batch = 1200;
  int axis_len = 100;

  const float LOWER_RANGE        = -88.37626;
  const float ROUNDING_BIAS      = 12582912.0;
  const float LOG2RECIPROCAL     = 1.44269504088896341;
  const float LOG2HIGH           = -6.93145752e-1;
  const float LOG2LOW            = -1.42860677e-6;
  const float POLY_0             = 0.0013780593872;
  const float POLY_1             = 0.0083731245250;
  const float POLY_2             = 0.0416695363820;
  const float POLY_3             = 0.1666647195816;
  const float POLY_4             = 0.4999998509884;
  const float POLY_56            = 1.0000000000000;
  const int32_t MAXIMUM_EXPONENT = 0x3F800000;

#pragma omp parallel for num_threads(8)
  for (int i = 0; i < batch; i++) {
    const float *input = input_name + i * axis_len;
    float *output = output_name + i * axis_len;

    float max = -INFINITY;
    for (int j = 0; j < axis_len; j++) {
      max = fmaxf(input[j], max);
    }

    float sum = 0.0;
#pragma clang loop vectorize(enable)
    for (int j = 0; j < axis_len; j++) {
      const int val0 = fmaxf(input[j] - max, LOWER_RANGE);
      const int biased = fmaf(val0, LOG2RECIPROCAL, ROUNDING_BIAS);
      const int m = biased - ROUNDING_BIAS;
      const int val1 = fmaf(m, LOG2HIGH, val0);
      const int val2 = fmaf(m, LOG2LOW, val1);
      const int32_t normal = (*(int *)&biased) << 23;
      const int32_t normal2 = normal + MAXIMUM_EXPONENT;
      const float p0 = POLY_0;
      const float p1 = fmaf(p0, val2, POLY_1);
      const float p2 = fmaf(p1, val2, POLY_2);
      const float p3 = fmaf(p2, val2, POLY_3);
      const float p4 = fmaf(p3, val2, POLY_4);
      const float p5 = fmaf(p4, val2, POLY_56);
      const float p6 = fmaf(p5, val2, POLY_56);
      const float p7 = p6 * (*(float *)&normal2);
      sum += p7;
      output[j] = p7;
    }

    const float recip_sum = 1.0 / sum;
#pragma clang loop vectorize(enable)
    for (int j = 0; j < axis_len; j++) {
      output[j] = output[j] * recip_sum;
    }
  }
}

int main() {
  float *input = (float *)malloc(120000 * sizeof(float));
  float *output = (float *)malloc(120000 * sizeof(float));

  {
    double elapsed = 0;
    int attempts = 10;
    for (int i = 0; i < attempts; i++) {
      double start = now_in_sec();
      softmax(input, output);
      elapsed += now_in_sec() - start;
    }
  }
  {
    double elapsed = 0;
    int attempts = 10000;
    for (int i = 0; i < attempts; i++) {
      double start = now_in_sec();
      softmax(input, output);
      elapsed += now_in_sec() - start;
    }
    printf("%f ms/call\n", (elapsed / attempts) * 1000.0);
  }

  return 0;
}

