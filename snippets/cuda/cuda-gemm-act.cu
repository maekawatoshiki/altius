#include <stdio.h>
#include <sys/time.h>

#include <cublas.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>

double now_in_sec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000.f / 1000.f;
}

__global__ void relu(float *x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = x[i] > 0.f ? x[i] : 0.f;
  }
}

__global__ void sigmoid(float *x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = 1.f / (1.f + expf(-x[i]));
  }
}

extern "C" void entry() {
  cublasHandle_t handle;
  cublasCreate(&handle);

  float *lhs, *rhs, *result;
  const int m = 1000, k = 200, n = 100;
  const float alpha = 1.f, beta = 0.f;

  cudaMalloc(&lhs, m * k * sizeof(float));
  cudaMalloc(&rhs, k * n * sizeof(float));
  cudaMalloc(&result, m * n * sizeof(float));

  {
    // fill lhs and rhs with random numbers
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, lhs, m * k);
    curandGenerateUniform(gen, rhs, k * n);
    curandDestroyGenerator(gen);
  }

  for (int attempt = 0; attempt < 100; attempt++) {
    const double start = now_in_sec();
    for (int i = 0; i < 1000; i++) {
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, lhs, m,
                  rhs, k, &beta, result, m);
      relu<<<(m * n + 31) / 32, 32>>>(result, m * n);
      sigmoid<<<(m * n + 31) / 32, 32>>>(result, m * n);
    }
    const double end = now_in_sec();
    printf("Time: %lf[ms]\n", (end - start) * 1000.0);
  }

  float *host = (float *)malloc(m * n * sizeof(float));
  cudaMemcpy(host, result, m * n * sizeof(float), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < m * n; i++) {
  //   printf("%f ", host[i]);
  // }
}
