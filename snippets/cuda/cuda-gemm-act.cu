#include <assert.h>
#include <blis/blis.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cudnn_backend.h>
#include <curand.h>
#include <stdio.h>
#include <sys/time.h>

#define ATTEMPT 10

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

  for (int attempt = 0; attempt < ATTEMPT; attempt++) {
    const double start = now_in_sec();
    for (int i = 0; i < 1000; i++) {
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, rhs, n,
                  lhs, k, &beta, result, n);
      // relu<<<(m * n + 31) / 32, 32>>>(result, m * n);
      // sigmoid<<<(m * n + 31) / 32, 32>>>(result, m * n);
    }
    const double end = now_in_sec();
    printf("GPU Time: %lf[ms]\n", (end - start) * 1000.0);
  }

  float *gpu_result = (float *)malloc(m * n * sizeof(float));
  cudaMemcpy(gpu_result, result, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  float *cpu_lhs = (float *)malloc(m * k * sizeof(float));
  float *cpu_rhs = (float *)malloc(k * n * sizeof(float));
  float *cpu_result = (float *)malloc(m * n * sizeof(float));

  cudaMemcpy(cpu_lhs, lhs, m * k * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_rhs, rhs, k * n * sizeof(float), cudaMemcpyDeviceToHost);

  for (int attempt = 0; attempt < ATTEMPT; attempt++) {
    const double start = now_in_sec();
    for (int i = 0; i < 1000; i++) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
                  cpu_lhs, k, cpu_rhs, n, beta, cpu_result, n);
    }
    const double end = now_in_sec();
    printf("CPU Time: %lf[ms]\n", (end - start) * 1000.0);
  }

  for (int i = 0; i < m * n; i++) {
    const float diff = fabs(gpu_result[i] - cpu_result[i]);
    assert(diff < 1e-3);
  }

  cudaFree(lhs);
  cudaFree(rhs);
  cudaFree(result);
}
