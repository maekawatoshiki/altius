// [2024-10-05T16:04:57Z DEBUG altius_session_clang::translator] m=197, k=1536, n=384
// [2024-10-05T16:04:57Z DEBUG altius_session_clang::translator] m=197, k=384, n=384

#include <blis/cblas.h>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>
#include <cblas.h>
#include <blis.h>

static const size_t N = 4000;
static const size_t ThreadBlockSizeI = N/16; // 250;  // 5周（×16スレッド）
static const size_t ThreadBlockSizeK = N; // 50周
static const size_t ThreadBlockSizeJ = N; // 50周
static const size_t L3CacheBlockSizeI = 50;  // 1周
static const size_t L3CacheBlockSizeK = 80;  // 2周
static const size_t L3CacheBlockSizeJ = 80;  // 2周
static const size_t L1DCacheBlockSizeI = 50; // 10周
static const size_t L1DCacheBlockSizeK = 40; // 10周
static const size_t L1DCacheBlockSizeJ = 80; // SIMD方向8要素×5周
static const size_t RegisterBlockSizeI = 5;  // 5レジスタ並列に
static const size_t RegisterBlockSizeK = 4;  // fma連鎖4回

void mm( const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c ) {
    for( int i1 = 0; i1 < ThreadBlockSizeI; i1 += L3CacheBlockSizeI )
        for( int k1 = 0; k1 < ThreadBlockSizeK; k1 += L3CacheBlockSizeK )
            for( int j1 = 0; j1 < ThreadBlockSizeJ; j1 += L3CacheBlockSizeJ )
                for( int i2 = 0; i2 < L3CacheBlockSizeI; i2 += L1DCacheBlockSizeI )
                    for( int k2 = 0; k2 < L3CacheBlockSizeK; k2 += L1DCacheBlockSizeK )
                        for( int j2 = 0; j2 < L3CacheBlockSizeJ; j2 += L1DCacheBlockSizeJ )
                            for( int i3 = 0; i3 < L1DCacheBlockSizeI; i3 += RegisterBlockSizeI )
                                for( int k3 = 0; k3 < L1DCacheBlockSizeK; k3 += RegisterBlockSizeK )
                                    for( int j3 = 0; j3 < L1DCacheBlockSizeJ; j3 += 1 )
                                        for( int i4 = 0; i4 < RegisterBlockSizeI; i4 += 1 )
                                            for( int k4 = 0; k4 < RegisterBlockSizeK; k4 += 1 )
                                            {
                                                int i = i1 + i2 + i3 + i4;
                                                int k = k1 + k2 + k3 + k4;
                                                int j = j1 + j2 + j3;

                                                c[i*N+j] = fma( a[i*N+k], b[k*N+j], c[i*N+j] );
                                            }
}

alignas(64) float ah[N*N];
alignas(64) float bh[N*N];
alignas(64) float ch[N*N], ch_cblas[N*N];

int main() {
    std::mt19937_64 mt;
    std::uniform_real_distribution dist(-1.0, 1.0);

    for( int i = 0; i < N*N; ++i ) {
        ah[i] = dist( mt );
        bh[i] = dist( mt );
        ch[i] = ch_cblas[i] = dist( mt );
    }

    std::cout << "initialized." << std::endl;

    while (true) {
        const auto start = std::chrono::system_clock::now();

#pragma omp parallel for num_threads(16)
        for( int tid = 0; tid < 16; ++tid )
        {
            int i0 = tid % 16 * ThreadBlockSizeI;
            int j0 = tid / 16 * ThreadBlockSizeJ;
            mm( &ah[i0*N], &bh[j0], &ch[i0*N+j0] );
        }

        const auto finish = std::chrono::system_clock::now();

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, ah, N, bh, N, 1.0, ch_cblas, N);

        for( int i = 0; i < N*N; ++i )
            if( std::abs( ch[i] - ch_cblas[i] ) > 1e-2 ) {
                std::cerr << "mismatch at " << i << ": " << ch[i] << " != " << ch_cblas[i] << std::endl;
                exit(EXIT_FAILURE);
            }

        const double s = std::chrono::duration_cast<std::chrono::nanoseconds>( finish - start ).count() * 1e-9;
        static constexpr double flop_per_fma = 2.0;
        std::cout << s << " seconds, " << N*N*N*flop_per_fma/s * 1e-9 << " GFLOPS" << std::endl;
    }
}


