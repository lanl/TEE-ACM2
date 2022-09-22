#ifndef __OPT_SINGLE_MAT_MULT_CUH__
#define __OPT_SINGLE_MAT_MULT_CUH__
namespace SingleMatMult{
    /*
    template<typename T>
    __device__ void print_mat(T * mat ,int m, int n){
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++)
                printf("%d ", int(mat[i*n +j]));
            printf("\n");
        }
        printf("\n");
    } */


    #define fetch(arr, col, m, n, bound) arr[min(n*col + m, bound)]
    //#define fetch(arr, col, m, n, bound) arr[n*col + m]

    /*#define DIM_X 16
    #define DIM_Y  16
    #define BLK_M  192
    #define BLK_N  192
    #define BLK_K  16
    #define DIM_XA  32
    #define DIM_YA  8
    #define DIM_XB  8
    #define DIM_YB  32
    #define THR_M  12
    #define THR_N  12*/

    /*#define DIM_X 16
    #define DIM_Y  16
    #define BLK_M  128
    #define BLK_N  128
    #define BLK_K  8
    #define DIM_XA  32
    #define DIM_YA  8
    #define DIM_XB  8
    #define DIM_YB  32
    #define THR_M  8
    #define THR_N  8

    #define DIM_X 16
    #define DIM_Y  16
    #define BLK_M  208
    #define BLK_N  208
    #define BLK_K  16
    #define DIM_XA  16
    #define DIM_YA  16
    #define DIM_XB  16
    #define DIM_YB  16
    #define THR_M  13
    #define THR_N  13*/




//nvcc main.cu -o test -I/projects/darwin-nv/centos8/x86_64/packages/cuda/11.4.2/include -L/projects/darwin-nv/centos8/x86_64/packages/cuda/11.4.2/targets/x86_64-linux/lib/stubs -lcuda -lcudart -lnvidia-ml -lcublas -DBLK_M=64
    __global__ void cupy_mm( int M, int N, int K, const float* A, const float* B,float * C)
    {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int idt = DIM_X * ty + tx;

        int idxA = idt % DIM_XA;
        int idyA = idt / DIM_XA;

        int idxB = idt % DIM_XB;
        int idyB = idt / DIM_XB;

        int blx = blockIdx.x;
        int bly = blockIdx.y;

        __shared__ float sA[BLK_K][BLK_M + 1];
        __shared__ float sB[BLK_N][BLK_K + 1];

        // registers for the innermost loop
        float rC[THR_N][THR_M];
        float rA[THR_M];
        float rB[THR_N];

        float ra[BLK_K / DIM_YA][BLK_M / DIM_XA];
        float rb[BLK_N / DIM_YB][BLK_K / DIM_XB];

        const float* offs_dA = A + blx * BLK_M + idyA * M + idxA;
        int boundA = (M * (K - 1) + M) - (blx * BLK_M + idyA * M + idxA) - 1;
        const float* offs_dB = B + bly * BLK_N * K + idyB * K + idxB;
        int boundB = (K * (N - 1) + K) - (bly * BLK_N * K + idyB * K + idxB) - 1;

        int m, n, k, kk;
        
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            #pragma unroll
            for (m = 0 ; m < THR_M; m++) {
                rC[n][m] = 0;
            }
        }

        // blockwise transpose to transpose load
        #pragma unroll
        for (n = 0; n < BLK_K; n += DIM_YA) {
            #pragma unroll
            for (m = 0; m < BLK_M; m += DIM_XA) {
                sA[n + idyA][m + idxA] = fetch(offs_dA, M, m, n, boundA);
            }
        }
        // blockwise transpose to transpose load
        #pragma unroll
        for (n = 0; n < BLK_N; n += DIM_YB) {
            #pragma unroll
            for (m = 0; m < BLK_K; m += DIM_XB) {
                sB[n + idyB][m + idxB] = fetch(offs_dB, K, m, n, boundB);
            }
        }
        __syncthreads();

        for (kk = 0; kk < K - BLK_K; kk += BLK_K)
        {
            offs_dA += BLK_K * M;
            boundA -= BLK_K * M;
            offs_dB += BLK_K;
            boundB -= BLK_K;
            
            #pragma unroll
            for (n = 0; n < BLK_K / DIM_YA; n++) {
                #pragma unroll
                for (m = 0; m < BLK_M / DIM_XA; m++) {
                    ra[n][m] = fetch(offs_dA, M, m * DIM_XA, n * DIM_YA, boundA);
                }
            }

            #pragma unroll
            for (n = 0; n < BLK_N / DIM_YB; n++) {
                #pragma unroll
                for (m = 0; m < BLK_K / DIM_XB; m++) {
                    rb[n][m] = fetch(offs_dB, K, m * DIM_XB, n * DIM_YB, boundB);
                }
            }

            // multiply
            #pragma unroll
            for (k = 0; k < BLK_K; k++)
            {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                    rA[m] = sA[k][m * DIM_X + tx];
                }
                
                #pragma unroll
                for (n = 0; n < THR_N; n++) {
                    rB[n] = sB[n * DIM_Y + ty][k];
                }

                #pragma unroll
                for (n = 0; n < THR_N; n++) {
                    #pragma unroll
                    for (m = 0; m < THR_M; m++) {
                        rC[n][m] += rA[m] * rB[n];
                    }
                }
            }
            __syncthreads();

            // store A regs->smem
            #pragma unroll
            for (n = 0; n < BLK_K / DIM_YA; n++)
            {
                //float * sA_slice = &sA[n * DIM_YA + idyA][idxA];
                #pragma unroll
                for (m = 0; m < BLK_M / DIM_XA; m++)
                {
                    sA[n * DIM_YA + idyA][m * DIM_XA + idxA] = ra[n][m];
                    //sA_slice[m * DIM_XA] = ra[n][m];
                }
            }

            #pragma unroll
            for (n = 0; n < BLK_N / DIM_YB; n++)
            {
                //float * sB_slice = &sB[n * DIM_YB + idyB][idxB];
                #pragma unroll
                for (m = 0; m < BLK_K / DIM_XB; m++)
                {
                    sB[n * DIM_YB + idyB][m * DIM_XB + idxB] = rb[n][m];
                    //sB_slice[m * DIM_XB] = rb[n][m];
                }
            }
            __syncthreads();
        }

        // Multiply last full (BLK_K) or partial block of columns of A and
        // rows of B.
        // It's okay that m,n exceed matrix bounds as all work is in registers
        // or shared memory, and out-of-bounds rC[n][m] will not be saved later.

        kk = K - kk;
        #pragma unroll
        for (k = 0; k < kk; k++)
        {
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                rA[m] = sA[k][m * DIM_X + tx];
            }

            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                rB[n] = sB[n * DIM_Y + ty][k];
            }
            
            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                    rC[n][m] += rA[m] * rB[n];
                }
            }
        }

        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            int coord_dCn = bly * BLK_N + n * DIM_Y + ty;
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                int coord_dCm = blx * BLK_M + m * DIM_X + tx;
                if (coord_dCm < M && coord_dCn < N) {
                    C[coord_dCn * M + coord_dCm] = rC[n][m];
                }
            }
        }
    }
    
}
#endif
