#ifndef __NO_ATOMIC_FUSED_LEFT_MAT_MULT_CUH__
#define __NO_ATOMIC_FUSED_LEFT_MAT_MULT_CUH__
#include "cuda_config.cuh"

// How to launch
// grid size : dim3 gridSize(1, ceil(P0/ (double)(BLK_M)), 1);
namespace FusedMatMult{
    #define DIM_X 1
    #define DIM_Y  208
    #define BLK_M  208
    #define BLK_N  208
    #define BLK_K  16
    #define DIM_XA  16
    #define DIM_YA  13
    #define DIM_XB  13
    #define DIM_YB  16
    #define THR_M  1
    #define THR_N  BLK_N
    #define BLK_TMP BLK_K
  
    #define rC(i) rB[i]
    #define sR(i,j) sA[i][j]

    __global__ void tee_acm_fused(int P0, int P1, int P2, int P3, float* A, float * B, float * C, float * R)
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
        
        __shared__ float sA[BLK_M][BLK_K];
        __shared__ float sB[BLK_K][BLK_N];
        
        // registers 
        float rR[THR_M][THR_N];
        float rA[THR_M];
        float rB[THR_N];

        int m, n, k, kk;
        for(int shift_N = 0; shift_N < P2; shift_N+=BLK_N){
            const float* offs_dA = A + bly * BLK_M * P1 + idyA * P1 + idxA;
            int boundA = (P1 * (P0 - 1) + P1) - (bly * BLK_M * P1 + idyA * P1 + idxA) - 1;

            const float* offs_dB = B + shift_N + idyB * P2 + idxB;
            int boundB = (P2 * (P1 - 1) + P2) - (shift_N  + idyB * P2 + idxB) - 1;

            /**************************************************************
            * Multiply A * B = R_tmp 
            ************************************************************* */
            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0 ; m < THR_M; m++) {
                    rR[m][n] = 0;
                }
            }

            for (kk = 0; kk < P1 - BLK_K; kk += BLK_K)
            {
                #pragma unroll
                for (n = 0; n < BLK_M; n += DIM_YA) {
                    #pragma unroll
                    for (m = 0; m < BLK_K; m += DIM_XA) {
                        sA[n + idyA][m + idxA] = fetch(offs_dA, P1, m, n, boundA);
                    }
                }

                #pragma unroll
                for (n = 0; n < BLK_K; n += DIM_YB) {
                    #pragma unroll
                    for (m = 0; m < BLK_N; m += DIM_XB) {
                        sB[n + idyB][m + idxB] = fetch(offs_dB, P2, m, n, boundB);
                    }
                }
                __syncthreads();

                // multiply
                #pragma unroll
                for (k = 0; k < BLK_K; k++)
                {
                    #pragma unroll
                    for (m = 0; m < THR_M; m++) {
                        rA[m] = sA[m * DIM_Y + ty][k];
                    }
                    
                    #pragma unroll
                    for (n = 0; n < THR_N; n++) {
                        rB[n] = sB[k][n];
                    }
                    
                    #pragma unroll
                    for (m = 0; m < THR_M; m++) {
                        #pragma unroll
                        for (n = 0; n < THR_N; n++) {    
                            rR[m][n] += rA[m] * rB[n];
                        }
                    }
                }
                __syncthreads();

                offs_dA += BLK_K;
                boundA -= BLK_K;
                offs_dB += BLK_K * P2;
                boundB -= BLK_K * P2;
            }

            
            #pragma unroll
            for (n = 0; n < BLK_M; n += DIM_YA) {
                #pragma unroll
                for (m = 0; m < BLK_K; m += DIM_XA) {
                    sA[n + idyA][m + idxA] = fetch(offs_dA, P1, m, n, boundA);
                }
            }
            // blockwise transpose to transpose load
            #pragma unroll
            for (n = 0; n < BLK_K; n += DIM_YB) {
                #pragma unroll
                for (m = 0; m < BLK_N; m += DIM_XB) {
                    sB[n + idyB][m + idxB] = fetch(offs_dB, P2, m, n, boundB);
                }
            }

            __syncthreads();
            
            kk = P1 - kk;
            #pragma unroll
            for (k = 0; k < kk; k++)
            {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                    rA[m] = sA[m * DIM_Y + ty][k];
                }
                
                #pragma unroll
                for (n = 0; n < THR_N; n++) {
                    //rB[n] = sB[k][n * DIM_X + tx];
                    rB[n] = sB[k][n];
                }
                
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                    #pragma unroll
                    for (n = 0; n < THR_N; n++) {    
                        rR[m][n] += rA[m] * rB[n];
                    }
                }
            }

            //if(blx == 0 && bly ==0 && tx ==0 && ty ==1) print_mat(rR);

            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                int coord_dCm =  bly * BLK_M + ty;
                #pragma unroll
                for (n = 0; n < THR_N; n++) {
                    int coord_dCn = shift_N + n ;

                    if (coord_dCm >= P0 || coord_dCn >= P2) 
                        rR[m][n] = 0;
                    //else 
                        //R[coord_dCm * P2 + coord_dCn] = rR[m][n];
                }
            }

            /**************************************************************
            * Multiply R_tmp * C = R 
            ************************************************************* */
            //__shared__ float sR[BLK_M][BLK_TMP]; // reuse sA
            float * sC = (float*)sB; // reuse sB

            //float rC[BLK_TMP]; // reuse rB 
            
            const float* offs_dC = C + shift_N * P3 + idyA * P3 + idxA;
            int boundC = (P3 * (P2 - 1) + P3) - (shift_N * P3 + idyA * P3 + idxA) - 1;

            for (kk = 0; kk < P3; kk += BLK_TMP)
            {
                #pragma unroll
                for (n = 0; n < BLK_N; n += DIM_YA) {
                    #pragma unroll
                    for (m = 0; m < BLK_TMP; m += DIM_XA) {
                        sC[(n + idyA)*BLK_TMP + m + idxA] = fetch(offs_dC, P3, m, n, boundC);
                    }
                }

                #pragma unroll
                for (n = 0; n < BLK_M; n += DIM_YA) {
                    #pragma unroll
                    for (m = 0; m < BLK_TMP; m += DIM_XA) {
                        sR(n + idyA, m + idxA) = 0;
                    }
                }
                __syncthreads();

                // multiply
                #pragma unroll
                for (k = 0; k < BLK_N; k++)
                {
                    float rT = rR[0][k];   
                    #pragma unroll
                    for (n = 0; n < BLK_TMP; n++) {
                        //rB[n] = sB[k][n * DIM_X + tx];
                        rC(n) = sC[k*BLK_TMP + n];
                    }
                    
                    #pragma unroll
                    for (n = 0; n < BLK_TMP; n++) {    
                        sR(ty, n) += rT * rC(n);
                        //rR[m][n] += sA[m * DIM_Y + ty][k]*sB[k][n * DIM_X + tx];
                    }
                    
                }
                __syncthreads();

                /*
                * Accumulate to R
                */
                int row_r = bly * BLK_M;
                #pragma unroll
                for (n = 0; n < BLK_M; n += DIM_YA) {
                    int row = row_r + n + idyA;
                    #pragma unroll
                    for (m = 0; m < BLK_TMP; m += DIM_XA) {
                        if( (row < P0) && (m + idxA + kk < P3))
                            R[row*P3 + m + idxA + kk] += sR(n + idyA, m + idxA);
                    }
                }
                __syncthreads();
                offs_dC += BLK_TMP;
                boundC -= BLK_TMP;
            }
        }
    }

}
#endif