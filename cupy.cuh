/*
    Original works by:
    --------------------------------------------------------
    MAGMA
    Copyright (c) 2017 The University of Tennessee. All rights reserved.
    Licensed under modified BSD license
*/

#ifndef __CUPY_CUH__
#define __CUPY_CUH__
#include "cuda_config.cuh"
namespace SingleMatMult{

    __global__ void cupy_mm_original( int P0, int P1, int P2, const float* A, const float* B,float * C)
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

        const float* offs_dA = A + blx * BLK_M + idyA * P2 + idxA;
        int boundA = (P2 * (P1 - 1) + P2) - (blx * BLK_M + idyA * P2 + idxA) - 1;
        const float* offs_dB = B + bly * BLK_N * P1 + idyB * P1 + idxB;
        int boundB = (P1 * (P0 - 1) + P1) - (bly * BLK_N * P1 + idyB * P1 + idxB) - 1;

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
                sA[n + idyA][m + idxA] = fetch(offs_dA, P2, m, n, boundA);
            }
        }
        // blockwise transpose to transpose load
        #pragma unroll
        for (n = 0; n < BLK_N; n += DIM_YB) {
            #pragma unroll
            for (m = 0; m < BLK_K; m += DIM_XB) {
                sB[n + idyB][m + idxB] = fetch(offs_dB, P1, m, n, boundB);
            }
        }
        __syncthreads();

        for (kk = 0; kk < P1 - BLK_K; kk += BLK_K)
        {
            offs_dA += BLK_K * P2;
            boundA -= BLK_K * P2;
            offs_dB += BLK_K;
            boundB -= BLK_K;
            
            #pragma unroll
            for (n = 0; n < BLK_K / DIM_YA; n++) {
                #pragma unroll
                for (m = 0; m < BLK_M / DIM_XA; m++) {
                    ra[n][m] = fetch(offs_dA, P2, m * DIM_XA, n * DIM_YA, boundA);
                }
            }

            #pragma unroll
            for (n = 0; n < BLK_N / DIM_YB; n++) {
                #pragma unroll
                for (m = 0; m < BLK_K / DIM_XB; m++) {
                    rb[n][m] = fetch(offs_dB, P1, m * DIM_XB, n * DIM_YB, boundB);
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

        kk = P1 - kk;
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
                if (coord_dCm < P2 && coord_dCn < P0) {
                    C[coord_dCn * P2 + coord_dCm] = rC[n][m];
                }
            }
        }
    }
}

#endif