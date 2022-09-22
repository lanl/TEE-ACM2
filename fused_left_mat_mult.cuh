#ifndef __FUSED_LEFT_MAT_MULT_CUH__
#define __FUSED_LEFT_MAT_MULT_CUH__
#include "cuda_config.cuh"


#define FUSED_TH_BLOCK_DIM_X 1
#define FUSED_TH_BLOCK_DIM_Y  208
#define FUSED_x_TILE  208
#define FUSED_y_TILE  208
#define FUSED_z_TILE  16
#define FUSED_LD_SIZE_XA  16
#define FUSED_LD_SIZE_YA  13
#define FUSED_LD_SIZE_XB  13
#define FUSED_LD_SIZE_YB  16
#define FUSED_x_REG_TILE  1
#define FUSED_y_REG_TILE  FUSED_y_TILE
#define FUSED_t_TILE FUSED_z_TILE
  

//namespace FusedMatMult{
namespace SingleMatMult{  
    /* reuse registers for the second multiplication (i.e. T*C) */
    #define rC(i) rB[i]
    /* reuse the shared memory block for storing the result of the matrix R */
    #define sR(i,j) sA[i][j]

    /**
    * TEE-ACM2 kernel : energy efficient fused matrix multiplication ( (A*B)*C = R)
    * Used tile sizes : FUSED_x_TILE, FUSED_z_TILE, FUSED_y_TILE, FUSED_t_TILE
    * @param P0 A height
    * @param P1 A width and B height
    * @param P2 B width and C height
    * @param P3 C width 
    * @param A Matrix A of size P0xP1.
    * @param B Matrix B of size P1xP2.
    * @param C Matrix C of size P2xP3
    * @param R Matrix multiplication results (i.e. R = (A*B)*C)
    */
    template<typename T>
    __global__ void tee_acm_fused(int P0, int P1, int P2, int P3, T* A, T * B, T * C, T * R)
    {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int idt = FUSED_TH_BLOCK_DIM_X * ty + tx;

        int idxA = idt % FUSED_LD_SIZE_XA;
        int idyA = idt / FUSED_LD_SIZE_XA;

        int idxB = idt % FUSED_LD_SIZE_XB;
        int idyB = idt / FUSED_LD_SIZE_XB;

        int blx = blockIdx.x;
        int bly = blockIdx.y;
        
        __shared__ T sA[FUSED_x_TILE][FUSED_z_TILE];
        __shared__ T sB[FUSED_z_TILE][FUSED_y_TILE];
        
        // registers for the innermost loop
        T rR[FUSED_x_REG_TILE][FUSED_y_REG_TILE];
        T rA[FUSED_x_REG_TILE];
        T rB[FUSED_y_REG_TILE];

        const T* offs_dA = A + bly * FUSED_x_TILE * P1 + idyA * P1 + idxA;
        int boundA = (P1 * (P0 - 1) + P1) - (bly * FUSED_x_TILE * P1 + idyA * P1 + idxA) - 1;

        const T* offs_dB = B + blx * FUSED_y_TILE + idyB * P2 + idxB;
        int boundB = (P2 * (P1 - 1) + P2) - (blx * FUSED_y_TILE + idyB * P2 + idxB) - 1;

        int m, n, k, shift;
        
        /**************************************************************
        * Multiply A * B = R_tmp 
        ************************************************************* */
        #pragma unroll
        for (n = 0; n < FUSED_y_REG_TILE; n++) {
            #pragma unroll
            for (m = 0 ; m < FUSED_x_REG_TILE; m++) {
                rR[m][n] = 0;
            }
        }
        /* Main Loop */
        for (shift = 0; shift < P1 - FUSED_z_TILE; shift += FUSED_z_TILE)
        {
            /* Load A from global memory */
            #pragma unroll
            for (n = 0; n < FUSED_x_TILE; n += FUSED_LD_SIZE_YA) {
                #pragma unroll
                for (m = 0; m < FUSED_z_TILE; m += FUSED_LD_SIZE_XA) {
                    sA[n + idyA][m + idxA] = fetch(offs_dA, P1, m, n, boundA);
                }
            }
            /* Load B from global memory */
            #pragma unroll
            for (n = 0; n < FUSED_z_TILE; n += FUSED_LD_SIZE_YB) {
                #pragma unroll
                for (m = 0; m < FUSED_y_TILE; m += FUSED_LD_SIZE_XB) {
                    sB[n + idyB][m + idxB] = fetch(offs_dB, P2, m, n, boundB);
                }
            }
            __syncthreads();

            // multiply : outer product
            #pragma unroll
            for (k = 0; k < FUSED_z_TILE; k++)
            {
                #pragma unroll
                for (m = 0; m < FUSED_x_REG_TILE; m++) {
                    rA[m] = sA[m * FUSED_TH_BLOCK_DIM_Y + ty][k];
                }
                
                #pragma unroll
                for (n = 0; n < FUSED_y_REG_TILE; n++) {
                    rB[n] = sB[k][n];
                }
                
                #pragma unroll
                for (m = 0; m < FUSED_x_REG_TILE; m++) {
                    #pragma unroll
                    for (n = 0; n < FUSED_y_REG_TILE; n++) {    
                        rR[m][n] += rA[m] * rB[n];
                    }
                }
            }
            __syncthreads();

            // next iteration
            offs_dA += FUSED_z_TILE;
            boundA -= FUSED_z_TILE;
            offs_dB += FUSED_z_TILE * P2;
            boundB -= FUSED_z_TILE * P2;
        }

        /* Load the last tile of A from global memory */
        #pragma unroll
        for (n = 0; n < FUSED_x_TILE; n += FUSED_LD_SIZE_YA) {
            #pragma unroll
            for (m = 0; m < FUSED_z_TILE; m += FUSED_LD_SIZE_XA) {
                sA[n + idyA][m + idxA] = fetch(offs_dA, P1, m, n, boundA);
            }
        }
        /* Load the last tile of B from global memory */
        #pragma unroll
        for (n = 0; n < FUSED_z_TILE; n += FUSED_LD_SIZE_YB) {
            #pragma unroll
            for (m = 0; m < FUSED_y_TILE; m += FUSED_LD_SIZE_XB) {
                sB[n + idyB][m + idxB] = fetch(offs_dB, P2, m, n, boundB);
            }
        }
        __syncthreads();
        
        // multiply : last outer product
        shift = P1 - shift;
        #pragma unroll
        for (k = 0; k < shift; k++)
        {
            #pragma unroll
            for (m = 0; m < FUSED_x_REG_TILE; m++) {
                rA[m] = sA[m * FUSED_TH_BLOCK_DIM_Y + ty][k];
            }
            
            #pragma unroll
            for (n = 0; n < FUSED_y_REG_TILE; n++) {
                rB[n] = sB[k][n];
            }
            
            #pragma unroll
            for (m = 0; m < FUSED_x_REG_TILE; m++) {
                #pragma unroll
                for (n = 0; n < FUSED_y_REG_TILE; n++) {    
                    rR[m][n] += rA[m] * rB[n];
                }
            }
        }
       #pragma unroll
        for (m = 0; m < FUSED_x_REG_TILE; m++) {
            int coord_dCm =  bly * FUSED_y_TILE + ty;
            #pragma unroll
            for (n = 0; n < FUSED_y_REG_TILE; n++) {
                int coord_dCn = blx * FUSED_x_TILE + n ;
                if (coord_dCm >= P0 || coord_dCn >= P2) 
                    rR[m][n] = 0;
            }
        }

        /**************************************************************
        * Multiply R_tmp * C = R 
        ************************************************************* */
        //__shared__ T sR[FUSED_x_TILE][FUSED_t_TILE]; // reuse sA
        T * sC = (T*)sB; // reuse sB
        //T rC[FUSED_t_TILE]; // reuse rB 
        
        const T* offs_dC = C + blx * FUSED_y_TILE * P3 + idyA * P3 + idxA;
        int boundC = (P3 * (P2 - 1) + P3) - (blx * FUSED_y_TILE * P3 + idyA * P3 + idxA) - 1;

        for (shift = 0; shift < P3; shift += FUSED_t_TILE)
        {
            /* Load C from global memory */
            #pragma unroll
            for (n = 0; n < FUSED_y_TILE; n += FUSED_LD_SIZE_YA) {
                #pragma unroll
                for (m = 0; m < FUSED_t_TILE; m += FUSED_LD_SIZE_XA) {
                    sC[(n + idyA)*FUSED_t_TILE + m + idxA] = fetch(offs_dC, P3, m, n, boundC);
                }
            }
            /* Set sR to 0 */
            #pragma unroll
            for (n = 0; n < FUSED_x_TILE; n += FUSED_LD_SIZE_YA) {
                #pragma unroll
                for (m = 0; m < FUSED_t_TILE; m += FUSED_LD_SIZE_XA) {
                    sR(n + idyA, m + idxA) = 0;
                }
            }
            __syncthreads();


            // multiply : outer product
            #pragma unroll
            for (k = 0; k < FUSED_y_TILE; k++)
            {
                T rT = rR[0][k];   
                #pragma unroll
                for (n = 0; n < FUSED_t_TILE; n++) {
                    rC(n) = sC[k*FUSED_t_TILE + n];
                }
                #pragma unroll
                for (n = 0; n < FUSED_t_TILE; n++) {    
                    sR(ty, n) += rT * rC(n);
                }
            }
            __syncthreads();

            // Accumulate to R (global read and write)
            int row_r = bly * FUSED_x_TILE;
            #pragma unroll
            for (n = 0; n < FUSED_x_TILE; n += FUSED_LD_SIZE_YA) {
                int row = row_r + n + idyA;
                #pragma unroll
                for (m = 0; m < FUSED_t_TILE; m += FUSED_LD_SIZE_XA) {
                    if( (row < P0) && (m + idxA + shift < P3))
                        atomicAdd(&R[row*P3 + m + idxA + shift], sR(n + idyA, m + idxA));
                }
            }
            __syncthreads();
            // next iteration
            offs_dC += FUSED_t_TILE;
            boundC -= FUSED_t_TILE;
        }
    }
}
#endif