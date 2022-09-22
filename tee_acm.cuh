#ifndef __TEE_ACM_CUH__
#define __TEE_ACM_CUH__
#include "cuda_config.cuh"
    /*#define TH_BLOCK_DIM_X 16
    #define TH_BLOCK_DIM_Y  16
    #define x_TILE  208
    #define y_TILE  208
    #define z_TILE  16
    #define LD_SIZE_XA  16
    #define LD_SIZE_YA  16
    #define LD_SIZE_XB  16
    #define LD_SIZE_YB  16
    #define x_REG_TILE  13
    #define y_REG_TILE  13*/

    #define TH_BLOCK_DIM_X DIM_X
    #define TH_BLOCK_DIM_Y  DIM_Y
    #define x_TILE  BLK_M
    #define y_TILE  BLK_N
    #define z_TILE  BLK_K
    #define LD_SIZE_XA  DIM_XA
    #define LD_SIZE_YA  DIM_YA
    #define LD_SIZE_XB  DIM_XB
    #define LD_SIZE_YB  DIM_YB
    #define x_REG_TILE  THR_M
    #define y_REG_TILE  THR_N

    /**
    * Wrapper function for the CUDA kernel function.
    * @param Mat matrix in global memory
    * @param width matrix width
    * @param col column index
    * @param row row index
    * @param bound matrix bound
    */
    #define global_load(Mat, width, col, row, bound) Mat[min(row*width + col, bound)]

    /**
    * Wrapper function for the CUDA kernel function.
    * @param P0 A height
    * @param P1 A width and B height
    * @param P2 B width 
    * @param A Matrix A of size P0xP1.
    * @param B Matrix B of size P1xP2.
    * @param R Matrix multiplication results (i.e. R = A*B)
    */
    template<typename T>
    __global__ void tee_acm(int P0, int P1, int P2, const T* A, const T* B, T * R)
    {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int idt = TH_BLOCK_DIM_X * ty + tx;

        int idxA = idt % LD_SIZE_XA;
        int idyA = idt / LD_SIZE_XA;

        int idxB = idt % LD_SIZE_XB;
        int idyB = idt / LD_SIZE_XB;

        int blx = blockIdx.x;
        int bly = blockIdx.y;
        
        // shared memory padding
        __shared__ T sA[x_TILE][z_TILE+1];
        __shared__ T sB[z_TILE][y_TILE+1];
        T elem_A;

        // register tile (x*y)
        T rR[x_REG_TILE][y_REG_TILE];

        // compute starts offsets for loading A and B
        const T* offs_dA = A + bly * y_TILE * P1 + idyA * P1 + idxA;
        int boundA = (P1 * (P0 - 1) + P1) - (bly * y_TILE * P1 + idyA * P1 + idxA) - 1;

        const T* offs_dB = B + blx * x_TILE + idyB * P2 + idxB;
        int boundB = (P2 * (P1 - 1) + P2) - (blx * x_TILE + idyB * P2 + idxB) - 1;

        int i, j, k, shift;
        
        // init R tile
        #pragma unroll
        for (i = 0; i < x_REG_TILE; i++) {
            #pragma unroll
            for (j = 0 ; j < y_REG_TILE; j++) {
                rR[i][j] = 0;
            }
        }

        /* Main Loop */
        for (shift = 0; shift < P1 - z_TILE; shift += z_TILE)
        {
            /* Load A from global memory */
            #pragma unroll
            for (i = 0; i < x_TILE; i += LD_SIZE_YA) {
                #pragma unroll
                for (j = 0; j < z_TILE; j += LD_SIZE_XA) {
                    sA[i + idyA][j + idxA] = global_load(offs_dA, P1, j, i, boundA);
                }
            }
            /* Load B from global memory */
            #pragma unroll
            for (i = 0; i < z_TILE; i += LD_SIZE_YB) {
                #pragma unroll
                for (j = 0; j < y_TILE; j += LD_SIZE_XB) {
                    sB[i + idyB][j + idxB] = global_load(offs_dB, P2, j, i, boundB);
                }
            }
            __syncthreads();

            /* Compute : outer product*/
            #pragma unroll
            for (k = 0; k < z_TILE; k++)
            {
                #pragma unroll
                for (i = 0; i < x_REG_TILE; i++) {
                    elem_A = sA[i * TH_BLOCK_DIM_Y + ty][k];
                    #pragma unroll
                    for (j = 0; j < y_REG_TILE; j++) {    
                        rR[i][j] += elem_A * sB[k][j * TH_BLOCK_DIM_X + tx];
                    }
                }
            }
            __syncthreads();

            // next tile size
            offs_dA += z_TILE;
            boundA -= z_TILE;
            offs_dB += z_TILE * P2;
            boundB -= z_TILE * P2;
        }
        
        /* Load the last tile of A from global memory */
        #pragma unroll
        for (i = 0; i < x_TILE; i += LD_SIZE_YA) {
            #pragma unroll
            for (j = 0; j < z_TILE; j += LD_SIZE_XA) {
                sA[i + idyA][j + idxA] = global_load(offs_dA, P1, j, i, boundA);
            }
        }

        /* Load the last tile of B from global memory */
        #pragma unroll
        for (i = 0; i < z_TILE; i += LD_SIZE_YB) {
            #pragma unroll
            for (j = 0; j < y_TILE; j += LD_SIZE_XB) {
                sB[i + idyB][j + idxB] = global_load(offs_dB, P2, j, i, boundB);
            }
        }
        __syncthreads();

        /* Compute the last tile */
        shift = P1 - shift;
        #pragma unroll
        for (k = 0; k < shift; k++)
        {
            #pragma unroll
            for (i = 0; i < x_REG_TILE; i++) {
                elem_A = sA[i * TH_BLOCK_DIM_Y + ty][k];
                #pragma unroll
                for (j = 0; j < y_REG_TILE; j++) {    
                    rR[i][j] += elem_A * sB[k][j * TH_BLOCK_DIM_X + tx];
                }
            }
        }

        /* Write restult to the global memory (exactly P0*P2 writes) */
        #pragma unroll
        for (i = 0; i < x_REG_TILE; i++) {
            int coord_dCm =  bly * y_TILE + i * TH_BLOCK_DIM_Y + ty;
            #pragma unroll
            for (j = 0; j < y_REG_TILE; j++) {
                int coord_dCn = blx * x_TILE + j * TH_BLOCK_DIM_X + tx;
                if (coord_dCm < P0 && coord_dCn < P2) {
                    R[coord_dCm * P2 + coord_dCn] = rR[i][j];
                }
            }
        }
    }


#endif
