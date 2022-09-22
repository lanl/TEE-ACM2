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

    #define NB_REGISTERS_PER_THREAD 64
    #define R_FRAGMENT_SIZE 8

    // mat_sizes mult de 2
    // block size 256x1
    // x*z*y 128x16x128

    /**
     * Two matrices multiplication : optimized keep R on shared memory
     * @param A Matrix A.
     * @param B Matrix B.
     * @param R Matrix multiplication results (i.e. R = A*B)
     * @param A_tile_height The height of the shared memory block used for storing a part of A (i.e. x)
     * @param A_tile_width  The width/height of the shared memory block used for storing a part of A/B (i.e. z)
     * @param B_tile_width  The width of the shared memory block used for storing a part of B (i.e. y)
     */
     //_A_transpose
     template<typename T>
     __global__ void  mat_mult_shared_mem_opt_keep_R_v3(T * A, T * B, T * R, const int P0, const int P1, const int P2, \
                                                 const int A_tile_height, const int A_tile_width, const int B_tile_width){
        extern __shared__ int s_mem[];

        int bx = blockIdx.x;
        int by = blockIdx.y;
   
        // Thread index
        int tx = threadIdx.x;

        /* declare shared memory blocks*/
        T * As = (T*)s_mem;
        T * Bs = (T*)&As[A_tile_height*A_tile_width];

        T Rs[NB_REGISTERS_PER_THREAD];
        for(int k=0; k<NB_REGISTERS_PER_THREAD; k++) Rs[k] = 0;
        __syncthreads();
        
        /* global id for an entier block */
        int glb_grid_id  = (by*gridDim.x +bx)*B_tile_width;
        int gi = (glb_grid_id / P2)*A_tile_height;
        int gj = glb_grid_id % P2;
        
        /* row and col of R to compute for each thread */
        int row = (tx*R_FRAGMENT_SIZE / B_tile_width) * R_FRAGMENT_SIZE;
        int col = (tx % 16);
        
        /* row and col at a tile level for each thread */
        int s_bi = tx / B_tile_width;
        int s_bj = tx % B_tile_width;
        int s_ai = tx / A_tile_width;
        int s_aj = tx % A_tile_width;
                                                    
        int bj = s_bj + gj;
        int ai = s_ai + gi;

        /* Loop over all tiles necessary for computing a whole block of R */
        #pragma unroll
        for(int tile_shift_from_beg = 0; tile_shift_from_beg<P1; tile_shift_from_beg+=A_tile_width){

            int bi = s_bi + tile_shift_from_beg;
            T * Bs_slice = &Bs[s_bi*B_tile_width + s_bj];  
            T * B_slice  = &B[bi*P2 + bj];

            int aj = s_aj + tile_shift_from_beg;
            T * As_slice = &As[s_aj*A_tile_height + s_ai]; 
            T * A_slice  = &A[ai*P1 + aj];

            /* Load A and B tiles */
            #pragma unroll
            for(int k=0; k < R_FRAGMENT_SIZE; k++){
                // Load a tile of A
                As_slice[0] = A_slice[0];
                As_slice += 16;
                A_slice += 16*P1;

                // Load a tile of B
                Bs_slice[0] = B_slice[0];
                Bs_slice += 256;
                B_slice  += 2 * P2;
            }
            __syncthreads();

            #pragma unroll
            for(int kk=0; kk < A_tile_width; kk++){ 
                Bs_slice = &Bs[kk*B_tile_width + col];
                #pragma unroll
                for(int jj = 0; jj<R_FRAGMENT_SIZE; jj++){
                    T b_elem = Bs_slice[jj*16];
                    As_slice = &As[kk*A_tile_height + row];
                    #pragma unroll
                    for(int ii = 0; ii<R_FRAGMENT_SIZE; ii++){
                        Rs[ii*R_FRAGMENT_SIZE+jj] += b_elem * As_slice[ii];
                    }
                }
            }
            __syncthreads();
         } 

         /* write results to R matrix */
        row += blockIdx.y*A_tile_height;
        col += blockIdx.x*B_tile_width;
        int r_elem = 0;
        #pragma unroll
        for(int ii = 0; ii<R_FRAGMENT_SIZE; ii++){
            #pragma unroll
            for(int jj = 0; jj<R_FRAGMENT_SIZE; jj++){
                R[(row+ii)*P2 + col + jj*16] = Rs[ii*R_FRAGMENT_SIZE+jj];
                r_elem++;     
            }
        }
    }
}
#endif
