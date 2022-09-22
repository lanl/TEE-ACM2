#ifndef __CUDA_CONFIG__CUH__
#define __CUDA_CONFIG__CUH__
#include <cuda.h>
#include <cuda_runtime.h>

constexpr bool DISPLAY = true;
constexpr bool TIME_MES = true;
constexpr bool USE_NVML = true;

#define DIM_X 16
#define DIM_Y  16
#define BLK_M  208 //176,192,208,224
#define BLK_N  208
#define BLK_K  16
#define DIM_XA  16
#define DIM_YA  16
#define DIM_XB  16
#define DIM_YB  16
#define THR_M  13
#define THR_N  13

#define BLK_TMP BLK_K
#define fetch(arr, col, m, n, bound) arr[min(n*col + m, bound)]


/* CUDA errors test */
#define cuda_err_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"\033[1;31mGPU Error:\033[0m %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* enum for selecting the kernel to launch*/
enum SingleMatMultKernel { TEE_ACM= 0,CUBLAS_SGEMM_TREE = 1,CUBLAS_SGEMM = 2, FUSED_TEE_ACM=3, ACM_RIGHT_F=4, CUBLAS_LEFT=5, CUBLAS_RIGHT=6, SGEMM_CUPY = 7};
static const char * StringsSingleMatMultKernel[] = {"tee_acm","cublas_with_tree","cublas_no_tree","acm_left_fuse","acm_right_fuse","cublas_left_fuse","cublas_right_fuse","cupy_matmult"};

const char * get_kernel_name(int kernel_enum){return StringsSingleMatMultKernel[kernel_enum];}
#endif
