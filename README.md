# TEE-ACM2
Transformations for Energy Efficient Accelerated Chain Matrix Multiplication

## Compile :
* $> module load cuda cmake 
* $> mkdir BUILD && cd BUILD && cmake ..
* $> make

## Execute:
* ./mat_test -n -k -run
* n : number of matrices to compute
* k : kernel to launch
*       0 : TEE-ACM2
*       1 : cuBLAS using the OP_Count Tree
*       2 : cuBLAS in order
* run : parameter to generate the sizes

## CUDA kernels:
### cupy.cuh
* Fast implementation (5% close to cuBLAS) for single matrix multiplication
* Manually overlap communication and computation (copy data from global memory to registers for the next iteration while computing the current one)
### tee_acm.cuh
* Energy efficient implementation for single matrix multiplication (up to 10% energy savings)
* No overlap
### fused_left_mat_mult.cuh
* Fused kernel implementation (80% slower than running two cuBLAS kernels)
* Each thread in a CUDA thread block compute an entire row of the final R tile
* Atomic adds to the global memory
### no_atomic_adds_left_mat_mult.cuh
* Fused kernel implementation (500% slower than running two cuBLAS kernels)
* Each thread in a CUDA thread block compute an entire row of the final R matrix 
* No atomic operations
* Occupancy problems

## Copyright

© 2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

This program is open source under BSD-3 License. Details can be found in [LICENSE](https://github.com/lanl/TEE-ACM2/blob/main/LICENSE).
