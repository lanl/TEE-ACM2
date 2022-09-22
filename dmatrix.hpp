#ifndef __DMATRIX_H__
#define __DMATRIX_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include <cublas_v2.h>
#include <stdio.h>

#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>

#include "cpu_mat_mult.hpp"
#include "cuda_config.cuh"


/* DeviceMatrix class */
template <typename T>
class DeviceMatrix{
    private:
        T *d_M = nullptr;
        int n_row, n_col;
        
    public:
        /* constructors */
        // constructor without direct cudaMemcpy
        DeviceMatrix(int rows, int cols): n_row(rows), n_col(cols) {}

        // constructor with direct cudaMemcpy
        DeviceMatrix(T * h_M,int rows, int cols): n_row(rows), n_col(cols) {
            //std::cout<<"Matrix created with sizes "<<n_row<<" x "<<n_col<<"is allocated on GPU"<<std::endl;
            cuda_err_check(cudaMalloc((void**)&d_M, rows*cols*sizeof(T)));
            cuda_err_check(cudaMemcpy(d_M, h_M, n_row * n_col * sizeof(T), cudaMemcpyHostToDevice));
        }
        // copy constructor
        DeviceMatrix(const DeviceMatrix & dm): d_M(dm.d_M), n_row(dm.n_row), n_col(dm.n_col){} 
        // default constructor
        DeviceMatrix(): d_M(nullptr), n_row(0), n_col(0){}

        void allocate_GPU(T * h_M){
            cuda_err_check(cudaMalloc((void**)&d_M, n_row*n_col*sizeof(T)));
            cuda_err_check(cudaMemcpy(d_M, h_M, n_row * n_col * sizeof(T), cudaMemcpyHostToDevice));
            //std::cout<<"Matrix with sizes "<<n_row<<" x "<<n_col<<"is allocated on GPU"<<std::endl;
        }
        // transpose cuda matrix
        __host__ void transpose(T * h_M){
            T * clone;
            cuda_err_check(cudaMalloc((void**)&clone, n_row*n_col*sizeof(T)));
            cuda_err_check(cudaMemcpy(clone, h_M, n_row * n_col * sizeof(T), cudaMemcpyHostToDevice));

            float const alpha(1.0);
            float const beta(0.0);
            cublasHandle_t handle;
            cublasCreate(&handle);
            cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, n_row, n_col, &alpha, clone, n_col, &beta, clone,  n_row, d_M, n_row );
            cudaDeviceSynchronize();
            cublasDestroy(handle);

            cuda_err_check(cudaFree(clone));
        }
        /* some common functionalities*/
        __host__ inline void copy_from_host(T * h_M){cuda_err_check(cudaMemcpy(d_M, h_M, n_row * n_col * sizeof(T), cudaMemcpyHostToDevice));}
        __host__ inline void copy_to_host(T * h_M){cuda_err_check(cudaMemcpy(h_M, d_M, n_row * n_col  * sizeof(T), cudaMemcpyDeviceToHost));}
        __host__ __device__ inline T *& get_raw_pointer(){return d_M;}
        __host__ __device__ inline T & operator()(const int & i, const int & j){ return d_M[i*n_col + j];}
        __host__ __device__ inline int get_n_rows() const {return n_row;}
        __host__ __device__ inline int get_n_cols() const {return n_col;}
};
#endif

