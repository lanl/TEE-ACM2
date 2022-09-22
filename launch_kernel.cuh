#ifndef __LAUNCH_KERNEL_CUH__
#define __LAUNCH_KERNEL_CUH__
#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include <stdio.h>
#include "dmatrix.hpp"
#include "nvmlPower.hpp"
#include "tee_acm.cuh"
#include "cupy.cuh"
#include "fused_left_mat_mult.cuh"

#define iter 10

namespace SingleMatMult{

    template<typename T>
    __device__ void print_mat(T * mat ,int m, int n){
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++)
                printf("%f ", mat[i*n +j]);
            printf("\n");
        }
        printf("\n");
    }

    #define N_tab 16
    

    /**
     * Wrapper function for the CUDA kernel function.
     * @param A Matrix A.
     * @param B Matrix B.
     * @param R Matrix multiplication results (i.e. R = A*B)
     */
    template<typename T>
    void launch_kernel(DeviceMatrix <T> & A, DeviceMatrix <T>  & B, DeviceMatrix <T> & R, SingleMatMultKernel kernel_to_launch) {
        /* set timers */
        cudaEvent_t start, stop;
        cublasHandle_t handle;
        cublasCreate(&handle);
        float time_ms;
        float time_mean;
         /* Variables for energy measurement */
        nvmlReturn_t nvmlResult;
        nvmlDevice_t nvmlDeviceID;
        unsigned long long energy_start;
        unsigned long long energy_end;
        unsigned long long energy_mean = 0;

        cuda_err_check(cudaEventCreate(&start));
        cuda_err_check(cudaEventCreate(&stop));

        if(USE_NVML){
            nvmlResult = nvmlInit();
            if (NVML_SUCCESS != nvmlResult)
            {
                printf("NVML Init fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
            }
            nvmlResult = nvmlDeviceGetHandleByIndex(0, &nvmlDeviceID); // change cuda device if need to use other GPUs
            if (NVML_SUCCESS != nvmlResult)
            {
                printf("Failed to get handle for device %d: %s\n", 0, nvmlErrorString(nvmlResult));
                exit(1);
            }
        }

        if(kernel_to_launch == TEE_ACM){
            int P0 = A.get_n_rows();
            int P1 = A.get_n_cols();
            int P2 = B.get_n_cols();

            dim3 blockSize(TH_BLOCK_DIM_X,TH_BLOCK_DIM_Y,1);
            dim3 gridSize( ceil( P2/ (double)(x_TILE)), ceil(P0/ (double)(y_TILE)), 1);
            for(int i=0; i<iter; i++){
                cuda_err_check(cudaEventRecord(start));
                if(USE_NVML)
                    nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_start);
                tee_acm<<<gridSize, blockSize>>>(P0, P1, P2, A.get_raw_pointer(), B.get_raw_pointer(),  R.get_raw_pointer());
                cudaDeviceSynchronize();

                if(USE_NVML)
                    nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_end);
                cuda_err_check(cudaEventRecord(stop));
                cuda_err_check(cudaEventSynchronize(stop));
                if(USE_NVML)
                    energy_mean += (energy_end - energy_start);
                cuda_err_check(cudaEventElapsedTime(&time_ms, start, stop));
                time_mean += time_ms;
            }
            
        }
        else if(kernel_to_launch == SGEMM_CUPY){
            int P0 = A.get_n_rows();
            int P1 = A.get_n_cols();
            int P2 = B.get_n_cols();

            dim3 blockSize(DIM_X,DIM_Y,1);
            dim3 gridSize( ceil( P2/ (double)(BLK_M)), ceil(P0/ (double)(BLK_N)), 1);
            for(int i=0; i<iter; i++){
                cuda_err_check(cudaEventRecord(start));
                if(USE_NVML)
                    nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_start);

                cupy_mm_original<<<gridSize, blockSize>>>(P0, P1, P2, B.get_raw_pointer(), A.get_raw_pointer(),  R.get_raw_pointer());
                cudaDeviceSynchronize();
                if(USE_NVML)
                    nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_end);
                cuda_err_check(cudaEventRecord(stop));
                cuda_err_check(cudaEventSynchronize(stop));
                if(USE_NVML)
                    energy_mean += (energy_end - energy_start);
                cuda_err_check(cudaEventElapsedTime(&time_ms, start, stop));
                time_mean += time_ms;
            }
            
        }
        else if( kernel_to_launch == CUBLAS_SGEMM_TREE){

            int P0 = A.get_n_rows();
            int P1 = A.get_n_cols();
            int P2 = B.get_n_cols();
            T const alpha(1.0);
            T const beta(0.0);
            for(int i=0; i<iter; i++){
                
                cuda_err_check(cudaEventRecord(start));
                if(USE_NVML)
                    nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_start);
                cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, P2, P0, P1, &alpha, B.get_raw_pointer(), P2, A.get_raw_pointer(), P1, &beta, R.get_raw_pointer(), P2);
                cudaDeviceSynchronize();
                
                if(USE_NVML)
                    nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_end);
                cuda_err_check(cudaEventRecord(stop));
                cuda_err_check(cudaEventSynchronize(stop));
                if(USE_NVML)
                    energy_mean += (energy_end - energy_start);
                cuda_err_check(cudaEventElapsedTime(&time_ms, start, stop));
                time_mean += time_ms;
            }

        }
        else if(kernel_to_launch == CUBLAS_SGEMM ){

            int P0 = A.get_n_rows();
            int P1 = A.get_n_cols();
            int P2 = B.get_n_cols();
            T const alpha(1.0);
            T const beta(0.0);
            for(int i=0; i<iter; i++){
                cuda_err_check(cudaEventRecord(start));
                if(USE_NVML)
                    nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_start);
                cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, P2, P0, P1, &alpha, B.get_raw_pointer(), P2, A.get_raw_pointer(), P1, &beta, R.get_raw_pointer(), P2);
                cudaDeviceSynchronize();
                
                if(USE_NVML)
                    nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_end);
                cuda_err_check(cudaEventRecord(stop));
                cuda_err_check(cudaEventSynchronize(stop));
                if(USE_NVML)
                    energy_mean += (energy_end - energy_start);
                cuda_err_check(cudaEventElapsedTime(&time_ms, start, stop));
                time_mean += time_ms;
            }
        }
        else{fprintf(stderr,"Unknown kernel \n");exit(1);}
        
        cudaDeviceSynchronize();
        //check kernel error
        cuda_err_check(cudaGetLastError()); 
        
        /* destroy event */
        cuda_err_check(cudaEventDestroy(start));
        cuda_err_check(cudaEventDestroy(stop));
        cublasDestroy(handle);
        
        /* calculate the theoretical number of data transfers*/
        int P0 = A.get_n_rows();
        int P1 = A.get_n_cols();
        int P2 = B.get_n_cols();
        int x = BLK_M;
        int y = BLK_N;
        double mat_sizes = P0 * (double)(P1/(double)x) * (double)((P2/(double)y));
        double block_sizes = (x+y) ;
        size_t e_ld = round((mat_sizes*block_sizes*sizeof(T))/32.0);
        size_t e_st = round((P0*P2*sizeof(T))/32.0);

         /* Finalize nvml */
        if(USE_NVML){
            nvmlResult = nvmlShutdown();
            if (NVML_SUCCESS != nvmlResult)
            {
                printf("Failed to shut down NVML: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
            } 
        }

        time_mean = (time_mean/iter)/1000; // to seconds
        if(USE_NVML)
            energy_mean /= iter; //mJ
        if constexpr (DISPLAY){
            std::cout<<"# Kernel name : "<<get_kernel_name(kernel_to_launch)<<std::endl;
            std::cout << std::defaultfloat;
            std::cout<<"# Expected global loads  : "<<e_ld<<std::endl;
            std::cout<<"# Expected global stores : "<<e_st<<std::endl;
            std::cout <<std::fixed << std::setprecision(4);
            std::cout<<"# Execution time [ms]: "<<time_ms<<std::endl;
            std::cout << std::defaultfloat;
            if(USE_NVML)
                std::cout<<"# NVML Energy : "<<energy_mean<<" mJ"<<std::endl;
        }
    }
    /**
     * Wrapper function for the CUDA kernel function.
     * @param A Matrix A.
     * @param B Matrix B.
     * @param C Matrix C.
     * @param R Matrix multiplication results (i.e. R = A*B)
     */
    template<typename T>
    void launch_kernel(DeviceMatrix <T> & A, DeviceMatrix <T>  & B, DeviceMatrix <T>  & C, DeviceMatrix <T> & R, SingleMatMultKernel kernel_to_launch) {
        /* set timers */
        
        cudaEvent_t start, stop;
        cublasHandle_t handle;
        cublasCreate(&handle);
        float time_ms;
        float time_mean;
         /* Variables for energy measurement */
        nvmlReturn_t nvmlResult;
        nvmlDevice_t nvmlDeviceID;
        unsigned long long energy_start;
        unsigned long long energy_end;
        unsigned long long energy_mean = 0;

        cuda_err_check(cudaEventCreate(&start));
        cuda_err_check(cudaEventCreate(&stop));

        cuda_err_check(cudaEventRecord(start));
        nvmlResult = nvmlInit();
        if (NVML_SUCCESS != nvmlResult)
        {
            printf("NVML Init fail: %s\n", nvmlErrorString(nvmlResult));
            exit(0);
        }
        nvmlResult = nvmlDeviceGetHandleByIndex(0, &nvmlDeviceID); // change cuda device if need to use other GPUs
		if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get handle for device %d: %s\n", 0, nvmlErrorString(nvmlResult));
			exit(1);
		} 
        if(kernel_to_launch == FUSED_TEE_ACM){
            int P0 = A.get_n_rows();
            int P1 = A.get_n_cols();
            int P2 = B.get_n_cols();
            int P3 = C.get_n_cols();

            dim3 blockSize(FUSED_TH_BLOCK_DIM_X, FUSED_TH_BLOCK_DIM_Y,1);
            dim3 gridSize( ceil( P2/ (double)(FUSED_y_TILE)), ceil(P0/ (double)(FUSED_x_TILE)), 1);
            
            //nvmlAPIRun(e_out_tee_acm.c_str());
            for(int i=0; i<iter; i++){
                cudaMemset ( R.get_raw_pointer(), 0, P0*P3*sizeof(T) );
                cuda_err_check(cudaEventRecord(start));
                nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_start);
                //tee_acm_fused<<<gridSize, blockSize>>>(P0, P1, P2, P3, A.get_raw_pointer(), B.get_raw_pointer(), Tmp.get_raw_pointer(),C.get_raw_pointer(), R.get_raw_pointer());
                tee_acm_fused<<<gridSize, blockSize>>>(P0, P1, P2, P3, A.get_raw_pointer(), B.get_raw_pointer(), C.get_raw_pointer(), R.get_raw_pointer());
                cudaDeviceSynchronize();
                //nvmlAPIEnd();
                nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_end);
                cuda_err_check(cudaEventRecord(stop));
                cuda_err_check(cudaEventSynchronize(stop));
                // energy
                energy_mean += (energy_end - energy_start);
                // time
                cuda_err_check(cudaEventElapsedTime(&time_ms, start, stop));
                time_mean += time_ms;
            }
            nvmlAPIEnd();
        }
        else if(kernel_to_launch == CUBLAS_LEFT){

            int P0 = A.get_n_rows();
            int P1 = A.get_n_cols();
            int P2 = B.get_n_cols();
            int P3 = C.get_n_cols();
            T const alpha(1.0);
            T const beta(0.0);
            
            DeviceMatrix <T> Temp(P0,P2);
            cuda_err_check(cudaMalloc((void**)&(Temp.get_raw_pointer()), P0*P2*sizeof(T)));
            
            //printf("CUBLASS Left with S= (%d,%d,%d,%d) \n",P0,P1,P2,P3 );

            for(int i=0; i<2; i++){
                cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, P2, P0, P1, &alpha, B.get_raw_pointer(), P2, A.get_raw_pointer(), P1, &beta, Temp.get_raw_pointer(), P2);
                cudaDeviceSynchronize();
            }
            cuda_err_check(cudaEventRecord(start));
            nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_start);
            
            cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, P2, P0, P1, &alpha, B.get_raw_pointer(), P2, A.get_raw_pointer(), P1, &beta, Temp.get_raw_pointer(), P2);  
            cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, P3, P0, P2, &alpha, C.get_raw_pointer(), P3, Temp.get_raw_pointer(), P2, &beta, R.get_raw_pointer(), P3);           
            cudaDeviceSynchronize();
            nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_end);
            cuda_err_check(cudaEventRecord(stop));
                cuda_err_check(cudaEventSynchronize(stop));
            // energy
            energy_mean += (energy_end - energy_start);
            cuda_err_check(cudaEventElapsedTime(&time_ms, start, stop));
            time_mean += time_ms;
        }
        else if( kernel_to_launch == CUBLAS_RIGHT){

            int P0 = A.get_n_rows();
            int P1 = A.get_n_cols();
            int P2 = B.get_n_cols();
            int P3 = C.get_n_cols();
            T const alpha(1.0);
            T const beta(0.0);

            DeviceMatrix<T> Temp (P1,P3) ;
            cuda_err_check(cudaMalloc((void**)&(Temp.get_raw_pointer()), P1*P3*sizeof(T)));
            //printf("CUBLASS Right with S= (%d,%d,%d,%d) \n",P0,P1,P2,P3 );
            
            for(int i=0; i<2; i++){
                cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, P3, P1, P2, &alpha, C.get_raw_pointer(), P3, B.get_raw_pointer(), P2, &beta, Temp.get_raw_pointer(), P3);
                cudaDeviceSynchronize();
            }
            cuda_err_check(cudaEventRecord(start));
            nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_start);
            cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, P3, P1, P2, &alpha, C.get_raw_pointer(), P3, B.get_raw_pointer(), P2, &beta, Temp.get_raw_pointer(), P3);  
            cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, P3, P0, P1, &alpha, Temp.get_raw_pointer(), P3, A.get_raw_pointer(), P1, &beta, R.get_raw_pointer(), P3);           
            cudaDeviceSynchronize();
            nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_end);
            cuda_err_check(cudaEventRecord(stop));
                cuda_err_check(cudaEventSynchronize(stop));
            // energy
            energy_mean += (energy_end - energy_start);
            cuda_err_check(cudaEventElapsedTime(&time_ms, start, stop));
            time_mean += time_ms;

        }
        else{fprintf(stderr,"Unknown kernel \n");exit(1);}
        
        cudaDeviceSynchronize();
        //check kernel error
        cuda_err_check(cudaGetLastError()); 
        cuda_err_check(cudaEventRecord(stop));
        cuda_err_check(cudaEventSynchronize(stop));
        cuda_err_check(cudaEventElapsedTime(&time_ms, start, stop));
        
        /* destroy event */
        cuda_err_check(cudaEventDestroy(start));
        cuda_err_check(cudaEventDestroy(stop));
        cublasDestroy(handle);
        
        /* calculate the theoretical number of data transfers*/
        int P0 = A.get_n_rows();
        int P1 = A.get_n_cols();
        int P2 = B.get_n_cols();
        int x = BLK_M;
        int y = BLK_N;
        double mat_sizes = P0 * (double)(P1/(double)x) * (double)((P2/(double)y));
        double block_sizes = (x+y) ;
        size_t e_ld = round((mat_sizes*block_sizes*sizeof(T))/32.0);
        size_t e_st = round((P0*P2*sizeof(T))/32.0);

         /* Finalize nvml */
        nvmlResult = nvmlShutdown();
        if (NVML_SUCCESS != nvmlResult)
        {
            printf("Failed to shut down NVML: %s\n", nvmlErrorString(nvmlResult));
            exit(0);
        } 
        if constexpr (DISPLAY){
            std::cout<<"# Kernel name : "<<get_kernel_name(kernel_to_launch)<<std::endl;
            std::cout << std::defaultfloat;
            std::cout<<"# Expected global loads  : "<<e_ld<<std::endl;
            std::cout<<"# Expected global stores : "<<e_st<<std::endl;
            std::cout <<std::fixed << std::setprecision(4);
            std::cout<<"# Execution time [ms]: "<<time_ms<<std::endl;
            std::cout << std::defaultfloat;
            std::cout<<"# NVML Energy : "<<energy_mean<<" mJ"<<std::endl;
        }
    }


}
#endif




