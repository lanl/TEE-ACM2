import subprocess
import io
import re
import sys
import os
from common_functions import *
from common_classes import *

############################################################################
# Local functions
#############################################################################
#
# run the experimental protocol
# @prefix : path used for storing the output csv file
# @csv_file : the csvfile name
#
def run_all_tests_multiple_exe(kernel_names, P0_tab, P1_tab, P2_tab, x_tab, y_tab, z_tab, xBlockDim_tab, yBlockDim_tab, csv_file, path_to_dir_with_exe, prefix=".",node_to_use="", with_slurm=True):
    csv_writer = CSVOutput(csv_file,prefix)
    header_already_set = False 

    #path_to_dir_with_exe="/vast/home/morarum/cuda_training/CUDA_Chain_Matrix_Mult/cupy_exes/square_sm"

    nb_of_exe = 0
    for mm_exe in os.listdir(path_to_dir_with_exe):
        if "mat_test" in mm_exe:
            nb_of_exe+=1
    tot = nb_of_exe*len(P0_tab) * len(P1_tab) * len(P2_tab)
    iter = 1

    init_power_prefix = prefix+"/power"
    for mm_exe in os.listdir(path_to_dir_with_exe):
        if "mat_test" in mm_exe:
            params = mm_exe.split("_")
            DIM_X=int(params[2])
            DIM_Y=int(params[3])
            BLK_M=int(params[4])
            BLK_N=int(params[5])
            BLK_K=int(params[6])
            DIM_XA=int(params[7])
            DIM_YA=int(params[8])
            DIM_XB=int(params[9])
            DIM_YB=int(params[10])
            THR_M=int(params[11])
            THR_N=int(params[12])

            for P0 in P0_tab:
                for P1 in P1_tab:
                    for P2 in P2_tab:
                        for x in x_tab:
                            for y in y_tab:
                                for z in z_tab:
                                    for xBlockDim in xBlockDim_tab:
                                        for yBlockDim in yBlockDim_tab:

                                            #P0 = (P0_tmp//BLK_M)*BLK_M 
                                            #P1 = (P1_tmp//BLK_K)*BLK_K
                                            #P2 = (P2_tmp//BLK_N)*BLK_N
                                            # run one sample
                                            if DISPLAY_PROGRESS:
                                                info_print("# Start sample {}/{}: P0,P1,P2 = {},{},{} - x,y,z = {},{},{} - bY,bX = {},{}  (exe : {} )\n"\
                                                        .format(iter,tot,P0,P1,P2,BLK_M,BLK_N,BLK_K,DIM_Y,DIM_X,mm_exe))

                                            
                                            iter += 1
                                            power_prefix = init_power_prefix+"/{}_{}_{}".format(P0,P1,P2)
                                            cublas_power = power_prefix+"/cublas"
                                            acm_power = power_prefix+"/tee_acm"
                                            exe_name=path_to_dir_with_exe+"/"+mm_exe
                                            kernels_info = run_kernels_and_get_info(kernel_names,P0,P1,P2,BLK_M,BLK_N,BLK_K,DIM_Y,DIM_X,cublas_power, acm_power,node_to_use=node_to_use, partition="clx-volta",with_slurm=with_slurm,path_to_executable=exe_name)
                                            if not header_already_set:
                                                csv_writer.write_csv_header_with_registers(kernels_info)
                                                header_already_set = True
                                            # write results into the csv file
                                            csv_writer.write_csv_line_with_registers(kernels_info,P0,P1,P2,BLK_M,BLK_N,BLK_K,DIM_Y,DIM_X,THR_M,THR_N)

#
# run the experimental protocol
# @prefix : path used for storing the output csv file
# @csv_file : the csvfile name
#
def run_all_tests(kernel_names, P0_tab, P1_tab, P2_tab, x_tab, y_tab, z_tab, xBlockDim_tab, yBlockDim_tab, csv_file, exe_name, prefix=".",with_slurm=True):
    csv_writer = CSVOutput(csv_file,prefix)
    header_already_set = False 

    init_power_prefix = prefix+"/power"
    tot = len(P0_tab) * len(P1_tab) * len(P2_tab) * len(x_tab) * len(y_tab) * len(z_tab) * len(xBlockDim_tab) * len(yBlockDim_tab)
    iter = 1
    for P0 in P0_tab:
        for P1 in P1_tab:
            for P2 in P2_tab:
                for x in x_tab:
                    for y in y_tab:
                        for z in z_tab:
                            for xBlockDim in xBlockDim_tab:
                                for yBlockDim in yBlockDim_tab:
                                    # run one sample
                                    if DISPLAY_PROGRESS:
                                        info_print("# Start sample {}/{}: P0,P1,P2 = {},{},{} - x,y,z = {},{},{} - bY,bX = {},{} \n"\
                                                   .format(iter,tot,P0,P1,P2,x,y,z,yBlockDim,xBlockDim))
                                    iter += 1
                                    power_prefix = init_power_prefix+"/{}_{}_{}".format(P0,P1,P2)
                                    cublas_power = power_prefix+"/cublas"
                                    acm_power = power_prefix+"/tee_acm"
                                    kernels_info = run_kernels_and_get_info(kernel_names,P0,P1,P2,x,y,z,yBlockDim,xBlockDim,cublas_power, acm_power, with_slurm=with_slurm,path_to_executable=exe_name)
                                    if not header_already_set:
                                        csv_writer.write_csv_header(kernels_info)
                                        header_already_set = True
                                    # write results into the csv file
                                    csv_writer.write_csv_line(kernels_info,P0,P1,P2,x,y,z,yBlockDim,xBlockDim)


#
# run the experimental protocol
# @prefix : path used for storing the output csv file
# @csv_file : the csvfile name
#
def fused_run_all_tests(kernel_names, P0_tab, P1_tab, P2_tab, P3_tab, x_tab, y_tab, z_tab, xBlockDim_tab, yBlockDim_tab, csv_file, exe_name, prefix=".",with_slurm=True):
    csv_writer = CSVOutput(csv_file,prefix)
    header_already_set = False 

    init_power_prefix = prefix+"/power"
    tot = len(P0_tab) * len(P1_tab) * len(P2_tab)* len(P3_tab) * len(x_tab) * len(y_tab) * len(z_tab) * len(xBlockDim_tab) * len(yBlockDim_tab)
    iter = 1
    for P0 in P0_tab:
        for P1 in P1_tab:
            for P2 in P2_tab:
                for P3 in P3_tab:
                    for x in x_tab:
                        for y in y_tab:
                            for z in z_tab:
                                for xBlockDim in xBlockDim_tab:
                                    for yBlockDim in yBlockDim_tab:
                                        # run one sample
                                        if DISPLAY_PROGRESS:
                                            info_print("# Start sample {}/{}: P0,P1,P2,P3 = {},{},{},{} - x,y,z = {},{},{} - bY,bX = {},{} \n"\
                                                    .format(iter,tot,P0,P1,P2,P3,x,y,z,yBlockDim,xBlockDim))
                                        iter += 1
                                        power_prefix = init_power_prefix+"/{}_{}_{}_{}".format(P0,P1,P2,P3)
                                        cublas_power = power_prefix+"/cublas"
                                        acm_power = power_prefix+"/tee_acm"
                                        kernels_info = fused_run_kernels_and_get_info(kernel_names,P0,P1,P2,P3,x,y,z,yBlockDim,xBlockDim,cublas_power, acm_power, with_slurm=with_slurm,path_to_executable=exe_name)
                                        if not header_already_set:
                                            csv_writer.fused_write_csv_header(kernels_info)
                                            header_already_set = True
                                        # write results into the csv file
                                        csv_writer.fused_write_csv_line(kernels_info,P0,P1,P2,P3,x,y,z,yBlockDim,xBlockDim)

#############################################################################
# Main
#############################################################################
if __name__ == '__main__':

    if len(sys.argv) != 8:
        print("usage : python {} <prefix> <out_file_name> <path_to_exe_or_dir> <bool_multiple_runs> <bool_with_slurm> <node_to_use> <bool_fused_launch>".format(sys.argv[0]))
        sys.exit(1)

    prefix = sys.argv[1]
    out_file = sys.argv[2]
    path_to_exe = sys.argv[3]
    multiple_runs = int(sys.argv[4])
    with_slurm = True if int(sys.argv[5]) else False
    node_to_use = sys.argv[6]
    fused_launch = True if int(sys.argv[7]) else False
    
    # set up all parameters
    
    #P0_tab = [8192, 16384]
    #P1_tab = [8192, 16384]
    #P2_tab = [8192, 16384]
    #P3_tab = [8192, 16384]
    P0_tab = [16384]
    P1_tab = [16384]
    P2_tab = [16384]
    #P3_tab = [256]
    #P0_tab = [256]
    #P1_tab = [512]
    #P2_tab = [512]
    x_tab  = [208]
    z_tab  = [16]
    y_tab  = [208]
    xBlockDim_tab = [16]
    yBlockDim_tab = [16]

    if fused_launch:
        kernel_names = [FUSED_CUBLAS, FUSED_ACM]
        print("# Running FUSED (with_slurm = {})".format(with_slurm))
        print("# Prefix : {}".format(prefix))
        print("# Out file name : {}".format(out_file))
        print("# Path to exe : {}".format(path_to_exe))
        print("# Kernels : {}".format(kernel_names))
        fused_run_all_tests(kernel_names, P0_tab, P1_tab, P2_tab, P3_tab, x_tab, y_tab, z_tab, xBlockDim_tab, yBlockDim_tab, out_file, path_to_exe, prefix=prefix,with_slurm=with_slurm)
    
    else:
        if multiple_runs:
            kernel_names = [TEE_ACM, CUBLASS_SGEMM]
            print("# Running on multiple exe (with_slurm = {})".format(with_slurm))
            print("# Prefix : {}".format(prefix))
            print("# Out file name : {}".format(out_file))
            print("# Path to directory containing all exe : {}".format(path_to_exe))
            print("# Kernels : {}".format(kernel_names))
            run_all_tests_multiple_exe(kernel_names, P0_tab, P1_tab, P2_tab, x_tab, y_tab, z_tab, xBlockDim_tab, yBlockDim_tab, out_file, path_to_exe,node_to_use=node_to_use, prefix=prefix,with_slurm=with_slurm)
        else:
            kernel_names = [CUBLASS_SGEMM, TEE_ACM]
            #kernel_names = [CUBLASS_SGEMM,CUPY_ORIGINAL,CUPY_SHARED_MEM,CUPY_ROW_MAJOR]
            print("# Running tests on a single exe (with_slurm = {})".format(with_slurm))
            print("# Prefix : {}".format(prefix))
            print("# Out file name : {}".format(out_file))
            print("# Path to exe : {}".format(path_to_exe))
            print("# Kernels : {}".format(kernel_names))
            run_all_tests(kernel_names, P0_tab, P1_tab, P2_tab, x_tab, y_tab, z_tab, xBlockDim_tab, yBlockDim_tab, out_file, path_to_exe, prefix=prefix,with_slurm=with_slurm)

    # run all tests
    #run_all_tests(kernel_names, P0_tab, P1_tab, P2_tab, x_tab, y_tab, z_tab, xBlockDim_tab, yBlockDim_tab, csv_file_name, prefix=prefix,with_slurm=True)
    #run_all_tests_multiple_exe(kernel_names, P0_tab, P1_tab, P2_tab, x_tab, y_tab, z_tab, xBlockDim_tab, yBlockDim_tab, csv_file_name, prefix=prefix,with_slurm=False)
