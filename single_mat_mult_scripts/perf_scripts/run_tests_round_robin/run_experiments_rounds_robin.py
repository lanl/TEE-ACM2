import subprocess
import io
import re
import sys
import os
import statistics
from common_functions import *
from common_classes import *
from config import *

############################################################################
# Local functions
#############################################################################



#
# run the experimental protocol
# @prefix : path used for storing the output csv file
# @csv_file : the csvfile name
#
def run_all_tests(kernel_names, P0_tab, P1_tab, P2_tab, csv_file, exe_name, prefix=".",with_slurm=True,total_iter = 10):
    csv_writer = CSVOutput(csv_file,prefix)
    header_already_set = False 

    init_power_prefix = prefix+"/power"
    iter = 1
    x = 224
    y = 224
    z = 16
    xBlockDim = 16
    yBlockDim = 16

    tot = len(P0_tab) * len(P1_tab) * len(P2_tab) * len(x_tab) * len(y_tab) * len(z_tab) * len(xBlockDim_tab) * len(yBlockDim_tab) * total_iter

    
    ###################################################
    #
    # Profiling Runs
    #
    ###################################################
    prof_data = OrderedDict()
    run_data = OrderedDict()
    init_power_prefix = prefix+"/power"
    for P0 in P0_tab:
        for P1 in P1_tab:
            for P2 in P2_tab:

                power_prefix = init_power_prefix+"/{}_{}_{}".format(P0,P1,P2)
                cublas_power = power_prefix+"/cublas"
                acm_power = power_prefix+"/tee_acm"

                tmp_info = run_kernels_and_profile(kernel_names,P0,P1,P2,x,y,z,yBlockDim,xBlockDim,cublas_power, acm_power, with_slurm=with_slurm,path_to_executable=exe_name)
                
                # init for future runs
                run_data["{}-{}-{}".format(P0,P1,P2)] = OrderedDict()
                for k_name in kernel_names:
                    run_data["{}-{}-{}".format(P0,P1,P2)][k_name] = OrderedDict()
                    run_data["{}-{}-{}".format(P0,P1,P2)][k_name][EXPECTED_EXEC_TIME] = []
                    run_data["{}-{}-{}".format(P0,P1,P2)][k_name][EXPECTED_GLOBAL_LD] = []
                    run_data["{}-{}-{}".format(P0,P1,P2)][k_name][EXPECTED_GLOBAL_ST] = []
                    run_data["{}-{}-{}".format(P0,P1,P2)][k_name][NVML_ENERGY]        = []
                    run_data["{}-{}-{}".format(P0,P1,P2)][k_name][POWER]              = []

                prof_data["{}-{}-{}".format(P0,P1,P2)] = tmp_info
    
    ###################################################
    #
    # Performance Runs (time, power, energy)
    #
    ###################################################
    iter=1
    for _ in range(total_iter):
        for P0 in P0_tab:
            for P1 in P1_tab:
                for P2 in P2_tab:
                    power_prefix = init_power_prefix+"/{}_{}_{}".format(P0,P1,P2)
                    cublas_power = power_prefix+"/cublas"
                    acm_power = power_prefix+"/tee_acm"
                    if DISPLAY_PROGRESS:
                        info_print("# Start sample {}/{}: P0,P1,P2 = {},{},{} - x,y,z = {},{},{} - bY,bX = {},{} \n"\
                                .format(iter,tot,P0,P1,P2,x,y,z,yBlockDim,xBlockDim))
                    iter += 1
                    run_single_and_get_expected_metrics(run_data["{}-{}-{}".format(P0,P1,P2)],P0,P1,P2,x,y,z,yBlockDim,xBlockDim, "volta-x86", with_slurm, kernel_names, exe_name, cublas_power, acm_power, _)

    ###################################################
    #
    # Write the results
    #
    ###################################################

    for P0 in P0_tab:
            for P1 in P1_tab:
                for P2 in P2_tab:
                    kernels_info = prof_data["{}-{}-{}".format(P0,P1,P2)]
                    tmp_info = run_data["{}-{}-{}".format(P0,P1,P2)]

                    for k_name in kernel_names:
                        kernels_info[k_name][EXPECTED_GLOBAL_LD] = statistics.mean(tmp_info[k_name][EXPECTED_GLOBAL_LD])
                        kernels_info[k_name][EXPECTED_GLOBAL_ST] = statistics.mean(tmp_info[k_name][EXPECTED_GLOBAL_ST])
                        
                        kernels_info[k_name][EXPECTED_EXEC_TIME] = statistics.mean(tmp_info[k_name][EXPECTED_EXEC_TIME])
                        kernels_info[k_name][MIN_EXEC_TIME] = min(tmp_info[k_name][EXPECTED_EXEC_TIME])
                        kernels_info[k_name][MAX_EXEC_TIME] = max(tmp_info[k_name][EXPECTED_EXEC_TIME])
                        
                        kernels_info[k_name][POWER] = statistics.mean(tmp_info[k_name][POWER])
                        kernels_info[k_name][POWER] = min(tmp_info[k_name][POWER])
                        kernels_info[k_name][POWER] = max(tmp_info[k_name][POWER])
                        
                        kernels_info[k_name][NVML_ENERGY]        = statistics.mean(tmp_info[k_name][NVML_ENERGY])
                        kernels_info[k_name][MIN_NVML_ENERGY]        = min(tmp_info[k_name][NVML_ENERGY])
                        kernels_info[k_name][MAX_NVML_ENERGY]        = max(tmp_info[k_name][NVML_ENERGY])

                    if not header_already_set:
                        csv_writer.write_csv_header(kernels_info)
                        header_already_set = True
                    csv_writer.write_csv_line(kernels_info,P0,P1,P2,x,y,z,yBlockDim,xBlockDim)





#############################################################################
# Main
#############################################################################
if __name__ == '__main__':

    if len(sys.argv) != 6:
        print("usage : python {} <prefix> <out_file_name> <path_to_exe_or_dir> <bool_multiple_runs> <bool_with_slurm>".format(sys.argv[0]))
        sys.exit(1)

    prefix = sys.argv[1]
    out_file = sys.argv[2]
    path_to_exe = sys.argv[3]
    multiple_runs = int(sys.argv[4])
    with_slurm = True if int(sys.argv[5]) else False
    
    # set up all parameters
    
    #P0_tab = [8192, 16384]
    #P1_tab = [8192, 16384]
    #P2_tab = [8192, 16384]
    #P3_tab = [8192, 16384]
    P0_tab = [8192, 16384]
    P1_tab = [8192, 16384]
    P2_tab = [8192, 16384]
    #P3_tab = [256]
    #P0_tab = [256,512]
    #P1_tab = [512]
    #P2_tab = [512]
    x_tab  = [224]
    z_tab  = [16]
    y_tab  = [224]
    xBlockDim_tab = [16]
    yBlockDim_tab = [16]
    fused_launch = False

    kernel_names = [CUBLASS_SGEMM, TEE_ACM]
    #kernel_names = [CUBLASS_SGEMM,CUPY_ORIGINAL,CUPY_SHARED_MEM,CUPY_ROW_MAJOR]
    print("# Running tests on a single exe (with_slurm = {})".format(with_slurm))
    print("# Prefix : {}".format(prefix))
    print("# Out file name : {}".format(out_file))
    print("# Path to exe : {}".format(path_to_exe))
    print("# Kernels : {}".format(kernel_names))
    run_all_tests(kernel_names, P0_tab, P1_tab, P2_tab, out_file, path_to_exe, prefix=prefix,with_slurm=with_slurm)

    # run all tests
    #run_all_tests(kernel_names, P0_tab, P1_tab, P2_tab, x_tab, y_tab, z_tab, xBlockDim_tab, yBlockDim_tab, csv_file_name, prefix=prefix,with_slurm=True)
    #run_all_tests_multiple_exe(kernel_names, P0_tab, P1_tab, P2_tab, x_tab, y_tab, z_tab, xBlockDim_tab, yBlockDim_tab, csv_file_name, prefix=prefix,with_slurm=False)
