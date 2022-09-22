import subprocess
import io
import re
import sys
import numpy as np
import os
import time
from collections import OrderedDict
from config import *
from common_classes import *
import statistics
#############################################################################
# Helper functions
#############################################################################

#
# get_expected_ld
#
def get_nvml_energy(run_info, cuda_kernel_name):
    nvml_energy = -1
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and  "# NVML Energy" in line:
            nvml_energy = int(float(re.findall(r'^\D*(\d+)', line)[0])) # get 1st int
            break
    
    if nvml_energy == -1:
        err_print("Error : Empty nvml energy for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)
    
    return nvml_energy

#
# get_expected_ld
#
def get_power(run_info, cuda_kernel_name):
    nvml_power = -1
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and  "# Mean power" in line:
            nvml_power = int(float(re.findall(r'^\D*(\d+)', line)[0])) # get 1st int
            break
    
    if nvml_power == -1:
        err_print("Error : Empty mean power for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)
    
    return nvml_power

#
# get_exec_time
#
def get_expected_exec_time(run_info, cuda_kernel_name):
    exec_t = -1
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "# Execution time" in line:
            exec_t= float(re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', line)[0])
            break
    
    if exec_t == -1:
        err_print("Error : Empty execution time for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)

    return exec_t

#
# get_expected_ld
#
def get_expected_ld(run_info, cuda_kernel_name):
    e_ld = -1
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and  "Expected global loads" in line:
            e_ld = int(float(re.findall(r'^\D*(\d+)', line)[0])) # get 1st int
            break
    
    if e_ld == -1:
        err_print("Error : Empty expected loads for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)
    
    return e_ld

#
# get_expected_st
#
def get_expected_st(run_info, cuda_kernel_name):
    e_st = -1
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "Expected global stores" in line:
            e_st = int(float(re.findall(r'^\D*(\d+)', line)[0])) # get 1st int
            break
    
    if e_st == -1:
        err_print("Error : Empty expected stores for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)
    else: 
        return e_st


#
# get_nvcc_profiled_ld: get the number of loads profiled by nvcc
#
def get_nvcc_profiled_ld(run_info, cuda_kernel_name):
    e_ld = -1
    searched_kernel = False
    for line in run_info.stderr.splitlines():
        if "Kernel:" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "gld_transactions" in line:
            e_ld = int(float(line.split()[-1]))
            break
    if e_ld == -1:
        err_print("Error : Empty nvcc profiled loads for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)
    else: 
        return e_ld

#
# get_nvcc_profiled_ld: get the number of loads profiled by nvcc
#
def get_nvcc_profiled_st(run_info, cuda_kernel_name):
    e_st = -1
    searched_kernel = False
    for line in run_info.stderr.splitlines():
        if "Kernel:" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "gst_transactions" in line:
            e_st = int(float(line.split()[-1]))
            break
    if e_st == -1:
        err_print("Error : Empty nvcc profiled stores for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)
    else: 
        return e_st

#
# get_nvcc_profiled_ld: get the number of loads profiled by nvcc
#
def get_nvcc_profiled_metric(run_info, cuda_kernel_name, metric):
    p_metric = ""
    searched_kernel = False
    for line in run_info.stderr.splitlines():
        if "Kernel:" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 
        if searched_kernel and metric in line:
            p_metric = line.split()[-1] #int(float(line.split()[-1]))
            break

    if p_metric == "":
        err_print("Error : Metric {} was not found for kernel {} ".format(p_metric,cuda_kernel_name))
        sys.exit(1)
  
    return p_metric

#
# add labels below the bars
#
def label_bar_plot(ax):
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                str(round((i.get_width()), 2)),
                fontsize = 10, fontweight ='bold',
                color ='grey')

#
# plot_and_save
#
def plot_and_save_figure(prefix,M,block_dim_xy_tab,prof_ld_tab,out_plot_file_name):
    if(out_plot_file_name):
        ld_min = min(prof_ld_tab)
        colors = ["green" if gld == ld_min else "blue" for gld in prof_ld_tab]
        # plot bars
        xy_labels= [ str(xy[0]) + ","+str(xy[1])  for xy in block_dim_xy_tab]
        y_pos = np.arange(len(xy_labels))
        fig, ax = plt.subplots()
        hbars = ax.barh(y_pos, prof_ld_tab, align='center', color=colors)
        # set labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(xy_labels)
        ax.invert_yaxis()  #read top-to-bottom
        ax.set_ylabel('(x,y)')
        ax.set_xlabel('global mem ld transactions')
        ax.set_title('# of global memory loads when varying x,y block sizes (M={} floats)'.format(M))
        label_bar_plot(ax)
        ax.grid(True)

        colors = {'min gld':'green', 'other':'blue'}         
        labels = list(colors.keys())
        handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
        plt.legend(handles, labels)

        # save figure
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        fig.savefig(prefix+"/"+"M_"+str(M)+"_"+out_plot_file_name,bbox_inches='tight')


#
# write header to csv file
#
def write_csv_header(fp):
    fp.write("P0,P1,P2,x,y,z,yBlockDim,xBlockDim,gld_theoretical,gst_theoretical,gld_nvprof,gst_nvprof\n")

#
# write line to csv file
#
def write_csv_line(fp,P0,P1,P2,x,y,z,yBlockDim,xBlockDim,gld_theoretical,gst_theoretical,gld_nvprof,gst_nvprof):
    fp.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(P0,P1,P2,x,y,z,yBlockDim,xBlockDim,gld_theoretical,gst_theoretical,gld_nvprof,gst_nvprof))


#############################################################################
# Profiling functions
#############################################################################
#
# run test with slurm
#
def run_sample_with_slurm(P0,P1,P2,x,y,z,yBlockDim,xBlockDim, e_out_cublas, e_out_tee_acm, node_to_use="", partition="volta-x86", get_metrics=True, with_slurm=True, run_kernels_multiple_times=False, cuda_exe_path=""):
    
    if cuda_exe_path:
        path_to_executable = cuda_exe_path
    else:
        path_to_executable = CUDA_MCM_EXE
    # prepare cmd
    run_cmd = []
    if get_metrics:
        # subprocess.run(["srun","-p", partition,"nvprof",
        run_cmd = ["nvprof", "-m", PROF_METRICS]
    else: # run the print-trace-gpu for getting execution time, grid size and block size
        run_cmd = ["nvprof", "--print-gpu-trace"]

    if run_kernels_multiple_times: # if iter parameter is not specified, the kernels will be run multiple times (mean will be printed)
        run_cmd = [path_to_executable, "-P0", "{}".format(P0), "-P1", "{}".format(P1), "-P2", "{}".format(P2),\
                                            "-x","{}".format(x), "-z","{}".format(z), "-y", "{}".format(y),\
                                            "-yBlockDim", "{}".format(yBlockDim), "-xBlockDim", "{}".format(xBlockDim),\
                                            "-e_out_cublas","{}".format(e_out_cublas),"-e_out_tee_acm","{}".format(e_out_tee_acm)]
    else:
        run_cmd += [path_to_executable, "-P0", "{}".format(P0), "-P1", "{}".format(P1), "-P2", "{}".format(P2),\
                                            "-x","{}".format(x), "-z","{}".format(z), "-y", "{}".format(y), "-iter", "{}".format(1),\
                                            "-yBlockDim", "{}".format(yBlockDim), "-xBlockDim", "{}".format(xBlockDim),\
                                            "-e_out_cublas","{}".format(e_out_cublas),"-e_out_tee_acm","{}".format(e_out_tee_acm)]
    if with_slurm:
        run_cmd = ["srun","-p", partition,"-w", node_to_use] + run_cmd

    # run cmd
    run_info = subprocess.run(run_cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) 
    if DISPLAY_USED_CMD:
        out = " "
        info_print(out.join(run_cmd))
    return run_info

#
# run test with slurm
#
def fused_run_sample_with_slurm(P0,P1,P2,P3,x,y,z,yBlockDim,xBlockDim, e_out_cublas, e_out_tee_acm, partition="volta-x86", get_metrics=True, with_slurm=True, run_kernels_multiple_times=False, cuda_exe_path=""):
    
    if cuda_exe_path:
        path_to_executable = cuda_exe_path
    else:
        path_to_executable = CUDA_MCM_EXE
    # prepare cmd
    run_cmd = []
    if get_metrics:
        # subprocess.run(["srun","-p", partition,"nvprof",
        run_cmd = ["nvprof", "-m", PROF_METRICS]
    else: # run the print-trace-gpu for getting execution time, grid size and block size
        run_cmd = ["nvprof", "--print-gpu-trace"]

    if run_kernels_multiple_times: # if iter parameter is not specified, the kernels will be run multiple times (mean will be printed)
        run_cmd = [path_to_executable, "-P0", "{}".format(P0), "-P1", "{}".format(P1), "-P2", "{}".format(P2),"-P3", "{}".format(P3),\
                                            "-x","{}".format(x), "-z","{}".format(z), "-y", "{}".format(y),\
                                            "-yBlockDim", "{}".format(yBlockDim), "-xBlockDim", "{}".format(xBlockDim),\
                                            "-e_out_cublas","{}".format(e_out_cublas),"-e_out_tee_acm","{}".format(e_out_tee_acm)]
    else:
        run_cmd += [path_to_executable, "-P0", "{}".format(P0), "-P1", "{}".format(P1), "-P2", "{}".format(P2),"-P3", "{}".format(P3),\
                                            "-x","{}".format(x), "-z","{}".format(z), "-y", "{}".format(y), "-iter", "{}".format(1),\
                                            "-yBlockDim", "{}".format(yBlockDim), "-xBlockDim", "{}".format(xBlockDim),\
                                            "-e_out_cublas","{}".format(e_out_cublas),"-e_out_tee_acm","{}".format(e_out_tee_acm)]
    if with_slurm:
        run_cmd = ["srun","-p", partition] + run_cmd

    # run cmd
    run_info = subprocess.run(run_cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) 
    if DISPLAY_USED_CMD:
        out = " "
        info_print(out.join(run_cmd))
    return run_info

#
# get nvprof exec time, Grid Size, Block Size, Regs, SSMem, DSMem,
#
def parse_general_kernel_run_info(run_info, kernel_names):
    kernels_info = OrderedDict()
    # go to profiling results:
    i = 0
    profiled_info_started = False
    lines = run_info.stderr.splitlines()
    while i < len(lines) and not "Profiling result:" in lines[i]:
        i+=1
    i+=2 #go to profiling info
    # Get all general profling metrics
    # nvprof exec time : 1 Grid Size : 2 - Block Size : 3 - Regs : 4 - SSMem : 5 - DSMem : 6
    while i < len(lines) and lines[i].strip() :
        prof_info = re.split(r'\s{2,}', lines[i])
        if prof_info[2].strip() != "-": # we are on a kernel info
            expected_kernel = False
            for k_name in kernel_names:
                if k_name in prof_info[-1]: # kernel found
                    
                    expected_kernel = True 
                    k_info = OrderedDict()
                    k_info[PROFILED_EXEC_TIME] = prof_info[1]
                    k_info[GRID_SIZE]  = prof_info[2]
                    k_info[BLOCK_SIZE] = prof_info[3]
                    k_info[REGS]       = prof_info[4]
                    k_info[SSMem]      = prof_info[5]
                    k_info[DSMem]      = prof_info[6]
                    # append kernel info
                    kernels_info[str(k_name)] = k_info
                    break 

            # found an unexcepcted kernel
            if not expected_kernel:  
                err_print("Warning : Found an unexpected kernel : {} ".format(prof_info[-1]))
                #sys.exit(1)
        i += 1
    
    # check that we got info for all kernels
    profiled_kernels = kernels_info.keys()

    for k_name in kernel_names:
        if k_name not in profiled_kernels:
            err_print("Error general parse info: no profiling info for kernel : {} ({})".format(k_name,kernels_info))
            sys.exit(1)
    
    return kernels_info
#
# get nvprof profiled metrics
#
def parse_nvprof_metrics(run_info, kernel_names, kernels_info):
    # sanity check
    for k_name in kernel_names:
        if k_name not in run_info.stderr:
            err_print("Error : no profiling info for kernel : {} ".format(k_name))
            sys.exit(1)
    
    
    nvprof_metrics = PROF_METRICS.split(",")
    lines = run_info.stderr.splitlines()
    i = 0
    while i < len(lines) and not "Metric result" in lines[i]:
        i+=1

    # sanity check
    while i < len(lines) and lines[i] and lines[i].strip():
        if "Kernel:" in lines[i]:
            knwon_kernel = False 
            for k_name in kernel_names:
                if k_name in lines[i]:
                    knwon_kernel = True 
            if not knwon_kernel:
                err_print("Error : Found an unexpected kernel : {} ".format(lines[i]))
                #sys.exit(1)
        i+=1
    
    # start parsing file
    for k_name in kernel_names:
        for metric in nvprof_metrics:
            m_info = get_nvcc_profiled_metric(run_info, k_name, metric)
            kernels_info[k_name][metric] = m_info


def sanity_check(run_info, kernel_names):
    # sanity check (no profiling info)
    for k_name in kernel_names:
        if k_name not in run_info.stdout:
            err_print("Error : no profiling info for kernel : {} ({})".format(k_name,run_info.stdout))
            sys.exit(1)

    # sanity check (unexpected kernel)
    lines = run_info.stdout.splitlines()
    i = 0
    while i < len(lines) and lines[i] and lines[i].strip():
        if "# Kernel name :" in lines[i]:
            knwon_kernel = False 
            for k_name in kernel_names:
                if k_name in lines[i]:
                    knwon_kernel = True 
            if not knwon_kernel:
                err_print("Error : Found an unexpected kernel : {} ".format(lines[i]))
                #sys.exit(1)
        i+=1

#
# get metrics printed by the program
#
def parse_expected_metrics(run_info, kernel_names, kernels_info):
    # sanity check
    for k_name in kernel_names:
        if k_name not in run_info.stdout:
            err_print("Error : no profiling info for kernel : {} ({})".format(k_name,run_info.stdout))
            sys.exit(1)

    # sanity check (unexpected kernel)
    lines = run_info.stdout.splitlines()
    i = 0
    while i < len(lines) and lines[i] and lines[i].strip():
        if "# Kernel name :" in lines[i]:
            knwon_kernel = False 
            for k_name in kernel_names:
                if k_name in lines[i]:
                    knwon_kernel = True 
            if not knwon_kernel:
                err_print("Error : Found an unexpected kernel : {} ".format(lines[i]))
                #sys.exit(1)
        i+=1
    # start parsing file
    for k_name in kernel_names:
        kernels_info[k_name][EXPECTED_EXEC_TIME] = get_expected_exec_time(run_info, k_name)
        kernels_info[k_name][EXPECTED_GLOBAL_LD] = get_expected_ld(run_info, k_name)
        kernels_info[k_name][EXPECTED_GLOBAL_ST] = get_expected_st(run_info, k_name)
        kernels_info[k_name][NVML_ENERGY]        = get_nvml_energy(run_info, k_name)

#
# get metrics printed by the program
#
def run_multiple_times_and_get_expected_metrics(P0,P1,P2,x,y,z,yBlockDim,xBlockDim, partition, with_slurm, kernel_names, kernels_info, path_to_executable,e_out_cublas, e_out_tee_acm,node_to_use="", iter=10, fused_kernel=False,P3=0):

    tmp_info = {}#OrderedDict()
    # start parsing file
    for k_name in kernel_names:
        tmp_info[k_name] = {}
        tmp_info[k_name][EXPECTED_EXEC_TIME] = []
        tmp_info[k_name][EXPECTED_GLOBAL_LD] = []
        tmp_info[k_name][EXPECTED_GLOBAL_ST] = []
        tmp_info[k_name][NVML_ENERGY]        = []
        tmp_info[k_name][POWER]              = []

    if not os.path.exists(e_out_cublas):
        os.makedirs(e_out_cublas)
    
    if not os.path.exists(e_out_tee_acm):
        os.makedirs(e_out_tee_acm)        
    # run iter number of itertions
    
    for _ in range(iter):
        csv_file = "/run_{}.csv".format(_)
        if fused_kernel:
            exp_metrics = fused_run_sample_with_slurm(P0,P1,P2,P3,x,y,z,yBlockDim,xBlockDim, e_out_cublas+csv_file, e_out_tee_acm+csv_file, partition=partition, get_metrics=True, with_slurm=with_slurm, run_kernels_multiple_times=True,cuda_exe_path=path_to_executable)
        else :
            exp_metrics = run_sample_with_slurm(P0,P1,P2,x,y,z,yBlockDim,xBlockDim, e_out_cublas+csv_file, e_out_tee_acm+csv_file, node_to_use=node_to_use,partition=partition, get_metrics=True, with_slurm=with_slurm, run_kernels_multiple_times=True,cuda_exe_path=path_to_executable)
        if DEBUG:
            for line in exp_metrics.stdout.splitlines():
                debug_print(line) 
        sanity_check(exp_metrics, kernel_names)
        for k_name in kernel_names:
            tmp_info[k_name][EXPECTED_EXEC_TIME].append(get_expected_exec_time(exp_metrics, k_name))
            tmp_info[k_name][EXPECTED_GLOBAL_LD].append(get_expected_ld(exp_metrics, k_name))
            tmp_info[k_name][EXPECTED_GLOBAL_ST].append(get_expected_st(exp_metrics, k_name))
            tmp_info[k_name][NVML_ENERGY].append(get_nvml_energy(exp_metrics, k_name))
            tmp_info[k_name][POWER].append(get_power(exp_metrics, k_name))
        time.sleep(1)

    # save only the mean
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


#
# run one data point and collect profiling metrics
#
def run_kernels_and_get_info(kernel_names,P0,P1,P2,x,y,z,yBlockDim,xBlockDim,e_out_cublas, e_out_tee_acm, node_to_use="", partition="volta-x86", with_slurm=True, path_to_executable=""):

    #print("NODE",node_to_use)
    # get --print-gpu-trace metrics
    general_info = run_sample_with_slurm(P0,P1,P2,x,y,z,yBlockDim,xBlockDim,"tmp1.csv","tmp2.csv",node_to_use=node_to_use, partition=partition, get_metrics=False, with_slurm=with_slurm,cuda_exe_path=path_to_executable)
    if DEBUG:
        for line in general_info.stdout.splitlines():
            debug_print(line)
        for line in general_info.stderr.splitlines():
            debug_print(line)
    kernels_info = parse_general_kernel_run_info(general_info,kernel_names)
    
    # get nvprof specific metrics (loads, stores, cache hits)
    nvprof_metrics = run_sample_with_slurm(P0,P1,P2,x,y,z,yBlockDim,xBlockDim,"tmp1.csv","tmp2.csv", node_to_use=node_to_use, partition=partition, get_metrics=True, with_slurm=with_slurm,cuda_exe_path=path_to_executable)
    if DEBUG:
        for line in nvprof_metrics.stderr.splitlines():
            debug_print(line) 
    parse_nvprof_metrics(nvprof_metrics, kernel_names, kernels_info)

    # e_out_cublas tmp1_cublas -e_out_tee_acm tee_energy
    run_multiple_times_and_get_expected_metrics(P0,P1,P2,x,y,z,yBlockDim,xBlockDim, partition, with_slurm, kernel_names, kernels_info, path_to_executable, e_out_cublas, e_out_tee_acm, node_to_use=node_to_use,iter=20)
    #exp_metrics = run_sample_with_slurm(P0,P1,P2,x,y,z,yBlockDim,xBlockDim, partition=partition, get_metrics=True, with_slurm=with_slurm, run_kernels_multiple_times=True,cuda_exe_path=path_to_executable)
    #parse_expected_metrics(exp_metrics, kernel_names, kernels_info)

    # relax the GPU
    time.sleep(0.5)
    return kernels_info


#
# run one data point and collect profiling metrics
#
def fused_run_kernels_and_get_info(kernel_names,P0,P1,P2,P3,x,y,z,yBlockDim,xBlockDim,e_out_cublas, e_out_tee_acm, partition="volta-x86", with_slurm=True, path_to_executable=""):
    kernels_info = OrderedDict()
    for k_name in kernel_names:
        kernels_info[k_name] = OrderedDict()
    # e_out_cublas tmp1_cublas -e_out_tee_acm tee_energy
    run_multiple_times_and_get_expected_metrics(P0,P1,P2,x,y,z,yBlockDim,xBlockDim, partition, with_slurm, kernel_names, kernels_info, path_to_executable, e_out_cublas, e_out_tee_acm, iter=5,fused_kernel=True,P3=P3)

    # relax the GPU
    time.sleep(0.5)
    return kernels_info