import subprocess
import io
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import OrderedDict
from config_kernel import *
from common_classes_kernel import *
#############################################################################
# Helper functions
#############################################################################
#
# get_exec_time
#

def get_seq_size(run_info, cuda_kernel_name):
    seq_size = -1
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "# Sequence Size" in line:
            #exec_tot_t.append(float(re.findall("\d+\,\d+", line)[0]))
            seq_size=float(re.findall('\d+\.\d+', line)[0])
            break
    
    if seq_size == -1:
        err_print("Error : No sequence size detected {} ".format(cuda_kernel_name))
        sys.exit(1)

    return seq_size

def get_expected_exec_time(run_info, cuda_kernel_name):
    exec_t = []
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "# Execution time [ms]" in line:
            #exec_t.append(float(re.findall('/^(?!0\d)\d*(\.\d+)?$/', line)[0])) #\d+\.\d+  /^(?!0\d)\d*(\.\d+)?$/
            exec_t.append(float(re.findall('\d+\.\d+', line)[0]))
            #exec_t=float(re.findall('\d+\.\d+', line)[0])
            
            #break
    
    if exec_t == -1:
        err_print("Error : Empty execution time for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)

    return exec_t

def get_total_exec_time(run_info, cuda_kernel_name):
    exec_tot_t = -1
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "# Traversal time [ms]" in line:
            #exec_tot_t.append(float(re.findall("\d+\,\d+", line)[0]))
            exec_tot_t=float(re.findall('\d+', line)[0])
            break
    
    if exec_tot_t == -1:
        err_print("Error : Empty total execution time for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)

    return exec_tot_t

def get_total_energy(run_info, cuda_kernel_name):
    exec_tot_t = -1
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "# Traversal energy [mJ]" in line:
            #exec_tot_t.append(float(re.findall("\d+\,\d+", line)[0]))
            exec_tot_t=float(re.findall('\d+', line)[0])
            break
    
    if exec_tot_t == -1:
        err_print("Error : Empty total execution time for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)

    return exec_tot_t

def get_creation_tree_time(run_info, cuda_kernel_name):
    exec_tot_t = -1
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "# Creation Tree" in line:
            #exec_tot_t.append(float(re.findall("\d+\,\d+", line)[0]))
            exec_tot_t=float(re.findall('\d+', line)[0])
            break
    
    if exec_tot_t == -1:
        err_print("Error : Empty creation time for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)

    return exec_tot_t

def get_total_creation_time(run_info, cuda_kernel_name):
    exec_tot_t = -1
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "# Entire creation" in line:
            #exec_tot_t.append(float(re.findall("\d+\,\d+", line)[0]))
            exec_tot_t=float(re.findall('\d+', line)[0])
            break
    
    if exec_tot_t == -1:
        err_print("Error : Empty creation time for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)

    return exec_tot_t





def get_nvml_energy(run_info, cuda_kernel_name):
    nvml_energy =  []
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and  "# NVML Energy" in line:
            nvml_energy.append(int(float(re.findall(r'^\D*(\d+)', line)[0]))) # get 1st int
            #break
    
    if nvml_energy == -1:
        err_print("Error : Empty nvml energy for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)
    
    return nvml_energy


def get_expected_compute(run_info, cuda_kernel_name):
    comp_t = -1
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "# Computation nb" in line:
            comp_t= int(re.findall('\d+', line)[0]) # get 1st int
    
    if comp_t == -1:
        err_print("Error : Empty computation number for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)

    return comp_t

def get_kernel_sizes(run_info, cuda_kernel_name):
    sizes_t=[]
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "# Sizes" in line:
            tmp=line.split(":")[1]
            tmp=tmp.split()
            s1= int(tmp[0]) # get 1st int
            s2= int(tmp[1]) # get 2nd int
            s3= int(tmp[2]) # get 3rd int
            sizes_t.append((s1,s2,s3))
            #break
    
    if sizes_t == -1:
        err_print("Error : Empty computation number for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)

    return sizes_t
def get_compute_order(run_info, cuda_kernel_name,seq_size):
    order_t=[]
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "# Sequence order" in line:
            #tmp=line.split(":")[1]
            #regex for xx.xx -> "\d+\.\d+"
            #tmp=tmp.split()
            #for nb in range(seq_size-1):
            #    order_t.append((tmp[nb]))
            #break
            order_t=re.findall('\d+.\d+',line)
            break

    
    if order_t == -1:
        err_print("Error : Empty computation number for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)

    return order_t

def get_sizes(run_info, cuda_kernel_name,seq_size):
    sizes_t=[]
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "# Matrix Sizes" in line:
            #tmp=line.split(":")[1]
            #tmp=tmp.split()
            #for nb in range(seq_size+1):
            #    sizes_t.append((tmp[nb][0]))
            #break
            sizes_t=re.findall('[0-9]+',line)
            break
    
    if sizes_t == -1:
        err_print("Error : Empty computation number for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)

    return sizes_t

def get_expected_dt(run_info, cuda_kernel_name):
    dt_t = -1
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "# Data transfers nb" in line:
            dt_t= int(re.findall('\d+', line)[0]) # get 1st int
    
    if dt_t == -1:
        err_print("Error : Empty data transfers for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)

    return dt_t

#
# get_expected_ld
#
def get_expected_ld(run_info, cuda_kernel_name):
    e_ld = []
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and  "Expected global loads" in line:
            e_ld.append(int(float(re.findall(r'^\D*(\d+)', line)[0]))) # get 1st int
            #break
    
    if e_ld == -1:
        err_print("Error : Empty expected loads for kernel {} ".format(cuda_kernel_name))
        sys.exit(1)
    
    return e_ld

#
# get_expected_st
#
def get_expected_st(run_info, cuda_kernel_name):
    e_st = []
    searched_kernel = False
    for line in run_info.stdout.splitlines():
        if "# Kernel name :" in line :
            if cuda_kernel_name in line:
                searched_kernel = True
            else: 
                searched_kernel = False 

        if searched_kernel and "Expected global stores" in line:
            e_st.append(int(float(re.findall(r'^\D*(\d+)', line)[0]))) # get 1st int
            #break
    
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
    searched_kernel = True
    for line in run_info.stderr.splitlines():
        #if "Kernel:" in line :
        #    if cuda_kernel_name in line:
        #        searched_kernel = True
        #    else: 
        #        searched_kernel = False 

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
    searched_kernel = True
    for line in run_info.stderr.splitlines():
        #if "Kernel:" in line :
            #if cuda_kernel_name in line:
            #    searched_kernel = True
            #else: 
            #    searched_kernel = False 

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
    fp.write("run,x,y,z,yBlockDim,xBlockDim,P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10\n")
    #fp.write("run,x,y,z,yBlockDim,xBlockDim\n")

#
# write line to csv file
#
def write_csv_line(fp,run,x,y,z,yBlockDim,xBlockDim,gld_theoretical,gst_theoretical,gld_nvprof,gst_nvprof,P):
    fp.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(run,x,y,z,yBlockDim,xBlockDim,P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8],P[9],P[10]))
    #fp.write("{},{},{},{},{},{},{},{},{},{}\n".format(run,x,y,z,yBlockDim,xBlockDim))


#############################################################################
# Profiling functions
#############################################################################
#
# run test with slurm
#
def run_sample_with_slurm(node,run,k,seq_size,x,y,z,yBlockDim,xBlockDim,P, partition="clx-volta", get_metrics=True, with_slurm=True):
    i=(int)((x-208)/16)
    # prepare cmd
    run_cmd = []
    #run_cmd = ["nvprof", "-m", PROF_METRICS]
    run_cmd += [CUDA_MCM_EXE[i], "-k", "{}".format(k),"-n", "{}".format(seq_size),"-run", "{}".format(run),\
                                            "-x","{}".format(x), "-z","{}".format(z), "-y", "{}".format(y),\
                                            "-yBlockDim", "{}".format(yBlockDim), "-xBlockDim", "{}".format(xBlockDim)]#,\
                                            #"-P0","{}".format(P[0]),"-P1","{}".format(P[1]),"-P2","{}".format(P[2]),"-P3","{}".format(P[3]),\
                                            #"-P4","{}".format(P[4]),"-P5","{}".format(P[5]),"-P6","{}".format(P[6]),"-P7","{}".format(P[7]),\
                                            #"-P8","{}".format(P[8]),"-P9","{}".format(P[9]),"-P10","{}".format(P[10])]
    if with_slurm:
        run_cmd = ["srun","-p", partition, "{}{}".format("-w cn",node)] + run_cmd #, "-w cn666" "{}{}".format("-w cn",node)

    # run cmd
    run_info = subprocess.run(run_cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) 
    if DISPLAY_USED_CMD:
        out = " "
        info_print(out.join(run_cmd))
    return run_info

#
# get nvprof exec time, Grid Size, Block Size, Regs, SSMem, DSMem,
#
def parse_general_kernel_run_info(run_info, k_name):
    
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
                kernels_info[k_name] = k_info
                break 

            # found an unexcepcted kernel
            if not expected_kernel:  
                err_print("Error : Found an unexpected kernel : {} ".format(prof_info[-1]))
                sys.exit(1)
        i += 1
    
    # check that we got info for all kernels
    profiled_kernels = kernels_info.keys()

    if k_name not in profiled_kernels:
        err_print("ERROR : no profiling info for kernel : {} ".format(k_name))
        sys.exit(1)
    
    return kernels_info
#
# get nvprof profiled metrics
#
def parse_nvprof_metrics(run_info, k_name, kernels_info):
    # sanity check
    if k_name not in run_info.stderr:
        err_print("ERror : no profiling info for kernel : {} ".format(k_name))
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
            if k_name in lines[i]:
                knwon_kernel = True 
            if not knwon_kernel:
                err_print("Error : Found an unexpected kernel : {} ".format(lines[i]))
                sys.exit(1)
        i+=1
    
    # start parsing file
    for metric in nvprof_metrics:
        m_info = get_nvcc_profiled_metric(run_info, k_name, metric)
        kernels_info[k_name][metric] = m_info

#
# get metrics printed by the program
#
def parse_expected_metrics(run_info, k_name, kernels_info,seq_size):
    # sanity check

    if k_name not in run_info.stdout:
        err_print("ErrOR : no profiling info for kernel : {} ".format(k_name))
        sys.exit(1)

    # sanity check (unexpected kernel)
    lines = run_info.stdout.splitlines()
    i = 0
    while i < len(lines) and lines[i] and lines[i].strip():
        if "# Kernel name :" in lines[i]:
            knwon_kernel = False 

            if k_name in lines[i]:
                knwon_kernel = True 
            if not knwon_kernel:
                err_print("Error : Found an unexpected kernel : {} ".format(lines[i]))
                sys.exit(1)
        i+=1

    kernels_info[k_name]= {}
    # start parsing file

    kernels_info[k_name][SIZES] = get_sizes(run_info, k_name,seq_size) 
    kernels_info[k_name][ORDER] = get_compute_order(run_info,k_name, seq_size) 
    kernels_info[k_name][EXPECTED_EXEC_TIME] = get_expected_exec_time(run_info, k_name)
    kernels_info[k_name][EXPECTED_GLOBAL_LD] = get_expected_ld(run_info, k_name)
    kernels_info[k_name][EXPECTED_GLOBAL_ST] = get_expected_st(run_info, k_name)

    #kernels_info[k_name][GLOBAL_LD] = get_nvcc_profiled_ld(run_info, k_name)
    #kernels_info[k_name][GLOBAL_ST] = get_nvcc_profiled_st(run_info, k_name) 
    kernels_info[k_name][NVML_ENERGY]        = get_nvml_energy(run_info, k_name)
    kernels_info[k_name][TOTAL_TIME] = get_total_exec_time(run_info, k_name)
    kernels_info[k_name][TOTAL_ENERGY] = get_total_energy(run_info, k_name)
    kernels_info[k_name][CREATION_TREE_TIME] = get_creation_tree_time(run_info, k_name) 
    kernels_info[k_name][TOTAL_CREATION_TIME] = get_total_creation_time(run_info, k_name)
        

        

#
# run one data point and collect profiling metrics
# 
def run_kernels_and_get_info(node,k_name,run,k,seq_size,x,y,z,yBlockDim,xBlockDim,P, partition="clx-volta", with_slurm=True):
    
    # get --print-gpu-trace metrics
   # general_info = run_sample_with_slurm(run,x,y,z,yBlockDim,xBlockDim, partition=partition, get_metrics=False, with_slurm=with_slurm)
    #if DEBUG:
    #    print(general_info)
    #    for line in general_info.stdout.splitlines():
    #        debug_print(line)
    #    for line in general_info.stderr.splitlines():
    #        debug_print(line)
    #kernels_info = parse_general_kernel_run_info(general_info,kernel_names)
    
    # get nvprof specific metrics (loads, stores, cache hits)
    kernels_info={}
    nvprof_and_exp_metrics = run_sample_with_slurm(node,run,k,seq_size,x,y,z,yBlockDim,xBlockDim,P, partition=partition, get_metrics=False, with_slurm=with_slurm)
    if DEBUG:
        print(nvprof_and_exp_metrics)
        for line in nvprof_and_exp_metrics.stdout.splitlines():
            debug_print(line) 
        for line in nvprof_and_exp_metrics.stderr.splitlines():
            debug_print(line) 
    #parse_nvprof_metrics(nvprof_and_exp_metrics, kernel_names, kernels_info)

    # get expected metrics printed by our implementation (expected loads, stores, execution time)
    parse_expected_metrics(nvprof_and_exp_metrics, k_name, kernels_info,seq_size)
    
    return kernels_info

    