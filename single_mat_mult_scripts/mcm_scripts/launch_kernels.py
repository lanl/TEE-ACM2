import subprocess
import io
import re
import sys
import random
from common_functions_kernel import *
from common_classes_kernel import *
import time

############################################################################
# Local functions
#############################################################################

#
# run the experimental protocol
# @prefix : path used for storing the output csv file
# @csv_file : the csvfile name
#
def run_all_tests(node,kernel_names,nb_mat, x_tab, xBlockDim_tab, yBlockDim_tab, csv_file,iteration, prefix=".",with_slurm=True):
    
    csv_writer = CSVOutput(csv_file,prefix)
    header_already_set = False 
    P=[]
    tot = 2 * 3  * (len(x_tab) + 2) * len(xBlockDim_tab) * len(yBlockDim_tab)
    
    max_seq_size=nb_mat
    seq_size=nb_mat
    for run in [0,1,4]:
        #for seq_size in nb_mat:
        for k in range(3): #kernels
            if (k!=0):
                x=208
                y=x
                z=16
                # run one sample
                if DISPLAY_PROGRESS:
                    info_print("# Start sample {}/{}: run = {} kernel = {} - nb_mat = {} - x,y,z = {},{},{} - bY,bX = {},{} \n"\
                                .format(iteration,tot,run,k,seq_size,x,y,z,yBlockDim,xBlockDim))
                    #info_print("with sizes P : {},{},{},{},{},{},{},{},{},{},{} \n"\
                            # .format(P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8],P[9],P[10]))
                iteration += 1
                k_name=kernel_names[k]
                kernels_info = run_kernels_and_get_info(node,k_name,run,k,seq_size,x,y,z,yBlockDim,xBlockDim,P,with_slurm=with_slurm)
                if not header_already_set:
                    csv_writer.write_csv_header_new(kernels_info,k_name,max_seq_size)
                    header_already_set = True
                # write results into the csv file
                csv_writer.write_csv_line_new(kernels_info,k_name,seq_size,x,y,z,yBlockDim,xBlockDim,P,max_seq_size)
                time.sleep(1)
            else:
                for x in (x_tab):
                    for xBlockDim in xBlockDim_tab:
                        for yBlockDim in yBlockDim_tab:
                            #for iter in range(3):
                            #generate sizes
                            #x=x_tab[xyz]
                            y=x
                            z=16
                            # run one sample
                            if DISPLAY_PROGRESS:
                                info_print("# Start sample {}/{}: run = {} kernel = {} - nb_mat = {} - x,y,z = {},{},{} - bY,bX = {},{} \n"\
                                            .format(iteration,tot,run,k,seq_size,x,y,z,yBlockDim,xBlockDim))
                                #info_print("with sizes P : {},{},{},{},{},{},{},{},{},{},{} \n"\
                                        # .format(P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8],P[9],P[10]))
                            iteration += 1
                            k_name=kernel_names[k]
                            kernels_info = run_kernels_and_get_info(node,k_name,run,k,seq_size,x,y,z,yBlockDim,xBlockDim,P,with_slurm=with_slurm)
                            if not header_already_set:
                                csv_writer.write_csv_header_new(kernels_info,k_name,max_seq_size)
                                header_already_set = True
                            # write results into the csv file
                            csv_writer.write_csv_line_new(kernels_info,k_name,seq_size,x,y,z,yBlockDim,xBlockDim,P,max_seq_size)
                            time.sleep(1)
        P.clear()


#############################################################################
# Main
#############################################################################
if __name__ == '__main__':
    prefix = "tempuris"
    
    # set up all desired parameters
    kernel_names = [ TEE_ACM,CUBLASS_SGEMM_TREE,CUBLASS_SGEMM]
    #run_tab = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    
    x_tab  = [208,224] #176,192,208,224
    #z_tab  = [16,16]
    #y_tab  = [176]
    xBlockDim_tab = [256]
    yBlockDim_tab = [1]
    # directory where to store the csv file
    prefix = "."
    #csv_file_name = "test_big_matrices_32k_30runs.csv"
    iteration = 1
    for node in [667]:
        info_print("# Start work on node {}\n"\
                                .format(node))
        for i in [12,6,20,15]:
            csv_file_name = "{}{}{}{}{}".format("NRun_",node,"_ACM_dataset_", i,"mat.csv")
            # run all tests for one size of sequence
            run_all_tests(node,kernel_names,i, x_tab, xBlockDim_tab, yBlockDim_tab, csv_file_name,iteration, prefix=prefix,with_slurm=True)
