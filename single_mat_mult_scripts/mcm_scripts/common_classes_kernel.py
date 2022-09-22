import os
import sys
from config_kernel import *
#############################################################################
# CSVOutput class
#############################################################################
class CSVOutput:
    def __init__(self, csv_file="", prefix=""):
        self.csv_file = csv_file
        self.prefix = prefix
        self.fp_csv = None
        if self.csv_file:
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            self.fp_csv = open(self.prefix+"/"+csv_file, 'w')
            #self.write_csv_header()
        else:
            err_print("Error : no valid or empty csv_file name : {}".format(csv_file))
            sys.exit(1)

    def __del__(self):
        if self.fp_csv:
            self.fp_csv.close()

    def write_csv_line(self,kernels_info,k_name,run,x,y,z,yBlockDim,xBlockDim,P,seq_size):
        #common_h = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(run,x,y,z,yBlockDim,xBlockDim,P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8],P[9],P[10])
        #common_h = "{},{},{},{},{},{}".format(run,x,y,z,yBlockDim,xBlockDim)
        common_h = "{}".format(run)
        final_h = common_h
        
        #for k_name in kernels_info:
        for compute_nb in range(seq_size-1):
            for metric in kernels_info[k_name]:
                if(metric=="matrix_sizes"):
                    final_h += ","+str(kernels_info[k_name][metric][compute_nb][0])
                    final_h += ","+str(kernels_info[k_name][metric][compute_nb][1])
                    final_h += ","+str(kernels_info[k_name][metric][compute_nb][2])
                if((metric != "computation_order") & (metric != "matrix_sizes") & (metric != "exp_gld_transactions") & (metric != "exp_gst_transactions") & (metric != "creation_tree_time")):
                    final_h += ","+str(kernels_info[k_name][metric][compute_nb])

        self.fp_csv.write(final_h+"\n")
    
    def write_csv_line_new(self,kernels_info,k_name,seq_size,x,y,z,yBlockDim,xBlockDim,P,max_seq_size):
        #common_h = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(run,x,y,z,yBlockDim,xBlockDim,P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8],P[9],P[10])
        #common_h = "{},{},{},{},{},{}".format(run,x,y,z,yBlockDim,xBlockDim)
        blank=""
        #for k_name in kernels_info:
            
        common_h= "{}".format(seq_size) + ","
        common_h+="{}".format(k_name) + ","
        common_h+= "{},{},{},{},{},".format(x,y,z,yBlockDim,xBlockDim)
        cmp_t=4
        #for i in range(seq_size+1):
        #    common_h+="{}".format(kernels_info[k_name][SIZES][i]) + ","
        #for i in range(seq_size-1):
        #    common_h+="{}".format(kernels_info[k_name][ORDER][i]) + ","
        
        for i in kernels_info[k_name][SIZES]:
            common_h+="{}".format(i) + ","
            cmp_t=cmp_t+2
        for i in range(seq_size,max_seq_size):
            common_h+=","
        for i in kernels_info[k_name][ORDER]:
            common_h+="{}".format(i) + ","
            cmp_t=cmp_t+2
        for i in range(seq_size,max_seq_size):
            common_h+=","
        common_h+=str((kernels_info[k_name][TOTAL_CREATION_TIME])) + ","    
        common_h+=str((kernels_info[k_name][CREATION_TREE_TIME])) + ","
        common_h+=str((kernels_info[k_name][TOTAL_TIME])) + ","
        common_h+=str((kernels_info[k_name][TOTAL_ENERGY])) 
        #common_h+=str((kernels_info[k_name][GLOBAL_LD])) + ","
        #common_h+=str((kernels_info[k_name][GLOBAL_ST]))
        cmp_t=cmp_t+5
        final_h = common_h

        tot_time=0
        tot_exp_load=0
        tot_exp_store=0
        ##for k_name in kernels_info:
        for compute_nb in range(seq_size-1):
            for metric in kernels_info[k_name]:
                if((metric != "computation_order") & (metric != "matrix_sizes")& (metric != "total_time [ms]")& (metric != "gld_transactions")& (metric != "gst_transactions")& (metric != "creation_tree_time")&(metric !="total_energy [mJ]")&(metric !="total_creation_time")): 
                    final_h += ","+str(kernels_info[k_name][metric][compute_nb])
                    cmp_t=cmp_t+2
                #if(metric == "exp_exec_time [ms]" ):
                    #tot_time+=float(str(kernels_info[k_name][metric][compute_nb]))
                if(metric == "exp_gld_transactions" ):
                    tot_exp_load+=int(str(kernels_info[k_name][metric][compute_nb]))
                if(metric == "exp_gst_transactions" ):
                    tot_exp_store+=int(str(kernels_info[k_name][metric][compute_nb]))
    
        #common_h+="{}".format(tot_exp_load) + ","
        #cmp_t=cmp_t+2
        #common_h+="{}".format(tot_exp_store) + ","
        #cmp_t=cmp_t+2
        self.fp_csv.write(final_h+"\n")
        #cmp_t=cmp_t+1

        #for i in range(cmp_t):
        #    blank+=" "
        #self.fp_csv.write(blank+"\n")

    def write_csv_header(self,kernels_info,seq_size):
        #second_h = "run,x,y,z,yBlockDim,xBlockDim,P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10"
        #second_h = "run,x,y,z,yBlockDim,xBlockDim"
        second_h = "run"
        # initial shift for the first header        
        N_c = len(second_h.split(","))
        first_h = "" 
        for _ in range(N_c):
            first_h += ","

        # get headers
        for k_name in kernels_info:
            first_h+= k_name
            for compute_nb in range(seq_size-1):
                for metric in kernels_info[k_name]:
                    first_h += ","
                    second_h += ","+metric
        first_h = first_h[:-1] 

        # write headers to file
        if self.fp_csv:
            self.fp_csv.write(first_h+"\n")
            self.fp_csv.write(second_h+"\n")

    def write_csv_header_new(self,kernels_info,k_name,max_seq_size):
        #second_h = "run,x,y,z,yBlockDim,xBlockDim,P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10"
        
        text_t = "nb_matrices,kernel,"
        text_t += "x,y,z,yBlockDim,xBlockDim,sizes"
        #add shift for sizes
        for i in range(max_seq_size+1):
            text_t += ","
        text_t+="order"
        #add shift for order
        for i in range(max_seq_size-1):
            text_t += ","
        #text_t += ","
        text_t+="total_creation_time,tree_creation"
        text_t+=",total_exec_time"
        text_t+=",total_energy"
        #text_t+=",gld_transactions"
        #text_t+=",gst_transactions"
        # get headers
        #for k_name in kernels_info:
        for compute_nb in range(max_seq_size-1):
            for metric in kernels_info[k_name]:
                if((metric != "computation_order") & (metric != "matrix_sizes") & (metric != "total_time [ms]")& (metric != "gld_transactions")& (metric != "gst_transactions") & (metric != "creation_tree_time")&(metric !="total_energy [mJ]")):
                    text_t += ","+metric
        
        text_t += ","
        # write headers to file
        if self.fp_csv:
            self.fp_csv.write(text_t+"\n")

        