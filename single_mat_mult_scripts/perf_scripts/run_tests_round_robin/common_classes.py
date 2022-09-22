import os
import sys
from config import *
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

    def write_csv_line(self,kernels_info,P0,P1,P2,x,y,z,yBlockDim,xBlockDim):
        common_h = "{},{},{},{},{},{},{},{}".format(P0,P1,P2,x,y,z,yBlockDim,xBlockDim)
        final_h = common_h
        
        for k_name in kernels_info:
            for metric in kernels_info[k_name]:
                final_h += ","+str(kernels_info[k_name][metric])

        self.fp_csv.write(final_h+"\n")

    def write_csv_header(self,kernels_info):
        second_h = "P0,P1,P2,x,y,z,yBlockDim,xBlockDim"
        # initial shift for the first header        
        N_c = len(second_h.split(","))
        first_h = "" 
        for _ in range(N_c):
            first_h += ","

        # get headers
        for k_name in kernels_info:
            first_h+= k_name
            for metric in kernels_info[k_name]:
                first_h += ","
                second_h += ","+metric
        first_h = first_h[:-1] 

        # write headers to file
        if self.fp_csv:
            self.fp_csv.write(first_h+"\n")
            self.fp_csv.write(second_h+"\n")
    
    def fused_write_csv_line(self,kernels_info,P0,P1,P2,P3,x,y,z,yBlockDim,xBlockDim):
        common_h = "{},{},{},{},{},{},{},{},{}".format(P0,P1,P2,P3,x,y,z,yBlockDim,xBlockDim)
        final_h = common_h
        
        for k_name in kernels_info:
            for metric in kernels_info[k_name]:
                final_h += ","+str(kernels_info[k_name][metric])

        self.fp_csv.write(final_h+"\n")

    def fused_write_csv_header(self,kernels_info):
        second_h = "P0,P1,P2,P3,x,y,z,yBlockDim,xBlockDim"
        # initial shift for the first header        
        N_c = len(second_h.split(","))
        first_h = "" 
        for _ in range(N_c):
            first_h += ","

        # get headers
        for k_name in kernels_info:
            first_h+= k_name
            for metric in kernels_info[k_name]:
                first_h += ","
                second_h += ","+metric
        first_h = first_h[:-1] 

        # write headers to file
        if self.fp_csv:
            self.fp_csv.write(first_h+"\n")
            self.fp_csv.write(second_h+"\n")


    def write_csv_header_with_registers(self,kernels_info):
        second_h = "P0,P1,P2,x,y,z,yBlockDim,xBlockDim,RegTileRows,RegTileCols"
        # initial shift for the first header        
        N_c = len(second_h.split(","))
        first_h = "" 
        for _ in range(N_c):
            first_h += ","

        # get headers
        for k_name in kernels_info:
            first_h+= k_name
            for metric in kernels_info[k_name]:
                first_h += ","
                second_h += ","+metric
        first_h = first_h[:-1] 

        # write headers to file
        if self.fp_csv:
            self.fp_csv.write(first_h+"\n")
            self.fp_csv.write(second_h+"\n") 
    
    def write_csv_line_with_registers(self,kernels_info,P0,P1,P2,x,y,z,yBlockDim,xBlockDim,RegTileRows,RegTileCols):
        common_h = "{},{},{},{},{},{},{},{},{},{}".format(P0,P1,P2,x,y,z,yBlockDim,xBlockDim,RegTileRows,RegTileCols)
        final_h = common_h
        
        for k_name in kernels_info:
            for metric in kernels_info[k_name]:
                final_h += ","+str(kernels_info[k_name][metric])

        self.fp_csv.write(final_h+"\n")