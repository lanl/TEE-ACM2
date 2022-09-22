import sys
#############################################################################
# Debug mode
#############################################################################
DEBUG = True
DISPLAY_USED_CMD = True
DISPLAY_PROGRESS = True
#############################################################################
# Constants
#############################################################################
#CUDA_MCM_EXE = ["/vast/home/mina/testing_MCM/build/new_176_acm","/vast/home/mina/testing_MCM/build/new_192_acm","/vast/home/mina/testing_MCM/build/new_208_acm","/vast/home/mina/testing_MCM/build/new_224_acm",] #(x,y)=176,192,208,224
#CUDA_MCM_EXE = ["/vast/home/mina/testing_MCM/build/graph_176","/vast/home/mina/testing_MCM/build/graph_192","/vast/home/mina/testing_MCM/build/graph_208","/vast/home/mina/testing_MCM/build/graph_224",] #(x,y)=176,192,208,224
CUDA_MCM_EXE = ["/vast/home/mina/testing_MCM/build/acm_208","/vast/home/mina/testing_MCM/build/acm_224",] #(x,y)=176,192,208,224
#CUDA_MCM_EXE = ["/vast/home/mina/testing_MCM/build/main_cupy_176","/vast/home/mina/testing_MCM/build/main_cupy_192","/vast/home/mina/testing_MCM/build/main_cupy_208","/vast/home/mina/testing_MCM/build/main_cupy_224",] #(x,y)=176,192,208,224
PROF_METRICS = "gld_transactions,gst_transactions,global_hit_rate,local_hit_rate"



# KERNEL NAMES 
SGEMM_CUPY = "cupy_matmult"
TEE_ACM = "tee_acm"
CUBLASS_SGEMM_TREE = "cublas_with_tree"
CUBLASS_SGEMM = "cublas_no_tree"
#COMPUTE_ALL= "compute_all"
#TOTAL_TIME= "total_execution"


#############################################################################
# Dictionary Keys (constants)
#############################################################################
PROFILED_EXEC_TIME = "nvrof_exec_time"
COMPUTATIONS = "COMP_number"
DATA_TRANSFERS = "DT_number"
GRID_SIZE = "grid_size"
BLOCK_SIZE = "block_size"
REGS = "registers"
SSMem = "SSMem"
DSMem = "DSMem"

EXPECTED_GLOBAL_LD = "exp_gld_transactions"
EXPECTED_GLOBAL_ST = "exp_gst_transactions"
GLOBAL_LD = "gld_transactions"
GLOBAL_ST = "gst_transactions"
EXPECTED_EXEC_TIME = "exp_exec_time [ms]"
NVML_ENERGY        = "avg_nvml_energy [mJ]"
TOTAL_ENERGY = "total_energy [mJ]"
TOTAL_TIME = "total_time [ms]"
CREATION_TREE_TIME = "creation_tree_time"
TOTAL_CREATION_TIME = "total_creation_time"
SIZES = "matrix_sizes"
ORDER = "computation_order"

#############################################################################
# Helper functions
#############################################################################
#
# err_print : print error on stderr
#
class TERMINAL_COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def err_print(*args, **kwargs):
    print(TERMINAL_COLORS.FAIL + str(*args) + TERMINAL_COLORS.ENDC, file=sys.stderr, **kwargs)

def debug_print(*args, **kwargs):
    print(TERMINAL_COLORS.WARNING + str(*args) + TERMINAL_COLORS.ENDC, file=sys.stderr, **kwargs)

def info_print(*args, **kwargs):
    print(TERMINAL_COLORS.OKBLUE + str(*args) + TERMINAL_COLORS.ENDC, file=sys.stderr, **kwargs)

