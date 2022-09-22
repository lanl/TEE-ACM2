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
#CUDA_MCM_EXE = "/vast/home/morarum/cuda_training/CUDA_Chain_Matrix_Mult/BUILD/mat_test_opt"
#CUDA_MCM_EXE = "/vast/home/morarum/cuda_training/CUDA_Chain_Matrix_Mult/BUILD/mat_test_cublass"
CUDA_MCM_EXE = "/vast/home/morarum/cuda_training/CUDA_Chain_Matrix_Mult/BUILD/mat_test_cupy"
PROF_METRICS = "gld_transactions,gst_transactions,global_hit_rate,local_hit_rate,local_load_transactions,local_store_transactions,shared_store_transactions,shared_load_transactions"

# KERNEL NAMES 
OPT_KEEP_R = "mat_mult_shared_mem_opt_keep_R"
CUBLASS_SGEMM = "sgemm"
OPT_KEEP_R_v3 = "mat_mult_shared_mem_opt_keep_R_v3"
OPT_KEEP_R_v3_small_tiles = "small_tiles_mat_mult_shared_mem_opt_keep_R_v3"
OPT_KEEP_R_v3_large_tiles = "large_tiles_mat_mult_shared_mem_opt_keep_R_v3"
CUSTOM_OPT_KEEP_R_v3 = "custom_mat_mult_shared_mem_opt_keep_R_v3"
CUPY_ORIGINAL = "cupy_mm_original"
CUPY_SHARED_MEM = "cupy_mm_shared_mem"
CUPY_ROW_MAJOR = "cupy_mm_row_major"
CUPY_SGEMM = "cupy_mm"
TEE_ACM = "tee_acm"

FUSED_CUBLAS="FUSED_CUBLAS"
FUSED_ACM="FUSED_CUPY"
#############################################################################
# Dictionary Keys (constants)
#############################################################################
PROFILED_EXEC_TIME = "nvrof_exec_time"
GRID_SIZE = "grid_size"
BLOCK_SIZE = "block_size"
REGS = "registers"
SSMem = "SSMem"
DSMem = "DSMem"

EXPECTED_GLOBAL_LD = "exp_gld_transactions"
EXPECTED_GLOBAL_ST = "exp_gst_transactions"
EXPECTED_EXEC_TIME = "avg exec_time [s]"
MIN_EXEC_TIME = "min exec_time [s]"
MAX_EXEC_TIME = "max exec_time [s]"
NVML_ENERGY        = "avg nvml_energy [mJ]"
MIN_NVML_ENERGY        = "min nvml_energy [mJ]"
MAX_NVML_ENERGY        = "max nvml_energy [mJ]"
POWER = "power [W]"

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

