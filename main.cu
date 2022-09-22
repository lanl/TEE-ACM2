// Include C++ header files.
#include <iostream>

// Include local CUDA header files.
#include "launch_kernel.cuh"
#include "nvmlPower.hpp"
#include "cmd_line_parser.hpp"
#include "cuda_config.cuh"
#include "mat_tree.hpp"

enum kernel_choice{
    tee_acm_strat=0,
    cublass=1,
    cublass_no_tree=2,
};
#define START 8
#define STOP 24

void gen_sizes(std::vector<int> &P, int n, int run){
    srand(n+run+5);
    switch(run){
        case(4): //random sizes
            for(int i=0;i<=n;i++){
                P.push_back((int)1024*(rand() % ((STOP - START +1)) + START));
            }
        break;
        case(0): //increasing sizes
            for(int i=0;i<=n;i++){
                if(n==12)
                    P.push_back((int)1024*(8+i));
                else if (n==6)
                    P.push_back((int)1024*(8+2*i));
                else if (n==15)
                    P.push_back((int)1024*(8+i));
                else if (n==20)
                    P.push_back((int)1024*(4+i));
            }
        break;
        case(1): //decreasing sizes
            for(int i=0;i<=n;i++){
                if(n==12)
                    P.push_back((int)1024*(20-i));
                else if (n==6)
                    P.push_back((int)1024*(20-2*i));
                else if (n==15)
                    P.push_back((int)1024*(23-i));
                else if (n==20)
                    P.push_back((int)1024*(24-i));
            }
        break;
    }
}

int main(int argc, char *argv[]) {
    bool use_tree =false;
    int n=10,kernel=2,run;
    std::vector<int> P;
    
    //MatChainEnv <float,false> env_Cu;
    /* Parse cmd line arguments*/
    CmdLineParser cmd_parser(argc, argv);
    cmd_parser.check_required_options();
    std::cout.precision(4);
    cudaSetDevice(0);
    /* Get cmd line arguments*/
    if(argc==7){ 
         n = cmd_parser.get_int("-n");
         kernel = cmd_parser.get_int("-k");
         run = cmd_parser.get_int("-run");
         kernel == 2?use_tree=false:use_tree=true;
    } //./main -run 4 -n 12 -k 0

    MatChainEnv <float,false> env_MCM;
    gen_sizes(P,n,run);
    int cpu_validate = 0;
    std::string e_out = "";
    if(cmd_parser.opt_exist("-e_out")) e_out = cmd_parser.get_str("-e_out");
    if(cmd_parser.opt_exist("-valid_res")) cpu_validate = cmd_parser.get_int("-valid_res");

    // MCM KeepR TREE
    
    auto begin_create = std::chrono::high_resolution_clock::now();
    env_MCM={P,kernel, use_tree};
    auto end_create = std::chrono::high_resolution_clock::now();
    if(!e_out.empty()) nvmlAPIRun(e_out.c_str());
    auto elapsed_creation = std::chrono::duration_cast<std::chrono::nanoseconds>(end_create - begin_create);
    std::cout << "# Kernel name : "<< get_kernel_name(kernel)<<std::endl;
    std::cout <<std::fixed << std::setprecision(4);
    std::cout <<"# Entire creation [ms]: "<< elapsed_creation.count()* 1e-6 << std::endl;
    
    //computation
    auto begin = std::chrono::high_resolution_clock::now();
    env_MCM.compute_sequence();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_traversal= std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    std::cout<<std::endl;
    
    std::cout <<std::fixed << std::setprecision(4);
    std::cout << "# Kernel name : "<< get_kernel_name(kernel)<<std::endl;
    std::cout <<"# Traversal time [ms]: "<< elapsed_traversal.count()* 1e-6 << std::endl;
    
    if(!e_out.empty()) {
        nvmlAPIEnd();
        std::cout<<"Power consumption written to \033[33m'"<<e_out<<"'\033[39m"<<std::endl;
    }
    if(cpu_validate) 
        env_MCM.validate_results_on_cpu(); 
    return 0;
}
