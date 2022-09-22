#ifndef __CMD_LINE_PARSER__
#define __CMD_LINE_PARSER__
class CmdLineParser{
    private:
        std::vector <std::string> options;
        std::vector <std::string> required_opts;
        std::string prog_name;
    public:
        CmdLineParser (int &argc, char **argv){
            //required_opts = {"-x","-z","-y", "-yBlockDim", "-xBlockDim"};
            required_opts = {"-n"};
            prog_name = argv[0];
            for (int i=1; i < argc; ++i)
                this->options.push_back(std::string(argv[i]));

            if(this->opt_exist("-h")){
                this->show_help();
                exit(0);
            }
        }
        // specify requirements
        CmdLineParser (int &argc, char **argv, std::vector <std::string> & req_opts){
            //required_opts = {"-x","-z","-y", "-yBlockDim", "-xBlockDim"};
            required_opts = req_opts;
            prog_name = argv[0];
            for (int i=1; i < argc; ++i)
                this->options.push_back(std::string(argv[i]));

            if(this->opt_exist("-h")){
                this->show_help();
                exit(0);
            }
        }

        void check_required_options(){
            for(const auto & req_opt : required_opts){
                if(!opt_exist(req_opt)){
                    std::cout<<"Required option missing : "<<req_opt<<std::endl;
                    std::cout<<"\tHelp: "<<prog_name<<" -h"<<std::endl;
                    exit(1);
                }
            }
        }
        const std::string & get_str(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->options.begin(), this->options.end(), option);
            if (itr != this->options.end() && ++itr != this->options.end()){
                return *itr;
            }
            static const std::string dummy_str("");
            return dummy_str;
        }

        int get_int(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->options.begin(), this->options.end(), option);
            if (itr != this->options.end() && ++itr != this->options.end()){
                try{
                    return stoi(*itr);
                }
                catch(...){
                    std::cout<<"Option '"<<option<<"' must be an integer "<<std::endl;
                    exit(1);
                }
            }
            return 0;
        }

        bool opt_exist(const std::string &option) const{
            return std::find(this->options.begin(), this->options.end(), option)
                   != this->options.end();
        }

        void show_help_SMM(){
            std::cout<<"# Single Matrix Multiplication : R = A*B "<<std::endl;
            std::cout<<"# Usage : "<<prog_name<<" -P0 <int> -P1 <int> -P2 <int> -x <int> -z <int> -y <int> -e_out <output_file> -valid_res <bool>"<<std::endl;
            std::cout<<"\t* P0,P1,P2 matrix sizes: A(P0,P1) - B(P1,P2)"<<std::endl;
            std::cout<<"\t* x,z,y shared memory sizes: (x,z) for A - (z,y) for B - (x,y) for R"<<std::endl;
            std::cout<<"\t* yBlockDim,xBlockDim : CUDA block sizes " <<std::endl;
            std::cout<<"\t* e_out : output file name for power consumption (no power mesurements if empty)" <<std::endl;
            std::cout<<"\t* valid_res (0|1) : validate result on CPU (default : 0) " <<std::endl;
            std::cout<<"\t* Restrictions :" <<std::endl;
            std::cout<<"\t\t -z must be a multiple of 8 " <<std::endl;
            //std::cout<<"\t\t -x must be a multiple of -yBlockDim" <<std::endl;
            //std::cout<<"\t\t -y must be a multiple of -xBlockDim" <<std::endl;
        }

        void show_help(){
            std::cout<<"# Matrix Chain Multiplication : R = A1*A2*...An "<<std::endl;
            std::cout<<"# Usage : "<<prog_name<<" -n <int> -k <int> -run <int>"<<std::endl;
            std::cout<<"\t* n number of matrices "<<std::endl;
            std::cout<<"\t* k kernel to run: 0 for TEE-ACM2 - 1 for cuBLAS using OP_Count Tree - 2 for cuBLAS in order"<<std::endl;
            std::cout<<"\t* run parameter to generate the sizes: 0 for increasing sizes - 1 for decreasing sizes - 4 for random sizes" <<std::endl;
            std::cout<<"\t* Restrictions :" <<std::endl;
            std::cout<<"\t\t sizes generation function is written for 6, 12, 15 or 20 matrices " <<std::endl;
        }
};

#endif