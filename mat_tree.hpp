#ifndef __MAT_TREE_H__
#define __MAT_TREE_H__

#include <array>
#include <chrono>
#include <string> 

#include "launch_kernel.cuh"
#include "dmatrix.hpp"

constexpr int VALUE_MAX = 100;
constexpr int SIZE_MINI = 5;
constexpr int SIZE_MAXI = 4*SIZE_MINI;
constexpr int MULTIPLYER = 1024;

enum fusion 
{   def = -1, 
    l_fused = 3, 
    r_fused = 4,
    p_fused = 0,
};

template< typename T>
struct Node {
    std::vector< T > * mat;
    DeviceMatrix<T> * d_mat;
    std::pair <int,int> seq;
    std::pair <int,int> size;
    int parent,child;
    int fuse;
    int visited;
};

/* Random Generator */
template< typename T>
struct RandomGenerator {
    int maxValue;
    RandomGenerator(int max) :maxValue(max) {}
    //T operator()() {return static_cast<T>(rand() % maxValue);}
    T operator()() {
    return static_cast <T> (rand()) / (static_cast <T> (RAND_MAX/maxValue));}
};
/* Unique Generator */
template< typename T>
struct ValueGenerator {
    int val;
    ValueGenerator(int v) :val(v) {}
    T operator()() {return static_cast<T>(val);}
    //return static_cast <T> (rand()) / (static_cast <T> (RAND_MAX/maxValue));
};


template <typename T, bool FUSE = false>
class MatChainEnv{
    private:
        std::vector< std::vector<T> > mat_chain;
        std::vector< DeviceMatrix<T> > d_mat_chain;
        std::vector<Node<T>> tree;
        std::vector<int> P;
        std::vector<unsigned long long>  opcount,  opdm, opdm_fused;
        std::vector<int> split, fusion;
        int nb_mat;
        bool use_tree;
        //int kernel;
        std::string compute_order="";
        int seq_size;
        SingleMatMultKernel kernel_to_launch;

 /*Function for Tree*/
    public:
        MatChainEnv()=default;
        __host__ MatChainEnv(int n, int k ,bool t): 
            nb_mat(n),use_tree(t),
            kernel_to_launch(static_cast<SingleMatMultKernel>(k)),seq_size(0)
            {
            int tree_size;
            int size = nb_mat+1;
		
            srand (time(NULL));

            P.resize(size);
            std::generate(P.begin(),P.end(),[](){
                static_cast<int>(rand() % (SIZE_MAXI - SIZE_MINI +1))+SIZE_MINI; 
            }); 
            init(size,tree_size); 
        }

        __host__ MatChainEnv(std::vector<int> sizes, int k, bool t): 
            nb_mat(sizes.size()-1),use_tree(t),
            kernel_to_launch(static_cast<SingleMatMultKernel>(k)),seq_size(0)
            {
            int tree_size;
            int size=nb_mat+1;

            srand (time(NULL));

            P.reserve(size);
            P.insert(P.end(), sizes.begin(), sizes.end());
            init(size,tree_size); 
        }

        __host__ MatChainEnv(std::initializer_list<int> sizes,  int k, bool t) : 
            nb_mat(sizes.size()-1),use_tree(t),
            kernel_to_launch(static_cast<SingleMatMultKernel>(k)),seq_size(0)
            {
            int tree_size;
            int size=nb_mat+1;
            std::cout<<nb_mat<<std::endl;

		    
            srand (time(NULL));

            P.reserve(size);
            P.insert(P.end(), sizes.begin(), sizes.end());
            init(size,tree_size);
        }

        void init(const int size, int& tree_size){
            //matrix generation
            auto begin_matrix = std::chrono::high_resolution_clock::now();
            mat_chain.reserve(2*nb_mat);
            d_mat_chain.reserve(2*nb_mat);
            
            mat_chain.emplace_back(P[0]*P.back(),0);//empty or R
            d_mat_chain.emplace_back(P[0], P.back());//empty or R
            
            for(int i=0 ; i<size-1; i++){
                int n_row = P[i];
                int n_col  = P[i+1];            
                mat_chain.emplace_back(std::vector<T>(n_row*n_col,0)); 
                generate(mat_chain[i+1].begin(), mat_chain[i+1].end(), RandomGenerator<T>(VALUE_MAX));
                //set context on device
                d_mat_chain.emplace_back(n_row,n_col);
            }

            auto end_matrix = std::chrono::high_resolution_clock::now();
            auto elapsed_matrix = std::chrono::duration_cast<std::chrono::nanoseconds>(end_matrix - begin_matrix);
            if constexpr (DISPLAY){
                std::cout <<std::fixed << std::setprecision(4);
                std::cout << "# Kernel name : "<< get_kernel_name(kernel_to_launch)<<std::endl;
                std::cout<<"# Matrix Creation [ms]: "<<elapsed_matrix.count()* 1e-6<<std::endl;
            }
                       
            //print_op(f);
            if(use_tree){
                //  TREE

                auto begin_create = std::chrono::high_resolution_clock::now();

                opcount= std::vector<unsigned long long>(size*size,0);
                split= std::vector<int>(size*size,0);
                opdm= std::vector<unsigned long long>(size*size,0);

                if constexpr (FUSE){
                    opdm_fused=std::vector<unsigned long long>(size*size,0);
                    fusion=std::vector<int>(size*size);
                }
                
                compute_op();

                //tree creation
                tree_size=2*nb_mat-1;
                tree.reserve(tree_size);
                tree_Creation();

                auto end_create = std::chrono::high_resolution_clock::now();
                auto elapsed_creation = std::chrono::duration_cast<std::chrono::nanoseconds>(end_create - begin_create);
                if constexpr (DISPLAY){
                    std::cout << "# Kernel name : "<< get_kernel_name(kernel_to_launch)<<std::endl;
                    std::cout<<"# Creation Tree [ms]: "<<elapsed_creation.count()* 1e-6<<std::endl;
                }
                
            }
            else{
                if constexpr (DISPLAY){
                    std::cout << "# Kernel name : "<< get_kernel_name(kernel_to_launch)<<std::endl;
                    std::cout<<"# Creation Tree [ms]: 0.00" <<std::endl;
                }
                //reserve temporary matrix sizes
                for(int i=1;i<nb_mat;i++){
                    mat_chain.emplace_back(std::vector<T>(P[0]*P[i+1],0));
                    d_mat_chain.emplace_back(P[0], P[i+1]);

                    printf("size = %d \n",d_mat_chain.size());
                }

            }
        }

        void compute_sequence(){
            nvmlReturn_t nvmlTotalResult;
            nvmlDevice_t nvmlDeviceID;
            unsigned long long energy_start;
            unsigned long long energy_end;
            unsigned long long energy_mean = 0;

            if constexpr (DISPLAY){
                std::cout<<std::endl<<"# Kernel name : "<<get_kernel_name(kernel_to_launch)<<std::endl;
                std::cout<< std::defaultfloat <<"# Matrix Sizes : ";
            }
            print_sizes();
            //std::cout<<"# Sequence Size : "<<seq_size<<std::endl;
            if(use_tree)
                get_order();
            else{
                compute_order.append(std::to_string(1));
                compute_order.append(".");
                compute_order.append(std::to_string(2));
                compute_order.append(" ");
                for(int i=2;i<nb_mat;i++){
                    compute_order.append(std::to_string(1));
                    compute_order.append(".");
                    compute_order.append(std::to_string(i+1));
                    compute_order.append(" ");
                }
            }
            if constexpr (DISPLAY){
            std::cout<<"# Sequence order : "<<compute_order<<std::endl;
            }
            nvmlTotalResult = nvmlInit();
            if (NVML_SUCCESS != nvmlTotalResult)
            {
                printf("NVML Init fail: %s\n", nvmlErrorString(nvmlTotalResult));
                exit(0);
            }
            nvmlTotalResult = nvmlDeviceGetHandleByIndex(0, &nvmlDeviceID);
            if (NVML_SUCCESS != nvmlTotalResult)
            {
                printf("Failed to get handle for device %d: %s\n", 0, nvmlErrorString(nvmlTotalResult));
                exit(1);
            } 
            nvmlTotalResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_start);
            if(use_tree){
                
                tree_traversal();
            }
            else{
                //compute sequence in order
                compute_SMM(1,2,nb_mat+1);//compute A1xA2, put res in
                /*
                compute_order.append(std::to_string(1));
                compute_order.append(".");
                compute_order.append(std::to_string(2));
                compute_order.append(" ");*/
                seq_size++;
                for(int i=2;i<nb_mat;i++){
                    /*
                    compute_order.append(std::to_string(1));
                    compute_order.append(".");
                    compute_order.append(std::to_string(i+1));
                    compute_order.append(" ");*/
                    seq_size++;
                    compute_SMM(nb_mat+i-1,i+1,nb_mat+i);//compute A1xA2, put res in
                }
            }
            nvmlTotalResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID,&energy_end);
            energy_mean += (energy_end - energy_start);
            if constexpr (DISPLAY){
                std::cout << "# Kernel name : "<< get_kernel_name(kernel_to_launch)<<std::endl;
                std::cout<<"# Traversal energy [mJ]: "<< energy_mean <<std::endl;
            }
        }

        void compute_SMM(int a, int b, int r){
            //set context on device -> d_mat[a] & d_mat[b]
            d_mat_chain[a].allocate_GPU(mat_chain[a].data());
            d_mat_chain[b].allocate_GPU(mat_chain[b].data());
            d_mat_chain[r].allocate_GPU(mat_chain[r].data());

            //call kernel
            if(kernel_to_launch==0) //use tee_acm
                SingleMatMult::launch_kernel(d_mat_chain[a], d_mat_chain[b], d_mat_chain[r], TEE_ACM);
            else //use cublass no tree
                SingleMatMult::launch_kernel(d_mat_chain[a], d_mat_chain[b], d_mat_chain[r],CUBLAS_SGEMM);
            cuda_err_check(cudaFree(d_mat_chain[a].get_raw_pointer()));
            cuda_err_check(cudaFree(d_mat_chain[b].get_raw_pointer()));
            //get results
            get_res_from_device(r);
            cuda_err_check(cudaFree(d_mat_chain[r].get_raw_pointer()));
        }

        void compute_FMM(int a, int b, int c, int r){
            //set context on device -> d_mat[a] & d_mat[b]
            d_mat_chain[a].allocate_GPU(mat_chain[a].data());
            d_mat_chain[b].allocate_GPU(mat_chain[b].data());
            d_mat_chain[c].allocate_GPU(mat_chain[c].data());
            d_mat_chain[r].allocate_GPU(mat_chain[r].data());

            //call kernel
            SingleMatMult::launch_kernel(d_mat_chain[a], d_mat_chain[b], d_mat_chain[c], d_mat_chain[r], FUSED_TEE_ACM);
            cuda_err_check(cudaFree(d_mat_chain[a].get_raw_pointer()));
            cuda_err_check(cudaFree(d_mat_chain[b].get_raw_pointer()));
            cuda_err_check(cudaFree(d_mat_chain[c].get_raw_pointer()));
            //get results
            get_res_from_device(r);
            cuda_err_check(cudaFree(d_mat_chain[r].get_raw_pointer()));
        }

        void compute_node(int r){ //to see later
            seq_size++;
            int i=tree[r].seq.first, j=tree[r].seq.second;
            if constexpr (! FUSE){
                int left=tree[r].child, right=left+1;
                tree[left].d_mat->allocate_GPU(tree[left].mat->data());
                tree[right].d_mat->allocate_GPU(tree[right].mat->data());
                tree[r].d_mat->allocate_GPU(tree[r].mat->data());
                if(kernel_to_launch==0) //use tee_acm
                    SingleMatMult::launch_kernel(*(tree[left].d_mat), *(tree[right].d_mat), *(tree[r].d_mat), TEE_ACM);
                else //use cublass
                    SingleMatMult::launch_kernel(*(tree[left].d_mat), *(tree[right].d_mat), *(tree[r].d_mat), CUBLAS_SGEMM_TREE);
                cuda_err_check(cudaFree(tree[left].d_mat->get_raw_pointer()));
                cuda_err_check(cudaFree(tree[right].d_mat->get_raw_pointer()));

                get_res_from_device(r);
                cuda_err_check(cudaFree(tree[r].d_mat->get_raw_pointer()));
            }
            else{//use fusion
                if(tree[r].fuse==p_fused){ //no_fuse -> normal kernel
                    int left=tree[r].child, right=left+1;
                    //set context on device -> d_mat[a] & d_mat[b]
                    //tree[left].d_mat & tree[right].d_mat
                    tree[left].d_mat->allocate_GPU(tree[left].mat->data());
                    tree[right].d_mat->allocate_GPU(tree[right].mat->data());
                    tree[r].d_mat->allocate_GPU(tree[r].mat->data());
                    if(kernel_to_launch==0) //use tee_acm
                        SingleMatMult::launch_kernel(*(tree[left].d_mat), *(tree[right].d_mat), *(tree[r].d_mat), TEE_ACM);
                    else //use cublass
                        SingleMatMult::launch_kernel(*(tree[left].d_mat), *(tree[right].d_mat), *(tree[r].d_mat), CUBLAS_SGEMM_TREE);
                    cuda_err_check(cudaFree(tree[left].d_mat->get_raw_pointer()));
                    cuda_err_check(cudaFree(tree[right].d_mat->get_raw_pointer()));
                }
                else {
                    int mat_a, mat_b, mat_c;
                    if(tree[r].fuse==l_fused){
                        mat_a= tree[tree[r].child].child;
                        mat_b= mat_a+1;
                        mat_c= tree[r].child +1;
                    }
                    else if(tree[r].fuse==r_fused){
                        mat_a= tree[r].child;
                        mat_b= tree[mat_a+1].child;
                        mat_c= mat_b +1;
                    }
                    tree[mat_a].d_mat->allocate_GPU(tree[mat_a].mat->data());
                    tree[mat_b].d_mat->allocate_GPU(tree[mat_b].mat->data());
                    tree[mat_c].d_mat->allocate_GPU(tree[mat_c].mat->data());
                    tree[r].d_mat->allocate_GPU(tree[r].mat->data());
                    if(kernel_to_launch==7){//use acm
                        SingleMatMult::launch_kernel(*(tree[mat_a].d_mat),*(tree[mat_b].d_mat),*(tree[mat_c].d_mat),*(tree[r].d_mat),static_cast<SingleMatMultKernel>(tree[r].fuse) );
                    }
                    else {//use cublass
                        SingleMatMult::launch_kernel(*(tree[mat_a].d_mat),*(tree[mat_b].d_mat),*(tree[mat_c].d_mat),*(tree[r].d_mat), static_cast<SingleMatMultKernel>(tree[r].fuse +2) );
                    }
                    cuda_err_check(cudaFree(tree[mat_a].d_mat->get_raw_pointer()));
                    cuda_err_check(cudaFree(tree[mat_b].d_mat->get_raw_pointer()));
                    cuda_err_check(cudaFree(tree[mat_c].d_mat->get_raw_pointer()));

                }
                get_res_from_device(r);
                cuda_err_check(cudaFree(tree[r].d_mat->get_raw_pointer()));
            }
        }

        /* getters */
        
        inline DeviceMatrix<T> & get_dev_mat(const int & i){return d_mat_chain[i];}
        inline DeviceMatrix<T> & get_dev_res_mat(){return d_mat_chain[nb_mat+1];}
        void get_res_from_device(int r){ 
            if(use_tree)
                tree[r].d_mat->copy_to_host(tree[r].mat->data());
            else
                d_mat_chain[r].copy_to_host(mat_chain[r].data());
        }

        /* Tree Creation*/

        void tree_Creation(){
            const int size=nb_mat+1;
            int empty=0;
            int r=0,k;

            create_Root(empty++, 1, nb_mat);
            k=split[1 *(size) + nb_mat];
            if constexpr (FUSE){
                if(tree[r].fuse==l_fused){
                    fusion[tree[r].seq.first * size + k ]=def;
                }
            }
            create_Node(empty++,r, 1, k);
            if constexpr (FUSE){
                if(tree[r].fuse==r_fused){
                    fusion[(k+1)* size + tree[r].seq.second ]=def;
                }
            }
            create_Node(empty++,r, k+1, nb_mat);

            r=gotochild(r);

            while(r!=0){
                if(isvisited(r)){
                    tree[r].visited=0;
                    r = isleft(r)?gotosibling(r):gotoparent(r);
                }
                else if(isleaf(r)){
                    r = isleft(r)?gotosibling(r):gotoparent(r);
                }
                else{
                    k=split[tree[r].seq.first *(size) + tree[r].seq.second];
                    if constexpr (FUSE){
                        if(tree[r].fuse==l_fused){
                            fusion[tree[r].seq.first * size + k ]=def;
                        }
                    }
                    create_Node(empty++,r,tree[r].seq.first,k);
                    if constexpr (FUSE){
                        if(tree[r].fuse==r_fused){
                            fusion[(k+1)* size + tree[r].seq.second ]=def;
                        }
                    }
                    create_Node(empty++,r,k+1,tree[r].seq.second);

                    tree[r].visited=1;
                    r=gotochild(r);
                }
            }
            
            for(int i=0;i<2*nb_mat-1;i++){
                tree[i].visited=0;
            }
        }

        void create_Root(int r, int i, int j){
            if constexpr (FUSE){
                tree[r].fuse=fusion[i*(nb_mat+1)+j];
            }
            tree[r].visited=0;
            tree[r].seq.first=i;
            tree[r].seq.second=j;
            tree[r].size.first=P[i-1];
            tree[r].size.second=P[j];
            //assign matrix
            mat_chain.emplace_back(std::vector<T>(tree[r].size.first*tree[r].size.second,0)); 
            d_mat_chain.emplace_back(tree[r].size.first, tree[r].size.second);
            tree[r].mat=&mat_chain.back();
            tree[r].d_mat=&d_mat_chain.back();
            
        }

        void create_Node(int r, int father, int i, int j){
            tree[r].parent=father;
            if(isleft(r)){
                tree[father].child=r; 
            }
            if constexpr (FUSE){
                tree[r].fuse=fusion[i*(nb_mat+1)+j];
            }
            tree[r].visited=0;
            tree[r].seq.first=i;
            tree[r].seq.second=j;
            tree[r].size.first=P[i-1];
            tree[r].size.second=P[j];
            
            //assign matrix
            if(i!=j){
                mat_chain.emplace_back(std::vector<T>(tree[r].size.first*tree[r].size.second,-1));
                d_mat_chain.emplace_back(tree[r].size.first, tree[r].size.second);
                tree[r].mat=&mat_chain.back(); 
                tree[r].d_mat=&d_mat_chain.back();
                //generate(tree[r].mat->begin(), tree[r].mat->end(), ValueGenerator<T>(r+1));
                
            }
            else{
                //if i==j -> leaf, created at first in order 1,2,,,n
                tree[r].mat=&mat_chain[i];
                tree[r].d_mat=&d_mat_chain[i];
                tree[r].fuse=def;
            }
            
        }

        void print_info_Node(int r){
            int i=tree[r].seq.first, j=tree[r].seq.second;
            int k=split[i*(nb_mat+1)+j];
            std::cout <<"!!!!!!!!!! Node "<< r <<" !!!!!!!!!"<<std::endl;
            if(i!=0)
                std::cout<<"parent "<<tree[r].parent<<std::endl;
            //if(i!=j && isleft(r))
                std::cout<<"child "<<tree[r].child<<std::endl;
            std::cout<<"visited "<<tree[r].visited<<std::endl;
            std::cout<<"seq "<<tree[r].seq.first<<","<<tree[r].seq.second<<std::endl;
            std::cout<<"size "<<tree[r].size.first<<"x"<<tree[r].size.second<<std::endl;
            //find matrix
        }

        void print_Node(int r){
            int i=tree[r].seq.first, j=tree[r].seq.second;
            int k=split[i*(nb_mat+1)+j];
            std::cout <<"Node "<< r<<" at value "<<tree[r].mat[0][0];
            if (i==j){
                std::cout <<" is a leaf."<<std::endl;
                std::cout <<"Matrix ("<< i;
                std::cout <<","<< j<<") of size"<< tree[r].size.first << "x"<< tree[r].size.second<<std::endl<<std::endl;
            }
            else{
                std::cout <<" is a T-matrix("<<i<<","<<j<<") of size "<< tree[r].size.first << "x"<< tree[r].size.second <<std::endl;
                std::cout <<"It will compute Matrix ("<<i<<","<<k<<")";
                std::cout <<"and Matrix ("<<k+1<<","<<j<<")"<<std::endl<<std::endl;
            }
        }

        void print_Node(int r, int f){
            int i=tree[r].seq.first, j=tree[r].seq.second;
            int k=split[i*(nb_mat+1)+j];
            int k1,k2;
            std::cout <<"Node "<< r<<" at value "<<tree[r]->mat[0];
            if (i==j){
                std::cout <<" is a leaf."<<std::endl;
                std::cout <<"Matrix ("<< i;
                std::cout <<","<< j<<") of size"<< tree[r].size.first << "x"<< tree[r].size.second<<std::endl<<std::endl;
            }
            else if (f==3){ //no fuse kernel
                std::cout <<" is a T-matrix("<<i<<","<<j<<") of size "<< tree[r].size.first << "x"<< tree[r].size.second <<std::endl;
                std::cout <<"It will compute Matrix ("<<i<<","<<k<<")";
                std::cout <<"and Matrix ("<<k+1<<","<<j<<")"<<std::endl<<std::endl;
            }
            else if (f==1){ //left fuse kernel
                k1=split[i*(nb_mat+1)+k];
                std::cout <<" is a T-matrix("<<i<<","<<j<<") of size "<< tree[r].size.first << "x"<< tree[r].size.second <<std::endl;
                std::cout <<"It will compute ( Matrix ("<<i<<","<<k1<<")";
                std::cout <<", Matrix ("<<k1+1<<","<<k<<") )";
                std::cout <<", then Matrix ("<<k+1<<","<<j<<")";
                std::cout <<" with the LEFT KERNEL"<<std::endl<<std::endl;
            }
            else{  //right fuse kernel
                k2=split[(k+1)*(nb_mat+1)+j];
                std::cout <<" is a T-matrix("<<i<<","<<j<<") of size "<< tree[r].size.first << "x"<< tree[r].size.second <<std::endl;
                std::cout <<"It will compute Matrix ("<<i<<","<<k<<")";
                std::cout <<", then ( Matrix ("<<k+1<<","<<k2<<")";
                std::cout <<"and Matrix ("<<k2+1<<","<<j<<"))";
                std::cout <<" with the RIGHT KERNEL"<<std::endl<<std::endl;
            }
        }

        bool isleaf(int r){return (tree[r].seq.first==tree[r].seq.second);}
        bool isvisited(int r){return (tree[r].visited);}
        bool isleft(int r){return (r%2);}
        bool isright(int r){return (r%2==0);}
        int gotoparent(int r){return (tree[r].parent);}
        int gotosibling(int r){return (r+1);}//right sibling
        int gotochild(int r){return (tree[r].child);}//left child
        
        

        void get_order(){
            int r=0;
            r=gotochild(r);
            int i,j;
            while(r!=0){
                if(isleaf(r)){ 
                    r = isleft(r)?gotosibling(r):gotoparent(r);
                }
                else if(!isvisited(r)){
                    tree[r].visited=1;
                    r=gotochild(r);
                }
                else{//visited -> compute (except if fuse=0)
                    tree[r].visited=0;
                    if constexpr (FUSE){
                        if(tree[r].fuse !=def){
                            i=tree[r].seq.first;
                            j=tree[r].seq.second;
                            compute_order.append(std::to_string(i));
                            compute_order.append(".");
                            compute_order.append(std::to_string(j));
                            compute_order.append(" ");
                        }
                    }
                    else{
                        i=tree[r].seq.first;
                        j=tree[r].seq.second;
                        compute_order.append(std::to_string(i));
                        compute_order.append(".");
                        compute_order.append(std::to_string(j));
                        compute_order.append(" ");
                    }
                    if(isright(r)){
                        r=gotoparent(r);
                    }
                    else{
                        r=gotosibling(r);
                    }
                }
            }
            compute_order.append(std::to_string(1));
            compute_order.append(".");
            compute_order.append(std::to_string(nb_mat));
            compute_order.append(" ");
        }

        void tree_traversal(){
            int r=0;
            r=gotochild(r);
            while(r!=0){
                if(isleaf(r)){ 
                    r = isleft(r)?gotosibling(r):gotoparent(r);
                }
                else if(!isvisited(r)){
                    tree[r].visited=1;
                    r=gotochild(r);
                }
                else{//visited -> compute (except if fuse=def)
                    if constexpr (FUSE){
                        if(tree[r].fuse !=def){
                            compute_node(r);
                        }
                    }
                    else{
                        compute_node(r);
                    }
                    if(isright(r)){
                        r=gotoparent(r);
                    }
                    else{
                        r=gotosibling(r);
                    }
                }
            }
            compute_node(r);
        }


        /*Function for OP, split and fusion*/
        void compute_op(){
            compute_opcount();
            compute_opdm();
            if constexpr (FUSE){
                compute_opdm_fusion();
            }
        }

        void compute_opcount(){
            unsigned long long min_count;
            int j,k;
            int size=nb_mat+1;
            if(use_tree){
                for(int l=1;l<nb_mat;l++){
                    for(int i=1;i<=nb_mat-l;i++){
                        j=i+l;
                        k=i;
                        
                        min_count=1;
                        min_count*= P[i-1]/MULTIPLYER;
                        min_count*=P[k]/MULTIPLYER;
                        min_count*=P[j]/MULTIPLYER;
                        min_count+= opcount[i*size+k] ;
                        min_count+= opcount[(k+1)*size+j] ;
                    
                        opcount[i*size+j]=min_count;
                        split[i*size+j]=k;
                        for(k=i+1;k<j;k++){
                            min_count=1;
                            min_count*= P[i-1]/MULTIPLYER;
                            min_count*=P[k]/MULTIPLYER;
                            min_count*=P[j]/MULTIPLYER;
                            min_count+= opcount[i*size+k] ;
                            min_count+= opcount[(k+1)*size+j] ;
                            if(min_count < opcount[i*size+j]){
                                opcount[i*size+j]=min_count;
                                split[i*size+j]=k;
                            }
                        }
                    }
                }
            }
            else{
                for(int l=1;l<nb_mat;l++){
                    for(int i=1;i<=nb_mat-l;i++){
                        j=i+l;
                        k=i;
                        min_count=1;
                        min_count*= P[i-1]/MULTIPLYER;
                        min_count*=P[k]/MULTIPLYER;
                        min_count*=P[j]/MULTIPLYER;
                        min_count+= opcount[i*size+k] ;
                        min_count+= opcount[(k+1)*size+j] ;
                        opcount[i*size+j]=min_count;
                        split[i*size+j]=k;
                    }
                }

            }
            
        }
        unsigned long long print_nb_computation(){
            return (opcount[1*(nb_mat+1)+nb_mat]*MULTIPLYER);
        }

        void compute_opdm(){
            int j,k,size=nb_mat+1;
            int M=65536;
            unsigned long long tmp;
            for(int l=1;l<nb_mat;l++){
                tmp=0;
                for(int i=1;i<=nb_mat-l;i++){
                    j=i+l;
                    k=split[i*size+j];
                    tmp+=opdm[i*size+k];
                    tmp+=opdm[(k+1)*size+j];
                    tmp+= ((P[j]/MULTIPLYER)/ sqrt(M)) *2 * (P[i-1]/MULTIPLYER) * (P[k]/MULTIPLYER) ;
                    tmp+=(P[i-1]/MULTIPLYER)*(P[j]/MULTIPLYER);
                    opdm[i*size+j]=tmp;
                }
            }
        }

        unsigned long long print_nb_dt(){
            return (opdm[1*(nb_mat+1)+nb_mat]*MULTIPLYER);
        }

        unsigned long long hp(int i, int j, int k, int M){
            unsigned long long tmp=0.f;
            if(i!=k)
                tmp+=(P[i-1]/MULTIPLYER)*(P[k]/MULTIPLYER);
            if(k+1!=j)
                tmp+=(P[k]/MULTIPLYER)*(P[j]/MULTIPLYER);
            tmp+=((P[j]/MULTIPLYER)/sqrt(M)) * 2*(P[i-1]/MULTIPLYER)*(P[k]/MULTIPLYER);
            return(floor(tmp));
        }

        unsigned long long hl(int i, int j, int k,  int k1, int M){
            unsigned long long tmp=0.f;
            float alpha=(P[j]/MULTIPLYER)/(P[k1]/MULTIPLYER);
            float alpha_prime=(1+2*alpha)/(1+alpha);
            if(i!=k1)
                tmp+=(P[i-1]/MULTIPLYER)*(P[k1]/MULTIPLYER);
            if(k1+1!=k)
                tmp+=(P[k1]/MULTIPLYER)*(P[k]/MULTIPLYER);
            if(k+1!=j)
                tmp+=(P[k]/MULTIPLYER)*(P[j]/MULTIPLYER);
            tmp+=((P[k]/MULTIPLYER)/sqrt(M))*2*(P[i-1]/MULTIPLYER)*(P[k1]/MULTIPLYER)*(1+alpha)*sqrt(alpha_prime);
            if(i!=j)
                tmp-=2*(P[i-1]/MULTIPLYER)*(P[j]/MULTIPLYER);
            return(floor(tmp));
        }

        unsigned long long hr(int i, int j, int k,  int k2, int M){
            unsigned long long tmp=0.f;
            float beta=(P[i-1]/MULTIPLYER)/(P[k]/MULTIPLYER);
            float beta_prime=(1+2*beta)/(1+beta);
            if(k+1!=k2)
                tmp+=(P[k]/MULTIPLYER)*(P[k2]/MULTIPLYER);
            if(k2+1!=j)
                tmp+=(P[k2]/MULTIPLYER)*(P[j]/MULTIPLYER);
            if(i!=k)
                tmp+=(P[i-1]/MULTIPLYER)*(P[k]/MULTIPLYER);
            tmp+=((P[j]/MULTIPLYER)/sqrt(M))*2*(P[k]/MULTIPLYER)*(P[k2]/MULTIPLYER)*(1+beta)*sqrt(beta_prime);
            if(i!=j)
                tmp-=2*(P[i-1]/MULTIPLYER)*(P[j]/MULTIPLYER);
            return(floor(tmp));
        }

        void compute_opdm_fusion(){
            int j,k, n=nb_mat, size=n+1;
            unsigned long long min_dm;
            int k1, k2;
            int M=65536;

            for(int l=1;l<n;l++){ //l begins at 2 ?
                for(int i=1;i<=n-l;i++){
                    j=i+l;
                    k=split[i*size+j];
                    if(i==k && k+1==j){ //Fp(i,j)
                        fusion[i*size+j]=p_fused;
                        opdm_fused[i*size+j]=hp(i, j, k, M);
                        opdm_fused[i*size+j]+=opdm_fused[i*size+k];
                        opdm_fused[i*size+j]+=opdm_fused[(k+1)*size+j];
                        
                    }
                    else if(i==k){ //min(Fp(i,j),Fr(i,j)) -> no fuse / right fuse

                        k2=split[(k+1)*size+j];

                        min_dm=hp(i, j, k, M);
                        min_dm+=opdm_fused[i*size+k];
                        min_dm+=opdm_fused[(k+1)*size+j]; //Fp    

                        //if(min_dm < opdm_fused[i*size+j]){
                        opdm_fused[i*size+j]=min_dm;
                        fusion[i*size+j]=p_fused;
                        //}

                        min_dm=hr(i, j, k,k2, M);
                        min_dm+=opdm_fused[i*size+k];
                        min_dm+=opdm_fused[(k+1)*size+k2];
                        min_dm+=opdm_fused[(k2+1)*size+j];//Fr
                        
                        if(min_dm < opdm_fused[i*size+j]){
                            opdm_fused[i*size+j]=min_dm;
                            fusion[i*size+j]=r_fused;
                        }
                    }
                    else if(k+1==j){ //min(Fp(i,j),Fl(i,j)) -> no fuse / left fuse
                        k1=split[i*size+k];

                        min_dm=hp(i, j, k, M);
                        min_dm+=opdm_fused[i*size+k];
                        min_dm+=opdm_fused[(k+1)*size+j];//Fp
                        //if(min_dm < opdm_fused[i*size+j]){
                        opdm_fused[i*size+j]=min_dm;
                        fusion[i*size+j]=p_fused;
                        //}

                        min_dm=hl( i, j, k,k1, M);
                        min_dm+=opdm_fused[i*size+k1];
                        min_dm+=opdm_fused[(k1+1)*size+k];
                        min_dm+=opdm_fused[(k+1)*size+j];//Fl

                        if(min_dm < opdm_fused[i*size+j]){
                            opdm_fused[i*size+j]=min_dm;
                            fusion[i*size+j]=l_fused;
                        }

                    }
                    else{ //min(Fp(i,j),Fr(i,j),Fl(i,j)) -> no fuse / right fuse / left fuse
                        k1=split[i*size+k];
                        k2=split[(k+1)*size+j];

                        min_dm=hp(i, j, k, M); //Fp
                        min_dm+=opdm_fused[i*size+k];
                        min_dm+=opdm_fused[(k+1)*size+j];

                       // if(min_dm < opdm_fused[i*size+j]){
                        opdm_fused[i*size+j]=min_dm;
                        fusion[i*size+j]=p_fused;
                        //}

                        min_dm=hr(i, j, k,k2, M);
                        min_dm+=opdm_fused[i*size+k];
                        min_dm+=opdm_fused[(k+1)*size+k2];
                        min_dm+=opdm_fused[(k2+1)*size+j];//Fr

                        if(min_dm < opdm_fused[i*size+j]){
                            opdm_fused[i*size+j]=min_dm;
                            fusion[i*size+j]=r_fused;
                        }

                        min_dm=hl(i, j, k,k1, M);//Fl
                        min_dm+=opdm_fused[i*size+k1];
                        min_dm+=opdm_fused[(k1+1)*size+k];
                        min_dm+=opdm_fused[(k+1)*size+j];

                        if(min_dm < opdm_fused[i*size+j]){
                            opdm_fused[i*size+j]=min_dm;
                            fusion[i*size+j]=l_fused;
                        }

                    }
                    
                }
            }
        }

        unsigned long long acc_hp(int i, int j, int k, int M){
            unsigned long long tmp=0.f;
            if(i!=k)
                tmp+=P[i-1]*P[k];
            if(k+1!=j)
                tmp+=P[k]*P[j];
            tmp+=(float(P[j])/sqrt(M)) * 2*P[i-1]*P[k];
            return(floor(tmp));
        }

        unsigned long long acc_hl(int i, int j, int k,  int k1, int M){
            unsigned long long tmp=0.f,res;
            float alpha=float(P[j])/P[k1];
            float alpha_prime=(1+2*alpha)/(1+alpha);
            int x=sqrt(M/alpha_prime);
            int y=sqrt(M*alpha_prime);
            if(i!=k1)
                tmp+=P[i-1]*P[k1];
            if(k1+1!=k)
                tmp+=P[k1]*P[k];
            if(k+1!=j)
                tmp+=P[k]*P[j];

            //tmp+=((P[k])/sqrt(M))*2*(P[i-1])*(P[k1])*(1+alpha)*sqrt(alpha_prime);
            res=float(P[k]);
            res*=P[i-1];
            res*=P[k1];
            res*=1+alpha;
            res*=alpha_prime*x+y;
            res/=x*y;
            tmp+=res;
            if(i!=j)

                tmp-=2*(P[i-1])*(P[j]);
            return(floor(tmp));
        }

        unsigned long long acc_hr(int i, int j, int k,  int k2, int M){
            unsigned long long tmp=0.f,res;
            float beta=float(P[i-1])/float(P[k]);
            float beta_prime=(1+2*beta)/(1+beta);
            int x=sqrt(M*beta_prime);
            int y=sqrt(M/beta_prime);
            if(k+1!=k2)
                tmp+=P[k]*P[k2];
            if(k2+1!=j)
                tmp+=P[k2]*P[j];
            if(i!=k)
                tmp+=P[i-1]*P[k];
            
            //tmp+=((P[j])/sqrt(M))*2*(P[k])*(P[k2])*(1+beta)*sqrt(beta_prime);
            res=float(P[j]);
            res*=P[k];
            res*=P[k2];
            res*=1+beta;
            res*=beta_prime*y+x;
            res/=x*y;
            tmp+=res;
            if(i!=j)
                tmp-=2*P[i-1]*P[j];
            
            return(floor(tmp));
        }

        void accurate_compute_opdm_fusion(){
            int j,k, n=nb_mat, size=n+1;
            unsigned long long min_dm;
            int k1, k2;
            int M=65536;
            //int x,y;//calc them for each strat

            for(int l=1;l<n;l++){ //l begins at 2 ?
                for(int i=1;i<=n-l;i++){
                    j=i+l;
                    k=split[i*size+j];
                    if(i==k && k+1==j){ //Fp(i,j)
                        fusion[i*size+j]=p_fused;
                        opdm_fused[i*size+j]=acc_hp(i, j, k, M);
                        opdm_fused[i*size+j]+=opdm_fused[i*size+k];
                        opdm_fused[i*size+j]+=opdm_fused[(k+1)*size+j];
                    }
                    else if(i==k){ //min(Fp(i,j),Fr(i,j)) -> no fuse / right fuse

                        k2=split[(k+1)*size+j];

                        min_dm=acc_hp(i, j, k, M);
                        min_dm+=opdm_fused[i*size+k];
                        min_dm+=opdm_fused[(k+1)*size+j]; //Fp    

                        //if(min_dm < opdm_fused[i*size+j]){
                        opdm_fused[i*size+j]=min_dm;
                        fusion[i*size+j]=p_fused;
                        //}

                        min_dm=acc_hr(i, j, k,k2,M);
                        min_dm+=opdm_fused[i*size+k];
                        min_dm+=opdm_fused[(k+1)*size+k2];
                        min_dm+=opdm_fused[(k2+1)*size+j];//Fr
                        
                        if(min_dm < opdm_fused[i*size+j]){
                            opdm_fused[i*size+j]=min_dm;
                            fusion[i*size+j]=r_fused;
                        }
                        
                    }
                    else if(k+1==j){ //min(Fp(i,j),Fl(i,j)) -> no fuse / left fuse
                        k1=split[i*size+k];

                        min_dm=acc_hp(i, j, k, M);
                        min_dm+=opdm_fused[i*size+k];
                        min_dm+=opdm_fused[(k+1)*size+j];//Fp
                        //if(min_dm < opdm_fused[i*size+j]){
                        opdm_fused[i*size+j]=min_dm;
                        fusion[i*size+j]=p_fused;

                        //}

                        min_dm=acc_hl( i, j, k,k1,M);
                        min_dm+=opdm_fused[i*size+k1];
                        min_dm+=opdm_fused[(k1+1)*size+k];
                        min_dm+=opdm_fused[(k+1)*size+j];//Fl

                        if(min_dm < opdm_fused[i*size+j]){
                            opdm_fused[i*size+j]=min_dm;
                            fusion[i*size+j]=l_fused;
                        }
                        
                    }
                    else{ //min(Fp(i,j),Fr(i,j),Fl(i,j)) -> no fuse / right fuse / left fuse
                        k1=split[i*size+k];
                        k2=split[(k+1)*size+j];

                        min_dm=acc_hp(i, j, k, M); //Fp
                        min_dm+=opdm_fused[i*size+k];
                        min_dm+=opdm_fused[(k+1)*size+j];

                       // if(min_dm < opdm_fused[i*size+j]){
                        opdm_fused[i*size+j]=min_dm;
                        fusion[i*size+j]=p_fused;
                        //}


                        min_dm=acc_hr(i, j, k,k2,M);
                        min_dm+=opdm_fused[i*size+k];
                        min_dm+=opdm_fused[(k+1)*size+k2];
                        min_dm+=opdm_fused[(k2+1)*size+j];//Fr

                        if(min_dm < opdm_fused[i*size+j]){
                            opdm_fused[i*size+j]=min_dm;
                            fusion[i*size+j]=r_fused;
                        }

                        min_dm=acc_hl(i, j, k,k1,M);//Fl
                        min_dm+=opdm_fused[i*size+k1];
                        min_dm+=opdm_fused[(k1+1)*size+k];
                        min_dm+=opdm_fused[(k+1)*size+j];

                        if(min_dm < opdm_fused[i*size+j]){
                            opdm_fused[i*size+j]=min_dm;
                            fusion[i*size+j]=l_fused;
                        }
                    }
                }
            }
        }

        
        void print_sizes(){
            for(int i=0;i<nb_mat+1;i++){
                std::cout <<P[i]<< " ";
            }
            std::cout<<std::endl;
        }
        void print_mat(const std::string& s, std::vector<T> mat){
            std::cout<<s<<std::endl;;
            for(int i=0;i<nb_mat;i++){
                for(int j=0;j<nb_mat;j++)
                    std::cout<< mat[i*nb_mat +j] << " ";
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }

        void print_op(){
            int n=nb_mat;
            int size=n+1;
            std::cout << "Normal Matrix sizes :"<<std::endl;
            for(int i=0;i<n;i++){
                std::cout <<P[i]<< ","<<P[i+1]<<" ";
            }
            std::cout<<std::endl << "Matrix sizes :"<<std::endl;
            for(int i=0;i<n;i++){
                std::cout <<P[i]/MULTIPLYER<< ","<<P[i+1]/MULTIPLYER<<" ";
            }
            print_mat("Computations ;",opcount); 
            print_mat("Splits ;",split); 
            print_mat("Data transfers;",opdm); 
            if constexpr (FUSE){
                print_mat("Data transfers with fusion;",opdm_fused); 
                print_mat("Fusions ;",fusion); 
                std::cout << std::endl;
            }
        }

        /* helper functions */
        void print_mat(std::vector<T> mat ,int m, int n){
            std::cout<<"sizes :"<< m << " "<<n<<std::endl;;
            for(int i=0;i<m;i++){
                for(int j=0;j<n;j++)
                    std::cout<< mat[i*n +j] << " ";
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
        
        void print_all(){
            for(int i=1 ; i<=nb_mat; i++){
                std::cout<<"# mat "<< i <<" :"<<std::endl;
                print_mat(mat_chain[i],P[i-1],P[i]);
                std::cout<<std::endl;
            }
            for(int i=nb_mat+1;i<=2*nb_mat-1;i++){
                std::cout<<"# mat "<< i <<" :"<<std::endl;
                print_mat(mat_chain[i],P[0],P[i-(nb_mat-1)]);
                std::cout<<std::endl;
            }
            //print_mat(mat_chain[0],P[0],P.back());
        }


        void validate_results_on_cpu(){
            std::vector<T> res;
            std::vector<T> tmp;
            if constexpr (DISPLAY){
                std::cout<<"start SMM "<<1<<","<<2<<";sizes="<<P[0]<<"x"<<P[1]<<" & "<<P[1]<<"x"<<P[1+1]<<std::endl;
            }
            res.resize(P[0] * P[2]);
            cpu_mat_mult(mat_chain[1],mat_chain[2],res, P[0],P[1],P[2]);
            if constexpr (DISPLAY){
                std::cout<<"val of res= "<<res[0]<<std::endl;
            }
            tmp.resize(P[0] * P[2]);
            tmp=res;
            //copy(res.begin(), res.end(), back_inserter(tmp));
            if constexpr (DISPLAY){
                std::cout<<"val of tmp= "<<tmp[0]<<std::endl;
                std::cout<<"end SMM "<<1<<" on "<<nb_mat-1<<std::endl;
            }

            
            for(int i=2 ; i<nb_mat ; i++){
                if constexpr (DISPLAY)
                    std::cout<<"start SMM "<<1<<","<<i+1<<";sizes="<<P[0]<<"x"<<P[i]<<" & "<<P[i]<<"x"<<P[i+1]<<std::endl;
                res.resize(P[0] * P[i+1]);
                cpu_mat_mult(tmp,mat_chain[i+1],res, P[0],P[i],P[i+1]);
                if constexpr (DISPLAY)
                    std::cout<<"val of res= "<<res[0]<<std::endl;
                tmp.resize(P[0] * P[i+1]);
                tmp=res;
            }
            bool validation_passed = true;
            double err = 0;
            for(int i=0 ; i<res.size(); i++){
                if(round_fl(res[i]) != round_fl(mat_chain[nb_mat+1][i])){
                    validation_passed = false;
                    if(!(i%1000))
                        std::cout<<"Fail : "<< round_fl(res[i]) <<" != "<<round_fl(mat_chain[nb_mat+1][i])<<std::endl;
                }
                err += abs(round_fl(mat_chain[nb_mat+1][i]) - round_fl(res[i]));
            }
            

            if(validation_passed){
                std::cout<<"CPU validation : \033[32m[Success] Validation Passed \033[39m"<<std::endl;
                std::cout<< (res[0]) <<" != "<<(mat_chain[nb_mat+1][0])<<std::endl;
            }
            else{ 
                std::cout<<"CPU validation : \033[1;31m[Fail] Validation failed\033[0m, err = "<<err<<std::endl;
                std::cout<< (res[0]) <<" != "<<(mat_chain[nb_mat+1][0])<<std::endl;
            }
        } 
};  

#endif
