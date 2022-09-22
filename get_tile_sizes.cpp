#include <iostream>
#include <vector>
#include <cmath>
//#include "cmd_line_parser.hpp"
//                                          1,          3,      2,      1
unsigned long long acc_hl(std::vector<int> P, int i, int j, int k,  int k1, int M, int *x, int *y){
    unsigned long long tmp=0.f,res;
    float alpha=float(P[j])/P[k1];
    float alpha_prime=(1+2*alpha)/(1+alpha);
    *x=sqrt(M/alpha_prime);
    *y=sqrt(M*alpha_prime);
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
    res*=alpha_prime*(*x)+(*y);
    res/=(*x)*(*y);
    tmp+=res;
    if(i!=j)
        tmp-=2*(P[i-1])*(P[j]);
    return(floor(tmp));
}

int main(int argc, char *argv[]){
    unsigned long long dt;
    int x,y,M;
    int P0,P1,P2,P3;
    std::vector<int> P;
    /* Parse cmd line arguments*/
    //CmdLineParser cmd_parser(argc, argv);
    //cmd_parser.check_required_options();
    /* Get cmd line arguments*/
    if(argc==11){ 
         //P0 = cmd_parser.get_int("-P0");
         //P1 = cmd_parser.get_int("-P1");
         //P2 = cmd_parser.get_int("-P2");
         //P3 = cmd_parser.get_int("-P3");
         //M = cmd_parser.get_int("-M");
         P0=atoi(argv[2]);
         P1=atoi(argv[4]);
         P2=atoi(argv[6]);
         P3=atoi(argv[8]);
         M=atoi(argv[10]);
    } 
    P.push_back(P0*1024);
    P.push_back(P1*1024);
    P.push_back(P2*1024);
    P.push_back(P3*1024);

    dt=acc_hl(P,1,3,2,1,M,&x,&y);
    //std::cout<<"The amount of DT is "<<dt<<std::endl;
    std::cout<<"The optimal tile sizes are "<<x<< ", "<<y<<std::endl;
}