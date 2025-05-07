// Kalle Fjäder
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <functional>
#include <omp.h>


static void naive_matmul(float* C,const float* A,const float* B,
                         uint32_t m,uint32_t n,uint32_t p)
{
    std::memset(C,0,sizeof(float)*m*p);
    for(uint32_t i=0;i<m;++i)
        for(uint32_t k=0;k<n;++k){
            float aik=A[i*n+k];
            for(uint32_t j=0;j<p;++j)
                C[i*p+j]+=aik*B[k*p+j];
        }
}

static void blocked_matmul(float* C,const float* A,const float* B,
                           uint32_t m,uint32_t n,uint32_t p,
                           uint32_t bs=64)
{
    std::memset(C,0,sizeof(float)*m*p);
    for(uint32_t ii=0;ii<m;ii+=bs)
        for(uint32_t kk=0;kk<n;kk+=bs)
            for(uint32_t jj=0;jj<p;jj+=bs){
                uint32_t i_max=std::min(ii+bs,m);
                uint32_t k_max=std::min(kk+bs,n);
                uint32_t j_max=std::min(jj+bs,p);
                for(uint32_t i=ii;i<i_max;++i)
                    for(uint32_t k=kk;k<k_max;++k){
                        float aik=A[i*n+k];
                        for(uint32_t j=jj;j<j_max;++j)
                            C[i*p+j]+=aik*B[k*p+j];
                    }
            }
}

static void parallel_matmul(float* C,const float* A,const float* B,
                            uint32_t m,uint32_t n,uint32_t p)
{
    std::memset(C,0,sizeof(float)*m*p);
    #pragma omp parallel for schedule(static)
    for(int i=0;i<(int)m;++i)
        for(uint32_t k=0;k<n;++k){
            float aik=A[i*n+k];
            for(uint32_t j=0;j<p;++j)
                C[i*p+j]+=aik*B[k*p+j];
        }
}

static bool read_matrix_ascii(const std::string& path,
                              uint32_t& rows,uint32_t& cols,
                              std::vector<float>& dst)
{
    std::ifstream in(path);
    if(!in||!(in>>rows>>cols)) return false;
    dst.resize((size_t)rows*cols);
    for(float& x:dst) if(!(in>>x)) return false;
    return true;
}

static bool write_matrix_ascii(const std::string& path,
                               uint32_t rows,uint32_t cols,
                               const float* data)
{
    std::ofstream out(path,std::ios::trunc);
    if(!out) return false;
    out<<rows<<' '<<cols<<'\n';

    for(uint32_t i=0;i<rows;++i){
        for(uint32_t j=0;j<cols;++j){
            float v = data[i*cols + j];
            int   scaled = int(std::floor(v * 100.f + 0.5f));
            int   cents  = scaled % 100;
            int   whole  = scaled / 100;


            if(cents == 0){
                out << whole << '.';
            } else if(cents % 10 == 0){
                out << whole << '.' << (cents / 10);
            } else {
                out << whole << '.' << std::setw(2) << std::setfill('0') << cents;
            }

            if(j + 1 != cols) out << ' ';
        }
        if (i + 1 != rows)
            out << '\n';
    }
    return true;
}

static bool identical_files(const std::string& a,const std::string& b)
{
    std::ifstream f1(a,std::ios::binary),f2(b,std::ios::binary);
    if(!f1||!f2) return false;
    char buf1[1<<14],buf2[1<<14];
    while(f1&&f2){
        f1.read(buf1,sizeof buf1);
        f2.read(buf2,sizeof buf2);
        if(f1.gcount()!=f2.gcount()) return false;
        if(std::memcmp(buf1,buf2,f1.gcount())) return false;
    }
    return f1.eof()&&f2.eof();
}


static double time_one(const std::function<void()>& fn)
{
    int iters=1; double t;
    do{
        double t0=omp_get_wtime();
        for(int i=0;i<iters;++i) fn();
        t=(omp_get_wtime()-t0)/iters;
        if(t>=1e-4||iters>=(1<<20)) break;
        iters*=2;
    }while(true);
    return t;
}


int main(int argc,char* argv[])
{
    if(argc!=2){ std::cerr<<"Usage: "<<argv[0]<<" <case>\n"; return 1; }
    int c=std::atoi(argv[1]);
    if(c<0||c>9){ std::cerr<<"Case 0‑9 only\n"; return 1; }

    std::string folder="data/"+std::to_string(c)+"/";
    std::string fA=folder+"input0.raw", fB=folder+"input1.raw";
    std::string fC=folder+"result.raw", fRef=folder+"output.raw";

    uint32_t m=0,n=0,p=0,n2=0; std::vector<float> A,B;
    if(!read_matrix_ascii(fA,m,n,A)||!read_matrix_ascii(fB,n2,p,B)||n2!=n){
        std::cerr<<"Failed to read input matrices\n"; return 2;
    }

    std::vector<float> C(m*p);

    double t_naive   =time_one([&]{ naive_matmul  (C.data(),A.data(),B.data(),m,n,p); });
    write_matrix_ascii(fC,m,p,C.data());
    bool ok_naive    =identical_files(fC,fRef);

    double t_blocked =time_one([&]{ blocked_matmul(C.data(),A.data(),B.data(),m,n,p,32); });
    write_matrix_ascii(fC,m,p,C.data());
    bool ok_blocked  =identical_files(fC,fRef);

    double t_parallel=time_one([&]{ parallel_matmul(C.data(),A.data(),B.data(),m,n,p); });
    write_matrix_ascii(fC,m,p,C.data());
    bool ok_parallel =identical_files(fC,fRef);

    std::cout << std::fixed << std::setprecision(6);
    auto same = [](bool ok){ return ok ? "same" : "diff"; };

    std::cout << "----- Summary -----";
    std::cout << "Matrix size : " << m << " x " << n << " x " << p << "";
    std::cout << "Naive     : " << t_naive    << " s  (" << same(ok_naive)   << ")";
    std::cout << "Blocked   : " << t_blocked  << " s  (" << same(ok_blocked) << ")  speedup " << t_naive / t_blocked  << "";
    std::cout << "Parallel  : " << t_parallel << " s  (" << same(ok_parallel)<< ")  speedup " << t_naive / t_parallel << "";

    return 0;
}