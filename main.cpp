/*
 * Homework 2 – Optimized Parallel Matrix Multiplication
 *
 * 
 * My work is done using Mac Air laptop with M1 chip
 * Compilers used: g++14 and gcc14.
 * 
 * This program reads two matrices A (m×n) and B (n×p) from “data/<case>/input0.raw” 
 * and “input1.raw”, computes their product C in three different ways, and writes 
 * out three result files for verification:
 *
 *   1) Naive triple‐loop implementation:
 *      - Function: naive_matmul
 *      - Measures wall‐clock time with omp_get_wtime()
 *      - Dumps C to data/<case>/result_naive.raw and compares to output.raw
 *
 *   2) Cache‐blocked (“tiled”) implementation:
 *      - Function: blocked_matmul with block size B=16
 *      - Improves data locality by operating on B×B sub‐blocks
 *      - Measures time, writes to result_blocked.raw, and validates
 *
 *   3) OpenMP‐parallel implementation:
 *      - Function: parallel_matmul using #pragma omp parallel for collapse(2)
 *      - Distributes (i,j) iterations across threads (set via OMP_NUM_THREADS)
 *      - Measures time, writes to result_parallel.raw, and validates
 *
 * Usage:
 *   mkdir -p build && cd build
 *   CC=gcc-14 CXX=g++-14 cmake ..
 *   make
 *   export OMP_NUM_THREADS=8
 *   ./matmul <case#>     # runs all three kernels, prints times + PASS/FAIL
 *
 * Validation:
 *   Each stage writes and then compares its own “result_*.raw” against the 
 *   reference output.raw (tolerance ±0.001). The program exits non‐zero if any 
 *   stage fails.
 * 
 * dependencies: 
 * Make sure you pass: export OMP_NUM_THREADS=8
 * I have used and attached a bash script "run_results.sh" inside the build folder that ran tests and 
 * saved and analyzed results.
 * 
 */




 #include <iostream>
 #include <fstream>
 #include <cstdint>
 #include <cmath>
 #include <cassert>
 #include <string>
 #include <sstream>
 #include <iomanip>
 #include <vector>
 #include <algorithm>
 #include <omp.h>

 
 //Naive triple-loop multiply C = A × B
 void naive_matmul(float* C, const float* A, const float* B,
                   uint32_t m, uint32_t n, uint32_t p) {
     for (uint32_t i = 0; i < m; ++i)
         for (uint32_t j = 0; j < p; ++j) {
             float sum = 0.0f;
             for (uint32_t k = 0; k < n; ++k)
                 sum += A[i*n + k] * B[k*p + j];
             C[i*p + j] = sum;
         }
 }

//Blocked Matrix Multiplication
/**
 * blocked_matmul: Cache-blocked (tiled) matrix multiplication
 *
 * Divides the m×p output matrix and the corresponding rows of A and columns of B
 * into Bsize×Bsize tiles to improve cache reuse.  For each tile (ii,jj) of C and
 * corresponding tiles in A and B, it multiplies sub‐blocks of size Bsize:
 *
 *   for ii in [0,m) step Bsize
 *     for jj in [0,p) step Bsize
 *       for kk in [0,n) step Bsize
 *         // multiply the small block A[ii..ii+Bsize][kk..kk+Bsize]
 *         // by B[kk..kk+Bsize][jj..jj+Bsize] into C[ii..ii+Bsize][jj..jj+Bsize]
 *
 * This ensures each Bsize×Bsize sub‐matrix fits in cache, reducing cache misses
 * by reusing loaded data across multiple inner products.
 */
 void blocked_matmul(float *C, const float *A, const float *B,
    uint32_t m, uint32_t n, uint32_t p, uint32_t Bsize) {
// Zero C
std::fill(C, C + m * p, 0.0f);

for (uint32_t ii = 0; ii < m; ii += Bsize) {
    for (uint32_t jj = 0; jj < p; jj += Bsize) {
        for (uint32_t kk = 0; kk < n; kk += Bsize) {
            uint32_t i_max = std::min(ii + Bsize, m);
            uint32_t j_max = std::min(jj + Bsize, p);
            uint32_t k_max = std::min(kk + Bsize, n);

for (uint32_t i = ii; i < i_max; ++i) {
    for (uint32_t j = jj; j < j_max; ++j) {
        float sum = C[i * p + j];
        for (uint32_t k = kk; k < k_max; ++k) {
            sum += A[i * n + k] * B[k * p + j];
        }
        C[i * p + j] = sum;
                    }
                }
            }
        }
    }
}

//Parallel Matrix Multiplication
/**
 * parallel_matmul: OpenMP‐parallelized matrix multiplication
 *
 * Computes C = A × B using the naive triple‐loop algorithm, but distributes the
 * independent (i,j) iterations across threads via OpenMP (best results were done
 * by 8 threads beucase the MAC I worked on has 8 threads, though 4 threads had almost a linear speed up).  Each thread accumulates
 * a distinct element C[i][j], so there’s no need for synchronization.alignas
 * 
 * Make sure you pass export OMP_NUM_THREADS=8
 */
void parallel_matmul(float *C, const float *A, const float *B,
    uint32_t m, uint32_t n, uint32_t p) {
// Zero output
std::fill(C, C + m*p, 0.0f);

#pragma omp parallel for collapse(2) schedule(static)
for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < p; ++j) {
          float sum = 0.0f;
          for (uint32_t k = 0; k < n; ++k)
              sum += A[i*n + k] * B[k*p + j];
          C[i*p + j] = sum;
        }
    }
}


 
 //Compare floats within ±0.001 (tolerance)
 bool compare_floats(float a, float b) {
     return std::fabs(a - b) <= 0.001f;
 }
 
 //Validation method: Line-by-line, tolerance-based compare
 bool compare_matrix_files(const std::string& p1, const std::string& p2) {
     std::ifstream f1(p1), f2(p2);
     if (!f1 || !f2) return false;
     std::string l1, l2; size_t line = 0;
     while (std::getline(f1, l1)) {
         ++line;
         if (!std::getline(f2, l2)) return false;
         if (l1 == l2) continue;
         std::istringstream s1(l1), s2(l2);
         std::vector<float> v1, v2; float x;
         while (s1 >> x) v1.push_back(x);
         while (s2 >> x) v2.push_back(x);
         if (v1.size()!=v2.size()) return false;
         for (size_t i=0; i<v1.size(); ++i)
             if (!compare_floats(v1[i], v2[i])) return false;
     }
     return f1.eof() && !std::getline(f2, l2);
 }
 
 int main(int argc, char* argv[]) {
     assert(argc==2 && "Usage: matmul <case#>");
     std::string dir = "../data/" + std::string(argv[1]) + "/";
 
     // read dims
     std::ifstream inA(dir+"input0.raw"), inB(dir+"input1.raw");
     assert(inA && inB);
     uint32_t m,n,n2,p;
     inA>>m>>n; inB>>n2>>p; assert(n==n2);
 
     // alloc & load
     float *A=new float[m*n], *B=new float[n*p], *C=new float[m*p];
     for(uint32_t i=0;i<m*n;++i) inA>>A[i];
     for(uint32_t i=0;i<n*p;++i) inB>>B[i];
 
     // Flags for validation
     bool ok_naive, ok_blocked, ok_parallel;


     // Execute then measure performance of naive_matmul
     double start_t1 = omp_get_wtime();
     naive_matmul(C,A,B,m,n,p);
     double end_t1 = omp_get_wtime();
     std::cout << "Naive matmul time: " << (end_t1 - start_t1) << " s\n";

     {
        // write result of naive_matmul
        std::string out_path = dir+"result_naive.raw";
        std::ofstream out(out_path);
        out<<m<<" "<<p<<"\n";
        for(uint32_t i=0;i<m;++i){
        for(uint32_t j=0;j<p;++j){
            float v2 = std::round(C[i*p+j]*100.0f)/100.0f;
            std::ostringstream ss;
            ss<<std::fixed<<std::setprecision(2)<<v2;
            std::string s=ss.str();
            while(s.back()=='0') s.pop_back();
            out<<s<<(j+1<p?" ":"\n");
        }
        }
        out.close();  // <-- ensure full flush
    
        // validate
        ok_naive = compare_matrix_files(out_path, dir+"output.raw");
        std::cout << (ok_naive ? "NAIVE PASS\n" : "NAIVE FAIL\n");
    }


//-------------------------------------------------------
//-------------------------------------------------------
//-------------------------------------------------------
//-------------------------------------------------------
//-------------------------------------------------------



     // Execute and measure performance of blocked_matmul
     // --- Blocked matmul (cache optimization) ---
     uint32_t Bsize = 16;  // tested with 32, 64 and 16
     //std::fill(C, C + m * p, 0.0f);
     double start_t2 = omp_get_wtime();
     blocked_matmul(C, A, B, m, n, p, Bsize);
     double end_t2 = omp_get_wtime();
     std::cout << "Blocked matmul time (B=" << Bsize << "): "
     << (end_t2 - start_t2) << " s\n";

    {
        // write result
        std::string out_path = dir+"result_blocked.raw";
        std::ofstream out(out_path);
        out<<m<<" "<<p<<"\n";
        for(uint32_t i=0;i<m;++i){
        for(uint32_t j=0;j<p;++j){
            float v2 = std::round(C[i*p+j]*100.0f)/100.0f;
            std::ostringstream ss;
            ss<<std::fixed<<std::setprecision(2)<<v2;
            std::string s=ss.str();
            while(s.back()=='0') s.pop_back();
            out<<s<<(j+1<p?" ":"\n");
        }
        }
        out.close();  // <-- ensure full flush
    
        // validate
        ok_blocked = compare_matrix_files(out_path, dir+"output.raw");
        std::cout << (ok_blocked ? "BLOCKED PASS\n" : "BLOCKED FAIL\n");
    }


//-------------------------------------------------------
//-------------------------------------------------------
//-------------------------------------------------------
//-------------------------------------------------------
//-------------------------------------------------------

    // Execute and measure performance of parallel_matmul
    // --- Parallel matmul (OpenMP) ---
     std::fill(C, C + m*p, 0.0f);
     double start_t3 = omp_get_wtime();
     parallel_matmul(C, A, B, m, n, p);
     double end_t3   = omp_get_wtime();
     std::cout << "Parallel matmul time: " << (end_t3 - start_t3) << " s\n";


     {
        // write result
        std::string out_path = dir+"result_parallel.raw";
        std::ofstream out(out_path);
        out<<m<<" "<<p<<"\n";
        for(uint32_t i=0;i<m;++i){
        for(uint32_t j=0;j<p;++j){
            float v2 = std::round(C[i*p+j]*100.0f)/100.0f;
            std::ostringstream ss;
            ss<<std::fixed<<std::setprecision(2)<<v2;
            std::string s=ss.str();
            while(s.back()=='0') s.pop_back();
            out<<s<<(j+1<p?" ":"\n");
        }
        }
        out.close();  // <-- ensure full flush
    
        // validate
         ok_parallel = compare_matrix_files(out_path, dir+"output.raw");
         std::cout << (ok_parallel ? "PARALLEL PASS\n" : "PARALLEL FAIL\n");
    }
     delete[] A; delete[] B; delete[] C;
     
     // overall return: success only if all three passed
     return (ok_naive && ok_blocked && ok_parallel) ? 0 : 1;
 }