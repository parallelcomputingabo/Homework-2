#include <iostream>//
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <iomanip>
#include <cstdint> // Pre-emptively adding this one, since this literally might just be a platform thing???

//Copied the old naive matmul implementation
void naive_matmul(float* C, float* A, float* B, uint32_t m, uint32_t n, uint32_t p)
{
    //A = (m*n)
    //B = (n*p)
    //C = (m*p)
    // TODO: Implement naive matrix multiplication C = A x B
    for (uint32_t i = 0; i < m; i++) {
        //Each column/row/whatever of A
        for (uint32_t j = 0; j < p; j++){
            C[i * p + j] =0.0f;
            for (uint32_t k = 0; k < n  ; k++){
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            };

        };

    };
}

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    // Initialize C with zeros
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < p; j++) {
            C[i * p + j] = 0.0f;
        }
    }

    // TODO: Implement blocked matrix multiplication
    // A is m x n, B is n x p, C is m x p
    // Use block_size to divide matrices into submatrices

    //Just added the type {}  and ; from the provided pseudocode, works really bad. Like half the performance of the base.

    for (uint32_t ii = 0; ii < m; ii += block_size)
    {
        for (uint32_t jj = 0; jj < p; jj += block_size)
        {
            for (uint32_t kk = 0; kk < n; kk += block_size)
            {

                //So I broke these up a bit


                //i < std::min(ii + block_size, m)
                //from ii block
                uint32_t block_i = std::min(ii + block_size, m);



                //j < std::min(jj + block_size, p)
                //from jj block
                uint32_t block_j = std::min(jj + block_size, p);


                //k < std::min(kk + block_size, n)
                //from kk block
                uint32_t block_k = std::min(kk + block_size, n);


                // Process block: C[ii:ii+block_size, jj:jj+block_size] += A[ii:ii+block_size, kk:kk+block_size] * B[kk:kk+block_size, jj:jj+block_size]

                for (uint32_t i = ii; i < block_i ; i++)
                {
                    for (uint32_t j = jj; j < block_j ;j++)
                    {
                        float sum = C[i * p + j];
                        uint32_t k = kk;


                        for (; k < block_k ; k++)
                        {
                            sum += A[i * n + k] * B[k * p + j];
                        }
                        C[i * p + j] = sum;
                    }
                }
            }
        }
    }
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // Initialize C with zeros

    //Apparently openmp likes to be told how many nestings there are? So I went with collapse 2, though it wasn't present in the pseudocode
    //(((I really should go and read/watch the course lectures again, and not just skim them. Maybe it's a bit more opened up there)))
#pragma omp parallel for
// though apparently my compiler doesn't like collapse(2)?
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0.0f;
        }
    }
    // TODO: Implement parallel matrix multiplication using OpenMP
    // A is m x n, B is n x p, C is m x p
#pragma omp parallel for
// though apparently my compiler doesn't like collapse(2)?
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            //I don't know if the pseudocodes way of doing it is better, but it kinda doesn't make sense to crank the memory address.
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; k++)
            {
                sum += A[i * n + k] * B[k * p + j];
            }

            C[i * p + j] = sum;
        }
    }

}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
   //TODO : Implement result validation

    std::ifstream mW(result_file);
    std::ifstream mT(reference_file);


    float test_v, wri_v;
    float rmT, cmT, rmW, cmW;
    bool suc=true;
    mT >> rmT, mT >>cmT, mW >> rmW, mW >> cmW;

    if (rmT!=rmW || cmT!=cmW)
    {
        std::cerr << "Dimension missmatch: ?" <<"\n result: ("<<cmT<<"//"<<cmW<<") output : ("<<rmT<<"//"<<cmT<<")" << std::endl;
        suc=false;
    }

    while(mT >>test_v && mW >> wri_v)
    {
        if (std::round(test_v) != std::round(wri_v) )
        {
            std::cerr << "Error in test: ?" <<"\n The differing values where C: "<< wri_v << " T:"<< test_v << std::endl;
            suc=false;
        }
    }

   return suc;


}




int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        return 1;
    }

    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9) {
        std::cerr << "Case number must be between 0 and 9" << std::endl;
        return 1;
    }

    // Construct file paths
    std::string folder = "data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";

    // yes
    std::ifstream reference_result(reference_file);
    int m, n, nt, p;


    std::ifstream input0(input0_file);
    input0 >> m >> n;


    std::ifstream input1(input1_file);
    input1 >> nt >> p;


    if (n != nt) {
      std::cerr << "Dimension missmatch!" << std::endl;
      return 1;
    }




    float* A = new float[ m * n ];
    for (uint32_t j = 0; j < m * n; ++j) {
        input0 >> A[j];
    }
    input0.close();     // TODO Read input0.raw (matrix A)



    float* B = new float[ n * p ];
    for (uint32_t j = 0; j < n * p; ++j) {
        input1 >> B[j];
    }
    input1.close();    // TODO Read input1.raw (matrix B)






    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];







    //-----------------


    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;


    std::ofstream mCout_paraller(result_file);     // TODO Write naive result to file
    if (!mCout_paraller.is_open())
    {
        std::cerr << "Opening result file failed" << std::endl; //I do kinda enjoy the aggressiveness of autocompleat in C-lion
    }
    mCout_paraller << m << " " << p << std::endl;

    for (int j = 0; j < m; ++j){
        for (int k = 0; k < p; ++k){
            mCout_paraller<< std::fixed << std::setprecision(2) << C_naive[j * p + k];
            if (k != p - 1)
            {
                mCout_paraller << " ";
            }
        }
        mCout_paraller << std::endl;
    }

    mCout_paraller.close();

    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }


    //-----------------


    // Measure performance of blocked_matmul (use block_size = 32 as default)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 8); // 8 seemed to work best at least on the larger test. I didn't test too thoroughly, and only with multiples of 2 :P
    double blocked_time = omp_get_wtime() - start_time;

    std::ofstream mCout_block(result_file); // TODO Write blocked result to file
    if (!mCout_block.is_open())
    {
        std::cerr << "Opening result file failed" << std::endl;
    }
    mCout_block << m << " " << p << std::endl;

    for (int j = 0; j < m; ++j){
        for (int k = 0; k < p; ++k){
            mCout_block<< std::fixed << std::setprecision(2) << C_blocked[j * p + k];
            if (k != p - 1)
            {
                mCout_block << " ";
            }
        }
        mCout_block << std::endl;
    }
    mCout_block.close(); // The autocorrent/fill in in c-lion is crazy, though tit doesn't seem to keep formatting


    // Validate blocked result
    bool blocked_correct = validate_result(result_file, reference_file);
    if (!blocked_correct) {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    //-----------------


    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    std::ofstream mCout_parallel(result_file); // TODO Write parallel result to file
    if (!mCout_parallel.is_open()){
      std::cerr << "Opening result file failed" << std::endl;
      }
      mCout_parallel << m << " " << p << std::endl;
      for (int j = 0; j < m; ++j){
        for (int k = 0; k < p; ++k){
          mCout_parallel<< std::fixed << std::setprecision(2) << C_parallel[j * p + k];
          if (k != p - 1){
            mCout_parallel << " ";
            }
      }
      mCout_parallel << std::endl;
    }
    mCout_parallel.close();


    // Validate parallel result
    bool parallel_correct = validate_result(result_file, reference_file);
    if (!parallel_correct) {
        std::cerr << "Parallel result validation failed for case " << case_number << std::endl;
    }

    //-----------------


    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive time: " << naive_time  * 1000 << " milliseconds\n";
    std::cout << "Blocked time: " << blocked_time * 1000 << " milliseconds\n";
    std::cout << "Parallel time: " << parallel_time  * 1000 << " milliseconds\n";
    std::cout << "Blocked speedup: " << (naive_time / blocked_time) << "x\n";
    std::cout << "Parallel speedup: " << (naive_time / parallel_time) << "x\n";

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;

    return 0;
}
