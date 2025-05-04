#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <cstdint>
using namespace std;

void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p)
{
    for (uint32_t i = 0; i < m * p; i++)
    {
        C[i] = 0.0f;
    }

    // Perform matrix multiplication C = A x B
    for (uint32_t i = 0; i < m; i++)
    { // For each row of A
        for (uint32_t j = 0; j < p; j++)
        { // For each column of B
            for (uint32_t k = 0; k < n; k++)
            { // For each element in row/column
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p)
{
    // TODO: Implement parallel matrix multiplication using OpenMP
    // A is m x n, B is n x p, C is m x p
     // Initialize C to 0 to avoid undefined behavior
     #pragma omp parallel for
     for (uint32_t i = 0; i < m; i++) {
         for (uint32_t j = 0; j < p; j++) {
             C[i * p + j] = 0.0f;
         }
     }
 
     // Perform parallel matrix multiplication
     #pragma omp parallel for
     for (uint32_t i = 0; i < m; i++) {
         for (uint32_t j = 0; j < p; j++) {
             for (uint32_t k = 0; k < n; k++) {
                 C[i * p + j] += A[i * n + k] * B[k * p + j];
             }
         }
     }
}



int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        return 1;
    }

    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9)
    {
        std::cerr << "Case number must be between 0 and 9" << std::endl;
        return 1;
    }

    // Construct file paths
    std::string folder = "data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";
    std::string result_block_file = folder + "result_block.raw"; //output for blocked matrix multiplication
    std::string result_omp_file = folder + "result_omp.raw"; //output for parrllel matrix multiplication
    
    int m, n, p; // A is m x n, B is n x p, C is m x p

    // TODO Read input0.raw (matrix A)
    ifstream fileA(input0_file);
    fileA >> m >> n;
    // TODO Read input1.raw (matrix B)
    ifstream fileB(input1_file);
    fileB >> n >> p;
    cout << "Matrix Dimensions: " << m << " x " << n << " x " << p << endl;
    // Allocate memory for result matrices
    float *A = new float[m * n];
    float *B = new float[n * p];
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // Read matrix in row-major order
    for (int i = 0; i < m * n; ++i) fileA >> A[i];
    for (int i = 0; i < n * p; ++i) fileB >> B[i];

    // Close input files
    fileA.close();
    fileB.close();

    const int NUM_RUNS = 5;
double total_naive_time = 0.0;
double total_parallel_time = 0.0;

// Run 5 times
for (int run = 0; run < NUM_RUNS; ++run) {
    // Measure naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;
    total_naive_time += naive_time;


    // Measure blocked_matmul (block size = 64)
    start_time = omp_get_wtime();
        parallel_matmul(C_parallel, A, B, m, n, p);
        double parallel_time = omp_get_wtime() - start_time;
        total_parallel_time += parallel_time;

}

double avg_naive_time = total_naive_time / NUM_RUNS;
    double avg_parallel_time = total_parallel_time / NUM_RUNS;
    double avg_speedup = avg_naive_time / avg_parallel_time;

    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Average Naive time: " << avg_naive_time << " seconds\n";
    std::cout << "Average Parallel time: " << avg_parallel_time << " seconds\n";
    std::cout << "Average Parallel speedup: " << avg_speedup << "x\n";


    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;

    return 0;
}