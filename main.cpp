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

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size)
{
    // TODO: Implement blocked matrix multiplication
    // A is m x n, B is n x p, C is m x p
    // Use block_size to divide matrices into submatrices
    for (uint32_t i = 0; i < m * p; i++)
    {
        C[i] = 0.0f;
    }

    // C = A * B
    for (uint32_t ii = 0; ii < m; ii += block_size) {
        for (uint32_t jj = 0; jj < p; jj += block_size) {
            for (uint32_t kk = 0; kk < n; kk += block_size) {
                
                uint32_t i_max = std::min(ii + block_size, m);
                uint32_t j_max = std::min(jj + block_size, p);
                uint32_t k_max = std::min(kk + block_size, n);

                for (uint32_t i = ii; i < i_max; ++i) {
                    for (uint32_t j = jj; j < j_max; ++j) {
                        for (uint32_t k = kk; k < k_max; ++k) {
                            C[i * p + j] += A[i * n + k] * B[k * p + j];
                        }
                    }
                }
            }
        }
    }
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p)
{
    // TODO: Implement parallel matrix multiplication using OpenMP
    // A is m x n, B is n x p, C is m x p
}

bool validate_result(const std::string &result_file, const std::string &reference_file)
{
    // TODO : Implement result validation
    //  Read result and reference files
    ifstream result(result_file);
    ifstream reference(reference_file);

    if (!result.is_open() || !reference.is_open())
    {
        std::cerr << "Error opening result or reference file!" << std::endl;
        return false;
    }

    // Read dimensions
    int m, n;
    result >> m >> n;
    int ref_m, ref_n;
    reference >> ref_m >> ref_n;

    if (m != ref_m || n != ref_n)
    {
        std::cerr << "Dimension mismatch!" << std::endl;
        return false;
    }

    // Compare matrices
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float res_val, ref_val;
            result >> res_val;
            reference >> ref_val;

            if (std::abs(res_val - ref_val) > 1e-5)
            {
                std::cerr << "Mismatch at (" << i << ", " << j << "): "
                          << res_val << " != " << ref_val << std::endl;
                return false;
            }
        }
    }

    // Close files
    result.close();
    reference.close();
    return true;
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
    std::string result_block_file = folder + "result_block.raw";

    int m, n, p; // A is m x n, B is n x p, C is m x p

    // TODO Read input0.raw (matrix A)
    ifstream fileA(input0_file);
    fileA >> m >> n;
    // TODO Read input1.raw (matrix B)
    ifstream fileB(input1_file);
    fileB >> n >> p;

    // Allocate memory for result matrices
    float *A = new float[m * n];
    float *B = new float[n * p];
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // Read matrix in row-major order
    for (int i = 0; i < m * n; ++i)
        fileA >> A[i];
    for (int i = 0; i < n * p; ++i)
        fileB >> B[i];

    // Close input files
    fileA.close();
    fileB.close();

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    // TODO Write naive result to file
    ofstream outFile(result_file);
    if (!outFile.is_open())
    {
        cout << "Error opening result file!" << std::endl;
        delete[] A;
        delete[] B;
        delete[] C_naive;
        return 1;
    }

    outFile << m << " " << p << "\n";
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            outFile << C_naive[i * p + j];
            if (j < p - 1)
                outFile << " ";
        }
        outFile << "\n";
    }
    outFile.close();

    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct)
    {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of blocked_matmul (use block_size = 32 as default)
   

    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 64);
    double blocked_time = omp_get_wtime() - start_time;
   
    std::cout << "64 Blocked time : " << blocked_time << " seconds\n";
    std::cout << "Blocked speedup: " << (naive_time / blocked_time) << "x\n";
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 32);
    blocked_time = omp_get_wtime() - start_time;
    
    std::cout << "32 Blocked time : " << blocked_time << " seconds\n";
    std::cout << "Blocked speedup: " << (naive_time / blocked_time) << "x\n";
    // TODO Write blocked result to file
    outFile.open(result_block_file);
    if (!outFile.is_open())
    {
        cout << "Error opening result file!" << std::endl;
        delete[] A;
        delete[] B;
        delete[] C_blocked;
        return 1;
    }

    outFile << m << " " << p << "\n";
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            outFile << C_blocked[i * p + j];
            if (j < p - 1)
                outFile << " ";
        }
        outFile << "\n";
    }
    outFile.close();
    // Validate blocked result
    bool blocked_correct = validate_result(result_block_file, reference_file);
    if (!blocked_correct)
    {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    // TODO Write parallel result to file

    // Validate parallel result
    // bool parallel_correct = validate_result(result_file, reference_file);
    // if (!parallel_correct)
    // {
    //     std::cerr << "Parallel result validation failed for case " << case_number << std::endl;
    // }

    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive time: " << naive_time << " seconds\n";
    std::cout << "Blocked time: " << blocked_time << " seconds\n";
    std::cout << "Parallel time: " << parallel_time << " seconds\n";
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