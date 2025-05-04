#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <cstdint>

void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p)
{
    // TODO : Implement naive matrix multiplication
    for (uint32_t i = 0; i < m; i++)
    {
        for (uint32_t j = 0; j < p; j++)
        {
            C[i * p + j] = 0.0f;
            for (uint32_t k = 0; k < n; k++)
            {
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
    for (uint32_t ii = 0; ii < m; ii += block_size)
    {
        for (uint32_t jj = 0; jj < p; jj += block_size)
        {
            for (uint32_t kk = 0; kk < n; kk += block_size)
            {
                // Process block: C[ii:ii+block_size, jj:jj+block_size] += A[ii:ii+block_size, kk:kk+block_size] * B[kk:kk+block_size, jj:jj+block_size]
                for (uint32_t i = ii; i < std::min(ii + block_size, m); i++)
                {
                    for (uint32_t j = jj; j < std::min(jj + block_size, p); j++)
                    {
                        float sum = 0.0f;
                        for (uint32_t k = kk; k < std::min(kk + block_size, n); k++)
                        {
                            sum += A[i * n + k] * B[k * p + j];
                        }
                        C[i * p + j] += sum;
                    }
                }
            }
        }
    }
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p)
{
#pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < m; i++)
    {
        for (uint32_t j = 0; j < p; j++)
        {
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; k++)
            {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file)
{
    // TODO : Implement result validation
    std::ifstream result_path(result_file, std::ios::binary);
    std::ifstream reference_path(reference_file, std::ios::binary);

    if (!result_path.is_open() || !reference_path.is_open())
    {
        std::cerr << "Failed to open result or reference file for validation" << std::endl;
        return false;
    }

    int result_m, result_n;
    result_path >> result_m >> result_n;

    int ref_m, ref_n;
    reference_path >> ref_m >> ref_n;

    if (result_m != ref_m || result_n != ref_n)
    {
        std::cerr << "Matrix dimensions don't match" << std::endl;
        return false;
    }

    for (int i = 0; i < result_m * result_n; i++)
    {
        float result_val, ref_val;
        result_path >> result_val;
        reference_path >> ref_val;

        if (std::abs(result_val - ref_val) > 1e-5)
        {
            std::cerr << "Values don't match at element " << i << ": "
                      << result_val << " vs " << ref_val << std::endl;
            return false;
        }
    }

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

    // TODO Read input0.raw (matrix A)
    std::ifstream input0(input0_file, std::ios::binary);
    if (!input0.is_open())
    {
        std::cerr << "Failed to open input0 file" << std::endl;
        return 1;
    }

    // TODO Read input1.raw (matrix B)
    std::ifstream input1(input1_file, std::ios::binary);
    if (!input1.is_open())
    {
        std::cerr << "Failed to open input1 file" << std::endl;
        return 1;
    }

    // Read dimensions from input0.raw and input1.raw
    int m, n, p;
    input0 >> m >> n;
    input1 >> n >> p;

    // Allocate memory for matrices A and B
    float *A = new float[m * n];
    float *B = new float[n * p];

    // Read matrix elements into A and B (row-major order)
    for (int i = 0; i < m * n; i++)
    {
        input0 >> A[i];
    }
    for (int i = 0; i < n * p; i++)
    {
        input1 >> B[i];
    }

    input0.close();
    input1.close();

    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    // TODO Write naive result to file
    std::ofstream result_path(result_file, std::ios::binary);
    if (!result_path.is_open())
    {
        std::cerr << "Failed to open result file for writing" << std::endl;
        delete[] A;
        delete[] B;
        delete[] C_naive;
        return 1;
    }

    // Set precision 2 decimal points
    result_path.precision(2);
    result_path << std::fixed;

    // Write dimensions and elements to result.raw
    result_path << m << " " << p << std::endl;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            result_path << C_naive[i * p + j];
            if (j < p - 1)
            {
                result_path << " ";
            }
        }
        result_path << std::endl;
    }
    result_path.close();

    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct)
    {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of blocked_matmul (use block_size = 32 as default)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 32);
    double blocked_time = omp_get_wtime() - start_time;

    // TODO Write blocked result to file
    result_path.open(result_file, std::ios::binary);
    if (!result_path.is_open())
    {
        std::cerr << "Failed to open result file for writing" << std::endl;
        delete[] A;
        delete[] B;
        delete[] C_blocked;
        return 1;
    }
    result_path.precision(2);
    result_path << std::fixed;
    result_path << m << " " << p << std::endl;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            result_path << C_blocked[i * p + j];
            if (j < p - 1)
            {
                result_path << " ";
            }
        }
        result_path << std::endl;
    }
    result_path.close();

    // Validate blocked result
    bool blocked_correct = validate_result(result_file, reference_file);
    if (!blocked_correct)
    {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    // TODO Write parallel result to file
    result_path.open(result_file, std::ios::binary);
    if (!result_path.is_open())
    {
        std::cerr << "Failed to open result file for writing" << std::endl;
        delete[] A;
        delete[] B;
        delete[] C_parallel;
        return 1;
    }
    result_path.precision(2);
    result_path << std::fixed;
    result_path << m << " " << p << std::endl;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            result_path << C_parallel[i * p + j];
            if (j < p - 1)
            {
                result_path << " ";
            }
        }
        result_path << std::endl;
    }
    result_path.close();

    // Validate parallel result
    bool parallel_correct = validate_result(result_file, reference_file);
    if (!parallel_correct)
    {
        std::cerr << "Parallel result validation failed for case " << case_number << std::endl;
    }

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
