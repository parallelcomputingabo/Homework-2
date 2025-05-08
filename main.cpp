#include <cmath>      // For std::fabs function
#include <iostream>   // For standard I/O
#include <fstream>    // For file stream handling
#include <cstdlib>    // For EXIT_SUCCESS, EXIT_FAILURE macros
#include <cstdint>    // For uint32_t type
#include <iomanip>    // For output formatting
#include <omp.h>      // For OpenMP parallelization
#include <algorithm>

// Function to perform naive matrix multiplication (C = A * B)
void naive_matmul(float* matrix_C, float* matrix_A, float* matrix_B, uint32_t m, uint32_t n, uint32_t p)
{
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < p; ++j)
        {
            float sum = 0.0f;
            for (std::size_t k = 0; k < n; ++k)
            {
                sum += matrix_A[i * n + k] * matrix_B[k * p + j];
            }
            matrix_C[i * p + j] = sum;
        }
    }
}

// Blocked matrix multiplication (Cache Optimization)
void blocked_matmul(float* matrix_C, float* matrix_A, float* matrix_B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size)
{
    for (std::size_t ii = 0; ii < m; ii += block_size)
    {
        for (std::size_t jj = 0; jj < p; jj += block_size)
        {
            for (std::size_t kk = 0; kk < n; kk += block_size)
            {
                for (std::size_t i = ii; i < std::min(static_cast<std::size_t>(ii + block_size), static_cast<std::size_t>(m)); ++i)
                {
                    for (std::size_t j = jj; j < std::min(static_cast<std::size_t>(jj + block_size), static_cast<std::size_t>(p)); ++j)
                    {
                        float sum = 0.0f;
                        for (std::size_t k = kk; k < std::min(static_cast<std::size_t>(kk + block_size), static_cast<std::size_t>(n)); ++k)
                        {
                            sum += matrix_A[i * n + k] * matrix_B[k * p + j];
                        }
                        matrix_C[i * p + j] += sum;
                    }
                }
            }
        }
    }
}

// Parallel matrix multiplication using OpenMP
void parallel_matmul(float* matrix_C, float* matrix_A, float* matrix_B, uint32_t m, uint32_t n, uint32_t p)
{
    #pragma omp parallel for
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < p; ++j)
        {
            float sum = 0.0f;
            for (std::size_t k = 0; k < n; ++k)
            {
                sum += matrix_A[i * n + k] * matrix_B[k * p + j];
            }
            matrix_C[i * p + j] = sum;
        }
    }
}

// Function to validate that the result matches the expected output
bool validate(const std::string& result_path, const std::string& output_path)
{
    std::ifstream result(result_path);
    std::ifstream output(output_path);

    if (!result || !output)
        return false;

    int m1, n1, m2, n2;
    result >> m1 >> n1;
    output >> m2 >> n2;

    if (m1 != m2 || n1 != n2)
        return false;

    constexpr float tolerance = 1e-3f;
    float val1 = 0.0f, val2 = 0.0f;

    while (result >> val1 && output >> val2)
    {
        if (std::fabs(val1 - val2) >= tolerance)
            return false;
    }

    return result.eof() && output.eof();
}

// Main function to manage matrix multiplication and performance measurement
int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <test_case_number>" << std::endl;
        return EXIT_FAILURE;
    }

    int test_case = std::atoi(argv[1]);
    if (test_case < 0 || test_case > 9)
    {
        std::cerr << "Test case number must be between 0 and 9." << std::endl;
        return EXIT_FAILURE;
    }

    const std::string base_path = "data/" + std::to_string(test_case) + "/";
    const std::string input0_path = base_path + "input0.raw";
    const std::string input1_path = base_path + "input1.raw";
    const std::string result_path = base_path + "result.raw";
    const std::string output_path = base_path + "output.raw";

    std::ifstream input0(input0_path);
    std::ifstream input1(input1_path);

    int m, n, p;
    input0 >> m >> n;
    input1 >> n >> p;

    float* matrix_A = new float[m * n];
    float* matrix_B = new float[n * p];
    float* matrix_C = new float[m * p];

    for (std::size_t i = 0; i < m * n; ++i)
        input0 >> matrix_A[i];
    for (std::size_t i = 0; i < n * p; ++i)
        input1 >> matrix_B[i];

    input0.close();
    input1.close();

    double start = omp_get_wtime();
    naive_matmul(matrix_C, matrix_A, matrix_B, m, n, p);
    double naive_time = omp_get_wtime() - start;

    start = omp_get_wtime();
    blocked_matmul(matrix_C, matrix_A, matrix_B, m, n, p, 64);
    double blocked_time = omp_get_wtime() - start;

    start = omp_get_wtime();
    parallel_matmul(matrix_C, matrix_A, matrix_B, m, n, p);
    double parallel_time = omp_get_wtime() - start;

    std::cout << "Naive Time: " << naive_time << " s\n";
    std::cout << "Blocked Time: " << blocked_time << " s\n";
    std::cout << "Parallel Time: " << parallel_time << " s\n";

    delete[] matrix_A;
    delete[] matrix_B;
    delete[] matrix_C;

    return EXIT_SUCCESS;
}
