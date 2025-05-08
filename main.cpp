#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>

void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p)
{
    // TODO : Implement naive matrix multiplication
    for (uint32_t i = 0; i < m; ++i)
    {
        for (uint32_t j = 0; j < p; ++j)
        {
            double sum = 0.0;
            for (uint32_t k = 0; k < n; ++k)
            {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

void blocked_matmul(float *C, float *A, float *B,
                    uint32_t m, uint32_t n, uint32_t p,
                    uint32_t block_size = 64)
{
    // Initialize result matrix to 0
    for (uint32_t i = 0; i < m * p; ++i)
        C[i] = 0.0f;

    for (uint32_t ii = 0; ii < m; ii += block_size)
    {
        for (uint32_t jj = 0; jj < p; jj += block_size)
        {
            for (uint32_t kk = 0; kk < n; kk += block_size)
            {
                uint32_t i_max = std::min(ii + block_size, m);
                uint32_t j_max = std::min(jj + block_size, p);
                uint32_t k_max = std::min(kk + block_size, n);

                for (uint32_t i = ii; i < i_max; ++i)
                {
                    for (uint32_t k = kk; k < k_max; ++k)
                    {
                        float a_ik = A[i * n + k];
                        for (uint32_t j = jj; j < j_max; ++j)
                        {
                            C[i * p + j] += a_ik * B[k * p + j];
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

#pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < m; ++i)
    {
        for (uint32_t j = 0; j < p; ++j)
        {
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; ++k)
            {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file, float epsilon = 1e-5)
{
    // TODO : Implement result validation
    std::ifstream result(result_file), reference(reference_file);
    if (!result || !reference)
    {
        std::cerr << "Validation failed: could not open files.\n";
        return false;
    }

    uint32_t r1, c1, r2, c2;
    result >> r1 >> c1;
    reference >> r2 >> c2;

    if (r1 != r2 || c1 != c2)
    {
        std::cerr << "Dimension mismatch during validation.\n";
        return false;
    }

    float a, b;
    for (uint32_t i = 0; i < r1 * c1; ++i)
    {
        result >> a;
        reference >> b;
        if (std::fabs(a - b) > epsilon)
        {
            std::cerr << "Mismatch at index " << i << ": " << a << " vs " << b << "\n";
            return false;
        }
    }

    return true;
}

bool read_matrix(const std::string &path,
                 float *&M, uint32_t &rows, uint32_t &cols)
{
    std::ifstream in(path);
    if (!in)
        return false;
    in >> rows >> cols;
    M = new float[static_cast<size_t>(rows) * cols];
    for (size_t i = 0; i < (size_t)rows * cols; ++i)
    {
        in >> M[i];
    }
    return true;
}

bool write_matrix(const std::string &path,
                  float *M, uint32_t rows, uint32_t cols)
{
    std::ofstream out(path);
    if (!out)
        return false;
    out << rows << " " << cols << '\n';
    for (size_t i = 0; i < (size_t)rows * cols; ++i)
    {
        out << M[i] << ((i + 1) % cols ? ' ' : '\n');
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

    float *A = nullptr, *B = nullptr, *C = nullptr;
    uint32_t m, n, p;

    // TODO Read input0.raw and input1.raw (matrix A & matrix B)
    if (!read_matrix(input0_file, A, m, n) ||
        !read_matrix(input1_file, B, n, p))
    {
        std::cerr << "Error reading input files" << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    // TODO Write naive result to file
    if (!write_matrix(result_file, C_naive, m, p))
    {
        std::cerr << "Error writing result.raw" << std::endl;
        return EXIT_FAILURE;
    }

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
    if (!write_matrix(result_file, C_blocked, m, p))
    {
        std::cerr << "Error writing result.raw" << std::endl;
        return EXIT_FAILURE;
    }

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
    if (!write_matrix(result_file, C_blocked, m, p))
    {
        std::cerr << "Error writing result.raw" << std::endl;
        return EXIT_FAILURE;
    }
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