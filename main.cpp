#include <iostream>
#include <fstream>
#include <format>
#include <string>
#include <omp.h>
#include <cmath>
#include <filesystem>

using namespace std;
namespace fs = filesystem;

void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p)
{
    // A is m x n, B is n x p, C is m x p
    for (uint32_t i = 0; i < m; i++) // Rows of A
    {
        for (uint32_t j = 0; j < p; j++) // Columns of B
        {
            for (uint32_t k = 0; k < n; k++) // Shared dimension
            {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size)
{
    // A is m x n, B is n x p, C is m x p
    // Use block_size to divide matrices into submatrices
    uint32_t ii, jj, kk, i, j, k;

    for (ii = 0; ii < m; ii += block_size)
        for (jj = 0; jj < p; jj += block_size)
            for (kk = 0; kk < n; kk += block_size)
                // Process block: C[ii:ii+block_size, jj:jj+block_size] += A[ii:ii+block_size, kk:kk+block_size] * B[kk:kk+block_size, jj:jj+block_size]
                for (i = ii; i < min(ii + block_size, m); i++)
                    for (j = jj; j < min(jj + block_size, p); j++)
                        for (k = kk; k < min(kk + block_size, n); k++)
                            C[i * p + j] += A[i * n + k] * B[k * p + j];
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p)
{
    // A is m x n, B is n x p, C is m x p
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) // Rows of A
    {
        for (int j = 0; j < p; j++) // Columns of B
        {
            for (int k = 0; k < n; k++) // Shared dimension
            {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

int read_matrix_dimensions(fs::path path, uint32_t &rows, uint32_t &cols)
{
    ifstream file(path);
    if (file.is_open())
    {
        file >> rows >> cols;
        file.close();
    }
    else
    {
        return -1;
    }

    return 0;
}

int read_matrix_elements(fs::path path, float *matrix, uint32_t rows, uint32_t cols)
{
    ifstream file(path);
    if (file.is_open())
    {

        int dummy1, dummy2;
        file >> dummy1 >> dummy2; // Skip the dimensions

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                file >> matrix[i * cols + j];
            }
        }

        file.close();
    }
    else
    {
        return -1;
    }

    return 0;
}

int write_matrix_result(fs::path filePath, const float *matrix, uint32_t rows, uint32_t cols)
{
    ofstream file(filePath);
    if (file.is_open())
    {
        // Write rows and columns
        file << rows << " " << cols << "\n";

        for (uint32_t i = 0; i < rows; i++)
        {
            for (uint32_t j = 0; j < cols; j++)
            {
                file << matrix[i * cols + j] << " ";
            }
            file << "\n"; // Newline after each row
        }

        file.close();
    }
    else
    {
        return -1;
    }

    return 0;
}

bool validate_result(fs::path result_file, fs::path reference_file)
{
    int result;
    uint32_t m, p;

    result = read_matrix_dimensions(result_file, m, p);
    if (result == -1)
    {
        cerr << "Failed to open file: " << result_file << endl;
        return 1;
    }

    float *C_result = new float[m * p];
    float *C_reference = new float[m * p];

    result = read_matrix_elements(result_file, C_result, m, p);
    if (result == -1)
    {
        cerr << "Failed to open file: " << result_file << endl;
        exit(1);
    }
    result = read_matrix_elements(result_file, C_reference, m, p);
    if (result == -1)
    {
        cerr << "Failed to open file: " << reference_file << endl;
        exit(1);
    }

    bool same = true;
    float c_res, c_ref;

    for (uint32_t i = 0; i < m; i++)
    {
        for (uint32_t j = 0; j < p; j++)
        {
            // Using a comparison with rounding because of precision error for floating point numbers
            c_res = C_result[i * p + j];
            c_ref = C_reference[i * p + j];
            c_res = std::round(c_res * 100.0f) / 100.0f;
            c_ref = std::round(c_ref * 100.0f) / 100.0f;
            if (c_res != c_ref)
            {
                cout << format("Difference at row = {} col = {} element1 = {} element2 = {}", i, j, c_res, c_ref) << endl;
                same = false;
                break;
            }
        }

        if (same == false)
            break;
    }

    delete[] C_result;
    delete[] C_reference;

    return same;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <case_number>" << endl;
        return 1;
    }

    int case_number = atoi(argv[1]);
    if (case_number < 0 || case_number > 9)
    {
        cerr << "Case number must be between 0 and 9" << endl;
        return 1;
    }

    // Construct file paths
    fs::path folder = fs::path(SOURCE_DIR) / "data" / to_string(case_number);
    fs::path input0_file = folder / "input0.raw";
    fs::path input1_file = folder / "input1.raw";
    fs::path result_file = folder / "result.raw";
    fs::path reference_file = folder / "output.raw";

    uint32_t m, n, p;
    // A is m x n, B is n x p, C is m x p

    // Read input0.raw (matrix A)
    int result = read_matrix_dimensions(input0_file, m, n);
    if (result == -1)
    {
        cerr << "Failed to open file: " << input0_file << endl;
        return 1;
    }

    float *A = new float[m * n];

    result = read_matrix_elements(input0_file, A, m, n);
    if (result == -1)
    {
        cerr << "Failed to open file: " << input0_file << endl;
        return 1;
    }

    // Read input1.raw (matrix B)
    result = read_matrix_dimensions(input1_file, n, p);
    if (result == -1)
    {
        cerr << "Failed to open file: " << input1_file << endl;
        return 1;
    }

    float *B = new float[n * p];

    result = read_matrix_elements(input1_file, B, n, p);
    if (result == -1)
    {
        cerr << "Failed to open file: " << input1_file << endl;
        return 1;
    }

    // Allocate memory for result matrices
    float *C_naive = new float[m * p]{0};
    float *C_blocked = new float[m * p]{0};
    float *C_parallel = new float[m * p]{0};

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    // Write naive result to file
    result = write_matrix_result(result_file, C_naive, m, p);
    if (result == -1)
    {
        cerr << "Failed to write to file: " << result_file << endl;
        return 1;
    }

    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct)
    {
        cerr << "Naive result validation failed for case " << case_number << endl;
    }

    // Measure performance of blocked_matmul (use block_size = 32 as default)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 64);
    double blocked_time = omp_get_wtime() - start_time;

    // Write blocked result to file
    result = write_matrix_result(result_file, C_blocked, m, p);
    if (result == -1)
    {
        cerr << "Failed to write to file: " << result_file << endl;
        return 1;
    }

    // Validate blocked result
    bool blocked_correct = validate_result(result_file, reference_file);
    if (!blocked_correct)
    {
        cerr << "Blocked result validation failed for case " << case_number << endl;
    }

    // Measure performance of parallel_matmul
    omp_set_num_threads(2);
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    // Write parallel result to file
    result = write_matrix_result(result_file, C_parallel, m, p);
    if (result == -1)
    {
        cerr << "Failed to write to file: " << result_file << endl;
        return 1;
    }

    // Validate parallel result
    bool parallel_correct = validate_result(result_file, reference_file);
    if (!parallel_correct)
    {
        cerr << "Parallel result validation failed for case " << case_number << endl;
    }

    // Print performance results
    cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    cout << "Naive time: " << naive_time << " seconds\n";
    cout << "Blocked time: " << blocked_time << " seconds\n";
    cout << "Parallel time: " << parallel_time << " seconds\n";
    cout << "Blocked speedup: " << (naive_time / blocked_time) << "x\n";
    cout << "Parallel speedup: " << (naive_time / parallel_time) << "x\n";

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;

    return 0;
}