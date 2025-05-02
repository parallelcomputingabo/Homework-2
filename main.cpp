#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>

// To read the matric from file
float *read_matrix(const std::string &filename, int &rows, int &cols)
{
    std::ifstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    file >> rows >> cols;
    float *mat = new float[rows * cols];

    for (int i = 0; i < rows * cols; ++i)
    {
        file >> mat[i];
    }

    file.close();
    return mat;
}

// To write the matrix to the file
void write_matrix(const std::string &filename, float *mat, int rows, int cols)
{
    std::ofstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    file << rows << " " << cols << "\n";
    for (int i = 0; i < rows * cols; ++i)
    {
        file << mat[i] << " ";
        if ((i + 1) % cols == 0)
        {
            file << "\n";
        }
    }

    file.close();
}

void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    //TODO : Implement naive matrix multiplication
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            float sum = 0;
            for (int k = 0; k < n; ++k)
            {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    // TODO: Implement blocked matrix multiplication
    // A is m x n, B is n x p, C is m x p
    // Use block_size to divide matrices into submatrices
    for (int ii=0; ii < m; ii += block_size) {
        for (int jj=0; jj < p; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                for (int i = ii; i < std::min(ii + block_size, m); i++) {
                    for (int j = jj; j < std::min(jj + block_size, p); ++j) {
                        float sum = 0.0f;   
                        for (int k = kk; k < std::min(kk + block_size, n); ++k) {
                            sum  += A[i * n + k] * B[k * p + j];
                        }
                        C[i * p + j] += sum;
                    }
                }
            }
        }
    }
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // TODO: Implement parallel matrix multiplication using OpenMP
    // A is m x n, B is n x p, C is m x p
    #pragma omp parallel for
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            float sum = 0;
            for (int k = 0; k < n; ++k)
            {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
   //TODO : Implement result validation
    int result_rows, result_cols, ref_rows, ref_cols;
    float *result = read_matrix(result_file, result_rows, result_cols);
    float *reference = read_matrix(reference_file, ref_rows, ref_cols);

    const float EPSILON = 1e-4f;

    for (int i = 0; i < result_rows * result_cols; ++i)
    {
        if (std::abs(result[i] - reference[i]) > EPSILON)
        {
            return false;
        }
    }

    return true;

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

    // TODO Read input0.raw (matrix A)
    int m, n, p; // A is m x n, B is n x p, C is m x p
    float *A = read_matrix(input0_file, m, n);
    
    // TODO Read input1.raw (matrix B)
    float *B = read_matrix(input1_file, n, p);


    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    // TODO Write naive result to file
    write_matrix(result_file, C_naive, m, p);


    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    //std::cout << "naive_correct " << naive_correct << std::endl;
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of blocked_matmul (use block_size = 32 as default)
    std::fill(C_blocked, C_blocked + m * p, 0.0f); 
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 64);
    double blocked_time = omp_get_wtime() - start_time;
    // TODO Write blocked result to file
    write_matrix(result_file, C_blocked, m, p);

    // Validate blocked result
    bool blocked_correct = validate_result(result_file, reference_file);
    std::cout << "blocked_correct " << blocked_correct << std::endl;
    if (!blocked_correct) {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    // TODO Write parallel result to file
    write_matrix(result_file, C_parallel, m, p);

    // Validate parallel result
    bool parallel_correct = validate_result(result_file, reference_file);
    if (!parallel_correct) {
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