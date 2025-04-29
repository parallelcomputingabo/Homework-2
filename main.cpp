#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <cstdint>

// Native matrix multiplication
void naive_matmul(double* C, double* A, double* B, uint32_t m, uint32_t n, uint32_t p) {
    // Initialize the result matrix C to zero
    for (uint32_t i = 0; i < m * p; ++i) {
        C[i] = 0.0f;
    }
    
    // Perform the matrix multiplication
    for (uint32_t i = 0; i < m; ++i) { 
        for (uint32_t j = 0; j < p; ++j) { 
            for (uint32_t k = 0; k < n; ++k) { 
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

void blocked_matmul(double *C, double *A, double *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    // TODO: Implement blocked matrix multiplication
    // A is m x n, B is n x p, C is m x p
    // Use block_size to divide matrices into submatrices
}

void parallel_matmul(double *C, double *A, double *B, uint32_t m, uint32_t n, uint32_t p) {
    // TODO: Implement parallel matrix multiplication using OpenMP
    // A is m x n, B is n x p, C is m x p
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
   //TODO : Implement result validation
}

// Read a matrix from text file (row-major)
double* read_matrix(const std::string& path, uint32_t& rows, uint32_t& cols) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Error: cannot open file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    in >> rows >> cols;
    double* mat = new double[static_cast<size_t>(rows) * cols];
    for (uint32_t i = 0; i < rows * cols; ++i) {
        in >> mat[i];
    }
    in.close();
    return mat;
}

// Write a matrix to text file (row-major)
void write_matrix(const std::string& path, const double* mat, uint32_t rows, uint32_t cols) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Error: cannot write to file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    out << rows << " " << cols << '\n';
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            out << mat[i * cols + j];
            if (j + 1 < cols) out << ' ';
        }
        out << '\n';
    }
    out.close();
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

    // Allocate memory for result matrices
    uint32_t m, n_A, n_B, n, p, m_D, p_D;  // A is m x n, B is n x p, C is m x p
    double *C_naive = new double[m * p];
    double *C_blocked = new double[m * p];
    double *C_parallel = new double[m * p];

    // TODO Read input0.raw (matrix A)
    std::cout << "Reading matrix A from: " << input0_file << std::endl;
    double* A = read_matrix(input0_file, m, n_A);

    // TODO Read input1.raw (matrix B)
    std::cout << "Reading matrix B from: " << input1_file << std::endl;
    double* B = read_matrix(input1_file, n_B, p);

    // Read dimensions and matrices from input0.raw and input1.raw
    std::cout << "Reading matrix D from: " << reference_file << std::endl;
    double* D = read_matrix(reference_file, m_D, p_D);

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    // TODO Write naive result to file


    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of blocked_matmul (use block_size = 32 as default)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 32);
    double blocked_time = omp_get_wtime() - start_time;

    // TODO Write blocked result to file


    // Validate blocked result
    bool blocked_correct = validate_result(result_file, reference_file);
    if (!blocked_correct) {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    // TODO Write parallel result to file


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