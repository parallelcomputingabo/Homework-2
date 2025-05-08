#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <omp.h>
using namespace std;

// Read a matrix: first two ints are rows, cols, then the whole matrix floats
bool read_matrix(const std::string& filename, float*& mat,
                 int& rows, int& cols)
{
    std::ifstream in(filename.c_str());
    if (!in) return false;
    in >> rows >> cols;
    mat = (float*)std::malloc(rows * cols * sizeof(float));
    if (!mat) return false;
    int total = rows * cols;
    for (int i = 0; i < total; ++i) {
        in >> mat[i];
    }
    return true;
}

// Write rows cols, then the whole matrix floats
bool write_matrix(const std::string& filename,
                  const float* mat, int rows, int cols)
{
    std::ofstream out(filename.c_str());
    if (!out) return false;
    out << rows << " " << cols << "\n"
        << std::fixed << std::setprecision(2);
    int total = rows * cols;
    for (int i = 0; i < total; ++i) {
        out << mat[i]
            << ((i % cols == cols-1) ? "\n" : " ");
    }
    return true;
}

// Naive triple-loop matmul: C = A * B
void naive_matmul(float *C, const float *A, const float *B,
                  int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i*n + k] * B[k*p + j];
            }
            C[i*p + j] = sum;
        }
    }
}

// Blocked matmul to improve cache reuse
void blocked_matmul(float *C, const float *A, const float *B,
                    int m, int n, int p,
                    int block_size) {
    std::memset(C, 0, m * p * sizeof(float));
    for (int ii = 0; ii < m; ii += block_size) {
        for (int kk = 0; kk < n; kk += block_size) {
            for (int jj = 0; jj < p; jj += block_size) {
                int i_max = std::min(ii + block_size, m);
                int k_max = std::min(kk + block_size, n);
                int j_max = std::min(jj + block_size, p);
                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        float a_val = A[i*n + k];
                        for (int j = jj; j < j_max; ++j) {
                            C[i*p + j] += a_val * B[k*p + j];
                        }
                    }
                }
            }
        }
    }
}

// Parallel matmul with OpenMP on the outer (i) loop
void parallel_matmul(float *C, const float *A, const float *B,
                     int m, int n, int p) {
    std::memset(C, 0, m * p * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < n; ++k) {
            float a_val = A[i*n + k];
            for (int j = 0; j < p; ++j) {
                C[i*p + j] += a_val * B[k*p + j];
            }
        }
    }
}

// Validate by reading two text matrices
bool validate_result(const std::string &result_file,
                     const std::string &reference_file) {
    float *C1, *C2;
    int r1, c1, r2, c2;
    if (!read_matrix(result_file, C1, r1, c1)) return false;
    if (!read_matrix(reference_file, C2, r2, c2)) return false;
    if (r1 != r2 || c1 != c2) { std::free(C1); std::free(C2); return false; }
    constexpr float EPS = 1e-5f;
    int total = r1 * c1;
    for (int i = 0; i < total; ++i) {
        if (std::fabs(C1[i] - C2[i]) > EPS) {
            std::free(C1);
            std::free(C2);
            return false;
        }
    }
    std::free(C1);
    std::free(C2);
    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <case_number>\n";
        return 1;
    }
    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9) {
        std::cerr << "Case number must be between 0 and 9\n";
        return 1;
    }
    // Construct file names based on case number
    std::string folder = "data/" + std::to_string(case_number) + "/";
    std::string input0_file    = folder + "input0.raw";
    std::string input1_file    = folder + "input1.raw";
    std::string result_file    = folder + "result.raw";
    std::string reference_file = folder + "output.raw";

    float *A, *B;
    int m, n, n2, p;
    if (!read_matrix(input0_file, A, m, n) ||
        !read_matrix(input1_file, B, n2, p) ||
        n2 != n) {
        std::cerr << "Failed to read input matrices or dimension mismatch\n";
        return 1;
    }
    //Allocate result matrices
    float *C_naive    = (float*)std::malloc(m * p * sizeof(float));
    float *C_blocked  = (float*)std::malloc(m * p * sizeof(float));
    float *C_parallel = (float*)std::malloc(m * p * sizeof(float));

    // Display OpenMP information
    #ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    std::cout << "Running with " << num_threads << " OpenMP threads" << std::endl;
    #else
    std::cout << "OpenMP not available. Running in single-threaded mode." << std::endl;
    #endif

    double start_time, naive_time, blocked_time, parallel_time;

    // Naive
    start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    naive_time = omp_get_wtime() - start_time;
    write_matrix(result_file, C_naive, m, p);
    if (!validate_result(result_file, reference_file))
        std::cerr << "[ERROR] Naive result incorrect\n";

    // Blocked
    // Blocked: try multiple block sizes and pick best
    uint32_t block_sizes[] = {16, 32, 64, 128};
    double best_blocked_time = __DBL_MAX__;
    uint32_t best_block_size = 32; // Default

    for (uint32_t block_size : block_sizes) {
        start_time = omp_get_wtime();
        blocked_matmul(C_blocked, A, B, m, n, p, block_size);
        double blocked_time = omp_get_wtime() - start_time;

        write_matrix(result_file, C_blocked, m, p);
        bool blocked_correct = validate_result(result_file, reference_file);

        std::cout << "block_size=(" << block_size << "): " 
                  << blocked_time << " s - " 
                  << (blocked_correct ? "Result is Correct" : "Result is Incorrect") << std::endl;

        if (blocked_time < best_blocked_time && blocked_correct) {
            best_blocked_time = blocked_time;
            best_block_size = block_size;
        }
    }
    // Parallel
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    parallel_time = omp_get_wtime() - start_time;
    write_matrix(result_file, C_parallel, m, p);
    if (!validate_result(result_file, reference_file))
        std::cerr << "[ERROR] Parallel result incorrect\n";

    std::cout << "Case " << case_number
              << " (" << m << "×" << n << "×" << p << "):\n"
              << "  Naive time:    " << naive_time    << " s\n"
              << "  Blocked time:  " << best_blocked_time  << " s\n"
              << "  Parallel time: " << parallel_time << " s\n"
              << "  Blocked speedup:  " << (naive_time/best_blocked_time)
              << "×\n"
              << "  Parallel speedup: " << (naive_time/parallel_time)
              << "×\n";
    std::free(A);
    std::free(B);
    std::free(C_naive);
    std::free(C_blocked);
    std::free(C_parallel);
    return 0;
}
