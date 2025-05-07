#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <cstdint>

void naive_matmul(float *C, const float *A, const float *B, uint32_t m, uint32_t n, uint32_t p) {
    // Implement naive matrix multiplication
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < p; j++) {
            float sum = 0.00f;
            for (uint32_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

void blocked_matmul(float *C, const float *A, const float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    // Implement blocked matrix multiplication
    // A is m x n, B is n x p, C is m x p
    // Use block_size to divide matrices into submatrices
    // C = A * B
    for (uint32_t ii = 0; ii < m; ii += block_size) {
        uint32_t i_max = (ii + block_size > m) ? m : ii + block_size;
        for (uint32_t jj = 0; jj < p; jj += block_size) {
            uint32_t j_max = (jj + block_size > p) ? p : jj + block_size;
            for (uint32_t kk = 0; kk < n; kk += block_size) {
                uint32_t k_max = (kk + block_size > n) ? n : kk + block_size;
                 for (uint32_t i = ii; i < i_max; i++) {
                    for (uint32_t j = jj; j < j_max; j++) {
                        float sum = C[i * p + j];
                        for (uint32_t k = kk; k < k_max; k++) {
                             sum += A[i * n + k] * B[k * p + j];
                        }
                        C[i * p + j] = sum;
                    }
                }
            }
        }
    }
}

void parallel_matmul(float *C, const float *A, const float *B, uint32_t m, uint32_t n, uint32_t p) {
    // Implement parallel matrix multiplication using OpenMP
    // A is m x n, B is n x p, C is m x p

    omp_set_num_threads(4);

    #pragma omp parallel for
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < p; j++) {
            float sum = 0.00f;
            for (uint32_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

// Writing the files values into the matrix
void read_file(std::ifstream &input, float *matrix, uint32_t x, uint32_t y) {
    for (uint32_t j = 0; j < x; j++) {
        for (uint32_t k = 0; k < y; k++) {
            input >> matrix[j * y + k];
        }
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    int x, y;
    float *R, *O;

    // Implement result validation
    std::ifstream result(result_file);
    if (result.is_open()) {
        result >> x >> y;
        R = new float[x * y];

        read_file(result, R, x, y);
    } else {
        exit(1);
    }

    std::ifstream output(reference_file);
    if (output.is_open()) {
        output >> x >> y;
        O = new float[x * y];

        read_file(output, O, x, y);
    } else {
        exit(1);
    }

    const float EPSILON = 1e-2f;

    for (int j = 0; j < x; j++) {
        for (int k = 0; k < y; k++) {
            if (std::fabs(R[j * y + k] - O[j * y + k]) > EPSILON) {
                std::cout << R[j * y + k] << " " << O[j * y + k] << std::endl;
                delete[] R;
                delete[] O;
                return false;
            }
        }
    }

    result.close();
    output.close();
    delete[] R;
    delete[] O;
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
    std::string result_file_naive = folder + "result_naive.raw";
    std::string result_file_blocked = folder + "result_blocked.raw";
    std::string result_file_parallel = folder + "result_parallel.raw";
    std::string reference_file = folder + "output.raw";

    // Read input0.raw (matrix A)
    float *A, *B;
    uint32_t m, n, p;

    std::ifstream input0(input0_file);
    if (input0.is_open()) {
        input0 >> m >> n;
        A = new float[m * n];

        read_file(input0, A, m, n);
    } else {
        exit(1);
    }

    input0.close();

    // Read input1.raw (matrix B)
    std::ifstream input1(input1_file);
    if (input1.is_open()) {
        input1 >> n >> p;
        B = new float[n * p];

        read_file(input1, B, n, p);
    } else {
        exit(1);
    }

    input1.close();

    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    // Write naive result to file
    std::ofstream result_naive(result_file_naive);

    result_naive << m << " " << p << "\n";
    for (uint32_t j = 0; j < m; j++) {
        for (uint32_t k = 0; k < p; k++) {
            result_naive << C_naive[j * p + k];

            if (k != p - 1) {
                result_naive << " ";
            }
        }
        if (j != m - 1) {
            result_naive << "\n";
        }
    }
    result_naive.close();

    // Validate naive result
    bool naive_correct = validate_result(result_file_naive, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of blocked_matmul (use block_size = 32 as default)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 64);
    double blocked_time = omp_get_wtime() - start_time;

    // Write blocked result to file
    std::ofstream result_blocked(result_file_blocked);

    result_blocked << m << " " << p << "\n";
    for (uint32_t j = 0; j < m; j++) {
        for (uint32_t k = 0; k < p; k++) {
            result_blocked << C_blocked[j * p + k];

            if (k != p - 1) {
                result_blocked << " ";
            }
        }
        if (j != m - 1) {
            result_blocked << "\n";
        }
    }
    result_blocked.close();

    // Validate blocked result
    bool blocked_correct = validate_result(result_file_blocked, reference_file);
    if (!blocked_correct) {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    // Write parallel result to file
    std::ofstream result_parallel(result_file_parallel);

    result_parallel << m << " " << p << "\n";
    for (uint32_t j = 0; j < m; j++) {
        for (uint32_t k = 0; k < p; k++) {
            result_parallel << C_parallel[j * p + k];

            if (k != p - 1) {
                result_parallel << " ";
            }
        }
        if (j != m - 1) {
            result_parallel << "\n";
        }
    }
    result_parallel.close();

    // Validate parallel result
    bool parallel_correct = validate_result(result_file_parallel, reference_file);
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
