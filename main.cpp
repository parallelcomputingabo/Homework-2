#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <cstdint>
#include <vector>


void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    //TODO : Implement naive matrix multiplication
    for (uint32_t i = 0; i < m*p; i++) {
        C[i] = 0.0f;
    }
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < p; j++) {
            float sum = 0.0;
            for (uint32_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] += sum;
        }
    }
}

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    // TODO: Implement blocked matrix multiplication
    // A is m x n, B is n x p, C is m x p
    // Use block_size to divide matrices into submatrices
    for (uint32_t i = 0; i < m*p; i++) {
        C[i] = 0.0f;
    }
    for (uint32_t ii = 0; ii < m; ii += block_size) {
        for (uint32_t jj = 0; jj < p; jj += block_size) {
            for (uint32_t kk = 0; kk < n; kk += block_size) {
                uint32_t i_end = std::min(ii + block_size, m);
                uint32_t j_end = std::min(jj + block_size, p);
                uint32_t k_end = std::min(kk + block_size, n);
                for (uint32_t i = ii; i < i_end; i++) {
                    for (uint32_t j = jj; j < j_end; j++) {
                        for (uint32_t k = kk; k < k_end; k++) {
                            C[i * p + j] += A[i * n + k] * B[k * p + j];
                        }
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
    for (uint32_t i = 0; i < m; i++)
        for (uint32_t j = 0; j < p; j++) {
            float sum = 0.0;
            for (uint32_t k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * p + j];
            C[i * p + j] = sum;
        }
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
   //TODO : Implement result validation
    std::ifstream res(result_file);
    std::ifstream ref(reference_file);

    if (!res.is_open() || !ref.is_open()) {
        return false;
    }

    // Introduced some tolerance since I don't seem to solve an issue with
    // getting the decimals written the same as the output.
    float tolerance = 1e-2f;
    float val1, val2;
    while (res >> val1 && ref >> val2) {
        if (std::abs(val1 - val2) > tolerance) {
            return false;
        }
    }

    // Check if one file had more lines
    if (res.eof() != ref.eof()) {
        return false;
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
    std::string folder = "../data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";

    // TODO Read input0.raw (matrix A)
    std::ifstream input0(input0_file);

    // TODO Read input1.raw (matrix B)
    std::ifstream input1(input1_file);

    if (!input0.is_open() || !input1.is_open()) {
        std::cerr << "Failed to open files" << std::endl;
        return 1;
    }
    uint32_t m, n, p;
    input0 >> m >> n;
    input1 >> n >> p;

    float *A = new float[m * n];
    float *B = new float[n * p];

    for (uint32_t i = 0; i < m * n; i++) {
        input0 >> A[i];
    }
    for (uint32_t i = 0; i < n * p; i++) {
        input1 >> B[i];
    }
    input0.close();
    input1.close();

    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // Measure performance of naive_matmul
    const int runs = 20;
    double tot_naive = 0;
    for (int i = 0; i < runs; i++) {
        double start_time = omp_get_wtime();
        naive_matmul(C_naive, A, B, m, n, p);
        double naive_time = omp_get_wtime() - start_time;
        tot_naive += naive_time;
    }


    // TODO Write naive result to file

    // Write results
    std::ofstream out(result_file);
    out << m << " " << p << std::endl;
    for (int i = 0; i < m;out<<std::endl, i++) {
        for (int j = 0; j < p; j++) {
            out<<C_naive[i * p + j]<<" ";
        }

    }

    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    out.close();
    out.clear();

    // Measure performance of blocked_matmul (use block_size = 32 as default)
    double tot_block = 0;
    for (int i = 0; i < runs; i++) {
        double start_time = omp_get_wtime();
        blocked_matmul(C_blocked, A, B, m, n, p, 32);
        double blocked_time = omp_get_wtime() - start_time;
        tot_block += blocked_time;

    }


    // TODO Write blocked result to file

    // Write results
    out.open(result_file);
    out << m << " " << p << std::endl;
    for (int i = 0; i < m;out<<std::endl, i++) {
        for (int j = 0; j < p; j++) {
            out<<C_blocked[i * p + j]<<" ";
        }

    }

    // Validate blocked result
    bool blocked_correct = validate_result(result_file, reference_file);
    if (!blocked_correct) {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }
    out.close();
    out.clear();

    // Measure performance of parallel_matmul
    double tot_parallel = 0;
    for (int i = 0; i < runs; i++) {
        double start_time = omp_get_wtime();
        parallel_matmul(C_parallel, A, B, m, n, p);
        double parallel_time = omp_get_wtime() - start_time;
        tot_parallel += parallel_time;
    }


    // TODO Write parallel result to file

    // Write results
    out.open(result_file);
    out << m << " " << p << std::endl;
    for (int i = 0; i < m;out<<std::endl, i++) {
        for (int j = 0; j < p; j++) {
            out<<C_parallel[i * p + j]<<" ";
        }

    }

    // Validate parallel result
    bool parallel_correct = validate_result(result_file, reference_file);
    if (!parallel_correct) {
        std::cerr << "Parallel result validation failed for case " << case_number << std::endl;
    }

    out.close();
    out.clear();

    double avg_naive = tot_naive / runs;
    double avg_blocked = tot_block / runs;
    double avg_parallel = tot_parallel / runs;

    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive time: " << avg_naive << " seconds\n";
    std::cout << "Blocked time: " << avg_blocked << " seconds\n";
    std::cout << "Parallel time: " << avg_parallel << " seconds\n";
    std::cout << "Blocked speedup: " << (avg_naive / avg_blocked) << "x\n";
    std::cout << "Parallel speedup: " << (avg_naive / avg_parallel) << "x\n";

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;

    return 0;
}