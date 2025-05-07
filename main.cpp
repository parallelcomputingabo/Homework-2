#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <omp.h>

void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    for (uint32_t i = 0; i < m * p; ++i) {
        C[i] = 0.0f;
    }
    // A is m x n, B is n x p, C is m x p
    // Use block_size to divide matrices into submatrices
    for (uint32_t ii = 0; ii < m; ii += block_size) {
        for (uint32_t kk = 0; kk < n; kk += block_size) {
            for (uint32_t jj = 0; jj < p; jj += block_size) {
                uint32_t i_max = std::min(ii + block_size, m);
                uint32_t k_max = std::min(kk + block_size, n);
                uint32_t j_max = std::min(jj + block_size, p);
                for (uint32_t i = ii; i < i_max; ++i) {
                    for (uint32_t k = kk; k < k_max; ++k) {
                        float a_val = A[i * n + k];
                        for (uint32_t j = jj; j < j_max; ++j) {
                            C[i * p + j] += a_val * B[k * p + j];
                        }
                    }
                }
            }
        }
    }
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    for (uint32_t i = 0; i < m * p; ++i) {
        C[i] = 0.0f;
    }
    // A is m x n, B is n x p, C is m x p
    #pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    std::ifstream res(result_file);
    std::ifstream ref(reference_file);
    if (!res.is_open() || !ref.is_open()) {
        return false;
    }
    uint32_t rm, rp, cm, cp;
    res >> rm >> rp;
    ref >> cm >> cp;
    if (rm != cm || rp != cp) return false;
    float vres, vref;
    const float eps = 1e-6f;
    for (uint32_t i = 0; i < rm * rp; ++i) {
        if (!(res >> vres) || !(ref >> vref)) return false;
        if (std::fabs(vres - vref) > eps) {
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

    // Read matrix A
    uint32_t m, n, p;
    float *A = nullptr;
    float *B = nullptr;
    {
        std::ifstream inA(input0_file);
        if (!inA.is_open()) {
            std::cerr << "Error opening " << input0_file << std::endl;
            return 1;
        }
        inA >> m >> n;
        A = new float[m * n];
        for (uint32_t i = 0; i < m * n; ++i) {
            inA >> A[i];
        }
    }

    // Read matrix B
    {
        std::ifstream inB(input1_file);
        if (!inB.is_open()) {
            std::cerr << "Error opening " << input1_file << std::endl;
            delete[] A;
            return 1;
        }
        uint32_t n2;
        inB >> n2 >> p;
        if (n2 != n) {
            std::cerr << "Inner dimensions do not match" << std::endl;
            delete[] A;
            return 1;
        }
        B = new float[n * p];
        for (uint32_t i = 0; i < n * p; ++i) {
            inB >> B[i];
        }
    }


    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    // Write naive result to file
    {
        std::ofstream out(result_file);
        out << m << " " << p << "\n";
        for (uint32_t i = 0; i < m * p; ++i) {
            float raw = C_naive[i];
            float scaled = std::floor(raw * 100.0f + 0.5f);
            int s = static_cast<int>(scaled);
            out << (s / 100) << "." << std::setw(2) << std::setfill('0') << (s % 100);
            if ((i + 1) % p == 0) out << "\n"; else out << " ";
        }
    }
    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of blocked_matmul (use block_size = 32 as default)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 32);
    double blocked_time = omp_get_wtime() - start_time;

    // Write blocked result to file
    {
        std::ofstream out(result_file);
        out << m << " " << p << "\n";
        for (uint32_t i = 0; i < m * p; ++i) {
            float raw = C_blocked[i];
            float scaled = std::floor(raw * 100.0f + 0.5f);
            int s = static_cast<int>(scaled);
            out << (s / 100) << "." << std::setw(2) << std::setfill('0') << (s % 100);
            if ((i + 1) % p == 0) out << "\n"; else out << " ";
        }
    }
    // Validate blocked result
    bool blocked_correct = validate_result(result_file, reference_file);
    if (!blocked_correct) {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    // Write parallel result to file
    {
        std::ofstream out(result_file);
        out << m << " " << p << "\n";
        for (uint32_t i = 0; i < m * p; ++i) {
            float raw = C_parallel[i];
            float scaled = std::floor(raw * 100.0f + 0.5f);
            int s = static_cast<int>(scaled);
            out << (s / 100) << "." << std::setw(2) << std::setfill('0') << (s % 100);
            if ((i + 1) % p == 0) out << "\n"; else out << " ";
        }
    }
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