#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
 
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
    for (uint32_t ii = 0; ii < m; ii += block_size) {
        for (uint32_t jj = 0; jj < p; jj += block_size) {
            for (uint32_t kk = 0; kk < n; kk += block_size) {
                for (uint32_t i = ii; i < std::min(ii + block_size, m); ++i) {
                    for (uint32_t j = jj; j < std::min(jj + block_size, p); ++j) {
                        float sum = 0.0f;
                        for (uint32_t k = kk; k < std::min(kk + block_size, n); ++k) {
                            sum += A[i * n + k] * B[k * p + j];
                        }
                        C[i * p + j] += sum;
                    }
                }
            }
        }
    }
}
 
void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < static_cast<int>(m); ++i) {
        for (int j = 0; j < static_cast<int>(p); ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}
 
bool validate_result(const std::string &result_file, const std::string &reference_file) {
    std::ifstream result(result_file);
    std::ifstream reference(reference_file);
    if (!result.is_open() || !reference.is_open()) return false;
 
    uint32_t m_res, p_res, m_ref, p_ref;
    result >> m_res >> p_res;
    reference >> m_ref >> p_ref;
    if (m_res != m_ref || p_res != p_ref) return false;
 
    for (uint32_t i = 0; i < m_res * p_res; ++i) {
        float val_res, val_ref;
        result >> val_res;
        reference >> val_ref;
        if (std::fabs(val_res - val_ref) > 1e-3) return false;
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
 
    std::string folder = "F:/Homework-2/data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";
 
    std::ifstream input0(input0_file);
    std::ifstream input1(input1_file);
 
    uint32_t m, n, n_check, p;
    input0 >> m >> n;
    input1 >> n_check >> p;
    if (n != n_check) {
        std::cerr << "Matrix dimension mismatch." << std::endl;
        return 1;
    }
 
    float* A = new float[m * n];
    float* B = new float[n * p];
    for (uint32_t i = 0; i < m * n; ++i) input0 >> A[i];
    for (uint32_t i = 0; i < n * p; ++i) input1 >> B[i];
    input0.close();
    input1.close();
 
    float *C_naive = new float[m * p]();
    float *C_blocked = new float[m * p]();
    float *C_parallel = new float[m * p]();
 
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;
 
    std::ofstream out_naive(result_file);
    out_naive << m << " " << p << "\n";
    for (uint32_t i = 0; i < m * p; ++i) out_naive << C_naive[i] << " ";
    out_naive << "\n";
    out_naive.close();
 
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }
 
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 32);
    double blocked_time = omp_get_wtime() - start_time;
 
    std::ofstream out_blocked(result_file);
    out_blocked << m << " " << p << "\n";
    for (uint32_t i = 0; i < m * p; ++i) out_blocked << C_blocked[i] << " ";
    out_blocked << "\n";
    out_blocked.close();
 
    bool blocked_correct = validate_result(result_file, reference_file);
    if (!blocked_correct) {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }
 
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;
 
    std::ofstream out_parallel(result_file);
    out_parallel << m << " " << p << "\n";
    for (uint32_t i = 0; i < m * p; ++i) out_parallel << C_parallel[i] << " ";
    out_parallel << "\n";
    out_parallel.close();
 
    bool parallel_correct = validate_result(result_file, reference_file);
    if (!parallel_correct) {
        std::cerr << "Parallel result validation failed for case " << case_number << std::endl;
    }
 
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive time: " << naive_time << " seconds\n";
    std::cout << "Blocked time: " << blocked_time << " seconds\n";
    std::cout << "Parallel time: " << parallel_time << " seconds\n";
    std::cout << "Blocked speedup: " << (naive_time / blocked_time) << "x\n";
    std::cout << "Parallel speedup: " << (naive_time / parallel_time) << "x\n";
 
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;
 
    return 0;
}