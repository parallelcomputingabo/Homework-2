#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <cassert>  // for assert
#include <cstring>  // for memset

void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    //TODO : Implement naive matrix multiplication
    for( uint32_t i = 0; i < m; i++) {
        for( uint32_t j = 0; j < p; j++) {
            for( uint32_t k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    // TODO: Implement blocked matrix multiplication
    // A is m x n, B is n x p, C is m x p
    // Use block_size to divide matrices into submatrices
    // C = A * B

    memset(C, 0, m * p * sizeof(float));
    for (int ii = 0; ii < m; ii += block_size)
        for (int jj = 0; jj < p; jj += block_size)
            for (int kk = 0; kk < n; kk += block_size)
                // Process block: C[ii:ii+block_size, jj:jj+block_size] += A[ii:ii+block_size, kk:kk    +block_size] * B[kk:kk+block_size, jj:jj+block_size]
                for (int i = ii; i < std::min(ii + block_size, m); i++)
                    for (int j = jj; j < std::min(jj + block_size, p); j++)
                        for (int k = kk; k < std::min(kk + block_size, n); k++)
                            C[i * p + j] += A[i * n + k] * B[k * p + j];
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, int num_threads) {
    // Set the number of threads
    omp_set_num_threads(num_threads);
    std::cout << "Running parallel_matmul with " << num_threads << " threads" << std::endl;

    // Clear the result matrix
    memset(C, 0, m * p * sizeof(float));

    // Parallelize the outer loop over rows
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++)
            for (int k = 0; k < n; k++)
                C[i * p + j] += A[i * n + k] * B[k * p + j];
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
   //TODO : Implement result validation
   std::ifstream result(result_file);
   std::ifstream reference(reference_file);
   int m_result, p_result;
   int m_reference, p_reference;
   result >> m_result >> p_result;
   reference >> m_reference >> p_reference;
   assert(m_result == m_reference && p_result == p_reference && "Matrix dimensions do not match");
   float* result_matrix = new float[m_result * p_result];
   float* reference_matrix = new float[m_reference * p_reference];
   if(result_matrix == nullptr || reference_matrix == nullptr) {
    std::cerr << "Memory allocation failed" << std::endl;
    return false;
   }
   for(int i = 0; i < m_result; i++) {
        for(int j = 0; j < p_result; j++) {
            result >> result_matrix[i * p_result + j];
        }
   }
   for(int i = 0; i < m_reference; i++) {
    for(int j = 0; j < p_reference; j++) {
        reference >> reference_matrix[i * p_reference + j];
    }
   }
   const float epsilon = 1e-3;
    for(int i = 0; i < m_result; i++) {
        for(int j = 0; j < p_result; j++) {
            if(std::fabs(result_matrix[i * p_result + j] - reference_matrix[i * p_reference + j]) > epsilon) {
                std::cerr << "Validation failed at (" << i << ", " << j << "): expected " << reference_matrix[i * p_reference + j] << ", got " << result_matrix[i * p_result + j] << std::endl;
                return false;
            }
        }
    }
    delete[] result_matrix;
    delete[] reference_matrix;
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
    std::string blocked_8_result_file = folder + "blocked_8_result.raw";
    std::string blocked_16_result_file = folder + "blocked_16_result.raw";
    std::string blocked_32_result_file = folder + "blocked_32_result.raw";
    std::string blocked_64_result_file = folder + "blocked_64_result.raw";
    std::string blocked_128_result_file = folder + "blocked_128_result.raw";
    std::string parallel_2_result_file = folder + "parallel_2_result.raw";
    std::string parallel_4_result_file = folder + "parallel_4_result.raw";
    std::string parallel_8_result_file = folder + "parallel_8_result.raw";
    std::string parallel_16_result_file = folder + "parallel_16_result.raw";
    std::string reference_file = folder + "output.raw";

    int m, n, p;  // A is m x n, B is n x p, C is m x p
    // TODO Read input0.raw (matrix A)
    int rol_input0, col_input0, rol_input1, col_input1;
    std::ifstream input0_file_ifstream(input0_file);
    std::ifstream input1_file_ifstream(input1_file);
    input0_file_ifstream >> rol_input0 >> col_input0;
    std::cout<< "Matrix A dimensions: " << rol_input0 << " x " << col_input0 << std::endl;
    

    // TODO Read input1.raw (matrix B)
    input1_file_ifstream >> rol_input1 >> col_input1;
    std::cout<< "Matrix B dimensions: " << rol_input1 << " x " << col_input1 << std::endl;

    assert( col_input0 == rol_input1 && "Matrix dimensions do not match for multiplication");
    m = rol_input0;
    n = col_input0;
    p = col_input1;
    float* A = new float[m * n];
    float* B = new float[n * p];
    if(A == nullptr || B == nullptr) {
        std::cerr << "Memory allocation for A and B failed" << std::endl;
        return 1;
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            input0_file_ifstream >> A[i * n + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            input1_file_ifstream >> B[i * p + j];
        }
    }
    input0_file_ifstream.close();
    input1_file_ifstream.close();

    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    memset(C_naive, 0, m * p * sizeof(float));
    memset(C_blocked, 0, m * p * sizeof(float));
    memset(C_parallel, 0, m * p * sizeof(float));

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    // TODO Write naive result to file
    std::ofstream result(result_file);
    if (!result.is_open()) {
        std::cerr << "Error: Could not open result.raw" << std::endl;
        return 1;
    }
    result << m << " " << p << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            result << C_naive[i * p + j] << " ";
        }
        result << std::endl;
    }
    result.close();

    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of blocked_matmul (use block_size = 8, 16, 32, 64, 128)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 8);
    double blocked_time_8 = omp_get_wtime() - start_time;

    // TODO Write blocked result to file
    std::ofstream blocked_8_result(blocked_8_result_file);
    if (!blocked_8_result.is_open()) {
        std::cerr << "Error: Could not open blocked_8_result.raw" << std::endl;
        return 1;
    }
    blocked_8_result << m << " " << p << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            blocked_8_result << C_blocked[i * p + j] << " ";
        }
        blocked_8_result << std::endl;
    }
    blocked_8_result.close(); 
    

    // Validate blocked result
    bool blocked_8_correct = validate_result(blocked_8_result_file, reference_file);
    if (!blocked_8_correct) {
        std::cerr << "Blocked (8) result validation failed for case " << case_number << std::endl;
    }

    memset(C_blocked, 0, m * p * sizeof(float));
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 16);
    double blocked_time_16 = omp_get_wtime() - start_time;

    // TODO Write blocked result to file
    std::ofstream blocked_16_result(blocked_16_result_file);
    if (!blocked_16_result.is_open()) {
        std::cerr << "Error: Could not open blocked_16_result.raw" << std::endl;
        return 1;
    }
    blocked_16_result << m << " " << p << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            blocked_16_result << C_blocked[i * p + j] << " ";
        }
        blocked_16_result << std::endl;
    }
    blocked_16_result.close(); 
    

    // Validate blocked result
    bool blocked_16_correct = validate_result(blocked_16_result_file, reference_file);
    if (!blocked_16_correct) {
        std::cerr << "Blocked (16) result validation failed for case " << case_number << std::endl;
    }

    memset(C_blocked, 0, m * p * sizeof(float));
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 32);
    double blocked_time_32 = omp_get_wtime() - start_time;

    // TODO Write blocked result to file
    std::ofstream blocked_32_result(blocked_32_result_file);
    if (!blocked_32_result.is_open()) {
        std::cerr << "Error: Could not open blocked_32_result.raw" << std::endl;
        return 1;
    }
    blocked_32_result << m << " " << p << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            blocked_32_result << C_blocked[i * p + j] << " ";
        }
        blocked_32_result << std::endl;
    }
    blocked_32_result.close(); 
    

    // Validate blocked result
    bool blocked_32_correct = validate_result(blocked_32_result_file, reference_file);
    if (!blocked_32_correct) {
        std::cerr << "Blocked (32) result validation failed for case " << case_number << std::endl;
    }

    memset(C_blocked, 0, m * p * sizeof(float));
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 64);
    double blocked_time_64 = omp_get_wtime() - start_time;

    // TODO Write blocked result to file
    std::ofstream blocked_64_result(blocked_64_result_file);
    if (!blocked_64_result.is_open()) {
        std::cerr << "Error: Could not open blocked_64_result.raw" << std::endl;
        return 1;
    }
    blocked_64_result << m << " " << p << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            blocked_64_result << C_blocked[i * p + j] << " ";
        }
        blocked_64_result << std::endl;
    }
    blocked_64_result.close(); 
    

    // Validate blocked result
    bool blocked_64_correct = validate_result(blocked_64_result_file, reference_file);
    if (!blocked_64_correct) {
        std::cerr << "Blocked (64) result validation failed for case " << case_number << std::endl;
    }

    memset(C_blocked, 0, m * p * sizeof(float));
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 128);
    double blocked_time_128 = omp_get_wtime() - start_time;
    
    // TODO Write blocked result to file
    std::ofstream blocked_128_result(blocked_128_result_file);
    if (!blocked_128_result.is_open()) {
        std::cerr << "Error: Could not open blocked_128_result.raw" << std::endl;
        return 1;
    }
    blocked_128_result << m << " " << p << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            blocked_128_result << C_blocked[i * p + j] << " ";
        }
        blocked_128_result << std::endl;
    }
    blocked_128_result.close(); 
    

    // Validate blocked result
    bool blocked_128_correct = validate_result(blocked_128_result_file, reference_file);
    if (!blocked_128_correct) {
        std::cerr << "Blocked (128) result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of parallel_matmul
    memset(C_parallel, 0, m * p * sizeof(float));
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p, 2);
    double parallel_time_2 = omp_get_wtime() - start_time;

    // TODO Write parallel result to file
    std::ofstream parallel_2_result(parallel_2_result_file);
    if (!parallel_2_result.is_open()) {
        std::cerr << "Error: Could not open parallel_2_result.raw" << std::endl;
        return 1;
    }
    parallel_2_result << m << " " << p << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            parallel_2_result << C_parallel[i * p + j] << " ";
        }
        parallel_2_result << std::endl;
    }
    parallel_2_result.close();


    // Validate parallel result
    bool parallel_2_correct = validate_result(parallel_2_result_file, reference_file);
    if (!parallel_2_correct) {
        std::cerr << "Parallel (2 threads) result validation failed for case " << case_number << std::endl;
    }

    memset(C_parallel, 0, m * p * sizeof(float));
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p, 4);
    double parallel_time_4 = omp_get_wtime() - start_time;

    // TODO Write parallel result to file
    std::ofstream parallel_4_result(parallel_4_result_file);
    if (!parallel_4_result.is_open()) {
        std::cerr << "Error: Could not open parallel_4_result.raw" << std::endl;
        return 1;
    }
    parallel_4_result << m << " " << p << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            parallel_4_result << C_parallel[i * p + j] << " ";
        }
        parallel_4_result << std::endl;
    }
    parallel_4_result.close();


    // Validate parallel result
    bool parallel_4_correct = validate_result(parallel_4_result_file, reference_file);
    if (!parallel_4_correct) {
        std::cerr << "Parallel (4 threads) result validation failed for case " << case_number << std::endl;
    }

    memset(C_parallel, 0, m * p * sizeof(float));
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p, 8);
    double parallel_time_8 = omp_get_wtime() - start_time;

    // TODO Write parallel result to file
    std::ofstream parallel_8_result(parallel_8_result_file);
    if (!parallel_8_result.is_open()) {
        std::cerr << "Error: Could not open parallel_8_result.raw" << std::endl;
        return 1;
    }
    parallel_8_result << m << " " << p << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            parallel_8_result << C_parallel[i * p + j] << " ";
        }
        parallel_8_result << std::endl;
    }
    parallel_8_result.close();


    // Validate parallel result
    bool parallel_8_correct = validate_result(parallel_8_result_file, reference_file);
    if (!parallel_8_correct) {
        std::cerr << "Parallel (8 threads) result validation failed for case " << case_number << std::endl;
    }

    memset(C_parallel, 0, m * p * sizeof(float));
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p, 16);
    double parallel_time_16 = omp_get_wtime() - start_time;

    // TODO Write parallel result to file
    std::ofstream parallel_16_result(parallel_16_result_file);
    if (!parallel_16_result.is_open()) {
        std::cerr << "Error: Could not open parallel_16_result.raw" << std::endl;
        return 1;
    }
    parallel_16_result << m << " " << p << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            parallel_16_result << C_parallel[i * p + j] << " ";
        }
        parallel_16_result << std::endl;
    }
    parallel_16_result.close();


    // Validate parallel result
    bool parallel_16_correct = validate_result(parallel_16_result_file, reference_file);
    if (!parallel_16_correct) {
        std::cerr << "Parallel (16 threads) result validation failed for case " << case_number << std::endl;
    }

    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive time: " << naive_time << " seconds\n";
    std::cout << "Blocked_8 time: " << blocked_time_8 << " seconds\n";
    std::cout << "Blocked_16 time: " << blocked_time_16 << " seconds\n";
    std::cout << "Blocked_32 time: " << blocked_time_32 << " seconds\n";
    std::cout << "Blocked_64 time: " << blocked_time_64 << " seconds\n";
    std::cout << "Blocked_128 time: " << blocked_time_128 << " seconds\n";
    std::cout << "Parallel_2 time: " << parallel_time_2 << " seconds\n";
    std::cout << "Parallel_4 time: " << parallel_time_4 << " seconds\n";
    std::cout << "Parallel_8 time: " << parallel_time_8 << " seconds\n";
    std::cout << "Parallel_16 time: " << parallel_time_16 << " seconds\n";
    std::cout << "Blocked_8 speedup: " << (naive_time / blocked_time_8) << "x\n";
    std::cout << "Blocked_16 speedup: " << (naive_time / blocked_time_16) << "x\n";
    std::cout << "Blocked_32 speedup: " << (naive_time / blocked_time_32) << "x\n";
    std::cout << "Blocked_64 speedup: " << (naive_time / blocked_time_64) << "x\n";
    std::cout << "Blocked_128 speedup: " << (naive_time / blocked_time_128) << "x\n";
    std::cout << "Parallel (2 threads) speedup: " << (naive_time / parallel_time_2) << "x\n";
    std::cout << "Parallel (4 threads) speedup: " << (naive_time / parallel_time_4) << "x\n";
    std::cout << "Parallel (8 threads) speedup: " << (naive_time / parallel_time_8) << "x\n";
    std::cout << "Parallel (16 threads) speedup: " << (naive_time / parallel_time_16) << "x\n";
    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;

    return 0;
}