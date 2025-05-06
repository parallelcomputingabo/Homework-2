#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <iomanip>

void naive_matmul(float *C, float *A, float *B, int m, int n, int p) {
    // Implement naive matrix multiplication C = A x B
    // A is m x n, B is n x p, C is m x p
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            // Initialize elements in C
            C[i * p + j] = 0;
            for (int k = 0; k < n; ++k) {
                // Access elements by Row-Major indexing, multiply using given formula
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

void blocked_matmul(float *C, float *A, float *B, int m, int n, int p, int block_size) {
    // Initialize elements in C
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            C[i * p + j] = 0;
        }
    }
    // Implement blocked matrix multiplication
    // Split matrices into blocks by block_size
    for (int ii = 0; ii < m; ii += block_size) {
        for (int jj = 0; jj < p; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                // Perform matrix multiplication on the smaller blocks, same principle as in naive_matmul
                for (int i = ii; i < std::min(ii + block_size, m); ++i) {
                    for (int j = jj; j < std::min(jj + block_size, p); ++j) {
                        for (int k = kk; k < std::min(kk + block_size, n); ++k) {
                            C[i * p + j] += A[i * n + k] * B[k * p + j];
                        }
                    }
                }
            }
        }
    }
    // Use block_size to divide matrices into submatrices
}

void parallel_matmul(float *C, float *A, float *B, int m, int n, int p) {
    // Initialize elements in C
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            C[i * p + j] = 0;
        }
    }
    // Implement parallel matrix multiplication using OpenMP
    // A is m x n, B is n x p, C is m x p
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file, int m, int p) {
   // Implement result validation
   std::ifstream comparison(reference_file);
    if (!comparison.is_open()) {
        // Validate that file opened correctly
        std::cerr << "Unable to open file";
        exit(1);
    }

    std::ifstream res(result_file);
    if (!res.is_open()) {
        // Validate that file opened correctly
        std::cerr << "Unable to open file";
        exit(1);
    }

    float Comp, ResValue;
    // Iterate using the dimensions of C.
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            // Get element from both matrix by index, compare values and throw error if values don't match
            res >> ResValue;
            comparison >> Comp;
            if (ResValue != Comp) {
                std::cerr << "Value mismatch";
                exit(1);
            }
        }
    }
    // Close both files once comparison is done
    comparison.close();
    res.close();
    return true;
}

int main(int argc, char *argv[]) {
    int m, n, p;
    
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

    // Read input0.raw (matrix A)
    std::ifstream FileA(input0_file);
    // Validate that file is opened correctly
    if (!FileA.is_open()) {
        std::cerr << "Error opening file";
        return 1;
    }

    // Read input1.raw (matrix B)
    std::ifstream FileB(input1_file);
    // Validate that file is opened correctly
    if (!FileB.is_open()) {
        std::cerr << "Error opening file";
        return 1;
    }

    FileA >> m >> n;
    FileB >> n >> p;

    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    float* A = (float*)malloc(m * n * sizeof(float));
    // Validate that memory is allocated correctly
    if (A == NULL) {
        std::cerr << "Memory allocation failed";
        return 1;
    }

    float* B = (float*)malloc(n * p * sizeof(float));
    if (B == NULL) {
        std::cerr << "Memory allocation failed";
        return 1;
    }

    //Read matrix elements into A and B (row-major order), close file after reading
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            FileA >> A[i * n + j];
        }
    }
    FileA.close();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            FileB >> B[i * p + j];
        }
    }
    FileB.close();

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    // Write naive result to file
    // Write dimensions and elements to result.raw
    std::ofstream result(result_file);
    // Validate that file is created correctly
    if (!result) {
        std::cerr << "Unable to open file";
        exit(1);
    }

    // Write the dimensions of C on the first line
    result << m << " " << p << std::endl;
    result << std::fixed << std::setprecision(2);
    // Iterate C and write each element to result.raw
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            result << C_naive[i * p + j] << " ";
        }
        result << std::endl;
    }
    // Close file after writing
    result.close();

    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file, m, p);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }


    // Measure performance of blocked_matmul (use block_size = 32 as default)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 64);
    double blocked_time = omp_get_wtime() - start_time;

    // Write blocked result to file
    std::ofstream result_block(result_file);
    // Validate that file is created correctly
    if (!result_block) {
        std::cerr << "Unable to open file";
        exit(1);
    }

    // Write the dimensions of C on the first line
    result_block << m << " " << p << std::endl;
    result_block << std::fixed << std::setprecision(2);
    // Iterate C and write each element to result.raw
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            result_block << C_blocked[i * p + j] << " ";
        }
        result_block << std::endl;
    }
    // Close file after writing
    result_block.close();

    // Validate blocked result
    bool blocked_correct = validate_result(result_file, reference_file, m, p);
    if (!blocked_correct) {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    // Write parallel result to file
    std::ofstream result_para(result_file);
    // Validate that file is created correctly
    if (!result_para) {
        std::cerr << "Unable to open file";
        exit(1);
    }

    // Write the dimensions of C on the first line
    result_para << m << " " << p << std::endl;
    result_para << std::fixed << std::setprecision(2);
    // Iterate C and write each element to result.raw
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            result_para << C_parallel[i * p + j] << " ";
        }
        result_para << std::endl;
    }
    // Close file after writing
    result_para.close();

    // Validate parallel result
    bool parallel_correct = validate_result(result_file, reference_file, m, p);
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